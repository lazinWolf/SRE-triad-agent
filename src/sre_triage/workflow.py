"""The incident-triage flow as a Google ADK ``Workflow`` graph.

    detect -> recall -> coordinator
    coordinator --investigate--> specialists (parallel workers, one ADK LlmAgent per task) -> collect -> coordinator
                --ask---------> question (human-in-the-loop) -> answer -> coordinator
                --review------> review (CrewAI review board over A2A; local critic as fallback)
                --approval----> approval
    review --revise--> coordinator                              (reflection loop, bounded)
           --accept--> approval --(human-in-the-loop RequestInput)--> decide
    decide --approve--> remediate -> verify --done--> learn
                                            --retry-> coordinator   (the fix did not work: rolled back, think again)
           --feedback--> coordinator                             (operator feedback loop)
           --reject--> learn
    approval --skip (nothing to approve)--> learn

The coordinator is an LLM agent that decides each turn whether to investigate further (and with whom), ask the
operator, or conclude. Loops are bounded in state. Workflow state lives in an ADK session stored in SQLite
(DatabaseSessionService), so a case waiting for a human survives restarts and can be resumed from the CLI or
the dashboard.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable

from google.adk.agents.context import Context
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.apps.app import App
from google.adk.events import RequestInput
from google.adk.models import BaseLlm
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.workflow import DEFAULT_ROUTE, START, Workflow, node
from google.genai import types

from . import agents, mcp_servers, remediation, tracing
from .a2a_critic import CARD_PATH, parse_review
from .callbacks import ActivityLog, Usage
from .cases import Cases
from .config import SERVICES, STORES, Settings, get_settings
from .guardrails import Guardrails
from .llm import make_model
from .memory import APP, USER, Memory
from .telemetry import Backends, Unavailable
from .tools import Evidence, Window, baseline_red, compare_red, health_scan

SPECIALISTS = ("metrics", "traces", "logs_changes")  # dispatched when the coordinator gives no valid plan
VERIFY_RECHECKS = 2  # after the settle time, look this many more times before calling a fix failed
NO_REPORT = {"status": "inconclusive", "summary": "The coordinator gave no valid report.", "root_cause_service": None,
             "category": "unknown", "mechanism": "", "confidence": 0, "evidence_ids": [], "hypotheses": [],
             "remediation": None, "alternative": None}


def _brief(evidence: list[dict]) -> str:
    return "\n".join(f"{e['id']} [{e['agent']}] {e['tool']}({json.dumps(e['args'])}): "
                     f"{json.dumps(e['result'], default=str)[:700]}" for e in evidence)


def _findings_text(findings: list[dict]) -> str:
    return "\n".join(f"- {f['agent']} (round {f.get('round', 1)}): "
                     f"{json.dumps({k: v for k, v in f.items() if k not in ('agent', 'round')})}"
                     for f in findings) or "none yet"


class Triage:
    def __init__(self, settings: Settings | None = None, backends: Backends | None = None,
                 model_factory: Callable[[str | None], BaseLlm] | None = None, db_path: str | None = None):
        self.settings = settings or get_settings()
        self.backends = backends or Backends(self.settings)
        self.model_factory = model_factory or (lambda model: make_model(self.settings, model))
        self.cases = Cases(self.settings.data_dir)
        self.memory = Memory(self.settings.data_dir, self.settings.memory_file)
        self.db_url = f"sqlite+aiosqlite:///{db_path or self.settings.data_dir / 'sessions.db'}"
        self._models: dict[str, BaseLlm] = {}
        self.usage: dict[str, Usage] = {}
        self.guard = Guardrails(self.cases, llm_factory=lambda: (
            self.model_factory(None) if self.settings.screen_with_llm else None))
        self.workflow = self._build()

    # ------------------------------------------------------------------ helpers
    def model(self, state) -> BaseLlm:
        case_id = state["case_id"]
        if case_id not in self._models:
            self._models[case_id] = self.model_factory(state.get("model") or None)
        return self._models[case_id]

    def emitter(self, case_id: str):
        return lambda kind, agent, text: self.cases.emit(case_id, kind, agent, text)

    def _enter(self, ctx: Context, name: str) -> tuple[str, Callable]:
        case_id = ctx.state["case_id"]
        self.cases.emit(case_id, "node", name, f"entered {name}")
        return case_id, self.emitter(case_id)

    def _context(self, state) -> str:
        memories = "\n".join(
            f"- [{m['id']}] {m['title']}: cause {m.get('root_cause_service')} / {m.get('category')}; "
            f"lessons: {m.get('lessons', '')}" + (f" (operator marked this {m['feedback']})" if m.get("feedback") else "")
            for m in state.get("memories") or [])
        text = (f"Alert: {state['symptom']}\nFiring alerts: {'; '.join(state.get('alerts') or []) or 'none'}\n"
                f"Anomaly scan (incident vs healthy baseline): "
                f"{'; '.join(state.get('anomalies') or []) or 'no service crosses the anomaly thresholds'}")
        if memories:
            text += f"\nSimilar past incidents (leads to verify, they may not apply):\n{memories}"
        return text

    async def _ask(self, ctx: Context, agent, sub: agents.Submission, prompt: str, run_id: str) -> Any:
        """Run an ADK agent as a dynamic child node of the workflow; return its validated submission."""
        await ctx.run_node(agent, node_input=prompt, run_id=run_id)
        return sub.value

    # ------------------------------------------------------------------ nodes
    async def detect(self, ctx: Context, node_input: Any = None) -> dict:
        case_id, emit = self._enter(ctx, "detect")
        window = Window(**ctx.state["window"])
        health = await asyncio.to_thread(health_scan, self.backends, window)
        anomalies = [f"{svc}: {', '.join(h['anomalies'])}" for svc, h in health.items() if h["anomalies"]]
        try:
            alerts = await asyncio.to_thread(self.backends.prometheus.alerts)
        except Unavailable:
            alerts = []
        ctx.state["anomalies"] = anomalies
        ctx.state["alerts"] = [f"{a['name']} on {a['service']} ({a['state']}): {a['summary']}" for a in alerts]
        emit("result", "detect", "Anomalies: " + ("; ".join(anomalies) if anomalies else "none above thresholds")
             + (f"\nAlerts: {'; '.join(ctx.state['alerts'])}" if alerts else ""))
        return {"anomalies": anomalies, "alerts": len(alerts)}

    async def recall(self, ctx: Context, node_input: Any = None) -> dict:
        case_id, emit = self._enter(ctx, "recall")
        if not self.settings.memory:
            ctx.state["memories"] = []
            emit("result", "memory", "Memory is switched off for this run")
            return {"recalled": 0}
        anomalies = ctx.state.get("anomalies") or []
        # ADK MemoryService search (SQLite full-text); the coordinator can also search it with load_memory
        memories = await self.memory.search(ctx.state["symptom"] + " " + " ".join(anomalies))
        ctx.state["memories"] = memories
        emit("result", "memory", "Recalled from ADK memory: " + (", ".join(m["title"] for m in memories)
                                                                 or "nothing similar"))
        return {"recalled": len(memories)}

    async def coordinator(self, ctx: Context, node_input: Any = None) -> Any:
        """The deciding agent. Each turn it investigates further, asks the operator, or concludes with a report."""
        case_id, emit = self._enter(ctx, "coordinator")
        state = ctx.state
        turn = state["turn"] = (state.get("turn") or 0) + 1
        round_ = state.get("round") or 0
        multi = state["mode"] == "multi"
        feedback, failed = state.get("operator_feedback") or [], state.get("failed_fixes") or []
        if not multi and round_ == 0 or state.get("solo_again"):  # single-agent baseline: one agent with every tool
            state["round"], state["solo_again"] = round_ + 1, False
            question = state["symptom"] + (f"\nOperator feedback: {feedback[-1]}" if feedback else "") \
                + (f"\nA fix was tried and did not help: {failed[-1]}" if failed else "")
            ctx.route = "investigate"
            return [{"agent": "solo", "question": question, "round": round_ + 1}]

        can_plan = multi and round_ < self.settings.max_rounds + len(failed) + len(feedback)
        can_ask = multi and not state.get("questions")
        evidence = state.get("evidence") or []
        prompt = (self._context(state) + f"\n\nHypotheses so far: {state.get('hypotheses') or 'none'}"
                  f"\n\nFindings so far:\n{_findings_text(state.get('findings') or [])}"
                  f"\n\nEvidence:\n{_brief(evidence) or 'none yet'}")
        for qa in state.get("questions") or []:
            prompt += f"\n\nYou asked the operator: {qa['question']}\nOperator answered: {qa['answer']}"
        critiques = state.get("critiques") or []
        if state.get("revising") and critiques:
            review = critiques[-1]
            prompt += (f"\n\nThe review board sent your report back. Issues: {'; '.join(review['issues']) or 'none given'}. "
                       f"Suggested follow-up: {json.dumps(review.get('follow_up') or [])}. Your report was: "
                       f"{json.dumps(state.get('report'))}")
        if state.get("feedback_pending"):
            prompt += f"\n\nThe operator REJECTED the previous diagnosis with this feedback: {feedback[-1]}"
        if failed:
            prompt += ("\n\nA remediation was approved and executed but the system did NOT recover, so it was rolled "
                       f"back: {'; '.join(failed)}. The diagnosis or the fix was wrong; reconsider.")
        prompt += (f"\n\nInvestigation rounds used: {round_}. " + ("" if can_plan else "No more rounds: conclude now. ")
                   + "Decide your next step.")
        state["revising"] = state["feedback_pending"] = False

        sub = agents.Submission()
        known = {e["id"] for e in evidence}
        agent = agents.coordinator(self.model(state), sub, known, self.backends, can_plan=can_plan, can_ask=can_ask,
                                   memory=self.settings.memory)
        try:
            await self._ask(ctx, agent, sub, prompt, f"coordinator-{turn}")
        except Exception:  # the plugin logged the error; fall back below
            sub.value = None
        if sub.value is None and round_ == 0 and can_plan:
            emit("error", "coordinator", "no valid decision; dispatching the standard specialists")
            sub.value, sub.kind = {"hypotheses": [], "tasks": [{"agent": a, "question": state["symptom"]}
                                                                 for a in SPECIALISTS]}, "plan"
        if sub.kind == "plan":
            state["round"] = round_ + 1
            state["hypotheses"] = sub.value["hypotheses"] or state.get("hypotheses")
            tasks = list({t["agent"]: t for t in sub.value["tasks"]}.values())  # one task per specialist per round
            emit("plan", "coordinator", "Decision: investigate. Hypotheses: " + " | ".join(state.get("hypotheses") or [])
                 + "\nTasks: " + "; ".join(f"{t['agent']}: {t['question']}" for t in tasks))
            ctx.route = "investigate"
            return [{**t, "round": round_ + 1} for t in tasks]
        if sub.kind == "question":
            state["question"] = sub.value["question"]
            emit("plan", "coordinator", f"Decision: ask the operator. {sub.value['question']}")
            ctx.route = "ask"
            return {"question": sub.value["question"]}
        report = sub.value if sub.kind == "report" else dict(NO_REPORT)
        checks = sub.problems  # guardrail failures the coordinator did not fix (see agents.coordinator)
        state["report"], state["checks"] = report, checks
        emit("report", "coordinator", f"Decision: conclude. {report['status']}: {report['summary']}"
             + (f"\nChecks failed: {'; '.join(checks)}" if checks else ""))
        nothing_wrong = report["status"] != "diagnosed" and not state.get("anomalies") and not state.get("alerts")
        ctx.route = "review" if multi and not nothing_wrong else "approval"
        return {"status": report["status"]}

    async def specialist(self, ctx: Context, node_input: dict) -> dict:
        """Runs once per task, in parallel (ADK parallel worker): one specialist LlmAgent with its own MCP tools."""
        state = ctx.state
        agent, case_id = node_input["agent"], state["case_id"]
        emit = self.emitter(case_id)
        self.cases.emit(case_id, "node", "specialists", f"{agent} task: {node_input['question']}")
        evidence = Evidence()
        prefix = agent[0].upper()
        evidence.counters[prefix] = sum(1 for e in state.get("evidence") or [] if e["id"].startswith(prefix))

        def log(tool: str, args: dict, result: dict, notes: list[str]) -> None:
            shown = ", ".join(f"{k}={str(v)[:60]}" for k, v in args.items())
            emit("tool", agent, f"{tool}({shown}) -> {result.get('evidence_id') or result.get('error')}")
            for note in notes:
                self.guard.record(case_id, "tool_output", f"{tool} ({agent}): {note}")

        prompt = (self._context(state) + f"\n\nYour task from the coordinator: {node_input['question']}"
                  + (f"\n\nEarlier findings:\n{_findings_text(state['findings'])}" if state.get("findings") else ""))
        sub = agents.Submission()
        steps = self.settings.agent_steps * (2 if agent == "solo" else 1)
        toolsets = agents.mcp_toolsets(agent, state["window"], self.mcp_urls)
        try:
            finding = await self._ask(ctx, agents.specialist(agent, self.model(state), steps, sub, evidence, toolsets, log),
                                      sub, prompt, f"{agent}-{node_input['round']}")
        except Exception:  # a failing agent must not sink the whole investigation (the plugin logged the error)
            finding = None
        finally:
            for toolset in toolsets:
                await toolset.close()
        finding = finding or {"summary": "no finding submitted", "suspect_service": None, "category": "unknown",
                              "confidence": 0, "evidence_ids": [], "ruled_out": []}
        finding.update(agent=agent, round=node_input["round"])
        emit("finding", agent, f"{finding['summary']} (suspect: {finding['suspect_service']}, "
                               f"{finding['category']}, confidence {finding['confidence']})")
        return {"finding": finding, "evidence": evidence.items}

    async def collect(self, ctx: Context, node_input: list[dict]) -> dict:
        """Join of the parallel specialists: their findings and evidence go into the shared state."""
        case_id, emit = self._enter(ctx, "collect")
        state = ctx.state
        state["findings"] = (state.get("findings") or []) + [r["finding"] for r in node_input or []]
        state["evidence"] = (state.get("evidence") or []) + [e for r in node_input or [] for e in r["evidence"]]
        return {"findings": len(state["findings"]), "evidence": len(state["evidence"])}

    async def question(self, ctx: Context, node_input: Any = None):
        case_id, emit = self._enter(ctx, "question")
        text = ctx.state["question"]
        request = {"interrupt_id": f"ask-{case_id}-{len(ctx.state.get('questions') or []) + 1}", "kind": "question",
                   "question": text, "description": text}
        self.cases.update(case_id, status="awaiting_answer", pending=request)
        emit("approval", "operator", f"Waiting for an answer: {text}")
        return RequestInput(interrupt_id=request["interrupt_id"], message=text, payload={"kind": "question"})

    async def answer(self, ctx: Context, node_input: Any = None) -> dict:
        case_id, emit = self._enter(ctx, "answer")
        reply = (node_input or {}).get("answer") if isinstance(node_input, dict) else None
        reply = reply or "No additional information; decide from the evidence."
        ctx.state["questions"] = (ctx.state.get("questions") or []) + [{"question": ctx.state["question"], "answer": reply}]
        self.cases.update(case_id, status="investigating", pending=None)
        emit("approval", "operator", f"answered: {reply}")
        return {"answer": reply}

    async def review(self, ctx: Context, node_input: Any = None):
        """Reflection: the review board checks the report. A CrewAI crew over A2A when configured, else a local LlmAgent."""
        case_id, emit = self._enter(ctx, "review")
        state = ctx.state
        prompt = (f"{self._context(state)}\n\nReport:\n{json.dumps(state['report'])}\n\nAutomatic checks: "
                  f"{state.get('checks') or 'passed'}\n\nFindings:\n{_findings_text(state.get('findings') or [])}"
                  f"\n\nEvidence:\n{_brief(state.get('evidence') or [])}")
        review = None
        if url := self.settings.reviewer_a2a_url:
            emit("a2a", "reviewer", f"Sending the report to the CrewAI review board over A2A at {url}")
            try:
                def only_the_request(_ctx, _name, convert):  # send this prompt, not the whole session history
                    parts = convert(types.Part(text=prompt))
                    return (parts if isinstance(parts, list) else [parts]), None

                remote = RemoteA2aAgent(name="crew_reviewer", agent_card=url.rstrip("/") + CARD_PATH,
                                        description="CrewAI review board served over A2A",
                                        context_builder=only_the_request)
                answer = await ctx.run_node(remote, node_input=prompt, run_id=f"a2a-review-{state.get('turn')}")
                review = parse_review(str(answer))
                review["via"] = "a2a"
                self.usage.setdefault(case_id, Usage()).add_remote("crew_reviewer", review.pop("usage", None), prompt, review)
                emit("a2a", "reviewer", "A2A answer received and validated against the Critique schema")
            except Exception as exc:  # unreachable service or invalid answer: fall back to the local critic
                emit("error", "reviewer", f"A2A review board unavailable ({str(exc)[:200]}); using the local critic")
        if review is None:
            sub = agents.Submission()
            try:
                review = await self._ask(ctx, agents.critic(self.model(state), sub), sub, prompt, f"critic-{state.get('turn')}")
            except Exception:
                review = None
        review = review or {"verdict": "accept", "issues": ["no valid review was produced"], "follow_up": [],
                            "risk": "medium", "risk_notes": ""}
        if state.get("checks"):
            review["verdict"] = "revise"
            review["issues"] = list(state["checks"]) + review["issues"]
        revisions = state.get("revisions") or 0
        if review["verdict"] == "revise" and revisions < self.settings.max_revisions:
            state["revisions"], state["revising"] = revisions + 1, True
        elif review["verdict"] == "revise":
            review["verdict"] = "accept_with_concerns"
            if state.get("checks"):  # never show an unchecked action to the operator
                state["report"] = {**state["report"], "remediation": None, "alternative": None}
        state["critiques"] = (state.get("critiques") or []) + [review]
        emit("critique", "reviewer", f"{review['verdict']}{' (via A2A)' if review.get('via') else ''}, "
             f"risk {review.get('risk', 'low')}: " + ("; ".join(review["issues"]) or "no issues"))
        ctx.route = "revise" if review["verdict"] == "revise" else "accept"
        yield review

    async def approval(self, ctx: Context, node_input: Any = None):
        case_id, emit = self._enter(ctx, "approval")
        report = ctx.state["report"]
        proposals = [p for p in (report.get("remediation"), report.get("alternative")) if p] \
            if report["status"] == "diagnosed" else []
        if not proposals or not report.get("remediation"):
            ctx.route = "skip"
            ctx.state["decision"] = {"approved": False, "reason": "no action proposed"}
            return {"approved": False}
        options = [{"proposal": p, "description": remediation.describe(p, self.backends), "risk": remediation.risk(p)}
                   for p in proposals]
        review = (ctx.state.get("critiques") or [{}])[-1]
        n = len(ctx.state.get("operator_feedback") or []) + len(ctx.state.get("failed_fixes") or []) + 1
        request = {"interrupt_id": f"approve-{case_id}-{n}", "kind": "approval", "options": options, **options[0],
                   "review_risk": review.get("risk"), "review_notes": review.get("risk_notes", "")}
        ctx.state["options"] = proposals
        self.cases.update(case_id, status="awaiting_approval", pending=request)
        emit("approval", "operator", "Waiting for approval: " + " | or: ".join(
            f"{o['description']} (risk {o['risk']})" for o in options))
        # Human in the loop: the workflow stops here until the operator answers this request.
        return RequestInput(interrupt_id=request["interrupt_id"], message=request["description"], payload=options[0]["proposal"])

    async def decide(self, ctx: Context, node_input: Any = None) -> dict:
        case_id, emit = self._enter(ctx, "decide")
        decision = node_input if isinstance(node_input, dict) else {"approved": False}
        options = ctx.state.get("options") or []
        choice = decision.get("choice") or 0
        choice = choice if isinstance(choice, int) and 0 <= choice < len(options) else 0
        ctx.state["decision"] = decision
        ctx.state["chosen"] = options[choice] if options else None
        self.cases.update(case_id, status="investigating", pending=None)
        emit("approval", "operator", (f"approved option {choice + 1}" if decision.get("approved") else "rejected")
             + (f" with feedback: {decision['feedback']}" if decision.get("feedback") else ""))
        feedback = ctx.state.get("operator_feedback") or []
        if decision.get("approved") and ctx.state["chosen"]:
            ctx.route = "approve"
        elif decision.get("feedback") and len(feedback) < 1:
            ctx.state["operator_feedback"] = feedback + [decision["feedback"]]
            ctx.state["feedback_pending"] = True
            ctx.state["solo_again"] = ctx.state["mode"] != "multi"
            ctx.route = "feedback"
        else:
            ctx.route = "reject"
        return decision

    async def remediate(self, ctx: Context, node_input: Any = None) -> dict:
        case_id, emit = self._enter(ctx, "remediate")
        proposal = ctx.state["chosen"]
        self.cases.update(case_id, status="remediating")
        problems = remediation.validate(proposal, self.backends)  # checked again right before acting
        try:
            if problems:
                raise ValueError("; ".join(problems))
            undo = await asyncio.to_thread(remediation.execute, proposal, self.backends)
        except Exception as exc:
            emit("error", "remediate", f"action failed: {exc}")
            ctx.state["outcome"] = "action_failed"
            return {"executed": False}
        ctx.state["undo"] = undo
        ctx.state["outcome"] = "executed"
        emit("action", "remediate", f"Executed: {remediation.describe(proposal, self.backends)}")
        return {"executed": True}

    async def verify(self, ctx: Context, node_input: Any = None) -> dict:
        case_id, emit = self._enter(ctx, "verify")
        state = ctx.state
        ctx.route = "done"
        if state.get("outcome") == "action_failed":
            return {}
        self.cases.update(case_id, status="verifying")
        emit("result", "verify", f"Waiting {self.settings.verify_wait_seconds:.0f}s for the system to settle")
        await asyncio.sleep(self.settings.verify_wait_seconds)
        window = Window(**state["window"])
        target = (state.get("chosen") or {}).get("service")
        watched = {a.split(":")[0] for a in state.get("anomalies") or []} | ({target} if target in SERVICES else set())
        watched = sorted(watched & set(SERVICES)) or ["checkout", "frontend"]

        def still_anomalous() -> dict:
            out, end = {}, time.time()
            for svc in watched:
                now = self.backends.prometheus.service_red(svc, end - 60, end)
                if flags := compare_red(now, baseline_red(self.backends, svc, window)):
                    out[svc] = flags
            if target in SERVICES + STORES and (status := self.backends.docker.inspect(target)["status"]) != "running":
                out[target] = out.get(target, []) + [f"container is {status}"]
            return out

        still = await asyncio.to_thread(still_anomalous)
        for _ in range(VERIFY_RECHECKS):  # a slow start or one unlucky minute is not a failed fix: look again
            if not still:
                break
            emit("result", "verify", f"Not back to baseline yet ({still}); checking again")
            await asyncio.sleep(self.settings.verify_wait_seconds * 0.6)
            still = await asyncio.to_thread(still_anomalous)
        recovered = not still
        if not recovered and state.get("undo"):
            await asyncio.to_thread(remediation.rollback, state["undo"], self.backends)
        emit("result", "verify", "Recovered: watched services are back to baseline" if recovered
             else f"Not recovered ({still}); action rolled back" if state.get("undo") else f"Not recovered ({still})")
        state["recovery"] = {"recovered": recovered, "still_anomalous": still}
        state["outcome"] = "resolved" if recovered else "not_recovered"
        failed = state.get("failed_fixes") or []
        if not recovered and not failed:  # one more attempt: back to the coordinator with what did not work
            state["failed_fixes"] = failed + [remediation.describe(state["chosen"], self.backends)]
            state["solo_again"] = state["mode"] != "multi"
            state["undo"] = None
            self.cases.update(case_id, status="investigating")
            ctx.route = "retry"
        return state["recovery"]

    async def learn(self, ctx: Context, node_input: Any = None) -> dict:
        case_id, emit = self._enter(ctx, "learn")
        state = ctx.state
        report, decision = state.get("report") or {}, state.get("decision") or {}
        if state.get("outcome") and state["outcome"] != "executed":
            outcome = state["outcome"]
        elif report.get("status") != "diagnosed":
            outcome = report.get("status") or "inconclusive"
        elif decision.get("reason") == "no action proposed":
            outcome = "diagnosed_no_action"
        else:
            outcome = "rejected"
        state["outcome"] = outcome
        # Only verified fixes become memories: a diagnosis that was rejected or never proven is not a lesson.
        if self.settings.memory and outcome == "resolved":
            cited = [e for e in state.get("evidence") or [] if e["id"] in set(report.get("evidence_ids") or [])]
            signals = "; ".join(f"{e['tool']}({', '.join(map(str, e['args'].values()))})" for e in cited[:4])
            await self.memory.add({
                "id": case_id, "title": report.get("summary", "")[:120], "symptom": state["symptom"],
                "root_cause_service": report.get("root_cause_service"), "category": report.get("category"),
                "mechanism": report.get("mechanism"), "services": [a.split(":")[0] for a in state.get("anomalies") or []],
                "remediation": state.get("chosen") or report.get("remediation"), "outcome": outcome, "source": "case",
                "lessons": f"{report.get('mechanism', '')} Telling evidence came from: {signals or 'n/a'}."
                           + (f" Operator feedback: {'; '.join(state['operator_feedback'])}" if state.get("operator_feedback") else ""),
                "feedback": None, "created": time.time()})
            emit("result", "memory", "Stored this verified incident in long-term memory")
        self.cases.update(case_id, status=outcome, report=report)
        return {"outcome": outcome}

    # ------------------------------------------------------------------ graph
    def _build(self) -> Workflow:
        # Nodes that run agents (dynamic child nodes) must be re-runnable; the human-in-the-loop nodes must not
        # be, so that on resume the operator's answer becomes their output instead of a new request.
        agentic = {"coordinator", "review"}
        n = {name: node(getattr(self, name), name=name, rerun_on_resume=name in agentic)
             for name in ("detect", "recall", "coordinator", "collect", "question", "answer", "review", "approval",
                          "decide", "remediate", "verify", "learn")}
        n["specialists"] = node(self.specialist, name="specialists", parallel_worker=True, rerun_on_resume=True)
        return Workflow(name="incident_triage", edges=[
            (START, n["detect"], n["recall"], n["coordinator"]),
            (n["coordinator"], {"investigate": n["specialists"], "ask": n["question"], "review": n["review"],
                                "approval": n["approval"]}),
            (n["specialists"], n["collect"], n["coordinator"]),
            (n["question"], n["answer"], n["coordinator"]),
            (n["review"], {"revise": n["coordinator"], "accept": n["approval"]}),
            (n["approval"], {"skip": n["learn"], DEFAULT_ROUTE: n["decide"]}),
            (n["decide"], {"approve": n["remediate"], "feedback": n["coordinator"], "reject": n["learn"]}),
            (n["remediate"], n["verify"]),
            (n["verify"], {"retry": n["coordinator"], "done": n["learn"]}),
        ])

    def edges(self) -> list[tuple[str, str, str | None]]:
        """(from, to, route) for drawing the graph."""
        out = []
        for e in self.workflow.graph.edges:
            route = e.route if not isinstance(e.route, list) else "/".join(map(str, e.route))
            out.append((e.from_node.name, e.to_node.name, None if route in (None, DEFAULT_ROUTE) else str(route)))
        return out

    # ------------------------------------------------------------------ running cases
    @property
    def mcp_urls(self) -> dict[str, str]:
        """The two MCP tool servers (external when SRE_MCP_URL is set, else started in this process)."""
        return mcp_servers.ensure(self.backends, self.settings.mcp_url)

    def open_case(self, symptom: str, *, start: float | None = None, mode: str = "multi", model: str | None = None,
                  scenario: str | None = None, source: str = "operator") -> str:
        """`source` says where the case came from: the operator typed it, or an alert / the fault lab raised it."""
        now = time.time()
        window = {"start": start or now - 180, "end": now, "baseline_seconds": self.settings.baseline_seconds}
        return self.cases.create(symptom=symptom, mode=mode, model=model or self.settings.model,
                                 scenario=scenario, window=window, source=source)

    async def _run(self, case_id: str, resume: dict | None) -> None:
        async with DatabaseSessionService(db_url=self.db_url) as sessions:
            await self._run_with(sessions, case_id, resume)

    async def _run_with(self, sessions: DatabaseSessionService, case_id: str, resume: dict | None) -> None:
        # Plugins are the Runner-wide callbacks: input guardrails first, then monitoring.
        app = App(name=APP, root_agent=self.workflow, plugins=[self.guard, ActivityLog(self.cases, self.usage)])
        runner = Runner(app=app, session_service=sessions,
                        memory_service=self.memory.service if self.settings.memory else None)
        meta = self.cases.get(case_id)
        if resume is None:
            meta["window"]["end"] = time.time()
            self.cases.update(case_id, window=meta["window"])
            await sessions.create_session(app_name=APP, user_id=USER, session_id=case_id, state={
                "case_id": case_id, "symptom": meta["symptom"], "mode": meta["mode"],
                "model": meta.get("model"), "window": meta["window"]})
            message = types.Content(role="user", parts=[types.Part(text=meta["symptom"])])
        else:
            pending = self.cases.get(case_id).get("pending") or {}
            message = types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
                id=pending.get("interrupt_id"), name="adk_request_input", response=resume))])
        with tracing.case_span(case_id, meta, resumed=resume is not None) as span:
            if span:
                self.cases.update(case_id, traces=[*(meta.get("traces") or []), span.trace_id])
            async for _ in runner.run_async(user_id=USER, session_id=case_id, new_message=message):
                pass
            if span:
                session = await sessions.get_session(app_name=APP, user_id=USER, session_id=case_id)
                span.update(output=session.state.get("report") or {"status": self.cases.get(case_id)["status"]})

    def run(self, case_id: str, resume: dict | None = None) -> dict:
        """Run (or resume after an approval decision) until the workflow ends or waits for a human."""
        started = time.time()
        try:
            asyncio.run(self._run(case_id, resume))
        except Exception as exc:
            self.cases.emit(case_id, "error", "workflow", f"{type(exc).__name__}: {exc}")
            self.cases.update(case_id, status="failed", error=str(exc))
        state = self.state(case_id)
        stats = self.cases.get(case_id).get("stats") or {}
        stats["seconds"] = round(stats.get("seconds", 0) + time.time() - started, 1)
        if usage := self.usage.get(case_id):
            stats.update(usage.as_stats())
        stats["guardrail_hits"] = len(self.guard.hits.get(case_id) or [])
        self.cases.update(case_id, stats=stats, report=state.get("report"))
        meta = self.cases.get(case_id)
        if traces := meta.get("traces"):  # how this run ended, as scores to filter on in Langfuse
            tracing.score(traces[-1], "status", meta["status"])
            tracing.score(traces[-1], "guardrail_hits", stats["guardrail_hits"])
        tracing.flush()
        return state

    def session(self, case_id: str):
        """The ADK session of a case: every Runner event plus the final state."""
        async def read():
            async with DatabaseSessionService(db_url=self.db_url) as sessions:
                return await sessions.get_session(app_name=APP, user_id=USER, session_id=case_id)
        return asyncio.run(read())

    def state(self, case_id: str) -> dict:
        async def read() -> dict:
            async with DatabaseSessionService(db_url=self.db_url) as sessions:
                session = await sessions.get_session(app_name=APP, user_id=USER, session_id=case_id)
            return dict(session.state) if session else {}
        return asyncio.run(read())

    def pending(self, case_id: str) -> dict | None:
        """The request a case is waiting on (an approval or a question), if any."""
        meta = self.cases.get(case_id)
        return meta.get("pending") if meta["status"] in ("awaiting_approval", "awaiting_answer") else None

    def decide_case(self, case_id: str, approved: bool, feedback: str = "", choice: int = 0) -> dict:
        return self.run(case_id, resume={"approved": approved, "feedback": feedback, "choice": choice})

    def answer_case(self, case_id: str, answer: str) -> dict:
        return self.run(case_id, resume={"answer": answer})

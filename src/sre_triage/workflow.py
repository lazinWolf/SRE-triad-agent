"""The incident-triage flow as a Google ADK ``Workflow`` graph.

    detect -> recall -> plan -> specialists (parallel workers, one ADK LlmAgent per task) -> synthesize
    synthesize --multi--> critic --revise--> plan              (reflection loop, bounded)
                                  --accept--> approval
    synthesize --single--> approval
    approval --(human-in-the-loop RequestInput)--> decide
    decide --approve--> remediate -> verify -> learn
           --feedback--> plan                                   (operator feedback loop)
           --reject--> learn
    approval --skip (nothing to approve)--> learn

Workflow state lives in an ADK session stored in SQLite (DatabaseSessionService), so a case that is
waiting for approval survives restarts and can be resumed from the CLI or the dashboard.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable

from google.adk.agents.context import Context
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.events import Event, RequestInput
from google.adk.models import BaseLlm
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.workflow import DEFAULT_ROUTE, START, Workflow, node
from google.genai import types

from . import agents, remediation
from .a2a_critic import CARD_PATH, parse_review
from .cases import Cases
from .config import Settings, get_settings
from .llm import make_model
from .memory import APP, USER, Memory
from .telemetry import Backends
from .tools import Toolbox, Window, compare_red, health_scan



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
        self.memory = Memory(self.settings.data_dir)
        self.db_url = f"sqlite+aiosqlite:///{db_path or self.settings.data_dir / 'sessions.db'}"
        self._models: dict[str, BaseLlm] = {}
        self.usage: dict[str, agents.Usage] = {}
        self.workflow = self._build()

    # ------------------------------------------------------------------ helpers
    def model(self, state) -> BaseLlm:
        case_id = state["case_id"]
        if case_id not in self._models:
            self._models[case_id] = self.model_factory(state.get("model") or None)
        return self._models[case_id]

    def emitter(self, case_id: str):
        return lambda kind, agent, text: self.cases.emit(case_id, kind, agent, text)

    def _usage(self, case_id: str) -> agents.Usage:
        return self.usage.setdefault(case_id, agents.Usage())

    def _enter(self, ctx: Context, name: str) -> tuple[str, Callable]:
        case_id = ctx.state["case_id"]
        self.cases.emit(case_id, "node", name, f"entered {name}")
        return case_id, self.emitter(case_id)

    def _context(self, state) -> str:
        memories = "\n".join(
            f"- [{m['id']}] {m['title']}: cause {m.get('root_cause_service')} / {m.get('category')}; "
            f"lessons: {m.get('lessons', '')}" + (f" (operator marked this {m['feedback']})" if m.get("feedback") else "")
            for m in state.get("memories") or []) or "none"
        return (f"Alert: {state['symptom']}\nAnomaly scan (incident vs healthy baseline): "
                f"{'; '.join(state.get('anomalies') or []) or 'no service crosses the anomaly thresholds'}\n"
                f"Similar past incidents (from memory; may not apply):\n{memories}")

    async def _ask(self, ctx: Context, agent, sub: agents.Submission, prompt: str, run_id: str) -> Any:
        """Run an ADK agent as a dynamic child node of the workflow; return its validated submission."""
        await ctx.run_node(agent, node_input=prompt, run_id=run_id)
        return sub.value

    # ------------------------------------------------------------------ nodes
    async def detect(self, ctx: Context, node_input: Any = None) -> dict:
        case_id, emit = self._enter(ctx, "detect")
        health = await asyncio.to_thread(health_scan, self.backends, Window(**ctx.state["window"]))
        anomalies = [f"{svc}: {', '.join(h['anomalies'])}" for svc, h in health.items() if h["anomalies"]]
        ctx.state["anomalies"] = anomalies
        emit("result", "detect", "Anomalies: " + ("; ".join(anomalies) if anomalies else "none above thresholds"))
        return {"anomalies": anomalies}

    async def recall(self, ctx: Context, node_input: Any = None) -> dict:
        case_id, emit = self._enter(ctx, "recall")
        anomalies = ctx.state.get("anomalies") or []
        # ADK MemoryService search (SQLite full-text); the coordinator can also search it with load_memory
        memories = await self.memory.search(ctx.state["symptom"] + " " + " ".join(anomalies))
        ctx.state["memories"] = memories
        emit("result", "memory", "Recalled from ADK memory: " + (", ".join(m["title"] for m in memories)
                                                                 or "nothing similar"))
        return {"recalled": len(memories)}

    async def plan(self, ctx: Context, node_input: Any = None) -> list[dict]:
        case_id, emit = self._enter(ctx, "plan")
        state = ctx.state
        round_ = (state.get("round") or 0) + 1
        state["round"] = round_
        feedback = state.get("operator_feedback") or []
        if state["mode"] == "single":
            question = state["symptom"] + (f"\nOperator feedback: {feedback[-1]}" if feedback else "")
            return [{"agent": "solo", "question": question, "round": round_}]
        critiques = state.get("critiques") or []
        if round_ > 1 and critiques and critiques[-1].get("follow_up") and not state.get("feedback_pending"):
            tasks = critiques[-1]["follow_up"]
            emit("plan", "coordinator", "Following up on the critic: "
                 + "; ".join(f"{t['agent']}: {t['question']}" for t in tasks))
        else:
            prompt = self._context(state) + f"\n\nFindings so far:\n{_findings_text(state.get('findings') or [])}"
            if state.get("feedback_pending"):
                prompt += f"\n\nThe operator REJECTED the previous diagnosis with this feedback: {feedback[-1]}"
                state["feedback_pending"] = False
            sub = agents.Submission()
            plan = await self._ask(ctx, agents.planner(self.model(state), emit, self._usage(case_id), sub),
                                   sub, prompt, f"plan-{round_}")
            if plan:
                tasks, state["hypotheses"] = plan["tasks"], plan["hypotheses"]
            else:
                emit("error", "coordinator", "no valid plan; dispatching every specialist")
                tasks = [{"agent": a, "question": state["symptom"]} for a in ("metrics", "traces", "logs_changes")]
            emit("plan", "coordinator", "Hypotheses: " + " | ".join(state.get("hypotheses") or []) + "\nTasks: "
                 + "; ".join(f"{t['agent']}: {t['question']}" for t in tasks))
        unique = {t["agent"]: t for t in tasks}  # one task per specialist per round
        return [{**t, "round": round_} for t in unique.values()]

    async def specialist(self, ctx: Context, node_input: dict) -> dict:
        """Runs once per task, in parallel (ADK parallel worker): one specialist LlmAgent with its own tools."""
        state = ctx.state
        agent, case_id = node_input["agent"], state["case_id"]
        emit = self.emitter(case_id)
        self.cases.emit(case_id, "node", "specialists", f"{agent} agent started: {node_input['question']}")
        box = Toolbox(self.backends, Window(**state["window"]), emit=emit)
        prefix = agent[0].upper()
        box._counters[prefix] = sum(1 for e in state.get("evidence") or [] if e["id"].startswith(prefix))
        prompt = (self._context(state) + f"\n\nYour task from the coordinator: {node_input['question']}"
                  + (f"\n\nEarlier findings:\n{_findings_text(state['findings'])}" if state.get("findings") else ""))
        sub = agents.Submission()
        steps = self.settings.agent_steps * (2 if agent == "solo" else 1)
        try:
            finding = await self._ask(ctx, agents.specialist(agent, self.model(state), box, emit, self._usage(case_id),
                                                             steps, sub), sub, prompt, f"{agent}-{node_input['round']}")
        except Exception as exc:  # a failing agent must not sink the whole investigation
            emit("error", agent, f"agent failed: {exc}")
            finding = None
        finding = finding or {"summary": "no finding submitted", "suspect_service": None, "category": "unknown",
                              "confidence": 0, "evidence_ids": [], "ruled_out": []}
        finding.update(agent=agent, round=node_input["round"])
        emit("finding", agent, f"{finding['summary']} (suspect: {finding['suspect_service']}, "
                               f"{finding['category']}, confidence {finding['confidence']})")
        return {"finding": finding, "evidence": box.evidence}

    async def synthesize(self, ctx: Context, node_input: list[dict]) -> dict:
        case_id, emit = self._enter(ctx, "synthesize")
        state = ctx.state
        state["findings"] = (state.get("findings") or []) + [r["finding"] for r in node_input or []]
        state["evidence"] = (state.get("evidence") or []) + [e for r in node_input or [] for e in r["evidence"]]
        prompt = (self._context(state) + f"\n\nHypotheses: {state.get('hypotheses')}\n\nFindings:\n"
                  f"{_findings_text(state['findings'])}\n\nEvidence:\n{_brief(state['evidence'])}")
        if state.get("operator_feedback"):
            prompt += f"\n\nOperator feedback on an earlier report: {state['operator_feedback'][-1]}"
        sub = agents.Submission()
        known = {e["id"] for e in state["evidence"]}
        report = await self._ask(ctx, agents.reporter(self.model(state), emit, self._usage(case_id), sub, known,
                                                      self.backends), sub, prompt, f"report-{state['round']}")
        report = report or {"status": "inconclusive", "summary": "The coordinator gave no valid report.",
                            "root_cause_service": None, "category": "unknown", "mechanism": "", "confidence": 0,
                            "evidence_ids": [], "hypotheses": [], "remediation": None}
        checks = self._checks(report, known)
        state["report"], state["checks"] = report, checks
        emit("report", "coordinator", f"{report['status']}: {report['summary']}"
             + (f"\nChecks failed: {'; '.join(checks)}" if checks else ""))
        ctx.route = "critic" if state["mode"] == "multi" else "approval"
        return {"status": report["status"]}

    def _checks(self, report: dict, known: set[str]) -> list[str]:
        """Deterministic guardrails: citations exist, a diagnosis has evidence, the action is allowlisted."""
        problems = []
        cited = set(report.get("evidence_ids") or []) | {i for h in report.get("hypotheses") or []
                                                         for i in h.get("evidence_ids") or []}
        if unknown := sorted(cited - known):
            problems.append(f"cites evidence that does not exist: {', '.join(unknown)}")
        if report["status"] == "diagnosed" and not set(report.get("evidence_ids") or []) & known:
            problems.append("diagnosis cites no evidence")
        return problems + remediation.validate(report.get("remediation"), self.backends)

    async def critic(self, ctx: Context, node_input: Any = None):
        """Reflection: review the report. Uses the Critic A2A service when configured, else a local LlmAgent."""
        case_id, emit = self._enter(ctx, "critic")
        state = ctx.state
        prompt = (f"{self._context(state)}\n\nReport:\n{json.dumps(state['report'])}\n\nAutomatic checks: "
                  f"{state.get('checks') or 'passed'}\n\nFindings:\n{_findings_text(state['findings'])}"
                  f"\n\nEvidence:\n{_brief(state['evidence'])}")
        review = None
        if url := self.settings.critic_a2a_url:
            # A2A: the remote agent reads the conversation from the session, so put the request there first.
            yield Event(author=self.workflow.name, content=types.Content(role="user", parts=[types.Part(text=prompt)]))
            emit("a2a", "critic", f"Sending the report to the Critic A2A service at {url}")
            try:
                remote = RemoteA2aAgent(name="remote_critic", agent_card=url.rstrip("/") + CARD_PATH,
                                        description="Critic served over A2A")
                answer = await ctx.run_node(remote, node_input=prompt, run_id=f"a2a-critic-{state['round']}")
                review = parse_review(str(answer))
                review["via"] = "a2a"
                emit("a2a", "critic", "A2A answer received and validated against the Critique schema")
            except Exception as exc:  # unreachable service or invalid answer: fall back to the local critic
                emit("error", "critic", f"A2A critic unavailable ({str(exc)[:200]}); using the local critic")
        if review is None:
            sub = agents.Submission()
            review = await self._ask(ctx, agents.critic(self.model(state), emit, self._usage(case_id), sub),
                                     sub, prompt, f"critic-{state['round']}")
        review = review or {"verdict": "accept", "issues": ["critic gave no valid review"], "follow_up": []}
        if state.get("checks"):
            review["verdict"] = "revise"
            review["issues"] = list(state["checks"]) + review["issues"]
        revisions = state.get("revisions") or 0
        if review["verdict"] == "revise" and revisions < self.settings.max_revisions:
            state["revisions"] = revisions + 1
        elif review["verdict"] == "revise":
            review["verdict"] = "accept_with_concerns"
            if state.get("checks"):  # never show an unchecked action to the operator
                state["report"] = {**state["report"], "remediation": None}
        state["critiques"] = (state.get("critiques") or []) + [review]
        emit("critique", "critic", f"{review['verdict']}{' (via A2A)' if review.get('via') else ''}: "
             + ("; ".join(review["issues"]) or "no issues"))
        ctx.route = "revise" if review["verdict"] == "revise" else "accept"
        yield review

    async def approval(self, ctx: Context, node_input: Any = None):
        case_id, emit = self._enter(ctx, "approval")
        report = ctx.state["report"]
        proposal = report.get("remediation") if report["status"] == "diagnosed" else None
        if not proposal:
            ctx.route = "skip"
            ctx.state["decision"] = {"approved": False, "reason": "no action proposed"}
            return {"approved": False}
        n = len(ctx.state.get("operator_feedback") or []) + 1
        request = {"interrupt_id": f"approve-{case_id}-{n}", "proposal": proposal,
                   "description": remediation.describe(proposal)}
        self.cases.update(case_id, status="awaiting_approval", pending=request)
        emit("approval", "operator", f"Waiting for approval: {request['description']}")
        # Human in the loop: the workflow stops here until the operator answers this request.
        return RequestInput(interrupt_id=request["interrupt_id"], message=request["description"], payload=proposal)

    async def decide(self, ctx: Context, node_input: Any = None) -> dict:
        case_id, emit = self._enter(ctx, "decide")
        decision = node_input if isinstance(node_input, dict) else {"approved": False}
        ctx.state["decision"] = decision
        self.cases.update(case_id, status="investigating", pending=None)
        emit("approval", "operator", ("approved" if decision.get("approved") else "rejected")
             + (f" with feedback: {decision['feedback']}" if decision.get("feedback") else ""))
        feedback = ctx.state.get("operator_feedback") or []
        if decision.get("approved"):
            ctx.route = "approve"
        elif decision.get("feedback") and len(feedback) < 1:
            ctx.state["operator_feedback"] = feedback + [decision["feedback"]]
            ctx.state["feedback_pending"] = True
            ctx.route = "feedback"
        else:
            ctx.route = "reject"
        return decision

    async def remediate(self, ctx: Context, node_input: Any = None) -> dict:
        case_id, emit = self._enter(ctx, "remediate")
        proposal = ctx.state["report"]["remediation"]
        self.cases.update(case_id, status="remediating")
        try:
            undo = await asyncio.to_thread(remediation.execute, proposal, self.backends)
        except Exception as exc:
            emit("error", "remediate", f"action failed: {exc}")
            ctx.state["outcome"] = "action_failed"
            return {"executed": False}
        ctx.state["undo"] = undo
        ctx.state["outcome"] = "executed"
        emit("action", "remediate", f"Executed: {remediation.describe(proposal)}")
        return {"executed": True}

    async def verify(self, ctx: Context, node_input: Any = None) -> dict:
        case_id, emit = self._enter(ctx, "verify")
        state = ctx.state
        if state.get("outcome") == "action_failed":
            return {}
        self.cases.update(case_id, status="verifying")
        emit("result", "verify", f"Waiting {self.settings.verify_wait_seconds:.0f}s for the system to settle")
        await asyncio.sleep(self.settings.verify_wait_seconds)
        w, end = state["window"], time.time()
        watched = [a.split(":")[0] for a in state.get("anomalies") or []] or ["checkout", "frontend"]

        def still_anomalous() -> dict:
            out = {}
            for svc in watched:
                now = self.backends.prometheus.service_red(svc, end - 60, end)
                before = self.backends.prometheus.service_red(svc, w["start"] - w["baseline_seconds"], w["start"])
                if flags := compare_red(now, before):
                    out[svc] = flags
            return out

        still = await asyncio.to_thread(still_anomalous)
        recovered = not still
        if not recovered and state.get("undo"):
            await asyncio.to_thread(remediation.rollback, state["undo"], self.backends)
        emit("result", "verify", "Recovered: watched services are back to baseline" if recovered
             else f"Not recovered ({still}); action rolled back" if state.get("undo") else f"Not recovered ({still})")
        state["recovery"] = {"recovered": recovered, "still_anomalous": still}
        state["outcome"] = "resolved" if recovered else "not_recovered"
        return state["recovery"]

    async def learn(self, ctx: Context, node_input: Any = None) -> dict:
        case_id, emit = self._enter(ctx, "learn")
        state = ctx.state
        report, decision = state.get("report") or {}, state.get("decision") or {}
        if state.get("outcome") and state["outcome"] != "executed":
            outcome = state["outcome"]
        elif report.get("status") != "diagnosed":
            outcome = "inconclusive"
        elif decision.get("reason") == "no action proposed":
            outcome = "diagnosed_no_action"
        else:
            outcome = "rejected"
        state["outcome"] = outcome
        if report.get("status") == "diagnosed":
            await self.memory.add({
                "id": case_id, "title": report.get("summary", "")[:120], "symptom": state["symptom"],
                "root_cause_service": report.get("root_cause_service"), "category": report.get("category"),
                "mechanism": report.get("mechanism"), "services": [a.split(":")[0] for a in state.get("anomalies") or []],
                "remediation": report.get("remediation"), "outcome": outcome, "source": "case",
                "lessons": "; ".join(state.get("operator_feedback") or []) or report.get("mechanism", ""),
                "feedback": "incorrect" if outcome == "rejected" else None, "created": time.time()})
            emit("result", "memory", "Stored this incident in long-term memory")
        self.cases.update(case_id, status=outcome, report=report)
        return {"outcome": outcome}

    # ------------------------------------------------------------------ graph
    def _build(self) -> Workflow:
        # Nodes that run agents (dynamic child nodes) must be re-runnable; the approval node must not be,
        # so that on resume the operator's answer becomes its output instead of a new request.
        agentic = {"plan", "synthesize", "critic"}
        n = {name: node(getattr(self, name), name=name, rerun_on_resume=name in agentic)
             for name in ("detect", "recall", "plan", "synthesize", "critic", "approval", "decide",
                          "remediate", "verify", "learn")}
        n["specialists"] = node(self.specialist, name="specialists", parallel_worker=True, rerun_on_resume=True)
        return Workflow(name="incident_triage", edges=[
            (START, n["detect"], n["recall"], n["plan"], n["specialists"], n["synthesize"]),
            (n["synthesize"], {"critic": n["critic"], "approval": n["approval"]}),
            (n["critic"], {"revise": n["plan"], "accept": n["approval"]}),
            (n["approval"], {"skip": n["learn"], DEFAULT_ROUTE: n["decide"]}),
            (n["decide"], {"approve": n["remediate"], "feedback": n["plan"], "reject": n["learn"]}),
            (n["remediate"], n["verify"], n["learn"]),
        ])

    def edges(self) -> list[tuple[str, str, str | None]]:
        """(from, to, route) for drawing the graph."""
        out = []
        for e in self.workflow.graph.edges:
            route = e.route if not isinstance(e.route, list) else "/".join(map(str, e.route))
            out.append((e.from_node.name, e.to_node.name, None if route in (None, DEFAULT_ROUTE) else str(route)))
        return out

    # ------------------------------------------------------------------ running cases
    def open_case(self, symptom: str, *, start: float | None = None, mode: str = "multi",
                  model: str | None = None, scenario: str | None = None) -> str:
        now = time.time()
        window = {"start": start or now - 180, "end": now, "baseline_seconds": self.settings.baseline_seconds}
        return self.cases.create(symptom=symptom, mode=mode, model=model or self.settings.model,
                                 scenario=scenario, window=window)

    async def _run(self, case_id: str, resume: dict | None) -> None:
        async with DatabaseSessionService(db_url=self.db_url) as sessions:
            await self._run_with(sessions, case_id, resume)

    async def _run_with(self, sessions: DatabaseSessionService, case_id: str, resume: dict | None) -> None:
        runner = Runner(node=self.workflow, app_name=APP, session_service=sessions,
                        memory_service=self.memory.service)
        if resume is None:
            meta = self.cases.get(case_id)
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
        async for _ in runner.run_async(user_id=USER, session_id=case_id, new_message=message):
            pass

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
            stats.update(llm_calls=usage.calls, tokens=usage.tokens, models_used=usage.models)
        self.cases.update(case_id, stats=stats, report=state.get("report"))
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
        """The approval request a case is waiting on, if any."""
        meta = self.cases.get(case_id)
        return meta.get("pending") if meta["status"] == "awaiting_approval" else None

    def decide_case(self, case_id: str, approved: bool, feedback: str = "") -> dict:
        return self.run(case_id, resume={"approved": approved, "feedback": feedback})

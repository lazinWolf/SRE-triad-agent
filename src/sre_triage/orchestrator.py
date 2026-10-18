"""Investigation control loop.

Multi-agent mode: the coordinator plans, specialists investigate in parallel,
the coordinator reconciles and either delegates targeted follow-ups or
concludes. Single-agent mode gives one agent every tool under the same budget.
Budgets, stop rules and validation are enforced here in code; the models decide
what to look at and what it means.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable

from .agents import prompts
from .agents.runtime import LlmBudget, ModelNotConfigured, Submission, make_model, run_agent
from .agents.team import make_coordinator, make_critic, make_single_agent, make_specialist
from .config import Settings
from .ledger import Ledger
from .memory import search_memory
from .models import CoordinatorDecision, Critique, Finding, Hypothesis, Report, Task
from .remediation import Remediator, describe
from .tools import Backends, CaseToolkit, RunBudget

log = logging.getLogger(__name__)


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def case_context(case: dict[str, Any], ledger: Ledger) -> str:
    t0 = case["incident_start"]
    lines = [
        f"Affected service: {case['service']}",
        f"Symptom: {case['symptom']}",
        f"Severity: {case['severity']} (source: {case['source']})",
        f"Incident window: t+0s .. t{case['incident_end'] - t0:+.0f}s  ({_iso(t0)} .. {_iso(case['incident_end'])})",
        "t+0s is the start of the investigation window (a lookback before detection), NOT the symptom onset. "
        "Establish when symptoms actually began from telemetry (e.g. the p95 trend) before judging change timing.",
        f"Baseline window: t{case['baseline_start'] - t0:+.0f}s .. t+0s",
        "Environment: OpenTelemetry Astronomy Shop in Docker Compose (frontend, checkout, cart, product-catalog, "
        "currency, payment, shipping, quote, email, recommendation, ad; valkey-cart cache; flagd feature flags).",
    ]
    history = []
    for run in ledger.runs(case["id"]):
        if run["id"] == case.get("run_id") or not run.get("stats"):
            continue
        summary = (run["stats"] or {}).get("summary")
        if summary:
            history.append(f"- Earlier run {run['id']} ({run['status']}): {summary}")
    for action in ledger.actions(case["id"]):
        if action["status"] in ("executed", "failed", "rolled_back"):
            history.append(f"- Action {describe(action['proposal'])}: {action['status']}"
                           f"{' at t' + format(action['executed_at'] - t0, '+.0f') + 's' if action.get('executed_at') else ''}")
    for check in ledger.recovery_checks(case["id"]):
        if check["status"] not in ("pending",):
            history.append(f"- Recovery check after action: {check['status']} {json.dumps((check['result'] or {}).get('checks', []))[:400]}")
    if history:
        lines.append("History of this case (earlier conclusions are not evidence for this run):")
        lines += history
    return "\n".join(lines)


def feedback_context(ledger: Ledger, case_id: str) -> str:
    items = ledger.feedback(case_id)
    return "\n".join(f"- {f['text']}" for f in items) if items else "(none)"


def board_text(board: list[Hypothesis]) -> str:
    if not board:
        return "(empty)"
    return "\n".join(json.dumps(h.model_dump(), separators=(",", ":")) for h in board)


def findings_text(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return "(none yet)"
    out = []
    for f in findings:
        out.append(f"[round {f['round']}] {f['agent']} — task: {f['task']}\n{json.dumps(f['finding'], separators=(',', ':'))}")
    return "\n".join(out)


def ledger_text(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "(no evidence yet)"
    return "\n".join(f"{e['id']}: {e['agent']} {e['tool']}({json.dumps(e['args'], separators=(',', ':'))}) -> {e['status']}"
                     for e in evidence[-80:])


def fallback_report(reason: str, board: list[Hypothesis]) -> Report:
    open_items = [h for h in board if h.status in ("candidate", "supported")]
    return Report(
        summary=f"Investigation ended without a validated conclusion: {reason}",
        diagnosis_status="insufficient_evidence", confidence="low",
        next_checks=[f"Verify {h.id}: {h.statement}" for h in open_items[:4]] or
                    ["Inspect the evidence ledger and rerun when telemetry is available."],
        limitations=[reason, "No remediation is proposed without a validated diagnosis."],
    )


class Investigation:
    def __init__(self, ledger: Ledger, settings: Settings, case_id: str, *, backends: Backends | None = None,
                 model_override: Any = None, on_progress: Callable[[str], None] | None = None):
        self.ledger, self.settings, self.case_id = ledger, settings, case_id
        self.backends = backends
        self.model_override = model_override
        self.on_progress = on_progress
        self.board: list[Hypothesis] = []
        self.findings: list[dict[str, Any]] = []
        self.critiques: list[dict[str, Any]] = []
        self.max_critiques = 1
        self.usage_start = _usage_snapshot()

    def event(self, actor: str, kind: str, message: str, data: dict | None = None) -> None:
        self.ledger.event(self.case_id, self.run_id, actor, kind, message, data)
        if self.on_progress:
            self.on_progress(f"[{actor}] {message}")

    async def run(self) -> dict[str, Any]:
        case = self.ledger.case(self.case_id)
        mode = case["mode"]
        from .llm import model_for_role

        model_name = ("scripted" if self.model_override is not None else
                      ", ".join(sorted({m for m in (model_for_role(self.settings, r, case.get("model")) for r in
                                                    ("coordinator", "metrics", "single_agent")) if m})) or "unconfigured")
        run = self.ledger.begin_run(self.case_id, mode, model_name)
        self.run_id = run["id"]
        case = self.ledger.case(self.case_id)
        own_backends = self.backends is None
        backends = self.backends or Backends.from_settings(self.settings)
        b = self.settings.budget
        self.tool_budget = RunBudget(max_tool_calls=b.max_tool_calls)
        self.llm_budget = LlmBudget(max_calls=b.max_llm_calls)
        memory = (lambda q, s: search_memory(self.ledger, q, s, exclude_case_id=self.case_id)) if case["use_memory"] else None
        self.tk = CaseToolkit(self.ledger, case, self.run_id, backends, self.tool_budget,
                              use_memory=bool(case["use_memory"]), memory_search=memory)
        self.case = case
        started = time.monotonic()
        report: Report | None = None
        error = None
        status = "completed"
        rounds = 0
        try:
            make_model(self.settings, self.model_override, "probe", case.get("model"))
            async with asyncio.timeout(b.wall_seconds):
                if mode == "single_agent":
                    report = await self._single_agent()
                else:
                    report, rounds = await self._multi_agent()
        except ModelNotConfigured as exc:
            error, status = str(exc), "failed"
        except TimeoutError:
            error, status = f"Wall-clock budget of {b.wall_seconds:.0f}s exhausted", "budget_exhausted"
        except asyncio.CancelledError:
            self._finish(fallback_report("investigation cancelled", self.board), "cancelled", "cancelled", started, rounds)
            raise
        except Exception as exc:
            log.exception("investigation failed")
            error, status = f"{type(exc).__name__}: {str(exc)[:500]}", "failed"
        if report is None:
            report = fallback_report(error or "no report produced", self.board)
            self.event("coordinator", "error", report.summary)
        result = self._finish(report, status, error, started, rounds)
        if own_backends:
            await backends.aclose()
        return result

    # ------------------------------------------------------------------ multi-agent
    async def _multi_agent(self) -> tuple[Report, int]:
        b = self.settings.budget
        decision = await self._coordinate("plan", 0)
        rounds = 0
        if decision is None:
            tasks = [Task(agent="metrics", question="Which services and resources changed versus the baseline, and how?"),
                     Task(agent="traces", question="Where in the request path is time or error introduced for the affected service?"),
                     Task(agent="logs_changes", question="What errors, container events and configuration changes align with the incident?")]
            self.event("coordinator", "decision", "Coordinator gave no valid plan; using the default triage tasks")
        elif decision.next_step == "conclude":
            return decision.report, 0
        else:
            tasks = decision.tasks
        critiques = 0
        phase = "reconcile"
        while True:
            rounds += 1
            await self._delegate(tasks[: b.max_tasks_per_round], rounds)
            last = (rounds >= b.max_rounds or self.tool_budget.tool_calls >= b.max_tool_calls
                    or self.llm_budget.calls >= b.max_llm_calls - 2)
            decision = await self._coordinate("final" if last else phase, rounds)
            if decision is None and not last:
                # One more chance, constrained to concluding, before giving up on the coordinator.
                decision = await self._coordinate("final", rounds)
            if decision is None:
                return fallback_report("the coordinator did not submit a valid decision", self.board), rounds
            if decision.next_step == "investigate":
                tasks, phase = decision.tasks, "reconcile"
                continue
            # Reflection: an independent critic tries to falsify a diagnosis before it is accepted.
            report = decision.report
            if report.diagnosis_status != "diagnosed" or critiques >= self.max_critiques:
                return report, rounds
            critiques += 1
            critique = await self._critique(report, rounds)
            if critique is None or critique.verdict == "accept":
                return report, rounds
            if last or not critique.requested_checks:
                # No budget to resolve the challenge: let the coordinator revise its conclusion once.
                revised = await self._coordinate("final", rounds)
                return (revised.report if revised else report), rounds
            tasks, phase = critique.requested_checks, "revise"

    async def _coordinate(self, phase: str, round_number: int) -> CoordinatorDecision | None:
        b = self.settings.budget
        instruction = prompts.COORDINATOR_TEMPLATE.format(
            rules=prompts.SHARED_RULES, case=case_context(self.case, self.ledger),
            feedback=feedback_context(self.ledger, self.case_id), phase=phase.upper(), round=round_number,
            max_rounds=b.max_rounds, tool_calls=self.tool_budget.tool_calls, max_tool_calls=b.max_tool_calls,
            phase_instructions=prompts.PHASE_INSTRUCTIONS[phase], board=board_text(self.board),
            findings=findings_text(self.findings), max_tasks=b.max_tasks_per_round,
            ledger=ledger_text(self.ledger.evidence(self.case_id, self.run_id)))
        submission = Submission()
        agent = make_coordinator(make_model(self.settings, self.model_override, "coordinator", self.case.get("model")), self.tk,
                                 self.llm_budget, instruction, submission, phase=phase,
                                 max_tasks=b.max_tasks_per_round, flagd_file=self.settings.flagd_file)
        self.event("coordinator", "phase", f"Coordinator {phase} (round {round_number})")
        result = await run_agent(agent, "Proceed.", submission, max_llm_calls=12,
                                 on_text=self._agent_text, nudge="Call submit_decision now.")
        if result.error:
            self.event("coordinator", "error", f"Coordinator error: {result.error}")
        decision: CoordinatorDecision | None = result.submitted
        if decision is None:
            if submission.last_errors:
                self.event("coordinator", "validation", "Decision rejected: " + "; ".join(submission.last_errors)[:1500])
            return None
        self.board = decision.hypotheses
        self.ledger.record_hypotheses(self.case_id, self.run_id, round_number, [h.model_dump() for h in self.board])
        self.event("coordinator", "decision", decision.reasoning,
                   {"next_step": decision.next_step, "tasks": [t.model_dump() for t in decision.tasks],
                    "board": [{"id": h.id, "status": h.status, "statement": h.statement} for h in self.board]})
        return decision

    def _models_used(self) -> dict[str, int]:
        now = _usage_snapshot()
        return {m: n - self.usage_start.get(m, 0) for m, n in now.items() if n - self.usage_start.get(m, 0) > 0}

    async def _critique(self, report: Report, round_number: int) -> Critique | None:
        instruction = prompts.CRITIC_TEMPLATE.format(
            rules=prompts.SHARED_RULES, case=case_context(self.case, self.ledger),
            feedback=feedback_context(self.ledger, self.case_id),
            report=json.dumps(report.model_dump(), separators=(",", ":")), board=board_text(self.board),
            findings=findings_text(self.findings))
        submission = Submission()
        agent = make_critic(make_model(self.settings, self.model_override, "critic", self.case.get("model")),
                            self.tk, self.llm_budget, instruction, submission)
        self.event("critic", "phase", "Critic reviewing the proposed diagnosis")
        result = await run_agent(agent, "Review the proposed diagnosis.", submission, max_llm_calls=8,
                                 on_text=self._agent_text, nudge="Call submit_critique now.")
        critique: Critique | None = result.submitted
        if critique is None:
            self.event("critic", "error", f"Critic produced no review: {result.error or 'no submission'}")
            return None
        payload = critique.model_dump()
        self.critiques.append({"round": round_number, **payload})
        # The critique becomes shared context for the coordinator's next decision.
        self.findings.append({"round": round_number, "agent": "critic", "task": "CRITIC REVIEW of the proposed diagnosis",
                              "finding": payload})
        self.ledger.record_finding(self.case_id, self.run_id, round_number, "critic", "Review of the proposed diagnosis", payload)
        self.event("critic", "critique", f"{critique.verdict.upper()}: {critique.summary}", payload)
        return critique

    async def _delegate(self, tasks: list[Task], round_number: int) -> None:
        for task in tasks:
            self.event("coordinator", "delegation", f"{task.agent}: {task.question}", task.model_dump())
        results = await asyncio.gather(*(self._specialist(task, round_number) for task in tasks))
        for task, finding in zip(tasks, results):
            payload = finding.model_dump()
            self.findings.append({"round": round_number, "agent": task.agent, "task": task.question, "finding": payload})
            self.ledger.record_finding(self.case_id, self.run_id, round_number, task.agent, task.question, payload)
            self.event(task.agent, "finding", finding.summary, payload)

    async def _specialist(self, task: Task, round_number: int) -> Finding:
        relevant = [h.model_dump() for h in self.board if not task.hypothesis_ids or h.id in task.hypothesis_ids]
        instruction = prompts.SPECIALIST_TEMPLATE.format(
            rules=prompts.SHARED_RULES, focus=prompts.ROLE_FOCUS[task.agent],
            case=case_context(self.case, self.ledger), question=task.question,
            hypotheses=json.dumps(relevant, separators=(",", ":")) if relevant else "(none yet)",
            feedback=feedback_context(self.ledger, self.case_id))
        submission = Submission()
        agent = make_specialist(task.agent, make_model(self.settings, self.model_override, task.agent, self.case.get("model")),
                                self.tk, self.llm_budget, instruction, submission)
        result = await run_agent(agent, "Begin.", submission, max_llm_calls=12, on_text=self._agent_text,
                                 nudge="Call submit_findings now with what you have found.")
        if result.submitted is not None:
            return result.submitted
        reason = result.error or "no findings were submitted"
        self.event(task.agent, "error", f"{task.agent} produced no valid findings: {reason}")
        return Finding(summary=f"No findings: {reason}", limitations=[reason])

    # ------------------------------------------------------------------ single agent
    async def _single_agent(self) -> Report:
        b = self.settings.budget
        instruction = prompts.SINGLE_AGENT_TEMPLATE.format(
            rules=prompts.SHARED_RULES, case=case_context(self.case, self.ledger),
            feedback=feedback_context(self.ledger, self.case_id), max_tool_calls=b.max_tool_calls)
        submission = Submission()
        agent = make_single_agent(make_model(self.settings, self.model_override, "single_agent", self.case.get("model")), self.tk,
                                  self.llm_budget, instruction, submission, flagd_file=self.settings.flagd_file,
                                  step_limit=max(4, b.max_llm_calls - 4))
        self.event("single_agent", "phase", "Single-agent investigation started")
        result = await run_agent(agent, "Begin.", submission, max_llm_calls=b.max_llm_calls,
                                 on_text=self._agent_text, nudge="Call submit_report now with what you have found.")
        if result.error:
            self.event("single_agent", "error", result.error)
        if result.submitted is None:
            return fallback_report("the agent did not submit a valid report", self.board)
        self.board = result.submitted["hypotheses"]
        self.ledger.record_hypotheses(self.case_id, self.run_id, 1, [h.model_dump() for h in self.board])
        return result.submitted["report"]

    # ------------------------------------------------------------------ persistence
    def _drain_model_notices(self) -> None:
        from .llm import NOTICES

        while NOTICES:
            self.event("model", "model_switch", NOTICES.popleft())

    def _agent_text(self, agent: str, text: str, kind: str = "agent_message") -> None:
        self._drain_model_notices()
        text = text.strip()
        if text:
            self.ledger.event(self.case_id, self.run_id, agent, kind, text[:4000])

    def _finish(self, report: Report, status: str, error: str | None, started: float, rounds: int) -> dict[str, Any]:
        self._drain_model_notices()
        payload = report.model_dump()
        payload["hypotheses"] = [h.model_dump() for h in self.board]
        payload["critiques"] = self.critiques
        case_status = "needs_review"
        if report.remediation is not None and status == "completed":
            proposal = report.remediation.model_dump()
            try:
                rollback = Remediator(self.tk.b.docker, self.settings.flagd_file).rollback_plan(proposal)
            except Exception as exc:
                rollback = {"available": False, "reason": f"rollback plan failed: {exc}"}
            action_id = self.ledger.propose_action(self.case_id, self.run_id, proposal, rollback)
            payload["action_id"] = action_id
            case_status = "awaiting_approval"
            self.event("coordinator", "proposal", f"Remediation proposed for operator approval: {describe(proposal)}",
                       {"action_id": action_id, "rollback": rollback})
        stats = {
            **self.tk.stats(), **self.llm_budget.stats(), "rounds": rounds,
            "duration_s": round(time.monotonic() - started, 1), "summary": report.summary[:500],
            "diagnosis_status": report.diagnosis_status, "models_used": self._models_used(),
        }
        self.ledger.set_case(self.case_id, status=case_status, report=payload, error=error)
        self.ledger.finish_run(self.run_id, status, stats, error)
        self.event("coordinator", "complete",
                   f"Investigation {status}: {report.diagnosis_status} after {stats['tool_calls']} tool calls, "
                   f"{stats['llm_calls']} model calls, {stats['duration_s']}s")
        return {"case_id": self.case_id, "run_id": self.run_id, "status": status, "report": payload, "stats": stats,
                "error": error}


def _usage_snapshot() -> dict[str, int]:
    from .llm import USAGE

    return dict(USAGE)


async def investigate(ledger: Ledger, settings: Settings, case_id: str, **kwargs: Any) -> dict[str, Any]:
    return await Investigation(ledger, settings, case_id, **kwargs).run()

"""Agent definitions: three specialists, the coordinator, and the single-agent baseline."""

from __future__ import annotations

import json
from typing import Any, Callable

from google.adk.tools import ToolContext
from pydantic import BaseModel, ValidationError

from ..models import CoordinatorDecision, Critique, Finding, Hypothesis, Report
from ..tools import CaseToolkit
from ..validation import EvidenceIndex, check_decision, check_finding, check_hypotheses, check_report
from .runtime import LlmBudget, Submission, build_agent


def telemetry_tools(tk: CaseToolkit, agent: str, kinds: set[str]) -> list[Callable]:
    """Tool functions bound to one agent name. Docstrings are what the model sees."""

    async def service_overview(window: str = "incident") -> dict:
        """RED metrics (request rate, error ratio, p95 latency) for every service, ranked by how much worse
        they are than the baseline. Every tool's window is 'incident', 'baseline', 'full', or a relative span
        such as 't-120s..t+60s'."""
        return await tk.service_overview(agent, window)

    async def service_red(service: str, window: str = "incident", start_offset_s: float = 0,
                          end_offset_s: float = 0) -> dict:
        """Request rate, error ratio, p50/p95 latency and a p95 trend for one service, compared with baseline.
        window: 'incident', 'baseline', 'full', or a relative span like 't-120s..t+60s' (seconds from incident start)."""
        return await tk.service_red(agent, service, window, start_offset_s, end_offset_s)

    async def operation_latency(service: str, span_kind: str = "server", window: str = "incident") -> dict:
        """Per-operation rate, error ratio and p95 for one service versus baseline. span_kind='server' for requests
        it handles, 'client' for calls it makes to dependencies (shows which downstream call got slower)."""
        return await tk.operation_latency(agent, service, span_kind, window)

    async def container_resources(service: str, window: str = "incident") -> dict:
        """Container CPU utilisation, CPU limit, CPU throttling ratio, memory and restarts versus baseline."""
        return await tk.container_resources(agent, service, window)

    async def cache_health(window: str = "incident") -> dict:
        """Valkey cache (used by cart) hit ratio, command rate, clients and memory versus baseline."""
        return await tk.cache_health(agent, window)

    async def find_traces(service: str, window: str = "incident", min_duration_ms: float = 0,
                          errors_only: bool = False, operation: str = "", limit: int = 10) -> dict:
        """Find traces through a service (optionally slow ones or failed ones) and the share of self time
        spent in each service across them. Self time localizes where latency originates."""
        return await tk.find_traces(agent, service, window, min_duration_ms, errors_only, operation, limit)

    async def compare_trace_profiles(service: str, operation: str = "") -> dict:
        """Compare average per-service self time in traces through a service between baseline and incident."""
        return await tk.compare_trace_profiles(agent, service, operation)

    async def trace_detail(trace_id: str) -> dict:
        """Breakdown of one trace: duration, self time by service, slowest spans and error messages."""
        return await tk.trace_detail(agent, trace_id)

    async def search_logs(service: str = "", window: str = "incident", min_severity: str = "WARN",
                          text: str = "", limit: int = 30) -> dict:
        """Search application logs. min_severity: ANY, WARN or ERROR. Results are grouped by message."""
        return await tk.search_logs(agent, service, window, min_severity, text, limit)

    async def config_changes(target: str = "", lookback_s: float = 1800) -> dict:
        """Recorded configuration and feature-flag changes from lookback_s before incident start until the
        incident end, with change IDs and times relative to incident start. target filters by service or flag."""
        return await tk.config_changes(agent, target, lookback_s)

    async def container_events(service: str = "", window: str = "full") -> dict:
        """Container lifecycle events (start, die, oom, kill, restart, update) in the window."""
        return await tk.container_events(agent, service, window)

    async def runtime_config(service: str) -> dict:
        """Current (present-time) container configuration and state: image, CPU and memory limits, restarts."""
        return await tk.runtime_config(agent, service)

    async def service_dependencies(service: str) -> dict:
        """Which services this service calls and which services call it."""
        return await tk.service_dependencies(agent, service)

    async def search_incident_memory(query: str, service: str = "") -> dict:
        """Search reviewed past incidents for similar symptoms. Results are leads to verify, not proof."""
        return await tk.search_incident_memory(agent, query, service)

    groups = {
        "overview": [service_overview],
        "metrics": [service_red, operation_latency, container_resources, cache_health],
        "traces": [find_traces, compare_trace_profiles, trace_detail],
        "logs_changes": [search_logs, config_changes, container_events, runtime_config],
        "shared": [service_dependencies],
        "memory": [search_incident_memory],
    }
    return [fn for kind in ("overview", "metrics", "traces", "logs_changes", "shared", "memory")
            if kind in kinds for fn in groups[kind]]


def coerce(model: type[BaseModel], value: Any) -> tuple[Any, list[str]]:
    """Accept the shapes models commonly send (object, JSON string, one-item list) for a structured argument."""
    if isinstance(value, model):
        return value, []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None, [f"Expected a JSON object for {model.__name__}, got unparseable text"]
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    try:
        return model.model_validate(value), []
    except ValidationError as exc:
        return None, [f"{model.__name__} schema error: " + "; ".join(
            f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:8])]


ROLE_TOOLKITS = {
    "metrics": {"overview", "metrics", "shared"},
    "traces": {"traces", "shared"},
    "logs_changes": {"logs_changes", "shared"},
}


def evidence_index(tk: CaseToolkit) -> EvidenceIndex:
    case = tk.case
    changes = tk.ledger.changes(case["incident_start"] - tk.change_lookback_s, case["incident_end"])
    return EvidenceIndex(tk.ledger.evidence(case["id"], tk.run_id), [c["id"] for c in changes])


def allowed_flag_variants(tk: CaseToolkit) -> dict[str, set[str]]:
    case = tk.case
    out: dict[str, set[str]] = {}
    for change in tk.ledger.changes(case["incident_start"] - tk.change_lookback_s, case["incident_end"]):
        if change["kind"] == "feature_flag":
            out.setdefault(change["target"].removeprefix("flag:"), set()).add(change["before_state"].get("variant"))
    return out


def make_specialist(role: str, model: Any, tk: CaseToolkit, budget: LlmBudget, instruction: str,
                    submission: Submission):
    async def submit_findings(findings: Finding, tool_context: ToolContext) -> dict:
        """Submit your answer to the assigned question. Call exactly once, at the end."""
        findings, errors = coerce(Finding, findings)
        if errors:
            return submission.reject(errors)
        errors = check_finding(findings, evidence_index(tk))
        if errors and submission.attempts + 1 < submission.max_attempts:
            return submission.reject(errors)
        return submission.accept(tool_context, findings)

    tools = telemetry_tools(tk, role, ROLE_TOOLKITS[role]) + [submit_findings]
    return build_agent(role, model, instruction, tools, budget, description=f"{role} specialist", step_limit=6)


def make_coordinator(model: Any, tk: CaseToolkit, budget: LlmBudget, instruction: str, submission: Submission,
                     *, phase: str, max_tasks: int, flagd_file):
    async def submit_decision(decision: CoordinatorDecision, tool_context: ToolContext) -> dict:
        """Submit the updated hypothesis board and either delegate tasks (next_step='investigate') or conclude
        with a report (next_step='conclude'). Call exactly once."""
        decision, errors = coerce(CoordinatorDecision, decision)
        if errors:
            return submission.reject(errors)
        if phase == "final" and decision.next_step != "conclude":
            return submission.reject(["This is the final round: next_step must be 'conclude' with a report."])
        if phase == "plan" and decision.next_step != "investigate":
            return submission.reject(["This is the planning phase: delegate tasks to the specialists "
                                      "(next_step='investigate'); the investigation has not started yet."])
        errors = check_decision(decision, evidence_index(tk), flagd_file, max_tasks=max_tasks,
                                allowed_flag_variants=allowed_flag_variants(tk))
        if errors:
            return submission.reject(errors)
        return submission.accept(tool_context, decision)

    kinds = {"overview", "shared", "memory"} if phase == "plan" else {"shared"}
    tools = telemetry_tools(tk, "coordinator", kinds) + [submit_decision]
    return build_agent("coordinator", model, instruction, tools, budget, description="incident coordinator",
                       step_limit=4 if phase == "plan" else 1)


def make_single_agent(model: Any, tk: CaseToolkit, budget: LlmBudget, instruction: str, submission: Submission,
                      *, flagd_file, step_limit: int):
    async def submit_report(hypotheses: list[Hypothesis], report: Report, tool_context: ToolContext) -> dict:
        """Submit the final hypothesis list and report. Call exactly once, at the end."""
        report, errors = coerce(Report, report)
        parsed = [coerce(Hypothesis, h) for h in (json.loads(hypotheses) if isinstance(hypotheses, str) else hypotheses or [])]
        errors += [e for _, errs in parsed for e in errs]
        if errors:
            return submission.reject(errors)
        hypotheses = [h for h, _ in parsed]
        index = evidence_index(tk)
        errors = check_hypotheses(hypotheses, index) + check_report(
            report, hypotheses, index, flagd_file, allowed_flag_variants(tk))
        if errors:
            return submission.reject(errors)
        return submission.accept(tool_context, {"hypotheses": hypotheses, "report": report})

    kinds = {"overview", "metrics", "traces", "logs_changes", "shared", "memory"}
    tools = telemetry_tools(tk, "single_agent", kinds) + [submit_report]
    return build_agent("single_agent", model, instruction, tools, budget, description="single-agent baseline",
                       step_limit=step_limit)


def make_critic(model: Any, tk: CaseToolkit, budget: LlmBudget, instruction: str, submission: Submission):
    async def submit_critique(critique: Critique, tool_context: ToolContext) -> dict:
        """Submit your review of the proposed diagnosis. Call exactly once."""
        critique, errors = coerce(Critique, critique)
        if errors:
            return submission.reject(errors)
        bad = evidence_index(tk).unknown(critique.evidence_ids)
        if bad:
            return submission.reject([f"Unknown evidence IDs: {', '.join(bad)}"])
        return submission.accept(tool_context, critique)

    kinds = {"overview", "metrics", "traces", "logs_changes", "shared"}
    tools = telemetry_tools(tk, "critic", kinds) + [submit_critique]
    return build_agent("critic", model, instruction, tools, budget, description="independent critic", step_limit=5)

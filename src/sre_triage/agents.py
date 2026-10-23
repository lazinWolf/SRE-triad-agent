"""The ADK agents: Coordinator (plan / report), three specialists, a single-agent baseline and a Critic.

Every agent is a Google ADK ``LlmAgent``. Specialists get read-only telemetry tools. Every agent
finishes by calling a ``submit_*`` tool; the tool validates the answer in code (schema, evidence IDs,
remediation allowlist) and returns the errors to the model so it can correct itself.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import ToolContext
from google.genai import types
from pydantic import BaseModel, TypeAdapter, ValidationError

from . import remediation
from .prompts import COORDINATOR_PLAN, COORDINATOR_REPORT, CRITIC, specialist_system
from .schemas import Critique, Finding, Hypothesis, Plan, Remediation, Report, Task
from .tools import AGENT_TOOLS, Toolbox

Emit = Callable[[str, str, str], None]  # (kind, agent, text)


@dataclass
class Submission:
    """Holds an agent's validated answer; rejected answers go back to the model (max 3 tries)."""

    value: Any = None
    attempts: int = 0

    def accept(self, tool_context: ToolContext, value: BaseModel) -> dict:
        self.value = value.model_dump()
        tool_context.actions.skip_summarization = True  # ends the agent's turn
        return {"status": "accepted"}

    def reject(self, errors: list[str]) -> dict:
        self.attempts += 1
        return {"status": "rejected", "errors": errors,
                "instruction": "Fix these problems and call the submit tool again." if self.attempts < 3
                else "Last attempt: submit a conservative answer that passes validation."}


@dataclass
class Usage:
    calls: int = 0
    tokens: int = 0
    models: dict[str, int] = field(default_factory=dict)


def _parse(value: Any) -> Any:
    """Models sometimes send structured arguments as JSON strings."""
    if isinstance(value, str) and value.strip()[:1] in "[{":
        try:
            return json.loads(value)
        except ValueError:
            return value
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_parse(v) for v in value]
    return value


def _errors(exc: ValidationError) -> list[str]:
    return [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()][:8]


# ---------------------------------------------------------------------------- callbacks
def _callbacks(name: str, emit: Emit, usage: Usage, step_limit: int):
    """Count model calls, log the agent's reasoning, and after `step_limit` calls offer only the submit tool."""
    calls = {"n": 0}

    def before(callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
        calls["n"] += 1
        if calls["n"] > step_limit + 3:  # it ignored the request to submit: stop the agent
            return LlmResponse(content=types.Content(role="model", parts=[types.Part(text="Step limit reached.")]))
        if calls["n"] > step_limit and llm_request.config and llm_request.config.tools:
            submit = [d for t in llm_request.config.tools for d in (t.function_declarations or [])
                      if d.name and d.name.startswith("submit_")]
            if submit:
                llm_request.config.tools = [types.Tool(function_declarations=submit)]
                llm_request.contents.append(types.Content(role="user", parts=[types.Part(text=(
                    f"Tool budget used. Call {submit[0].name} now with what you found, citing only evidence IDs you received."))]))
        return None

    def after(callback_context: CallbackContext, llm_response: LlmResponse) -> LlmResponse | None:
        usage.calls += 1
        meta = llm_response.usage_metadata
        if meta:
            usage.tokens += (meta.prompt_token_count or 0) + (meta.candidates_token_count or 0)
        served = (llm_response.custom_metadata or {}).get("served_by")
        if served:
            usage.models[served] = usage.models.get(served, 0) + 1
        for part in (llm_response.content.parts if llm_response.content else None) or []:
            if part.text and part.text.strip():
                emit("thought", name, part.text.strip())
        return None

    return before, after


def _agent(name: str, model: BaseLlm, instruction: str, tools: list, emit: Emit, usage: Usage,
           step_limit: int) -> LlmAgent:
    before, after = _callbacks(name, emit, usage, step_limit)
    return LlmAgent(name=name, model=model, instruction=lambda _ctx: instruction, tools=tools,
                    disallow_transfer_to_parent=True, disallow_transfer_to_peers=True,
                    before_model_callback=before, after_model_callback=after)


# ---------------------------------------------------------------------------- telemetry tools
def telemetry_tools(box: Toolbox, agent: str) -> list[Callable]:
    """Toolbox methods as plain functions, so ADK builds their declarations from signature + docstring."""
    out = []
    for name in AGENT_TOOLS[agent]:
        method = getattr(Toolbox, name)
        sig = inspect.signature(method)
        params = [p for p in sig.parameters.values() if p.name != "self"]

        def tool(_name=name, **kwargs) -> dict:
            return box.call(agent, _name, kwargs)

        tool.__name__ = name
        tool.__doc__ = method.__doc__
        tool.__signature__ = sig.replace(parameters=params, return_annotation=dict)
        tool.__annotations__ = {p.name: str for p in params} | {"return": dict}
        out.append(tool)
    return out


# ---------------------------------------------------------------------------- agents
def specialist(agent: str, model: BaseLlm, box: Toolbox, emit: Emit, usage: Usage, steps: int,
               sub: Submission) -> LlmAgent:
    known = lambda: {e["id"] for e in box.evidence}  # noqa: E731

    def submit_finding(summary: str, tool_context: ToolContext, suspect_service: str = "",
                       category: str = "unknown", confidence: float = 0.5,
                       evidence_ids: list[str] | None = None, ruled_out: list[str] | None = None) -> dict:
        """Submit your finding. suspect_service is the service you believe is the origin ("" if none);
        evidence_ids must be IDs returned by your tool calls."""
        try:
            finding = Finding(summary=summary, suspect_service=suspect_service or None, category=category,
                              confidence=confidence, evidence_ids=_parse(evidence_ids) or [],
                              ruled_out=_parse(ruled_out) or [])
        except ValidationError as exc:
            return sub.reject(_errors(exc))
        unknown = sorted(set(finding.evidence_ids) - known())
        if unknown and sub.attempts < 2:
            return sub.reject([f"evidence IDs you did not receive: {', '.join(unknown)}"])
        finding.evidence_ids = [i for i in finding.evidence_ids if i in known()]
        return sub.accept(tool_context, finding)

    tools = telemetry_tools(box, agent) + [submit_finding]
    return _agent(agent, model, specialist_system(agent), tools, emit, usage, steps)


def planner(model: BaseLlm, emit: Emit, usage: Usage, sub: Submission) -> LlmAgent:
    def submit_plan(hypotheses: list[str], tasks: list[Task], tool_context: ToolContext) -> dict:
        """Submit 2-4 competing hypotheses and 1-3 tasks, each {agent: metrics|traces|logs_changes, question}."""
        try:
            plan = Plan(hypotheses=_parse(hypotheses), tasks=_parse(tasks))
        except ValidationError as exc:
            return sub.reject(_errors(exc))
        return sub.accept(tool_context, plan)

    return _agent("coordinator", model, COORDINATOR_PLAN, [submit_plan], emit, usage, 2)


def reporter(model: BaseLlm, emit: Emit, usage: Usage, sub: Submission, known_ids: set[str],
             backends) -> LlmAgent:
    def submit_report(status: str, summary: str, tool_context: ToolContext, root_cause_service: str = "",
                      category: str = "unknown", mechanism: str = "", confidence: float = 0.5,
                      evidence_ids: list[str] | None = None, hypotheses: list[Hypothesis] | None = None,
                      remediation_action: Remediation | None = None) -> dict:
        """Submit the root-cause report. status is "diagnosed" or "inconclusive". remediation_action is at most
        one allowlisted action ({action, service, cpus, flag, variant}) or null."""
        try:
            report = Report(status=status, summary=summary, root_cause_service=root_cause_service or None,
                            category=category, mechanism=mechanism, confidence=confidence,
                            evidence_ids=_parse(evidence_ids) or [], hypotheses=_parse(hypotheses) or [],
                            remediation=_parse(remediation_action) or None)
        except ValidationError as exc:
            return sub.reject(_errors(exc))
        if report.status != "diagnosed":
            report.remediation = None
        problems = []
        cited = set(report.evidence_ids) | {i for h in report.hypotheses for i in h.evidence_ids}
        if unknown := sorted(cited - known_ids):
            problems.append(f"evidence IDs that do not exist: {', '.join(unknown)}")
        if report.status == "diagnosed" and not set(report.evidence_ids) & known_ids:
            problems.append("a diagnosis must cite evidence")
        problems += remediation.validate(report.remediation.model_dump() if report.remediation else None, backends)
        if problems and sub.attempts < 2:
            return sub.reject(problems)
        return sub.accept(tool_context, report)

    return _agent("coordinator", model, COORDINATOR_REPORT, [submit_report], emit, usage, 2)


def critic(model: BaseLlm, emit: Emit, usage: Usage, sub: Submission) -> LlmAgent:
    def submit_review(verdict: str, tool_context: ToolContext, issues: list[str] | None = None,
                      follow_up: list[Task] | None = None) -> dict:
        """Submit your review: verdict "accept" or "revise", concrete issues, and at most 2 follow-up tasks."""
        try:
            review = Critique(verdict=verdict, issues=_parse(issues) or [],
                              follow_up=TypeAdapter(list[Task]).validate_python(_parse(follow_up) or []))
        except ValidationError as exc:
            return sub.reject(_errors(exc))
        return sub.accept(tool_context, review)

    return _agent("critic", model, CRITIC, [submit_review], emit, usage, 2)

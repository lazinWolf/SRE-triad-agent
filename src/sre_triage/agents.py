"""The ADK agents: a Coordinator, four specialists, a single-agent baseline and a local Critic.

Every agent is a Google ADK ``LlmAgent``. Specialists get read-only tools from the MCP servers (``McpToolset``);
the coordinator gets ADK's ``load_memory`` tool. Every agent finishes by calling a ``submit_*`` tool; the tool
validates the answer in code (schema, evidence IDs, remediation allowlist) and returns the errors to the model so
it can correct itself. The coordinator has three of them, and which one it calls decides where the workflow goes.
(The review board is a CrewAI crew behind A2A: see ``a2a_critic``; the Critic here is its local fallback.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import ToolContext, load_memory
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.genai import types
from pydantic import BaseModel, TypeAdapter, ValidationError

from . import remediation
from .guardrails import check_tool_output
from .mcp_servers import WINDOW_HEADER
from .prompts import COORDINATOR, COORDINATOR_MEMORY, CRITIC, specialist_system
from .schemas import Critique, Finding, Hypothesis, Plan, Question, Remediation, Report, Task
from .tools import AGENT_TOOLS, RESULT_MODELS, SERVER_TOOLS, Evidence


@dataclass
class Submission:
    """Holds an agent's validated answer; rejected answers go back to the model (max 3 tries)."""

    value: Any = None
    kind: str = ""       # which submit tool was accepted: plan, question, report, finding, review
    attempts: int = 0
    problems: list[str] = field(default_factory=list)  # checks still failing when the last attempt was accepted

    def accept(self, tool_context: ToolContext, value: BaseModel, kind: str = "") -> dict:
        self.value, self.kind = value.model_dump(), kind
        tool_context.actions.skip_summarization = True  # ends the agent's turn
        return {"status": "accepted"}

    def reject(self, errors: list[str]) -> dict:
        self.attempts += 1
        return {"status": "rejected", "errors": errors,
                "instruction": "Fix these problems and call the submit tool again." if self.attempts < 3
                else "Last attempt: submit a conservative answer that passes validation."}


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


# ---------------------------------------------------------------------------- step limit
def _step_limit(step_limit: int):
    """Agent callback (guardrail): after `step_limit` model calls offer only the submit tools, then stop the agent.
    Monitoring of all agents (thoughts, tools, usage, errors) is done by the Runner plugin in ``callbacks``."""
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
                    f"Tool budget used. Call {submit[-1].name} now with what you found, citing only evidence IDs you received."))]))
        return None

    return before


def _agent(name: str, model: BaseLlm, instruction: str, tools: list, step_limit: int, **kw) -> LlmAgent:
    return LlmAgent(name=name, model=model, instruction=lambda _ctx: instruction, tools=tools,
                    disallow_transfer_to_parent=True, disallow_transfer_to_peers=True,
                    before_model_callback=_step_limit(step_limit), **kw)


# ---------------------------------------------------------------------------- MCP tools
def mcp_toolsets(agent: str, window: dict, urls: dict[str, str]) -> list[McpToolset]:
    """The agent's tools on each MCP server. The incident window travels as a header, not as a model argument."""
    headers = {WINDOW_HEADER: json.dumps(window)}
    return [McpToolset(connection_params=StreamableHTTPConnectionParams(url=urls[kind], headers=headers, timeout=20),
                       tool_filter=wanted)
            for kind, names in SERVER_TOOLS.items() if (wanted := [n for n in AGENT_TOOLS[agent] if n in names])]


def _payload(response: Any) -> Any:
    """The tool's own result out of an MCP call result ({content, structuredContent, isError})."""
    if not isinstance(response, dict) or "content" not in response:
        return response
    text = " ".join(c.get("text", "") for c in response.get("content") or [] if isinstance(c, dict))
    if response.get("isError") or response.get("is_error"):
        return {"error": text[:400] or "tool failed"}
    structured = response.get("structuredContent") or response.get("structured_content")
    if isinstance(structured, dict):
        return structured
    try:
        return json.loads(text)
    except ValueError:
        return text


def _tool_guard(agent: str, evidence: Evidence, log: Callable[[str, dict, dict, list[str]], None]):
    """Agent callback (guardrail) on every MCP tool result: type-check it against the tool's Pydantic model, redact
    PII, neutralise instruction-like text, cap its size; then store it as evidence and give the model its ID."""

    def after(tool, args: dict, tool_context: ToolContext, tool_response: Any) -> dict | None:
        if tool.name not in RESULT_MODELS:
            return None
        clean, notes = check_tool_output(tool.name, _payload(tool_response), RESULT_MODELS[tool.name])
        if "error" not in clean:
            clean = {"evidence_id": evidence.record(agent, tool.name, dict(args), clean), **clean}
        log(tool.name, dict(args), clean, notes)
        return clean

    return after


# ---------------------------------------------------------------------------- agents
def specialist(agent: str, model: BaseLlm, steps: int, sub: Submission, evidence: Evidence, toolsets: list[McpToolset],
               log: Callable[[str, dict, dict, list[str]], None]) -> LlmAgent:
    known = lambda: {e["id"] for e in evidence.items}  # noqa: E731

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
        return sub.accept(tool_context, finding, "finding")

    return _agent(agent, model, specialist_system(agent), [*toolsets, submit_finding], steps,
                  after_tool_callback=_tool_guard(agent, evidence, log))


def coordinator(model: BaseLlm, sub: Submission, known_ids: set[str], backends, *, can_plan: bool, can_ask: bool,
                memory: bool) -> LlmAgent:
    """The coordinator's turn: it calls exactly one submit tool, and that choice routes the workflow."""

    def submit_plan(hypotheses: list[str], tasks: list[Task], tool_context: ToolContext) -> dict:
        """Investigate further: 2-4 competing hypotheses and 1-3 tasks, each
        {agent: metrics|traces|logs_changes|infra, question}."""
        try:
            plan = Plan(hypotheses=_parse(hypotheses), tasks=_parse(tasks))
        except ValidationError as exc:
            return sub.reject(_errors(exc))
        return sub.accept(tool_context, plan, "plan")

    def submit_question(question: str, tool_context: ToolContext) -> dict:
        """Ask the human operator one question that telemetry cannot answer."""
        try:
            return sub.accept(tool_context, Question(question=question), "question")
        except ValidationError as exc:
            return sub.reject(_errors(exc))

    def submit_report(status: str, summary: str, tool_context: ToolContext, root_cause_service: str = "",
                      category: str = "unknown", mechanism: str = "", confidence: float = 0.5,
                      evidence_ids: list[str] | None = None, hypotheses: list[Hypothesis] | None = None,
                      remediation_action: Remediation | None = None,
                      alternative_action: Remediation | None = None) -> dict:
        """Conclude with the root-cause report. status is "diagnosed", "inconclusive" or "no_incident".
        remediation_action is at most one allowlisted action ({action, service, flag, restart_after}) or null;
        alternative_action is an optional second choice for the operator."""
        try:
            report = Report(status=status, summary=summary, root_cause_service=root_cause_service or None,
                            category=category, mechanism=mechanism, confidence=confidence,
                            evidence_ids=_parse(evidence_ids) or [], hypotheses=_parse(hypotheses) or [],
                            remediation=_parse(remediation_action) or None,
                            alternative=_parse(alternative_action) or None)
        except ValidationError as exc:
            return sub.reject(_errors(exc))
        if report.status != "diagnosed":
            report.remediation = report.alternative = None
        problems = []
        cited = set(report.evidence_ids) | {i for h in report.hypotheses for i in h.evidence_ids}
        if unknown := sorted(cited - known_ids):
            problems.append(f"evidence IDs that do not exist: {', '.join(unknown)}")
        if report.status == "diagnosed" and not set(report.evidence_ids) & known_ids:
            problems.append("a diagnosis must cite evidence" + ("; investigate first with submit_plan" if can_plan else ""))
        problems += remediation.validate(report.remediation.model_dump() if report.remediation else None, backends)
        if report.alternative and remediation.validate(report.alternative.model_dump(), backends):
            report.alternative = None  # an invalid second choice is simply not offered
        if problems and sub.attempts < 2:
            return sub.reject(problems)
        sub.problems = problems
        return sub.accept(tool_context, report, "report")

    # load_memory is ADK's built-in memory tool: it searches the Runner's MemoryService (memory as tool use).
    tools = ([load_memory] if memory else []) + ([submit_plan] if can_plan else []) \
        + ([submit_question] if can_ask else []) + [submit_report]
    return _agent("coordinator", model, COORDINATOR + (COORDINATOR_MEMORY if memory else ""), tools, 3)


def critic(model: BaseLlm, sub: Submission) -> LlmAgent:
    def submit_review(verdict: str, tool_context: ToolContext, issues: list[str] | None = None,
                      follow_up: list[Task] | None = None, risk: str = "low", risk_notes: str = "") -> dict:
        """Submit your review: verdict "accept" or "revise", concrete issues, at most 2 follow-up tasks, and the
        risk (low, medium, high) of the proposed remediation."""
        try:
            review = Critique(verdict=verdict, issues=_parse(issues) or [], risk=risk or "low", risk_notes=risk_notes,
                              follow_up=TypeAdapter(list[Task]).validate_python(_parse(follow_up) or []))
        except ValidationError as exc:
            return sub.reject(_errors(exc))
        return sub.accept(tool_context, review, "review")

    return _agent("critic", model, CRITIC, [submit_review], 2)

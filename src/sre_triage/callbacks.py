"""Task monitoring with ADK callbacks: one plugin on the Runner logs how every case executes.

ADK calls the plugin for every agent, model call and tool call in the workflow, and for every error, so the
execution flow is recorded in one place. The workflow nodes only report their decisions (plan, finding,
report, ...). Both go to the case's activity log, which the dashboard and the CLI show live.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from google.adk.plugins.base_plugin import BasePlugin

from .cases import Cases


def _short(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


@dataclass
class Usage:
    calls: int = 0
    tokens: int = 0
    models: dict[str, int] = field(default_factory=dict)


class ActivityLog(BasePlugin):
    def __init__(self, cases: Cases, usage: dict[str, Usage]):
        super().__init__(name="activity_log")
        self.cases = cases
        self.usage = usage  # case id -> model calls, tokens and models used

    def _emit(self, ctx, kind: str, agent: str, text: str) -> None:
        self.cases.emit(ctx.session.id, kind, agent, text)

    async def before_agent_callback(self, *, agent, callback_context):
        self._emit(callback_context, "agent", agent.name, "started")

    async def after_agent_callback(self, *, agent, callback_context):
        self._emit(callback_context, "agent", agent.name, "finished")

    async def after_model_callback(self, *, callback_context, llm_response):
        usage = self.usage.setdefault(callback_context.session.id, Usage())
        usage.calls += 1
        if meta := llm_response.usage_metadata:
            usage.tokens += (meta.prompt_token_count or 0) + (meta.candidates_token_count or 0)
        if served := (llm_response.custom_metadata or {}).get("served_by"):
            usage.models[served] = usage.models.get(served, 0) + 1
        for part in (llm_response.content.parts if llm_response.content else None) or []:
            if part.text and part.text.strip():
                self._emit(callback_context, "thought", callback_context.agent_name, part.text.strip())

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        if hasattr(result, "memories"):  # ADK load_memory
            result = {"result": f"{len(result.memories)} memories"}
        result = result if isinstance(result, dict) else {"result": result}
        if tool.name.startswith("submit_"):  # the answer's Pydantic / guardrail validation outcome
            text = f"{tool.name}: {result.get('status')}" + (f" ({'; '.join(result['errors'])})"
                                                              if result.get("errors") else "")
        else:
            args = ", ".join(f"{k}={_short(v, 60)}" for k, v in tool_args.items())
            text = f"{tool.name}({args}) -> {result.get('evidence_id') or result.get('error') or _short(result)}"
        self._emit(tool_context, "tool", tool_context.agent_name, text)

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error):
        self._emit(tool_context, "error", tool_context.agent_name, f"tool {tool.name} failed: {error}")

    async def on_model_error_callback(self, *, callback_context, llm_request, error):
        self._emit(callback_context, "error", callback_context.agent_name, f"model call failed: {_short(str(error), 300)}")

    async def on_agent_error_callback(self, *, agent, callback_context, error):
        self._emit(callback_context, "error", agent.name, f"agent failed: {_short(str(error), 300)}")

    async def on_run_error_callback(self, *, invocation_context, error):
        self.cases.emit(invocation_context.session.id, "error", "workflow", f"{type(error).__name__}: {_short(str(error), 300)}")

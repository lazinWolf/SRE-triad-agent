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

from . import tracing
from .cases import Cases


def _short(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """List price of a model call from LiteLLM's price table (what a free-tier call would cost when paid)."""
    try:
        import litellm

        prompt, completion = litellm.cost_per_token(model=model, prompt_tokens=prompt_tokens,
                                                    completion_tokens=completion_tokens)
        return prompt + completion
    except Exception:  # model not in the price table
        return 0.0


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    models: dict[str, int] = field(default_factory=dict)   # model -> calls it served
    agents: dict[str, dict] = field(default_factory=dict)  # agent -> {calls, tokens}

    def add_remote(self, agent: str, used: dict | None, asked: Any = None, answer: Any = None) -> None:
        """Count (and trace) the LLM usage a remote agent reported for its own work (the CrewAI review board)."""
        if not used:
            return
        calls, sent, got = (int(used.get(k) or 0) for k in ("llm_requests", "prompt_tokens", "completion_tokens"))
        cost = cost_usd(used.get("model") or "", sent, got)
        self.calls += calls
        self.prompt_tokens += sent
        self.completion_tokens += got
        self.cost += cost
        tracing.remote_call(agent, used.get("model") or "", sent, got, cost, asked, answer)
        self.agents[agent] = {"calls": self.agents.get(agent, {}).get("calls", 0) + calls,
                              "tokens": self.agents.get(agent, {}).get("tokens", 0) + sent + got}

    def as_stats(self) -> dict:
        return {"llm_calls": self.calls, "tokens": self.prompt_tokens + self.completion_tokens,
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "cost_usd": round(self.cost, 5), "models_used": self.models, "agents": self.agents}


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
        meta = llm_response.usage_metadata
        sent, got = (meta.prompt_token_count or 0, meta.candidates_token_count or 0) if meta else (0, 0)
        served = (llm_response.custom_metadata or {}).get("served_by")
        usage.calls += 1
        usage.prompt_tokens += sent
        usage.completion_tokens += got
        if served:
            cost = cost_usd(served, sent, got)
            usage.models[served] = usage.models.get(served, 0) + 1
            usage.cost += cost
            tracing.model_call(served, cost)
        per_agent = usage.agents.setdefault(callback_context.agent_name, {"calls": 0, "tokens": 0})
        per_agent["calls"] += 1
        per_agent["tokens"] += sent + got
        for part in (llm_response.content.parts if llm_response.content else None) or []:
            if part.text and part.text.strip():
                self._emit(callback_context, "thought", callback_context.agent_name, part.text.strip())

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        if hasattr(result, "memories"):  # ADK load_memory
            result = {"result": f"{len(result.memories)} memories"}
        if isinstance(result, dict) and "content" in result and not tool.name.startswith("submit_"):
            return  # an MCP tool: the agent's own callback checks it, stores the evidence and logs it
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

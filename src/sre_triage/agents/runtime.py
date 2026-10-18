"""Running ADK LlmAgents with a shared model-call budget and structured submissions.

Every agent finishes by calling a ``submit_*`` tool. The tool validates the
payload, returns errors to the model for correction, and on success ends the
agent's turn (``skip_summarization``). This works with any tool-capable model
behind LiteLLM, without relying on provider-specific JSON modes.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.run_config import RunConfig
from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types

from ..config import Settings


class ModelNotConfigured(RuntimeError):
    pass


def make_model(settings: Settings, override: BaseLlm | Callable[[str], BaseLlm] | None = None,
               role: str = "", case_model: str | None = None) -> BaseLlm | str:
    """Return the model for an agent: a test/scripted model, or a LiteLLM model for the role."""
    if override is not None:
        return override(role) if callable(override) and not isinstance(override, BaseLlm) else override
    from ..llm import litellm_model, model_for_role

    model = model_for_role(settings, role, case_model)
    if not model:
        raise ModelNotConfigured(
            "Set SRE_MODEL (a LiteLLM model string such as 'gemini/<model>', 'groq/<model>' or "
            "'openrouter/<model>') and the provider's API key in .env.")
    return litellm_model(settings, model)


@dataclass
class LlmBudget:
    max_calls: int
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_agent: dict[str, int] = field(default_factory=dict)

    def exhausted(self) -> bool:
        return self.calls >= self.max_calls

    def before_model(self, agent_name: str):
        def callback(callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
            if self.calls >= self.max_calls:
                return LlmResponse(content=types.Content(role="model", parts=[types.Part(
                    text="Model-call budget exhausted; stopping without further reasoning.")]))
            self.calls += 1
            self.by_agent[agent_name] = self.by_agent.get(agent_name, 0) + 1
            return None
        return callback

    def after_model(self, callback_context: CallbackContext, llm_response: LlmResponse) -> LlmResponse | None:
        usage = llm_response.usage_metadata
        if usage is not None:
            self.prompt_tokens += usage.prompt_token_count or 0
            self.completion_tokens += usage.candidates_token_count or 0
        return None

    def stats(self) -> dict[str, Any]:
        return {"llm_calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "llm_calls_by_agent": dict(self.by_agent)}


@dataclass
class Submission:
    """Holds the validated payload an agent submitted, if any."""

    value: Any = None
    attempts: int = 0
    last_errors: list[str] = field(default_factory=list)
    max_attempts: int = 3

    def accept(self, tool_context: ToolContext, value: Any) -> dict[str, Any]:
        self.value = value
        tool_context.actions.skip_summarization = True
        return {"status": "accepted"}

    def reject(self, errors: list[str]) -> dict[str, Any]:
        self.attempts += 1
        self.last_errors = errors
        remaining = self.max_attempts - self.attempts
        return {"status": "rejected", "errors": errors, "attempts_remaining": max(0, remaining),
                "instruction": "Fix these problems and call the submit tool again." if remaining > 0
                else "No attempts remain; submit a conservative result that passes validation."}


@dataclass
class AgentResult:
    submitted: Any
    texts: list[str]
    error: str | None = None


async def run_agent(agent: LlmAgent, prompt: str, submission: Submission, *, max_llm_calls: int,
                    on_text: Callable[[str, str, str], None] | None = None, nudge: str | None = None) -> AgentResult:
    """Run one agent in a fresh session until it submits, gives up, or hits its call limit."""
    sessions = InMemorySessionService()
    runner = Runner(agent=agent, app_name="sre_triage", session_service=sessions)
    session = await sessions.create_session(app_name="sre_triage", user_id="triage", session_id=uuid.uuid4().hex)
    texts: list[str] = []
    error = None

    async def turn(message: str) -> None:
        nonlocal error
        content = types.Content(role="user", parts=[types.Part(text=message)])
        async for event in runner.run_async(user_id="triage", session_id=session.id, new_message=content,
                                            run_config=RunConfig(max_llm_calls=max_llm_calls)):
            if event.error_code:
                error = f"{event.error_code}: {event.error_message}"
                if on_text:
                    on_text(agent.name, f"[model error] {error}", "agent_error")
            if event.content and event.author == agent.name:
                for part in event.content.parts or []:
                    if part.text and part.thought:
                        if on_text:
                            on_text(agent.name, part.text, "agent_thought")
                    elif part.text:
                        texts.append(part.text)
                        if on_text:
                            on_text(agent.name, part.text, "agent_message")
                    if part.function_call and part.function_call.name.startswith("submit_") and on_text:
                        on_text(agent.name, f"{part.function_call.name} called", "submit")
            for response in (event.get_function_responses() if event.content else []):
                payload = response.response or {}
                if response.name.startswith("submit_") and payload.get("status") == "rejected" and on_text:
                    on_text(agent.name, "Submission rejected: " + "; ".join(payload.get("errors", []))[:1500], "validation")
                elif payload.get("error") and on_text and not payload.get("evidence_id"):
                    on_text(agent.name, f"Tool {response.name} argument error: {str(payload.get('error'))[:500]}", "validation")

    try:
        await turn(prompt)
        if submission.value is None and nudge and submission.attempts < submission.max_attempts:
            await turn(nudge)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # model/provider errors end this agent, not the investigation
        error = f"{type(exc).__name__}: {str(exc)[:400]}"
    return AgentResult(submission.value, texts, error)


def finalize_after(limit: int):
    """After `limit` model calls, offer only the submit tool and ask the agent to submit.

    Smaller models in particular keep querying; this bounds each agent's investigation
    while still giving it one clean chance to report what it found."""
    calls = {"n": 0}

    def callback(callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
        calls["n"] += 1
        if calls["n"] <= limit or not llm_request.config or not llm_request.config.tools:
            return None
        submit = None
        for tool in llm_request.config.tools:
            for decl in tool.function_declarations or []:
                if decl.name and decl.name.startswith("submit_"):
                    submit = decl
        if submit is None:
            return None
        llm_request.config.tools = [types.Tool(function_declarations=[submit])]
        llm_request.contents.append(types.Content(role="user", parts=[types.Part(text=(
            f"Tool-call limit for this step reached. Do not call other tools. Call {submit.name} now, "
            "citing only evidence IDs you received and listing what remains unknown."))]))
        return None
    return callback


def strip_thoughts(callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
    """Drop earlier reasoning text from the history; some providers (e.g. Groq) reject it on input."""
    for content in llm_request.contents or []:
        if content.parts:
            kept = [p for p in content.parts if not (p.thought and not p.function_call)]
            content.parts = kept or [types.Part(text="")]
    return None


THOUGHT_INTOLERANT_PROVIDERS = ("groq/",)
TOOL_CALL_REJECTIONS = ("tool_use_failed", "did not match schema", "Failed to call a function", "failed_generation")


def recover_tool_call_errors(model: BaseLlm, attempts: int = 2):
    """When a provider rejects a malformed tool call server-side (HTTP 400), show the model the error
    and let it try again, instead of ending the agent. Mirrors in-band validation feedback."""

    async def callback(callback_context: CallbackContext, llm_request: LlmRequest, error: Exception):
        message = str(error)
        if not any(marker in message for marker in TOOL_CALL_REJECTIONS):
            return None
        for _ in range(attempts):
            llm_request.contents.append(types.Content(role="user", parts=[types.Part(text=(
                "Your previous tool call was rejected by the API because its arguments did not match the tool "
                f"schema: {message[:800]}\nCall the tool again with arguments that exactly match its schema."))]))
            try:
                final = None
                async for response in model.generate_content_async(llm_request, stream=False):
                    final = response
                if final is not None:
                    return final
            except Exception as retry_error:  # still malformed: show the new error next time
                message = str(retry_error)
                if not any(marker in message for marker in TOOL_CALL_REJECTIONS):
                    return None
        return None
    return callback


def build_agent(name: str, model: BaseLlm | str, instruction: str, tools: list, budget: LlmBudget,
                description: str = "", step_limit: int | None = None) -> LlmAgent:
    # A provider function bypasses ADK's {state} templating, so JSON in the text stays literal.
    return LlmAgent(
        name=name, model=model, instruction=lambda _ctx: instruction, tools=tools, description=description,
        disallow_transfer_to_parent=True, disallow_transfer_to_peers=True,
        before_model_callback=[budget.before_model(name)] + ([finalize_after(step_limit)] if step_limit else [])
        + ([strip_thoughts] if any(str(m).startswith(THOUGHT_INTOLERANT_PROVIDERS)
                                   for m in getattr(model, "model_names", [getattr(model, "model", model)])) else []),
        after_model_callback=budget.after_model,
        on_model_error_callback=recover_tool_call_errors(model) if isinstance(model, BaseLlm) else None,
    )

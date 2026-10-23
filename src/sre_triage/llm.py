"""Models for the ADK agents: LiteLLM through ADK's ``LiteLlm``, with a fallback chain.

Any LiteLLM model string works ("gemini/...", "groq/...", "openrouter/...", "ollama_chat/...").
Free tiers rate-limit bursts of parallel agents, so a failing model is skipped for a while
and the next model in SRE_FALLBACK_MODELS serves the call.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import AsyncGenerator
from typing import Any

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from google.adk.models import BaseLlm  # noqa: E402
from google.adk.models.lite_llm import LiteLlm  # noqa: E402
from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from pydantic import PrivateAttr  # noqa: E402

from .config import Settings  # noqa: E402

log = logging.getLogger(__name__)

PROVIDER_KEYS = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY",
                 "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
SUGGESTED_MODELS = [
    "gemini/gemini-3.5-flash", "gemini/gemini-flash-latest", "gemini/gemini-3.5-flash-lite",
    "gemini/gemini-3.1-flash-lite", "groq/openai/gpt-oss-120b", "groq/qwen/qwen3.8-27b",
    "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
]


def has_key(model: str) -> bool:
    var = PROVIDER_KEYS.get(model.split("/", 1)[0])
    return var is None or bool(os.environ.get(var))


_cooldown: dict[str, float] = {}  # model -> time it may be used again (shared by all agents in the process)
_lock = threading.Lock()


def _cooldown_for(error: Exception) -> float:
    text = str(error).lower()
    if "tool_use_failed" in text or "failed_generation" in text:
        return 0  # malformed tool call: the model, not the provider, is at fault
    if "per day" in text or "perday" in text or "daily" in text:
        return 6 * 3600
    if "429" in text or "rate" in text or "quota" in text or "resource_exhausted" in text:
        return 60
    if "401" in text or "403" in text or "api key" in text:
        return 24 * 3600
    return 30


class FallbackLlm(BaseLlm):
    """An ADK model that tries each LiteLLM model in order, skipping ones that recently failed."""

    _models: list[LiteLlm] = PrivateAttr()

    def __init__(self, models: list[str], settings: Settings):
        super().__init__(model=models[0])
        self._models = []
        for name in models:
            kwargs: dict[str, Any] = {"timeout": 60, "num_retries": 0}
            if not name.startswith("gemini/gemini-3"):  # Gemini 3 deprecates sampling parameters
                kwargs["temperature"] = settings.temperature
            self._models.append(LiteLlm(model=name, **kwargs))

    @property
    def names(self) -> list[str]:
        return [m.model for m in self._models]

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False
                                     ) -> AsyncGenerator[LlmResponse, None]:
        last: Exception | None = None
        for attempt in range(2):
            for inner in self._models:
                with _lock:
                    if _cooldown.get(inner.model, 0) > time.time():
                        continue
                request = llm_request.model_copy(deep=True)
                request.model = inner.model
                if inner.model.startswith("groq/"):  # Groq rejects reasoning text in the history
                    for content in request.contents:
                        content.parts = [p for p in content.parts or [] if not p.thought or p.function_call] or content.parts
                try:
                    responses = [r async for r in inner.generate_content_async(request, stream=False)]
                except Exception as exc:  # rate limits, overload, timeouts: next model
                    last = exc
                    seconds = _cooldown_for(exc)
                    if not seconds:
                        raise
                    log.warning("model %s unavailable for %.0fs: %s", inner.model, seconds, str(exc)[:160])
                    with _lock:
                        _cooldown[inner.model] = time.time() + seconds
                    continue
                for response in responses:
                    response.custom_metadata = {**(response.custom_metadata or {}), "served_by": inner.model}
                    yield response
                return
            wait = min((_cooldown.get(m, 0) for m in self.names), default=0) - time.time()
            if attempt == 0 and 0 < wait <= 65:  # everything cooling down briefly: wait once
                await asyncio.sleep(wait + 1)
                continue
            break
        raise RuntimeError(f"no model available ({', '.join(self.names)}): {str(last)[:300]}")


def make_model(settings: Settings, model: str | None = None) -> FallbackLlm:
    primary = model or settings.model
    chain = [primary] + [m for m in settings.fallback_models if m != primary]
    return FallbackLlm([m for m in chain if has_key(m)] or [primary], settings)

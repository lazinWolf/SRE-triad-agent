"""LLM selection through LiteLLM: model catalog, provider keys, roles, retries, fallbacks and probing.

Any LiteLLM model string works ("<provider>/<model>"). The catalog lists models known to
work with this project's tool-calling agents, so the dashboard and CLI can offer them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from pydantic import BaseModel, PrivateAttr

log = logging.getLogger(__name__)

from .config import Settings


class ProbeAnswer(BaseModel):
    service: str
    p95_ms: float
    evidence_ids: list[str]

# Provider prefix -> environment variable LiteLLM reads for that provider.
PROVIDER_KEYS = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
    "ollama_chat": None,
}


@dataclass(frozen=True)
class CatalogModel:
    id: str
    label: str
    notes: str = ""


CATALOG: list[CatalogModel] = [
    CatalogModel("gemini/gemini-3.8-flash", "Gemini 3.8 Flash (preview)", "slow at test time (~50 s per call)"),
    CatalogModel("gemini/gemini-3.5-flash", "Gemini 3.5 Flash", "recommended default; tested end to end"),
    CatalogModel("gemini/gemini-flash-latest", "Gemini Flash (latest alias)"),
    CatalogModel("gemini/gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite"),
    CatalogModel("gemini/gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite", "cheapest Gemini"),
    CatalogModel("gemini/gemma-4-31b-it", "Gemma 4 31B (Gemini API)", "open model; separate quota"),
    CatalogModel("groq/openai/gpt-oss-120b", "GPT-OSS 120B (Groq)", "very fast; low free-tier tokens/minute"),
    CatalogModel("groq/qwen/qwen3.8-27b", "Qwen 3.8 27B (Groq)"),
    CatalogModel("groq/openai/gpt-oss-20b", "GPT-OSS 20B (Groq)"),
    CatalogModel("openrouter/nvidia/nemotron-3-super-120b-a12b:free", "Nemotron 3 Super 120B (OpenRouter free)"),
    CatalogModel("openrouter/google/gemma-4-31b-it:free", "Gemma 4 31B (OpenRouter free)"),
    CatalogModel("openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", "Nemotron 3 Ultra (OpenRouter free)"),
]


def prepare_environment() -> None:
    """LiteLLM's Gemini provider reads GEMINI_API_KEY; accept the common GOOGLE_API_KEY too."""
    if not os.environ.get("GEMINI_API_KEY") and os.environ.get("GOOGLE_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
    # Keep LiteLLM quiet and avoid its remote model-cost fetch at import time.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def provider(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else "openai"


def has_credentials(model: str) -> bool:
    prepare_environment()
    var = PROVIDER_KEYS.get(provider(model), "")
    return var is None or bool(var and os.environ.get(var))


def available_models() -> list[CatalogModel]:
    return [m for m in CATALOG if has_credentials(m.id)]


def model_for_role(settings: Settings, role: str, case_model: str | None = None) -> str:
    """Per-case choice wins, then a per-role override, then the default model."""
    if case_model:
        return case_model
    if role == "coordinator" and settings.coordinator_model:
        return settings.coordinator_model
    if role in ("metrics", "traces", "logs_changes") and settings.specialist_model:
        return settings.specialist_model
    return settings.model


# --------------------------------------------------------------------------- rotation across models
NOTICES: deque[str] = deque(maxlen=200)  # model switches, drained into case events by the orchestrator
USAGE: dict[str, int] = {}
_LAST_SERVED: dict[str, str] = {}  # primary model -> model that served the previous call


class Cooldowns:
    """Models temporarily unusable (quota, rate limit, outage), shared across processes via a file."""

    def __init__(self, path: Path):
        self.path = path

    def _load(self) -> dict[str, float]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        now = time.time()
        return {m: t for m, t in data.items() if t > now}

    def until(self, model: str) -> float:
        return self._load().get(model, 0.0)

    def set(self, model: str, seconds: float) -> None:
        data = self._load()
        data[model] = max(data.get(model, 0), time.time() + seconds)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data))
        except OSError:
            pass

    def active(self) -> dict[str, float]:
        return self._load()


def classify_error(error: Exception) -> tuple[str, float]:
    """(kind, cooldown seconds) for a provider error."""
    text = str(error)
    lowered = text.lower()
    if any(m in text for m in ("tool_use_failed", "did not match schema", "Failed to call a function", "failed_generation")):
        return "tool_call", 0
    if "PerDay" in text or "per day" in lowered or "free-models-per-day" in lowered or "daily" in lowered:
        return "daily_quota", 6 * 3600
    if "429" in text or "rate" in lowered or "quota" in lowered or "resource_exhausted" in lowered:
        return "rate_limit", 75
    if "401" in text or "403" in text or "api key" in lowered or "authentication" in lowered:
        return "auth", 24 * 3600
    if "context" in lowered and ("length" in lowered or "too long" in lowered or "maximum" in lowered):
        return "context", 3600
    return "error", 45


class RotatingLlm(BaseLlm):
    """Tries the configured models in order, skipping ones on cooldown.

    Free tiers cap requests per minute and per day per model; instead of retrying an
    exhausted model with backoff, switch to the next one at once and remember why."""

    _models: list[Any] = PrivateAttr()
    _cooldowns: Cooldowns = PrivateAttr()

    def __init__(self, models: list[str], settings: Settings):
        super().__init__(model=models[0])
        from google.adk.models.lite_llm import LiteLlm

        self._models = []
        for name in models:
            kwargs: dict[str, Any] = {"temperature": settings.model_temperature, "max_tokens": settings.model_max_tokens,
                                      "num_retries": 0, "timeout": settings.model_timeout}
            if provider(name) in ("ollama", "ollama_chat"):
                kwargs["num_ctx"] = settings.ollama_num_ctx
            self._models.append(LiteLlm(model=name, **kwargs))
        self._cooldowns = Cooldowns(settings.data_dir / "model_cooldowns.json")

    @property
    def model_names(self) -> list[str]:
        return [m.model for m in self._models]

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        last_error: Exception | None = None
        for attempt in range(2):
            for inner in self._models:
                name = inner.model
                if self._cooldowns.until(name) > time.time():
                    continue
                request = llm_request.model_copy(deep=True)
                request.model = name
                try:
                    responses = [r async for r in inner.generate_content_async(request, stream=False)]
                except Exception as exc:
                    kind, cooldown = classify_error(exc)
                    if kind == "tool_call":
                        raise  # handled by the agent's tool-call recovery callback
                    last_error = exc
                    if cooldown:
                        self._cooldowns.set(name, cooldown)
                    NOTICES.append(f"{name} unavailable ({kind}); switching model")
                    log.warning("model %s failed (%s): %s", name, kind, str(exc)[:200])
                    continue
                primary = self.model_names[0]
                if _LAST_SERVED.get(primary, primary) != name:
                    NOTICES.append(f"now served by {name}" + (" (fallback)" if name != primary else ""))
                _LAST_SERVED[primary] = name
                USAGE[name] = USAGE.get(name, 0) + 1
                for response in responses:
                    yield response
                return
            # Everything is cooling down: wait for the soonest short cooldown once, then retry.
            soonest = min((self._cooldowns.until(m) for m in self.model_names), default=0) - time.time()
            if attempt == 0 and 0 < soonest <= 90:
                NOTICES.append(f"all models rate-limited; waiting {soonest:.0f}s")
                await asyncio.sleep(soonest + 1)
                continue
            break
        raise RuntimeError(f"No model available ({', '.join(self.model_names)}): {last_error}")


def model_chain(settings: Settings, primary: str) -> list[str]:
    chain = [primary] + [m for m in settings.fallback_models if m != primary]
    return [m for m in chain if has_credentials(m)] or [primary]


def litellm_model(settings: Settings, model: str) -> BaseLlm:
    """An ADK model for `model`, rotating to the configured fallbacks on quota/rate limits."""
    prepare_environment()
    return RotatingLlm(model_chain(settings, model), settings)


async def probe(settings: Settings, model: str) -> dict[str, Any]:
    """Check a model end to end the way agents use it: a tool call, then a structured submission."""
    from google.adk.tools import ToolContext

    from .agents.runtime import LlmBudget, Submission, build_agent, run_agent

    if not has_credentials(model):
        return {"model": model, "ok": False, "error": f"missing {PROVIDER_KEYS.get(provider(model))}"}
    submission = Submission()

    async def service_latency(service: str) -> dict:
        """Return the measured p95 latency of a service."""
        return {"evidence_id": "ev_probe1", "service": service, "p95_ms": 412.5}

    async def submit_answer(answer: ProbeAnswer, tool_context: ToolContext) -> dict:
        """Submit the structured answer."""
        from .agents.team import coerce

        value, errors = coerce(ProbeAnswer, answer)
        if errors:
            return submission.reject(errors)
        return submission.accept(tool_context, value)

    budget = LlmBudget(6)
    agent = build_agent("probe", litellm_model(settings.model_copy(update={"fallback_models": []}), model),
                        "Call service_latency for the requested service, then call submit_answer citing the evidence_id.",
                        [service_latency, submit_answer], budget)
    started = time.monotonic()
    try:
        result = await asyncio.wait_for(run_agent(agent, "What is the p95 latency of cart?", submission, max_llm_calls=5),
                                        timeout=120)
    except TimeoutError:
        return {"model": model, "ok": False, "error": "timed out after 120s"}
    value = result.submitted
    ok = bool(value) and value.service == "cart" and abs(value.p95_ms - 412.5) < 1 and "ev_probe1" in value.evidence_ids
    return {"model": model, "ok": ok, "seconds": round(time.monotonic() - started, 1), "llm_calls": budget.calls,
            "tokens": budget.prompt_tokens + budget.completion_tokens, "error": None if ok else (result.error or "wrong or no submission")}

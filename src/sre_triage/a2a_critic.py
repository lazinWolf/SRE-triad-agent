"""The Critic as a standalone A2A service (Agent2Agent protocol), built with ADK's ``to_a2a``.

    uv run sre-triage a2a-critic --port 8001      # agent card: http://127.0.0.1:8001/.well-known/agent-card.json
    SRE_CRITIC_A2A_URL=http://127.0.0.1:8001      # the workflow then reviews reports through RemoteA2aAgent

The service runs in its own process with its own ADK Runner and session. The workflow sends it the report
and evidence as an A2A message and validates the JSON answer with the same Pydantic ``Critique`` schema.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from google.adk.agents import LlmAgent
from google.adk.models import BaseLlm
from pydantic import ValidationError

from .config import Settings
from .llm import make_model
from .prompts import CRITIC_A2A
from .schemas import Critique

CARD_PATH = "/.well-known/agent-card.json"


def critic_agent(model: BaseLlm) -> LlmAgent:
    return LlmAgent(name="critic", model=model, instruction=lambda _ctx: CRITIC_A2A,
                    description="Skeptical senior SRE that reviews an incident root-cause report against its "
                                "evidence and answers accept/revise with follow-up tasks, as JSON.")


def build_app(model: BaseLlm, host: str = "127.0.0.1", port: int = 8001):
    from google.adk.a2a.utils.agent_to_a2a import to_a2a

    return to_a2a(critic_agent(model), host=host, port=port)


def serve(settings: Settings, host: str = "127.0.0.1", port: int = 8001) -> None:
    import uvicorn

    uvicorn.run(build_app(make_model(settings), host, port), host=host, port=port, log_level="warning")


def agent_card(url: str) -> dict[str, Any]:
    response = httpx.get(url.rstrip("/") + CARD_PATH, timeout=5)
    response.raise_for_status()
    return response.json()


def parse_review(text: str) -> dict[str, Any]:
    """Pydantic validation at the service boundary: the remote answer must be a valid Critique."""
    body = (text or "").strip()
    if body.startswith("```"):
        body = body.strip("`").removeprefix("json").strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object in A2A answer: {body[:200]!r}")
    try:
        return Critique.model_validate(json.loads(body[start:end + 1])).model_dump()
    except (ValueError, ValidationError) as exc:
        raise ValueError(f"A2A answer failed Critique validation: {exc}") from exc

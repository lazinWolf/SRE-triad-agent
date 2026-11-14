"""Client side of the review board: a CrewAI crew served over A2A (Agent2Agent) by ``services/crew_reviewer``.

    cd services/crew_reviewer && uv run python server.py --port 8002
    SRE_REVIEWER_A2A_URL=http://127.0.0.1:8002      # the workflow then reviews reports through RemoteA2aAgent

The crew runs in its own process and Python environment (CrewAI and ADK cannot share one). The workflow sends
it the report and evidence as an A2A message and validates the JSON answer with the Pydantic ``Critique`` schema.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import ValidationError

from .schemas import Critique

CARD_PATH = "/.well-known/agent-card.json"


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
        data = json.loads(body[start:end + 1])
        review = Critique.model_validate(data).model_dump()
    except (ValueError, ValidationError) as exc:
        raise ValueError(f"A2A answer failed Critique validation: {exc}") from exc
    if isinstance(data.get("usage"), dict):  # the crew reports its own LLM usage: {model, llm_requests, *_tokens}
        review["usage"] = {k: data["usage"].get(k) for k in ("model", "llm_requests", "prompt_tokens", "completion_tokens")}
    return review

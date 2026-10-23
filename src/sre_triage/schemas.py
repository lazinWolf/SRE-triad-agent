"""Structured outputs exchanged between agents (validated; invalid answers go back to the model)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["cpu_limit", "cpu_saturation", "memory_exhaustion", "dependency_errors",
                   "dependency_unavailable", "lock_contention", "cache_failure", "config_change", "unknown"]
CATEGORY_HELP = {
    "cpu_limit": "container CPU limit too low: CPU throttling, usage pinned at the limit",
    "cpu_saturation": "process burns CPU (busy loop/load) with a normal limit",
    "memory_exhaustion": "memory growth, OOM kills or restarts",
    "dependency_errors": "a service returns errors to its callers (failed operations)",
    "dependency_unavailable": "a service cannot be reached at all (connection refused/timeouts)",
    "lock_contention": "requests serialise on a lock: latency up, CPU not saturated",
    "cache_failure": "cache misses/evictions make a service slow",
    "config_change": "a feature flag or setting changed behaviour (when no category above fits)",
    "unknown": "not determined",
}
AgentName = Literal["metrics", "traces", "logs_changes"]


def _confidence(value):
    if isinstance(value, str):
        return {"high": 0.85, "medium": 0.6, "low": 0.3}.get(value.lower().strip(), 0.5)
    return value


class Finding(BaseModel):
    summary: str = Field(description="what you found, with numbers")
    suspect_service: str | None = Field(None, description="service you believe is the origin, or null")
    category: Category = "unknown"
    confidence: float = Field(0.5, ge=0, le=1)
    evidence_ids: list[str] = []
    ruled_out: list[str] = Field([], description="leads you checked and rejected")

    _conf = field_validator("confidence", mode="before")(_confidence)


class Task(BaseModel):
    agent: AgentName
    question: str


class Plan(BaseModel):
    hypotheses: list[str] = Field(description="2-4 competing explanations to test")
    tasks: list[Task] = Field(min_length=1, max_length=3)


class Remediation(BaseModel):
    action: Literal["set_cpu_limit", "set_feature_flag", "restart_service"]
    service: str | None = None
    cpus: float | None = None
    flag: str | None = None
    variant: str | None = None


class Hypothesis(BaseModel):
    statement: str
    status: Literal["supported", "rejected", "open"]
    evidence_ids: list[str] = []


class Report(BaseModel):
    status: Literal["diagnosed", "inconclusive"]
    summary: str
    root_cause_service: str | None = None
    category: Category = "unknown"
    mechanism: str = ""
    confidence: float = Field(0.5, ge=0, le=1)
    evidence_ids: list[str] = []
    hypotheses: list[Hypothesis] = []
    remediation: Remediation | None = None

    _conf = field_validator("confidence", mode="before")(_confidence)


class Critique(BaseModel):
    verdict: Literal["accept", "revise"]
    issues: list[str] = []
    follow_up: list[Task] = Field([], max_length=2)


def schema_hint(model: type[BaseModel]) -> str:
    """Compact JSON schema for prompts."""
    import json

    return json.dumps(model.model_json_schema(), separators=(",", ":"))

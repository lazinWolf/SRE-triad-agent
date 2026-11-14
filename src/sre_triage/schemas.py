"""Structured outputs exchanged between agents (validated; invalid answers go back to the model)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["cpu_limit", "cpu_saturation", "memory_limit", "memory_exhaustion", "dependency_errors",
                   "dependency_unavailable", "service_down", "lock_contention", "cache_failure", "traffic_spike",
                   "config_change", "unknown"]
CATEGORY_HELP = {
    "cpu_limit": "container CPU limit too low: CPU throttling, usage pinned at the limit",
    "cpu_saturation": "process burns CPU (busy loop/load) with a normal limit",
    "memory_limit": "container memory limit too low: OOM kills and restarts right after start",
    "memory_exhaustion": "memory grows (leak) until OOM kills or restarts",
    "dependency_errors": "a service returns errors to its callers (failed operations)",
    "dependency_unavailable": "a running service cannot be reached (bad address, connection refused, timeouts)",
    "service_down": "a container is stopped, crashed or paused: its traffic drops to zero",
    "lock_contention": "requests serialise on a lock: latency up, CPU not saturated",
    "cache_failure": "cache misses/evictions or a broken cache make a service slow",
    "traffic_spike": "request rate far above normal overloads the system",
    "config_change": "a feature flag or setting changed behaviour (when no category above fits)",
    "unknown": "not determined",
}
AgentName = Literal["metrics", "traces", "logs_changes", "infra"]
Action = Literal["set_cpu_limit", "set_memory_limit", "set_feature_flag", "restart_service", "start_service",
                 "unpause_service"]


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
    """One investigation round the coordinator dispatches."""

    hypotheses: list[str] = Field(description="2-4 competing explanations to test")
    tasks: list[Task] = Field(min_length=1, max_length=3)


class Question(BaseModel):
    """A question the coordinator asks the human operator."""

    question: str = Field(min_length=8)


class Remediation(BaseModel):
    action: Action
    service: str | None = None
    cpus: float | None = None
    memory_mb: float | None = None
    flag: str | None = None
    variant: str | None = None
    restart_after: bool = Field(False, description="also restart `service` after turning the flag off")


class Hypothesis(BaseModel):
    statement: str
    status: Literal["supported", "rejected", "open"]
    evidence_ids: list[str] = []


class Report(BaseModel):
    status: Literal["diagnosed", "inconclusive", "no_incident"]
    summary: str
    root_cause_service: str | None = None
    category: Category = "unknown"
    mechanism: str = ""
    confidence: float = Field(0.5, ge=0, le=1)
    evidence_ids: list[str] = []
    hypotheses: list[Hypothesis] = []
    remediation: Remediation | None = None
    alternative: Remediation | None = Field(None, description="a second option the operator may choose instead")

    _conf = field_validator("confidence", mode="before")(_confidence)


class Critique(BaseModel):
    verdict: Literal["accept", "revise"]
    issues: list[str] = []
    follow_up: list[Task] = Field([], max_length=2)
    risk: Literal["low", "medium", "high"] = "low"   # of the proposed remediation
    risk_notes: str = ""

"""Domain schemas shared by the ledger, agents, validation and UI."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from .config import APP_SERVICES, DATA_STORES

KNOWN_SERVICES = APP_SERVICES + DATA_STORES

AgentRole = Literal["metrics", "traces", "logs_changes"]
ROLES: tuple[AgentRole, ...] = ("metrics", "traces", "logs_changes")

# Mechanism categories, not scenario names: a diagnosis is scored on the service
# and mechanism it identifies, independent of how the fault was injected.
CauseCategory = Literal[
    "cpu_limit",            # CPU quota/limit too low -> throttling
    "cpu_saturation",       # CPU exhausted by workload or code path
    "memory_exhaustion",    # leak, OOM kills, restarts
    "dependency_errors",    # a service returns errors to its callers
    "dependency_unavailable",  # a service cannot be reached
    "lock_contention",      # contention inside a service or its database
    "cache_failure",        # cache misbehaves
    "slow_dependency",      # latency injected in a downstream call
    "traffic_surge",        # offered load exceeds capacity
    "other",
]
CAUSE_CATEGORIES: tuple[str, ...] = CauseCategory.__args__  # type: ignore[attr-defined]
CATEGORY_HELP = (
    "Mechanism at the root-cause service: cpu_limit = container CPU limit/quota too low (throttling); "
    "cpu_saturation = CPU exhausted by work; memory_exhaustion = leak, OOM kills, restarts; "
    "dependency_errors = the service returns errors; dependency_unavailable = it cannot be reached; "
    "lock_contention = locks in the service or its database serialise work; cache_failure = cache misbehaves; "
    "slow_dependency = injected call/network delay; traffic_surge = load exceeds capacity; "
    "other = only if none fits.")

HypothesisStatus = Literal["candidate", "supported", "weakened", "rejected"]
CaseStatus = Literal[
    "open", "investigating", "needs_review", "awaiting_approval", "remediating",
    "verifying", "recovered", "reopened", "closed",
]
EvidenceStatus = Literal["ok", "empty", "unavailable", "error", "invalid", "budget_exhausted"]
ActionKind = Literal["set_cpu_limit", "set_feature_flag", "restart_service"]


class CaseInput(BaseModel):
    """An alert or operator report that opens an investigation."""

    service: str
    symptom: str = Field(min_length=3, max_length=2000)
    incident_start: float
    incident_end: float
    baseline_seconds: float = 600
    severity: Literal["critical", "warning", "info"] = "warning"
    source: Literal["alert_rule", "operator", "evaluation"] = "operator"
    mode: Literal["multi_agent", "single_agent"] = "multi_agent"
    use_memory: bool = True
    model: str = Field("", description="LiteLLM model for this case; empty uses the configured default")

    @field_validator("service")
    @classmethod
    def known_service(cls, value: str) -> str:
        if value not in KNOWN_SERVICES:
            raise ValueError(f"Unknown service {value!r}; expected one of {', '.join(KNOWN_SERVICES)}")
        return value

    @model_validator(mode="after")
    def valid_window(self):
        values = (self.incident_start, self.incident_end, self.baseline_seconds)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Times must be finite")
        if self.incident_end <= self.incident_start:
            raise ValueError("incident_end must be after incident_start")
        if self.incident_end - self.incident_start > 3 * 3600:
            raise ValueError("Incident windows are limited to three hours")
        if not 60 <= self.baseline_seconds <= 3 * 3600:
            raise ValueError("Baseline must be between one minute and three hours")
        return self


class Observation(BaseModel):
    statement: str = Field(description="One concrete observation or inference.")
    kind: Literal["observation", "inference"] = Field(
        "observation", description="observation = directly read from evidence; inference = your interpretation.")
    evidence_ids: list[str] = Field(default_factory=list, description="Evidence IDs returned by your tool calls.")


class HypothesisSignal(BaseModel):
    hypothesis: str = Field(description="Hypothesis ID such as H1, or a short new hypothesis statement.")
    effect: Literal["supports", "contradicts", "inconclusive"]
    evidence_ids: list[str] = Field(default_factory=list)
    note: str = ""


class Finding(BaseModel):
    """Specialist output for one delegated task."""

    summary: str = Field(description="Two to four sentences answering the assigned question.")
    observations: list[Observation] = Field(default_factory=list)
    hypothesis_signals: list[HypothesisSignal] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list, description="Missing or ambiguous data.")
    suggested_checks: list[str] = Field(default_factory=list)


class Hypothesis(BaseModel):
    id: str = Field(description="Stable ID such as H1. Keep IDs stable across rounds.")
    # Models often name this field differently; accept the common variants.
    statement: str = Field(validation_alias=AliasChoices(
        "statement", "hypothesis", "description", "text", "cause", "title", "name", "summary", "label"))
    service: str = Field(description="Service the cause is located in.")
    category: CauseCategory = Field(description=CATEGORY_HELP)
    status: HypothesisStatus
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    rationale: str = ""


class Task(BaseModel):
    agent: AgentRole
    question: str = Field(min_length=8, max_length=1200, description="A specific question that can distinguish hypotheses.")
    hypothesis_ids: list[str] = Field(default_factory=list)


class RemediationProposal(BaseModel):
    action: ActionKind
    service: str
    cpus: float | None = Field(None, description="For set_cpu_limit: target CPU limit, e.g. 1.0")
    flag: str | None = Field(None, description="For set_feature_flag: flag key")
    variant: str | None = Field(None, description="For set_feature_flag: target variant, normally 'off'")
    rationale: str
    evidence_ids: list[str] = Field(min_length=1)
    expected_effect: str = ""


class RootCause(BaseModel):
    service: str
    category: CauseCategory = Field(description=CATEGORY_HELP)
    description: str
    trigger_change_id: str | None = Field(None, description="Change ID from get_config_changes, if a change triggered it.")
    evidence_ids: list[str] = Field(min_length=1)


class Report(BaseModel):
    summary: str
    diagnosis_status: Literal["diagnosed", "insufficient_evidence"]
    confidence: Literal["low", "medium", "high"] = "low"
    root_cause: RootCause | None = None
    impact: str = ""
    timeline: list[str] = Field(default_factory=list)
    next_checks: list[str] = Field(default_factory=list)
    remediation: RemediationProposal | None = None
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent(self):
        if self.diagnosis_status == "diagnosed" and self.root_cause is None:
            raise ValueError("A diagnosed report needs root_cause")
        return self


class CoordinatorDecision(BaseModel):
    reasoning: str = Field(description="Brief reconciliation of findings; name conflicts explicitly.")
    hypotheses: list[Hypothesis] = Field(default_factory=list, description="The full current hypothesis board.")
    next_step: Literal["investigate", "conclude"]
    tasks: list[Task] = Field(default_factory=list, description="Targeted tasks when next_step is investigate.")
    report: Report | None = Field(None, description="Required when next_step is conclude.")

    @model_validator(mode="after")
    def coherent(self):
        if self.next_step == "investigate" and not self.tasks:
            raise ValueError("next_step=investigate needs at least one task")
        if self.next_step == "conclude" and self.report is None:
            raise ValueError("next_step=conclude needs a report")
        return self


class Critique(BaseModel):
    """The critic's independent review of a proposed diagnosis."""

    verdict: Literal["accept", "challenge"] = Field(
        description="accept = the diagnosis survives your checks; challenge = a material gap remains.")
    summary: str = Field(description="Two or three sentences on what you checked and concluded.")
    issues: list[str] = Field(default_factory=list, description="Concrete weaknesses: timing, magnitude, alternatives, missing checks.")
    alternative_hypotheses: list[str] = Field(default_factory=list)
    requested_checks: list[Task] = Field(default_factory=list, max_length=3,
                                         description="Specific follow-up tasks for specialists when challenging.")
    evidence_ids: list[str] = Field(default_factory=list, description="Evidence you gathered or relied on.")


class FeedbackInput(BaseModel):
    text: str = Field(min_length=3, max_length=4000)


class ReviewInput(BaseModel):
    outcome: Literal["resolved", "mitigated", "unresolved", "false_alarm"]
    confirmed_service: str = ""
    confirmed_category: str = ""
    confirmed_cause: str = Field("", max_length=2000)
    notes: str = Field("", max_length=4000)

"""Validated case inputs and coordinator decisions."""

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class CaseInput(BaseModel):
    service: Literal["checkout", "inventory"] = "checkout"
    environment: Literal["local"] = "local"
    summary: str = Field(min_length=3, max_length=2000)
    start: float
    end: float
    incident_start: float | None = None
    mode: Literal["live", "replay"] = "replay"

    @model_validator(mode="after")
    def valid_window(self):
        if not all(math.isfinite(x) for x in (self.start, self.end)):
            raise ValueError("Timestamps must be finite")
        if self.start >= self.end or self.end - self.start > 3600:
            raise ValueError("Use a positive investigation window of at most one hour")
        if self.incident_start is not None and not (
            math.isfinite(self.incident_start) and self.start < self.incident_start < self.end
        ):
            raise ValueError("incident_start must be inside the investigation window")
        return self


class Hypothesis(BaseModel):
    cause: str
    service: str
    status: Literal["candidate", "supported", "weakened", "rejected"]
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)


class ActionProposal(BaseModel):
    service: str
    description: str
    reason: str
    rollback: str
    evidence_ids: list[str] = Field(min_length=1)


class InvestigationReport(BaseModel):
    summary: str
    diagnosis_status: Literal["supported", "insufficient_evidence"]
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    next_checks: list[str] = Field(default_factory=list)
    proposed_action: ActionProposal | None = None
    limitations: list[str] = Field(default_factory=list)
    scripted_mode: bool = False


class FollowUp(BaseModel):
    agent: Literal["metrics", "traces", "changes"]
    question: str = Field(min_length=5, max_length=1000)


class CoordinatorDecision(BaseModel):
    next_step: Literal["follow_up", "report"]
    reasoning: str
    follow_ups: list[FollowUp] = Field(default_factory=list, max_length=3)
    report: InvestigationReport | None = None

    @model_validator(mode="after")
    def coherent_decision(self):
        if self.next_step == "follow_up" and not self.follow_ups:
            raise ValueError("Follow-up decisions need a concrete task")
        if self.next_step == "report" and self.report is None:
            raise ValueError("Report decisions need a report")
        return self


class FeedbackInput(BaseModel):
    text: str = Field(min_length=3, max_length=4000)


class ReviewInput(BaseModel):
    confirmed_cause: str = Field(default="", max_length=2000)
    outcome: Literal["unresolved", "mitigated", "resolved"]
    notes: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def resolved_context(self):
        # Recovery and a confirmed cause are separate facts. Unknown is legitimate.
        if self.confirmed_cause and len(self.confirmed_cause.strip()) < 3:
            raise ValueError("Describe the confirmed cause or leave it unknown")
        return self

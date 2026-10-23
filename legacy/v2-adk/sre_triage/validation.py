"""Deterministic checks on agent outputs before they are accepted.

Errors are returned to the submitting model so it can correct itself; anything
still invalid after the allowed attempts is replaced by an explicit
insufficient-evidence result rather than shown as a diagnosis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .models import KNOWN_SERVICES, CoordinatorDecision, Finding, Hypothesis, Report
from .remediation import validate_proposal


class EvidenceIndex:
    def __init__(self, evidence: Iterable[dict[str, Any]], change_ids: Iterable[str] = ()):
        self.rows = {e["id"]: e for e in evidence}
        self.change_ids = set(change_ids)

    def unknown(self, ids: Iterable[str]) -> list[str]:
        return [i for i in ids if i not in self.rows]

    def usable(self, ids: Iterable[str]) -> list[dict[str, Any]]:
        return [self.rows[i] for i in ids if i in self.rows and self.rows[i]["status"] == "ok"]


def check_finding(finding: Finding, index: EvidenceIndex) -> list[str]:
    cited = [i for o in finding.observations for i in o.evidence_ids]
    cited += [i for s in finding.hypothesis_signals for i in s.evidence_ids]
    bad = index.unknown(cited)
    return [f"Unknown evidence IDs (cite only IDs your tools returned): {', '.join(sorted(set(bad)))}"] if bad else []


def check_hypotheses(hypotheses: list[Hypothesis], index: EvidenceIndex) -> list[str]:
    errors = []
    ids = [h.id for h in hypotheses]
    if len(ids) != len(set(ids)):
        errors.append("Hypothesis IDs must be unique")
    for h in hypotheses:
        if h.service not in KNOWN_SERVICES:
            errors.append(f"{h.id}: unknown service {h.service!r}")
        bad = index.unknown(h.supporting_evidence + h.contradicting_evidence)
        if bad:
            errors.append(f"{h.id}: unknown evidence IDs {', '.join(sorted(set(bad)))}")
        if h.status == "supported":
            tools = {e["tool"] for e in index.usable(h.supporting_evidence)}
            if len(tools) < 2:
                errors.append(f"{h.id}: 'supported' needs usable supporting evidence from at least two different tools "
                              f"(has {len(tools)}). Mark it 'candidate' or gather more evidence.")
        if h.status in ("weakened", "rejected") and not h.contradicting_evidence:
            errors.append(f"{h.id}: '{h.status}' needs contradicting_evidence")
    return errors


def check_report(report: Report, hypotheses: list[Hypothesis], index: EvidenceIndex, flagd_file: Path,
                 allowed_flag_variants: dict[str, set[str]] | None = None) -> list[str]:
    errors = []
    if report.root_cause is not None:
        rc = report.root_cause
        if rc.service not in KNOWN_SERVICES:
            errors.append(f"root_cause.service {rc.service!r} is unknown")
        bad = index.unknown(rc.evidence_ids)
        if bad:
            errors.append(f"root_cause cites unknown evidence {', '.join(bad)}")
        if report.diagnosis_status == "diagnosed":
            if not index.usable(rc.evidence_ids):
                errors.append("root_cause needs at least one usable (status ok) evidence ID")
            if not any(h.status == "supported" and h.service == rc.service for h in hypotheses):
                errors.append("A diagnosed root cause must match a 'supported' hypothesis on the board for the same service")
        if rc.trigger_change_id and rc.trigger_change_id not in index.change_ids:
            errors.append(f"trigger_change_id {rc.trigger_change_id!r} is not a recorded change ID from config_changes")
    if report.remediation is not None:
        proposal = report.remediation.model_dump()
        errors += [f"remediation: {e}" for e in validate_proposal(proposal, flagd_file,
                                                                   allowed_flag_variants=allowed_flag_variants)]
        bad = index.unknown(report.remediation.evidence_ids)
        if bad:
            errors.append(f"remediation cites unknown evidence {', '.join(bad)}")
        elif not index.usable(report.remediation.evidence_ids):
            errors.append("remediation needs usable (status ok) supporting evidence")
        if report.diagnosis_status != "diagnosed":
            errors.append("Only a diagnosed report may propose remediation")
    return errors


def check_decision(decision: CoordinatorDecision, index: EvidenceIndex, flagd_file: Path, *,
                   max_tasks: int, allowed_flag_variants: dict[str, set[str]] | None = None) -> list[str]:
    errors = check_hypotheses(decision.hypotheses, index)
    if decision.next_step == "investigate" and len(decision.tasks) > max_tasks:
        errors.append(f"At most {max_tasks} tasks per round")
    if decision.report is not None and decision.next_step == "conclude":
        errors += check_report(decision.report, decision.hypotheses, index, flagd_file, allowed_flag_variants)
    return errors


def unsupported_claims(report: dict[str, Any] | None, hypotheses: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> int:
    """Count claims that cite missing or unusable evidence (for evaluation)."""
    index = EvidenceIndex(evidence)
    count = 0
    for h in hypotheses or []:
        if h.get("status") == "supported":
            if len({e["tool"] for e in index.usable(h.get("supporting_evidence", []))}) < 2:
                count += 1
        count += len(index.unknown(h.get("supporting_evidence", []) + h.get("contradicting_evidence", [])))
    rc = (report or {}).get("root_cause") or None
    if rc and (report or {}).get("diagnosis_status") == "diagnosed":
        if not index.usable(rc.get("evidence_ids", [])):
            count += 1
        count += len(index.unknown(rc.get("evidence_ids", [])))
    return count

"""Approved action execution, recovery verification and reopening.

Recovery is judged against the original symptom (p95 latency and error ratio of
the case's service) over a fresh observation window after the action, compared
with the case baseline. Persistent symptoms reopen the investigation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

from . import analysis
from .config import Settings
from .ledger import Ledger
from .remediation import RemediationError, Remediator, describe, validate_proposal
from .telemetry import Prometheus, TelemetryUnavailable, red_queries

log = logging.getLogger(__name__)


def execute_action(ledger: Ledger, settings: Settings, remediator: Remediator, action_id: str) -> dict[str, Any]:
    action = ledger.action(action_id)
    if action["status"] != "approved":
        raise ValueError(f"Action {action_id} is {action['status']}; only approved actions execute")
    case_id = action["case_id"]
    proposal = action["proposal"]
    # Re-validate at execution time: the environment may have changed since the proposal.
    errors = validate_proposal(proposal, settings.flagd_file, allowed_flag_variants={
        proposal.get("flag"): {proposal.get("variant")}} if proposal.get("action") == "set_feature_flag" else None)
    if errors:
        ledger.set_action(action_id, status="failed", error="; ".join(errors))
        ledger.set_case(case_id, status="needs_review")
        ledger.event(case_id, None, "remediation", "action_failed", "Validation failed at execution: " + "; ".join(errors))
        return ledger.action(action_id)
    ledger.set_case(case_id, status="remediating")
    try:
        before = remediator.current_state(proposal)
        after = remediator.execute(proposal)
    except (RemediationError, Exception) as exc:
        ledger.set_action(action_id, status="failed", error=str(exc)[:500], executed_at=time.time())
        ledger.set_case(case_id, status="needs_review")
        ledger.event(case_id, None, "remediation", "action_failed", f"{describe(proposal)} failed: {exc}")
        return ledger.action(action_id)
    ledger.set_action(action_id, status="executed", executed_at=time.time(), before_state=before, after_state=after)
    ledger.set_case(case_id, status="verifying")
    ledger.event(case_id, None, "remediation", "action_executed", f"Executed: {describe(proposal)}",
                 {"before": before, "after": after})
    return ledger.action(action_id)


def rollback_action(ledger: Ledger, remediator: Remediator, action_id: str) -> dict[str, Any]:
    action = ledger.action(action_id)
    plan = action.get("rollback") or {}
    if action["status"] != "executed" or not plan.get("available"):
        raise ValueError("Only executed actions with an available rollback can be rolled back")
    proposal = {k: plan.get(k) for k in ("action", "service", "cpus", "flag", "variant") if plan.get(k) is not None}
    remediator.execute(proposal)
    ledger.set_action(action_id, status="rolled_back")
    ledger.event(action["case_id"], None, "remediation", "rollback", f"Rolled back: {plan.get('description')}")
    return ledger.action(action_id)


async def window_red(prometheus: Prometheus, service: str, start: float, end: float) -> dict[str, float | None]:
    queries = red_queries(service, end - start)
    values = await asyncio.gather(*(prometheus.scalar(q, end) for q in queries.values()))
    return dict(zip(queries, values))


async def verify_recovery(ledger: Ledger, settings: Settings, prometheus: Prometheus, case_id: str,
                          action_id: str | None, *, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                          clock: Callable[[], float] = time.time) -> dict[str, Any]:
    policy = settings.recovery
    case = ledger.case(case_id)
    check_id = ledger.add_recovery_check(case_id, action_id)
    executed_at = ledger.action(action_id)["executed_at"] if action_id else clock()
    start = executed_at + policy.settle_seconds
    end = start + policy.observe_seconds
    ledger.event(case_id, None, "recovery", "recovery_started",
                 f"Observing {case['service']} for {policy.observe_seconds:.0f}s after a {policy.settle_seconds:.0f}s settle period")
    wait = end - clock() + 20  # allow span metrics to flush past the window end
    if wait > 0:
        await sleep(wait)
    try:
        baseline, after = await asyncio.gather(
            window_red(prometheus, case["service"], case["baseline_start"], case["baseline_end"]),
            window_red(prometheus, case["service"], start, end))
    except TelemetryUnavailable as exc:
        result = {"verdict": "inconclusive", "error": str(exc)}
    else:
        result = analysis.recovered(baseline, after, latency_ratio=policy.latency_ratio,
                                    latency_slack_ms=policy.latency_slack_ms, error_rate_slack=policy.error_rate_slack)
        result.update({"baseline": baseline, "after": after, "window": [start, end]})
    verdict = result["verdict"]
    ledger.set_recovery_check(check_id, verdict, result)
    if verdict == "recovered":
        ledger.set_case(case_id, status="recovered")
        ledger.event(case_id, None, "recovery", "recovered", "Symptoms returned to baseline after the action")
    elif verdict == "not_recovered":
        # Persistent symptoms reopen the investigation over a window that includes the post-action period.
        ledger.set_case(case_id, status="reopened", incident_end=max(case["incident_end"], end))
        ledger.event(case_id, None, "recovery", "reopened",
                     "Symptoms persisted after the action; the investigation is reopened with this result as context")
    else:
        ledger.set_case(case_id, status="needs_review")
        ledger.event(case_id, None, "recovery", "inconclusive", "Recovery could not be judged; operator review needed")
    return {"check_id": check_id, **result}

"""Deterministic alert rules that open cases from live telemetry.

A rule fires when a user-facing service's recent p95 latency or error ratio is
clearly worse than its own recent history, on consecutive evaluations. Alerts
for one underlying problem are grouped onto one open case.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from .ledger import Ledger
from .models import CaseInput
from .telemetry import ERROR_STATUS, SERVER_KIND, SPAN_CALLS, SPAN_DURATION_BUCKET, Prometheus, red_queries

WATCHED = ("frontend", "checkout", "cart", "product-catalog", "payment", "currency", "shipping", "recommendation", "ad")


@dataclass
class AlertRules:
    recent_s: float = 120
    history_s: float = 900
    history_gap_s: float = 300
    latency_ratio: float = 2.5
    latency_min_increase_ms: float = 200
    error_ratio_min: float = 0.05
    min_rate: float = 0.05
    consecutive: int = 2
    group_window_s: float = 1200
    pending: dict[str, int] = field(default_factory=dict)

    async def evaluate(self, prometheus: Prometheus, now: float | None = None) -> list[dict[str, Any]]:
        now = now or time.time()
        firing = []
        for service in WATCHED:
            recent_q = red_queries(service, self.recent_s)
            hist_end = now - self.history_gap_s
            hist_q = history_queries(service, self.history_s)
            recent, history = await asyncio.gather(
                asyncio.gather(*(prometheus.scalar(q, now) for q in recent_q.values())),
                asyncio.gather(*(prometheus.scalar(q, hist_end) for q in hist_q.values())))
            r, h = dict(zip(recent_q, recent)), dict(zip(hist_q, history))
            reasons = []
            if (r.get("request_rate_per_s") or 0) >= self.min_rate:
                rp, hp = r.get("p95_latency_ms"), h.get("p95_latency_ms")
                if rp is not None and hp is not None and rp > max(hp * self.latency_ratio, hp + self.latency_min_increase_ms):
                    reasons.append(f"p95 latency {rp:.0f} ms vs {hp:.0f} ms earlier")
                re_, he = r.get("error_ratio") or 0, h.get("error_ratio") or 0
                if re_ >= max(self.error_ratio_min, he + self.error_ratio_min):
                    reasons.append(f"error ratio {re_:.1%} vs {he:.1%} earlier")
            if reasons:
                self.pending[service] = self.pending.get(service, 0) + 1
                if self.pending[service] >= self.consecutive:
                    firing.append({"service": service, "reasons": reasons, "recent": r, "history": h})
            else:
                self.pending.pop(service, None)
        return firing


def history_queries(service: str, history_s: float) -> dict[str, str]:
    """Typical behaviour over the history window: the median of per-minute values, so an
    earlier short incident inside the window does not become the reference."""
    base = f'service_name="{service}",span_kind="{SERVER_KIND}"'
    w = f"{int(history_s)}s"
    p95_1m = f"histogram_quantile(0.95, sum by (le) (rate({SPAN_DURATION_BUCKET}{{{base}}}[1m])))"
    err_1m = (f"(sum(rate({SPAN_CALLS}{{{base},status_code=\"{ERROR_STATUS}\"}}[1m])) or vector(0))"
              f" / sum(rate({SPAN_CALLS}{{{base}}}[1m]))")
    return {
        "request_rate_per_s": f"sum(rate({SPAN_CALLS}{{{base}}}[{w}]))",
        "error_ratio": f"quantile_over_time(0.5, ({err_1m})[{w}:60s])",
        "p95_latency_ms": f"quantile_over_time(0.5, ({p95_1m})[{w}:60s])",
    }


def open_cases_for_alerts(ledger: Ledger, firing: list[dict[str, Any]], rules: AlertRules,
                          baseline_seconds: float, now: float | None = None) -> list[str]:
    """Open at most one case per alert group; attach later alerts to the open case."""
    now = now or time.time()
    opened = []
    recent_open = [c for c in ledger.cases(50) if c["source"] == "alert_rule"
                   and c["status"] not in ("closed", "recovered") and now - c["created_at"] < rules.group_window_s]
    for alert in firing:
        service = alert["service"]
        symptom = f"{service}: " + "; ".join(alert["reasons"])
        target = next(iter(recent_open), None)
        if target is not None:
            if not any(a["service"] == service and a["case_id"] == target["id"] for a in ledger.alerts(100)):
                ledger.add_alert("red_regression", service, target["id"], alert)
                ledger.event(target["id"], None, "alerting", "alert", f"Related alert: {symptom}")
            continue
        severity = "critical" if any("error" in r for r in alert["reasons"]) else "warning"
        case = ledger.create_case(CaseInput(
            service=service, symptom=symptom, severity=severity, source="alert_rule",
            incident_start=now - rules.recent_s - 60, incident_end=now, baseline_seconds=baseline_seconds))
        ledger.add_alert("red_regression", service, case["id"], alert)
        recent_open.append(case)
        opened.append(case["id"])
    return opened

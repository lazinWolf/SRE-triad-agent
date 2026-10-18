"""Deterministic comparisons used by tools, recovery checks and alert rules.

The model decides what to compare; this code decides how numbers compare, so
"increased 4x" in a report always means the same thing.
"""

from __future__ import annotations

from typing import Any


def compare(baseline: float | None, current: float | None, *, higher_is_worse: bool = True,
            min_abs_delta: float = 0.0, ratio_threshold: float = 1.5) -> dict[str, Any]:
    """Classify a change between a baseline and a current value."""
    if baseline is None or current is None:
        return {"baseline": baseline, "current": current, "change": "unknown",
                "note": "missing data in one window; this is not evidence of health"}
    delta = current - baseline
    ratio = (current / baseline) if baseline not in (0, 0.0) else (None if current == 0 else float("inf"))
    significant = abs(delta) >= min_abs_delta and (
        ratio is None or ratio == float("inf") or ratio >= ratio_threshold or ratio <= 1 / ratio_threshold)
    if not significant:
        change = "unchanged"
    elif (delta > 0) == higher_is_worse:
        change = "worse"
    else:
        change = "better"
    return {
        "baseline": round(baseline, 4), "current": round(current, 4), "delta": round(delta, 4),
        "ratio": None if ratio is None else ("inf" if ratio == float("inf") else round(ratio, 2)),
        "change": change,
    }


# Minimum absolute deltas that count as meaningful for each signal.
SIGNAL_RULES: dict[str, dict[str, Any]] = {
    "p95_latency_ms": {"higher_is_worse": True, "min_abs_delta": 20},
    "p50_latency_ms": {"higher_is_worse": True, "min_abs_delta": 10},
    "error_ratio": {"higher_is_worse": True, "min_abs_delta": 0.01},
    "request_rate_per_s": {"higher_is_worse": False, "min_abs_delta": 0.2},
    "cpu_cores_used_avg": {"higher_is_worse": True, "min_abs_delta": 0.05},
    "cpu_cores_used_peak_1m": {"higher_is_worse": True, "min_abs_delta": 0.05},
    "cpu_throttled_ratio": {"higher_is_worse": True, "min_abs_delta": 0.05},
    "cpu_limit_cpus": {"higher_is_worse": False, "min_abs_delta": 0.05},
    "memory_max_bytes": {"higher_is_worse": True, "min_abs_delta": 20 * 2**20},
    "restarts_in_window": {"higher_is_worse": True, "min_abs_delta": 1, "ratio_threshold": 1.0},
    "hit_ratio": {"higher_is_worse": False, "min_abs_delta": 0.05},
    "commands_per_s": {"higher_is_worse": False, "min_abs_delta": 1},
}


def compare_signals(baseline: dict[str, float | None], current: dict[str, float | None]) -> dict[str, Any]:
    out = {}
    for key in sorted(set(baseline) | set(current)):
        rule = SIGNAL_RULES.get(key, {})
        out[key] = compare(baseline.get(key), current.get(key), **rule)
    return out


def series_stats(values: list[tuple[float, float]]) -> dict[str, Any]:
    if not values:
        return {"points": 0}
    nums = [v for _, v in values]
    ordered = sorted(nums)
    return {
        "points": len(nums), "min": round(ordered[0], 4), "max": round(ordered[-1], 4),
        "mean": round(sum(nums) / len(nums), 4), "median": round(ordered[len(ordered) // 2], 4),
        "first": round(nums[0], 4), "last": round(nums[-1], 4),
    }


def shift_point(values: list[tuple[float, float]], factor: float = 2.0) -> float | None:
    """First timestamp where the value exceeds `factor` x the early median, if any."""
    if len(values) < 4:
        return None
    head = sorted(v for _, v in values[: max(2, len(values) // 4)])
    reference = head[len(head) // 2]
    if reference <= 0:
        return None
    for t, v in values:
        if v >= reference * factor:
            return t
    return None


def recovered(baseline: dict[str, float | None], after: dict[str, float | None], *, latency_ratio: float,
              latency_slack_ms: float, error_rate_slack: float) -> dict[str, Any]:
    """Decide recovery against the original symptom with explicit, auditable thresholds."""
    checks = []
    b95, a95 = baseline.get("p95_latency_ms"), after.get("p95_latency_ms")
    if b95 is None or a95 is None:
        checks.append({"signal": "p95_latency_ms", "ok": None, "detail": "missing data"})
    else:
        limit = b95 * latency_ratio + latency_slack_ms
        checks.append({"signal": "p95_latency_ms", "ok": a95 <= limit, "baseline": round(b95, 2),
                       "after": round(a95, 2), "limit": round(limit, 2)})
    be, ae = baseline.get("error_ratio"), after.get("error_ratio")
    if be is None or ae is None:
        checks.append({"signal": "error_ratio", "ok": None, "detail": "missing data"})
    else:
        limit = be + error_rate_slack
        checks.append({"signal": "error_ratio", "ok": ae <= limit, "baseline": round(be, 4),
                       "after": round(ae, 4), "limit": round(limit, 4)})
    rate = after.get("request_rate_per_s")
    if rate is not None and rate < 0.05:
        checks.append({"signal": "request_rate_per_s", "ok": None,
                       "detail": "too little traffic after the action to judge recovery"})
    known = [c["ok"] for c in checks if c["ok"] is not None]
    if any(c["ok"] is None for c in checks) and not known:
        verdict = "inconclusive"
    elif all(known) and all(c["ok"] is not None for c in checks):
        verdict = "recovered"
    elif not all(known):
        verdict = "not_recovered"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "checks": checks}

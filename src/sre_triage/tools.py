"""Tools the agents call. Each result is stored as evidence with an ID agents must cite.

Every tool compares the incident window with the healthy baseline window just before
it, so agents never have to reason about timestamps.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import BASELINE_CPUS, DEPENDENCIES, SERVICES
from .telemetry import Backends, Unavailable, read_flags, summarize_trace


@dataclass
class Window:
    start: float          # incident start (unix seconds)
    end: float            # investigation time
    baseline_seconds: float = 300

    @property
    def baseline(self) -> tuple[float, float]:
        return self.start - self.baseline_seconds, self.start


def compare_red(now: dict, before: dict) -> list[str]:
    """Deterministic anomaly flags for one service."""
    flags = []
    p95, p95_b = now.get("p95_ms"), before.get("p95_ms")
    if p95 and p95_b and p95 > 1.5 * p95_b and p95 - p95_b > 25:
        flags.append(f"p95 latency {p95_b:.0f} -> {p95:.0f} ms (x{p95 / p95_b:.1f})")
    err, err_b = now.get("error_ratio") or 0, before.get("error_ratio") or 0
    if err - err_b > 0.02:
        flags.append(f"error ratio {err_b:.1%} -> {err:.1%}")
    return flags


def health_scan(backends: Backends, window: Window) -> dict[str, dict]:
    """RED metrics of every service, incident vs baseline, with anomaly flags (used by the detect step)."""
    out = {}
    for service in SERVICES:
        now = backends.prometheus.service_red(service, window.start, window.end)
        before = backends.prometheus.service_red(service, *window.baseline)
        out[service] = {"incident": now, "baseline": before, "anomalies": compare_red(now, before)}
    return out


@dataclass
class Toolbox:
    backends: Backends
    window: Window
    evidence: list[dict] = field(default_factory=list)
    emit: Callable[[str, str, str], None] | None = None   # activity log (kind, agent, text)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _counters: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------ evidence
    def record(self, agent: str, tool: str, args: dict, result: Any) -> dict:
        with self._lock:
            prefix = agent[0].upper()
            self._counters[prefix] = self._counters.get(prefix, 0) + 1
            eid = f"{prefix}{self._counters[prefix]}"
            self.evidence.append({"id": eid, "agent": agent, "tool": tool, "args": args, "result": result,
                                  "at": time.time()})
        return {"evidence_id": eid, **(result if isinstance(result, dict) else {"result": result})}

    def call(self, agent: str, name: str, args: dict) -> dict:
        result = self._call(agent, name, args)
        if self.emit:
            self.emit("tool", agent, f"{name}({', '.join(f'{k}={v}' for k, v in args.items())}) -> "
                                     f"{result.get('evidence_id') or result.get('error')}")
        return result

    def _call(self, agent: str, name: str, args: dict) -> dict:
        fn = getattr(self, name, None)
        if name not in TOOL_NAMES or fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            result = fn(**args)
        except TypeError as exc:
            return {"error": f"bad arguments: {exc}"}
        except Unavailable as exc:
            return {"error": f"backend unavailable: {exc}"}
        except ValueError as exc:
            return {"error": str(exc)}
        return self.record(agent, name, args, result)

    # ------------------------------------------------------------------ tools
    def service_health(self, service: str) -> dict:
        """Request rate, error ratio and p50/p95 latency of a service: incident window vs healthy baseline."""
        _check(service)
        now = self.backends.prometheus.service_red(service, self.window.start, self.window.end)
        before = self.backends.prometheus.service_red(service, *self.window.baseline)
        return {"service": service, "incident": now, "baseline": before, "anomalies": compare_red(now, before)}

    def downstream_health(self, service: str) -> dict:
        """p95 latency and error ratio of every service this one calls, incident vs baseline."""
        _check(service)
        rows = {}
        for dep in DEPENDENCIES.get(service, []):
            if dep not in SERVICES:
                rows[dep] = "data store (no span metrics; check the caller's traces)"
                continue
            now = self.backends.prometheus.service_red(dep, self.window.start, self.window.end)
            before = self.backends.prometheus.service_red(dep, *self.window.baseline)
            rows[dep] = {"p95_ms": f"{before['p95_ms']} -> {now['p95_ms']}",
                         "error_ratio": f"{before['error_ratio']} -> {now['error_ratio']}",
                         "anomalies": compare_red(now, before)}
        callers = sorted(s for s, deps in DEPENDENCIES.items() if service in deps)
        return {"service": service, "called_by": callers, "calls": rows}

    def container_resources(self, service: str) -> dict:
        """CPU used vs limit, CPU throttling, memory and restarts of a service's container, incident vs baseline."""
        _check(service, allow_stores=True)
        now = self.backends.prometheus.container(service, self.window.start, self.window.end)
        before = self.backends.prometheus.container(service, *self.window.baseline)
        config = self.backends.docker.inspect(service)
        notes = []
        if (now.get("cpu_throttled_ratio") or 0) > 0.1:
            notes.append(f"CPU throttled in {now['cpu_throttled_ratio']:.0%} of periods")
        if config["cpu_limit"] and config["cpu_limit"] < BASELINE_CPUS:
            notes.append(f"CPU limit {config['cpu_limit']} is below the standard {BASELINE_CPUS}")
        if now.get("cpu_cores_used") and config["cpu_limit"] and now["cpu_cores_used"] > 0.8 * config["cpu_limit"]:
            notes.append("CPU usage is at the container limit")
        return {"service": service, "incident": now, "baseline": before, "current_config": config, "notes": notes}

    def slow_traces(self, service: str) -> dict:
        """Where time goes in the slowest traces through a service: self time per service, incident vs baseline."""
        _check(service)

        def profile(start: float, end: float) -> tuple[dict, list]:
            traces = self.backends.jaeger.search(service, start, end, limit=40)
            traces = sorted(traces, key=lambda t: max((s["duration"] for s in t.get("spans", [])), default=0))[-8:]
            summaries = [summarize_trace(t) for t in traces]
            totals: dict[str, float] = {}
            for s in summaries:
                for svc, ms in s["self_time_ms"].items():
                    totals[svc] = totals.get(svc, 0) + ms / max(1, len(summaries))
            return {k: round(v, 1) for k, v in sorted(totals.items(), key=lambda kv: -kv[1])[:6]}, summaries

        now, examples = profile(self.window.start, self.window.end)
        before, _ = profile(*self.window.baseline)
        grew = {svc: f"{before.get(svc, 0)} -> {ms} ms" for svc, ms in now.items() if ms > 2 * before.get(svc, 0) + 20}
        return {"service": service, "avg_self_time_ms_incident": now, "avg_self_time_ms_baseline": before,
                "self_time_grew": grew,
                "slowest_examples": [{k: e[k] for k in ("trace_id", "root", "duration_ms")} for e in examples[-3:]]}

    def error_traces(self, service: str) -> dict:
        """Error traces through a service in the incident window and which service/operation the errors come from."""
        _check(service)
        traces = self.backends.jaeger.search(service, self.window.start, self.window.end, limit=20, errors=True)
        origins: dict[str, int] = {}
        messages: dict[str, str] = {}
        for t in traces:
            for err in summarize_trace(t)["errors"]:
                key = f"{err['service']} {err['operation']}"
                origins[key] = origins.get(key, 0) + 1
                messages.setdefault(key, err["message"])
        top = sorted(origins.items(), key=lambda kv: -kv[1])[:6]
        return {"service": service, "error_traces_found": len(traces),
                "error_spans_by_origin": [{"span": k, "count": n, "message": messages[k]} for k, n in top]}

    def search_logs(self, service: str = "", text: str = "") -> dict:
        """Warning/error log lines in the incident window (optionally for one service or containing text), vs baseline counts."""
        if service:
            _check(service, allow_stores=True)
        now = self.backends.logs.search(self.window.start, self.window.end, service=service or None, text=text or None)
        before = self.backends.logs.search(*self.window.baseline, service=service or None, text=text or None, limit=1)
        return {"filter": {"service": service, "text": text}, "incident_total": now["total"],
                "baseline_total": before["total"], "incident_by_service": now["by_service"],
                "baseline_by_service": before["by_service"], "top_messages": now["messages"]}

    def recent_changes(self) -> dict:
        """Configuration that differs from the standard setup (CPU limits, feature flags), plus container events."""
        drift = []
        for service in SERVICES:
            try:
                cfg = self.backends.docker.inspect(service)
            except Unavailable:
                continue
            if cfg["cpu_limit"] and abs(cfg["cpu_limit"] - BASELINE_CPUS) > 1e-6:
                drift.append({"target": service, "change": f"CPU limit {BASELINE_CPUS} -> {cfg['cpu_limit']}"})
        flags_file = self.backends.flagd_file
        for flag, info in read_flags(flags_file).items():
            if info["variant"] != "off":
                drift.append({"target": f"flag:{flag}", "change": f"feature flag {flag} = {info['variant']} (standard: off)"})
        since = self.window.start - self.window.baseline_seconds
        events = self.backends.docker.events(since, self.window.end, list(SERVICES))
        mtime = flags_file.stat().st_mtime
        return {
            "drift_from_standard_config": drift,
            "container_events": [{"seconds_from_incident_start": round(e["time"] - self.window.start),
                                  "container": e["container"], "action": e["action"]} for e in events][-15:],
            "flag_file_last_modified_seconds_from_incident_start": round(mtime - self.window.start),
            "note": "A change is a suspect only if its service or flag is on the failing request path.",
        }


def _check(service: str, allow_stores: bool = False) -> None:
    stores = {"valkey-cart", "astronomy-db"} if allow_stores else set()
    if service not in SERVICES and service not in stores:
        raise ValueError(f"unknown service {service!r}; known: {', '.join(SERVICES)}")


TOOL_NAMES = ("service_health", "downstream_health", "container_resources", "slow_traces",
              "error_traces", "search_logs", "recent_changes")

AGENT_TOOLS = {
    "metrics": ["service_health", "downstream_health", "container_resources"],
    "traces": ["slow_traces", "error_traces", "downstream_health"],
    "logs_changes": ["recent_changes", "search_logs", "container_resources"],
    "solo": list(TOOL_NAMES),
}


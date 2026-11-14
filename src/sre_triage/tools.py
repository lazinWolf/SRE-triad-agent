"""The read-only investigation tools. Agents reach them through the MCP servers in ``mcp_servers``.

Every tool compares the incident window with a healthy baseline, so agents never have to reason about
timestamps. Results are typed (``RESULT_MODELS``): the server builds them and the agent side checks them again.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict

from .config import BASELINE_CPUS, DEPENDENCIES, PROJECT_ROOT, SERVICES, STORE_CLIENTS, STORES
from .telemetry import CPU_USAGE, MEMORY_USAGE, SPAN_BUCKET, SPAN_CALLS, Backends, Unavailable, read_flags, summarize_trace


@dataclass
class Window:
    start: float          # incident start (unix seconds)
    end: float            # investigation time
    baseline_seconds: float = 300

    @property
    def baseline(self) -> tuple[float, float]:
        return self.start - self.baseline_seconds, self.start


BUSY_RATE = 0.5  # requests/s in the baseline from which the p95 is trustworthy


def compare_red(now: dict, before: dict) -> list[str]:
    """Deterministic anomaly flags for one service."""
    flags = []
    rate, rate_b = now.get("requests_per_s") or 0, before.get("requests_per_s") or 0
    if rate_b > 0.05 and rate < 0.2 * rate_b:
        flags.append(f"request rate {rate_b:.2f} -> {rate:.2f} /s (traffic dropped)")
    elif rate_b > 0.05 and rate > 3 * rate_b:
        flags.append(f"request rate {rate_b:.2f} -> {rate:.2f} /s (traffic spike)")
    # Under ~0.5 requests/s a 1-minute p95 is the slowest of a handful of requests and jumps 100 -> 350 ms on a
    # healthy shop, so quiet services are judged on their median. The ServiceHighLatency alert does the same.
    busy = rate_b >= BUSY_RATE
    key, name, margin = ("p95_ms", "p95", 25) if busy else ("p50_ms", "median", 50)
    lat, lat_b = now.get(key), before.get(key)
    if lat and lat_b and lat > 1.5 * lat_b and lat - lat_b > margin:
        flags.append(f"{name} latency {lat_b:.0f} -> {lat:.0f} ms (x{lat / lat_b:.1f})")
    err, err_b = now.get("error_ratio") or 0, before.get("error_ratio") or 0
    if err - err_b > 0.02:
        flags.append(f"error ratio {err_b:.1%} -> {err:.1%}")
    return flags


def baseline_red(backends: Backends, service: str, window: Window) -> dict:
    """Healthy reference for a service: the recorded baseline (learned from alert-free periods), or, when the
    Prometheus rules are not loaded, the window just before the incident."""
    return (backends.prometheus.service_baseline(service, window.start)
            or backends.prometheus.service_red(service, *window.baseline))


def health_scan(backends: Backends, window: Window) -> dict[str, dict]:
    """RED metrics of every service, incident vs baseline, with anomaly flags (used by the detect step)."""
    out = {}
    for service in SERVICES:
        now = backends.prometheus.service_red(service, window.start, window.end)
        before = baseline_red(backends, service, window)
        out[service] = {"incident": now, "baseline": before, "anomalies": compare_red(now, before)}
    return out


@dataclass
class Evidence:
    """Tool results of a case; each gets an ID (M1, T2, ...) that agents must cite."""

    items: list[dict] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, agent: str, tool: str, args: dict, result: Any) -> str:
        with self._lock:
            prefix = agent[0].upper()
            self.counters[prefix] = self.counters.get(prefix, 0) + 1
            eid = f"{prefix}{self.counters[prefix]}"
            self.items.append({"id": eid, "agent": agent, "tool": tool, "args": args, "result": result,
                               "at": time.time()})
        return eid


TIMELINE = {  # metric -> PromQL over a 1-minute rate; SEL = span selector or container selector
    "p95_ms": f"histogram_quantile(0.95, sum by (le) (rate({SPAN_BUCKET}{{SEL}}[1m])))",
    "error_ratio": f'(sum(rate({SPAN_CALLS}{{SEL,status_code="STATUS_CODE_ERROR"}}[1m])) or vector(0))'
                   f" / sum(rate({SPAN_CALLS}{{SEL}}[1m]))",
    "requests_per_s": f"sum(rate({SPAN_CALLS}{{SEL}}[1m]))",
    "cpu_cores": f"sum(rate({CPU_USAGE}{{SEL}}[1m])) / 1e9",
    "memory_mb": f"max({MEMORY_USAGE}{{SEL}}) / 1048576",
}


@dataclass
class Tools:
    backends: Backends
    window: Window

    # ------------------------------------------------------------------ telemetry (Prometheus, Jaeger, OpenSearch)
    def service_health(self, service: str) -> dict:
        """Request rate, error ratio and p50/p95 latency of a service: incident window vs healthy baseline."""
        _check(service)
        now = self.backends.prometheus.service_red(service, self.window.start, self.window.end)
        before = baseline_red(self.backends, service, self.window)
        return {"service": service, "incident": now, "baseline": before, "anomalies": compare_red(now, before)}

    def downstream_health(self, service: str) -> dict:
        """p95 latency and error ratio of every service this one calls, incident vs baseline."""
        _check(service)
        rows = {}
        for dep in DEPENDENCIES.get(service, []):
            if dep not in SERVICES:
                rows[dep] = "data store (no server spans; use datastore_health)"
                continue
            now = self.backends.prometheus.service_red(dep, self.window.start, self.window.end)
            before = baseline_red(self.backends, dep, self.window)
            rows[dep] = {"p95_ms": f"{before['p95_ms']} -> {now['p95_ms']}",
                         "error_ratio": f"{before['error_ratio']} -> {now['error_ratio']}",
                         "requests_per_s": f"{before['requests_per_s']} -> {now['requests_per_s']}",
                         "anomalies": compare_red(now, before)}
        callers = sorted(s for s, deps in DEPENDENCIES.items() if service in deps)
        return {"service": service, "called_by": callers, "calls": rows}

    def metric_timeline(self, service: str, metric: str = "p95_ms") -> dict:
        """How one metric of a service moved from 10 minutes before the incident until now (shows when it started).
        metric: p95_ms, error_ratio, requests_per_s, cpu_cores or memory_mb."""
        _check(service, allow_stores=metric in ("cpu_cores", "memory_mb"))
        if metric not in TIMELINE:
            raise ValueError(f"unknown metric {metric!r}; known: {', '.join(TIMELINE)}")
        sel = (f'container_name="{service}"' if metric in ("cpu_cores", "memory_mb")
               else f'service_name="{service}",span_kind="SPAN_KIND_SERVER"')
        start = self.window.start - 600
        step = max(15, (self.window.end - start) / 24)
        points = self.backends.prometheus.series(TIMELINE[metric].replace("SEL", sel), start, self.window.end, step)
        values = [v for _, v in points]
        return {"service": service, "metric": metric,
                "points": [{"seconds_from_incident_start": round(t - self.window.start), "value": _r(v)} for t, v in points],
                "min": _r(min(values)) if values else None, "max": _r(max(values)) if values else None,
                "note": "negative seconds are before the incident started; gaps mean no data (no traffic or container down)"}

    def container_resources(self, service: str) -> dict:
        """CPU used vs limit, CPU throttling and memory of a container, incident vs the 5 minutes before."""
        _check(service, allow_stores=True)
        now = self.backends.prometheus.container(service, self.window.start, self.window.end)
        before = self.backends.prometheus.container(service, *self.window.baseline)
        config = self.backends.docker.inspect(service)
        notes = []
        if (now.get("cpu_throttled_ratio") or 0) > 0.1:
            notes.append(f"CPU throttled in {now['cpu_throttled_ratio']:.0%} of periods")
        if now.get("cpu_cores_used") and config["cpu_limit"] and now["cpu_cores_used"] > 0.8 * config["cpu_limit"]:
            notes.append("CPU usage is at the container limit")
        if now.get("memory_mb") and now.get("memory_limit_mb") and now["memory_mb"] > 0.9 * now["memory_limit_mb"]:
            grew = not before.get("memory_mb") or now["memory_mb"] > 1.15 * before["memory_mb"]
            notes.append("memory usage is at the container limit" + ("" if grew else " (it was before the incident too)"))
        notes += self._drift(service, config)
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
        examples: dict[str, str] = {}
        for t in traces:
            for err in summarize_trace(t)["errors"]:
                key = f"{err['service']} {err['operation']}"
                origins[key] = origins.get(key, 0) + 1
                messages.setdefault(key, err["message"])
                examples.setdefault(key, t.get("traceID"))
        top = sorted(origins.items(), key=lambda kv: -kv[1])[:6]
        return {"service": service, "error_traces_found": len(traces),
                "error_spans_by_origin": [{"span": k, "count": n, "message": messages[k], "example_trace_id": examples[k]}
                                          for k, n in top]}

    def get_trace(self, trace_id: str) -> dict:
        """One trace in detail: its slowest spans, time per service and errors (use a trace_id from another tool)."""
        if not re.fullmatch(r"[0-9a-fA-F]{8,32}", trace_id or ""):
            raise ValueError("trace_id must be the hex id returned by slow_traces or error_traces")
        trace = self.backends.jaeger.get(trace_id)
        if not trace:
            return {"trace_id": trace_id, "found": False}
        procs = {pid: p.get("serviceName", "?") for pid, p in (trace.get("processes") or {}).items()}
        spans = sorted(trace.get("spans") or [], key=lambda s: -s["duration"])[:12]
        return {"found": True, **summarize_trace(trace),
                "slowest_spans": [{"service": procs.get(s.get("processID"), "?"), "operation": s.get("operationName"),
                                   "duration_ms": round(s["duration"] / 1000, 1)} for s in spans]}

    def search_logs(self, service: str = "", text: str = "") -> dict:
        """Warning/error log lines in the incident window (optionally for one service or containing text), vs baseline counts."""
        if service:
            _check(service, allow_stores=True)
        now = self.backends.logs.search(self.window.start, self.window.end, service=service or None, text=text or None)
        before = self.backends.logs.search(*self.window.baseline, service=service or None, text=text or None, limit=1)
        return {"filter": {"service": service, "text": text}, "incident_total": now["total"],
                "baseline_total": before["total"], "incident_by_service": now["by_service"],
                "baseline_by_service": before["by_service"], "top_messages": now["messages"]}

    def datastore_health(self, store: str) -> dict:
        """Health of a data store (valkey-cart or astronomy-db) as its client service sees it, plus its container state."""
        if store not in STORES:
            raise ValueError(f"unknown data store {store!r}; known: {', '.join(STORES)}")
        client, span = STORE_CLIENTS[store]
        sel = f'service_name="{client}",span_kind="SPAN_KIND_CLIENT",span_name="{span}"'
        now = self.backends.prometheus.red(sel, self.window.start, self.window.end)
        before = self.backends.prometheus.red(sel, *self.window.baseline)
        config = self.backends.docker.inspect(store)
        notes = compare_red(now, before) + self._drift(store, config)
        return {"store": store, "client_service": client, "client_calls_incident": now, "client_calls_baseline": before,
                "container": config, "notes": notes}

    def service_map(self) -> dict:
        """Which service calls which, as observed in traces during the incident (with call counts)."""
        lookback = self.window.end - self.window.start + self.window.baseline_seconds
        edges = [e for e in self.backends.jaeger.dependencies(self.window.end, lookback) if e["parent"] != e["child"]]
        observed: dict[str, dict[str, int]] = {}
        for e in edges:
            observed.setdefault(e["parent"], {})[e["child"]] = e["callCount"]
        missing = [f"{a} -> {b}" for a, deps in DEPENDENCIES.items() for b in deps
                   if b in SERVICES and b not in observed.get(a, {})]
        return {"observed_calls": observed, "expected_but_not_observed": missing,
                "note": "an expected call that is not observed means no successful traffic on that edge in the window"}

    def active_alerts(self) -> dict:
        """Alerts that are pending or firing right now (Prometheus alert rules)."""
        rows = [{"alert": a["name"], "service": a["service"], "kind": a["kind"], "severity": a["severity"],
                 "state": a["state"], "summary": a["summary"],
                 "seconds_from_incident_start": round(a["since"] - self.window.start) if a["since"] else None}
                for a in self.backends.prometheus.alerts()]
        return {"alerts": rows, "note": "the earliest alert is often closest to the cause; later ones are usually victims"}

    # ------------------------------------------------------------------ operations (Docker, flagd, runbooks)
    def recent_changes(self) -> dict:
        """Configuration that differs from the standard setup (limits, container state, feature flags), plus container events."""
        drift = []
        for name in (*SERVICES, *STORES):
            try:
                cfg = self.backends.docker.inspect(name)
            except Unavailable:
                continue
            drift += [{"target": name, "change": change} for change in self._drift(name, cfg)]
        drift += [{"target": f"flag:{f['flag']}", "change": f"feature flag {f['flag']} = {f['variant']} (standard: {f['standard']})"}
                  for f in self._flags() if f["changed"]]
        since = self.window.start - self.window.baseline_seconds
        events = self.backends.docker.events(since, self.window.end, [*SERVICES, *STORES])
        mtime = self.backends.flagd_file.stat().st_mtime
        return {
            "drift_from_standard_config": drift,
            "container_events": [{"seconds_from_incident_start": round(e["time"] - self.window.start),
                                  "container": e["container"], "action": e["action"]} for e in events][-15:],
            "flag_file_last_modified_seconds_from_incident_start": round(mtime - self.window.start),
            "note": "A change is a suspect only if its service or flag is on the failing request path.",
        }

    def container_status(self, service: str) -> dict:
        """Is a container running, paused or stopped; restarts, OOM kills and its limits vs the standard ones."""
        _check(service, allow_stores=True)
        config = self.backends.docker.inspect(service)
        events = self.backends.docker.events(self.window.start - self.window.baseline_seconds, self.window.end, [service])
        return {"service": service, **config, "standard": self._standard(service), "notes": self._drift(service, config),
                "events": [{"seconds_from_incident_start": round(e["time"] - self.window.start), "action": e["action"]}
                           for e in events][-10:]}

    def container_logs(self, service: str, text: str = "") -> dict:
        """Last stdout/stderr lines of a container in the incident window (optionally only lines containing text)."""
        _check(service, allow_stores=True)
        lines = self.backends.docker.logs(service, self.window.start, self.window.end, tail=200)
        if text:
            lines = [line for line in lines if text.lower() in line.lower()]
        return {"service": service, "filter": text, "lines_found": len(lines), "lines": lines[-25:]}

    def list_feature_flags(self) -> dict:
        """Every feature flag with its current value, its standard value and whether it was changed."""
        flags = self._flags()
        return {"flags": flags, "changed": [f["flag"] for f in flags if f["changed"]]}

    def search_runbooks(self, query: str) -> dict:
        """Search the team's runbooks (how to confirm and fix known failure modes) by keywords."""
        words = {w for w in re.findall(r"[a-z0-9_]+", query.lower()) if len(w) > 2}
        if not words:
            raise ValueError("give a few keywords, e.g. 'cpu throttling latency'")
        hits = []
        for path in sorted((PROJECT_ROOT / "runbooks").glob("*.md")):
            if path.name == "README.md":
                continue
            body = path.read_text()
            found = re.findall(r"[a-z0-9_]+", body.lower())
            score = sum(found.count(w) for w in words) + 5 * sum(w in path.stem for w in words)
            if score:
                hits.append((score, path.name, body))
        hits.sort(key=lambda h: -h[0])
        return {"query": query, "runbooks": [{"file": name, "title": body.splitlines()[0].lstrip("# "), "text": body[:1500]}
                                             for _, name, body in hits[:2]]}

    # ------------------------------------------------------------------ helpers
    def _standard(self, name: str) -> dict:
        known = (self.backends.standard.get("services") or {}).get(name)
        return known or {"cpus": BASELINE_CPUS if name in SERVICES else None, "memory_mb": None}

    def _drift(self, name: str, config: dict) -> list[str]:
        """How a container differs from the standard configuration."""
        standard, out = self._standard(name), []
        if config.get("status") and config["status"] != "running":
            out.append(f"container is {config['status']} (standard: running)")
        if standard["cpus"] and config["cpu_limit"] and abs(config["cpu_limit"] - standard["cpus"]) > 1e-6:
            out.append(f"CPU limit {standard['cpus']} -> {config['cpu_limit']}")
        if standard["memory_mb"] and config.get("memory_limit_mb") and abs(config["memory_limit_mb"] - standard["memory_mb"]) > 1:
            out.append(f"memory limit {standard['memory_mb']} -> {config['memory_limit_mb']} MB")
        if config.get("oom_killed"):
            out.append("last exit was an OOM kill")
        return out

    def _flags(self) -> list[dict]:
        pristine = (self.backends.standard.get("flags") or {}).get("flags") or {}
        out = []
        for flag, info in read_flags(self.backends.flagd_file).items():
            standard = pristine.get(flag, {}).get("defaultVariant", "off")
            out.append({"flag": flag, "variant": info["variant"], "standard": standard,
                        "changed": info["variant"] != standard, "variants": info["variants"]})
        return out


def _r(value: float) -> float:
    return round(value, 4 if abs(value) < 1 else 1)


def _check(service: str, allow_stores: bool = False) -> None:
    if service not in SERVICES and not (allow_stores and service in STORES):
        known = ", ".join(SERVICES + (STORES if allow_stores else ()))
        raise ValueError(f"unknown service {service!r}; known: {known}")


# ---------------------------------------------------------------------------- typed results
class _Result(BaseModel):
    model_config = ConfigDict(extra="allow")


class ServiceHealth(_Result):
    service: str
    incident: dict
    baseline: dict
    anomalies: list[str]


class DownstreamHealth(_Result):
    service: str
    called_by: list[str]
    calls: dict


class MetricTimeline(_Result):
    service: str
    metric: str
    points: list[dict]


class ContainerResources(_Result):
    service: str
    incident: dict
    baseline: dict
    current_config: dict
    notes: list[str]


class SlowTraces(_Result):
    service: str
    avg_self_time_ms_incident: dict[str, float]
    avg_self_time_ms_baseline: dict[str, float]
    self_time_grew: dict[str, str]


class ErrorTraces(_Result):
    service: str
    error_traces_found: int
    error_spans_by_origin: list[dict]


class TraceDetail(_Result):
    found: bool


class LogSearch(_Result):
    incident_total: int
    baseline_total: int
    top_messages: list[dict]


class DatastoreHealth(_Result):
    store: str
    client_service: str
    client_calls_incident: dict
    client_calls_baseline: dict
    container: dict
    notes: list[str]


class ServiceMap(_Result):
    observed_calls: dict[str, dict[str, int]]
    expected_but_not_observed: list[str]


class ActiveAlerts(_Result):
    alerts: list[dict]


class RecentChanges(_Result):
    drift_from_standard_config: list[dict]
    container_events: list[dict]


class ContainerStatus(_Result):
    service: str
    status: str | None
    restart_count: int
    oom_killed: bool
    notes: list[str]


class ContainerLogs(_Result):
    service: str
    lines_found: int
    lines: list[str]


class FeatureFlags(_Result):
    flags: list[dict]
    changed: list[str]


class Runbooks(_Result):
    query: str
    runbooks: list[dict]


RESULT_MODELS: dict[str, type[BaseModel]] = {
    "service_health": ServiceHealth, "downstream_health": DownstreamHealth, "metric_timeline": MetricTimeline,
    "container_resources": ContainerResources, "slow_traces": SlowTraces, "error_traces": ErrorTraces,
    "get_trace": TraceDetail, "search_logs": LogSearch, "datastore_health": DatastoreHealth,
    "service_map": ServiceMap, "active_alerts": ActiveAlerts, "recent_changes": RecentChanges,
    "container_status": ContainerStatus, "container_logs": ContainerLogs, "list_feature_flags": FeatureFlags,
    "search_runbooks": Runbooks,
}
# The two MCP servers and which tools each one serves.
SERVER_TOOLS = {
    "telemetry": ("service_health", "downstream_health", "metric_timeline", "container_resources", "slow_traces",
                  "error_traces", "get_trace", "search_logs", "datastore_health", "service_map", "active_alerts"),
    "ops": ("recent_changes", "container_status", "container_logs", "list_feature_flags", "search_runbooks"),
}
TOOL_NAMES = SERVER_TOOLS["telemetry"] + SERVER_TOOLS["ops"]
AGENT_TOOLS = {
    "metrics": ["service_health", "downstream_health", "metric_timeline", "container_resources", "active_alerts"],
    "traces": ["slow_traces", "error_traces", "get_trace", "service_map", "downstream_health"],
    "logs_changes": ["recent_changes", "list_feature_flags", "search_logs", "container_logs", "search_runbooks"],
    "infra": ["container_status", "container_resources", "datastore_health", "container_logs", "metric_timeline"],
    "solo": list(TOOL_NAMES),
}

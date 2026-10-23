"""Read-only adapters for Prometheus, Jaeger, OpenSearch logs and the Docker engine.

Adapters return plain dictionaries and raise ``TelemetryUnavailable`` when a
backend cannot answer. They never interpret incidents; that is the agents' job,
supported by deterministic helpers in ``analysis``.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any
from urllib.parse import quote

import httpx

# Metric names as exported by the demo's collector into Prometheus (OTLP receiver,
# underscore translation with unit suffixes). Kept in one place so they can be
# verified against a live stack with `sre-triage doctor`.
SPAN_CALLS = "traces_span_metrics_calls_total"
SPAN_DURATION_BUCKET = "traces_span_metrics_duration_milliseconds_bucket"
CPU_USAGE = "container_cpu_usage_nanoseconds_total"
CPU_LIMIT = "container_cpu_limit"
CPU_PERIODS = "container_cpu_throttling_data_periods_total"
CPU_THROTTLED_PERIODS = "container_cpu_throttling_data_throttled_periods_total"
MEMORY_USAGE = "container_memory_usage_total_bytes"
MEMORY_LIMIT = "container_memory_usage_limit_bytes"
CONTAINER_RESTARTS = "container_restarts_total"
REDIS_HITS = "redis_keyspace_hits_total"
REDIS_MISSES = "redis_keyspace_misses_total"
REDIS_COMMANDS = "redis_commands_processed_total"
REDIS_CLIENTS = "redis_clients_connected"
REDIS_MEMORY = "redis_memory_used_bytes"

SERVER_KIND = "SPAN_KIND_SERVER"
CLIENT_KIND = "SPAN_KIND_CLIENT"
ERROR_STATUS = "STATUS_CODE_ERROR"

LOG_INDEX = "otel-logs-*"


class TelemetryUnavailable(RuntimeError):
    """The backend could not be reached or rejected the query."""


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _label_value(value: str) -> str:
    """Escape a value for a PromQL double-quoted label matcher."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


# ---------------------------------------------------------------------------
# Prometheus
# ---------------------------------------------------------------------------
class Prometheus:
    def __init__(self, base_url: str, client: httpx.AsyncClient):
        self.base = base_url.rstrip("/")
        self.client = client

    async def _get(self, path: str, params: dict[str, Any]) -> Any:
        try:
            response = await self.client.get(f"{self.base}{path}", params=params)
        except httpx.HTTPError as exc:
            raise TelemetryUnavailable(f"Prometheus unreachable: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            raise TelemetryUnavailable(f"Prometheus HTTP {response.status_code}: {response.text[:300]}")
        body = response.json()
        if body.get("status") != "success":
            raise TelemetryUnavailable(f"Prometheus error: {body.get('error', 'unknown')}")
        return body["data"]

    async def instant(self, expr: str, at: float) -> list[dict[str, Any]]:
        data = await self._get("/api/v1/query", {"query": expr, "time": f"{at:.3f}"})
        out = []
        for item in data.get("result", []):
            value = _finite(item["value"][1])
            if value is not None:
                out.append({"labels": item.get("metric", {}), "value": value})
        return out

    async def scalar(self, expr: str, at: float) -> float | None:
        rows = await self.instant(expr, at)
        return rows[0]["value"] if rows else None

    async def series(self, expr: str, start: float, end: float, points: int = 12) -> list[dict[str, Any]]:
        step = max(15.0, (end - start) / max(points, 1))
        data = await self._get("/api/v1/query_range", {
            "query": expr, "start": f"{start:.3f}", "end": f"{end:.3f}", "step": f"{step:.0f}s"})
        out = []
        for item in data.get("result", []):
            values = [(float(t), _finite(v)) for t, v in item.get("values", [])]
            out.append({"labels": item.get("metric", {}), "values": [(t, v) for t, v in values if v is not None]})
        return out

    async def metric_names(self) -> list[str]:
        return await self._get("/api/v1/label/__name__/values", {})

    def graph_link(self, expr: str, start: float, end: float) -> str:
        span = max(60, int(end - start))
        return (f"{self.base}/graph?g0.expr={quote(expr)}&g0.tab=0&g0.range_input={span}s"
                f"&g0.end_input={quote(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(end)))}")


def window_selector(seconds: float) -> str:
    return f"{max(30, int(round(seconds)))}s"


def red_queries(service: str, window_s: float, kind: str = SERVER_KIND) -> dict[str, str]:
    svc = _label_value(service)
    w = window_selector(window_s)
    base = f'service_name="{svc}",span_kind="{kind}"'
    return {
        "request_rate_per_s": f"sum(rate({SPAN_CALLS}{{{base}}}[{w}]))",
        "error_ratio": (f"(sum(rate({SPAN_CALLS}{{{base},status_code=\"{ERROR_STATUS}\"}}[{w}])) or vector(0))"
                        f" / sum(rate({SPAN_CALLS}{{{base}}}[{w}]))"),
        "p50_latency_ms": f"histogram_quantile(0.50, sum by (le) (rate({SPAN_DURATION_BUCKET}{{{base}}}[{w}])))",
        "p95_latency_ms": f"histogram_quantile(0.95, sum by (le) (rate({SPAN_DURATION_BUCKET}{{{base}}}[{w}])))",
    }


def operation_queries(service: str, window_s: float, kind: str) -> dict[str, str]:
    svc = _label_value(service)
    w = window_selector(window_s)
    base = f'service_name="{svc}",span_kind="{kind}"'
    return {
        "rate": f"sum by (span_name) (rate({SPAN_CALLS}{{{base}}}[{w}]))",
        "errors": f"sum by (span_name) (rate({SPAN_CALLS}{{{base},status_code=\"{ERROR_STATUS}\"}}[{w}]))",
        "p95": f"histogram_quantile(0.95, sum by (span_name, le) (rate({SPAN_DURATION_BUCKET}{{{base}}}[{w}])))",
    }


def container_queries(container: str, window_s: float) -> dict[str, str]:
    c = _label_value(container)
    w = window_selector(window_s)
    sel = f'container_name="{c}"'
    return {
        "cpu_cores_used_avg": f"sum(rate({CPU_USAGE}{{{sel}}}[{w}])) / 1e9",
        "cpu_cores_used_peak_1m": f"max_over_time((sum(rate({CPU_USAGE}{{{sel}}}[1m])) / 1e9)[{w}:15s])",
        "cpu_limit_cpus": f"max(last_over_time({CPU_LIMIT}{{{sel}}}[{w}]))",
        "cpu_throttled_ratio": (f"sum(increase({CPU_THROTTLED_PERIODS}{{{sel}}}[{w}]))"
                                f" / sum(increase({CPU_PERIODS}{{{sel}}}[{w}]))"),
        "memory_max_bytes": f"max(max_over_time({MEMORY_USAGE}{{{sel}}}[{w}]))",
        "memory_limit_bytes": f"max(last_over_time({MEMORY_LIMIT}{{{sel}}}[{w}]))",
        "restarts_in_window": f"sum(increase({CONTAINER_RESTARTS}{{{sel}}}[{w}]))",
    }


def cache_queries(window_s: float) -> dict[str, str]:
    w = window_selector(window_s)
    return {
        "hit_ratio": (f"sum(increase({REDIS_HITS}[{w}])) / "
                      f"(sum(increase({REDIS_HITS}[{w}])) + sum(increase({REDIS_MISSES}[{w}])))"),
        "commands_per_s": f"sum(rate({REDIS_COMMANDS}[{w}]))",
        "clients_connected": f"max(max_over_time({REDIS_CLIENTS}[{w}]))",
        "memory_used_bytes": f"max(max_over_time({REDIS_MEMORY}[{w}]))",
    }


# ---------------------------------------------------------------------------
# Jaeger
# ---------------------------------------------------------------------------
class Jaeger:
    def __init__(self, base_url: str, client: httpx.AsyncClient):
        self.base = base_url.rstrip("/")
        self.client = client

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        try:
            response = await self.client.get(f"{self.base}{path}", params=params)
        except httpx.HTTPError as exc:
            raise TelemetryUnavailable(f"Jaeger unreachable: {type(exc).__name__}") from exc
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise TelemetryUnavailable(f"Jaeger HTTP {response.status_code}: {response.text[:300]}")
        return response.json()

    async def services(self) -> list[str]:
        body = await self._get("/api/services")
        return sorted((body or {}).get("data") or [])

    async def search(self, service: str, start: float, end: float, *, limit: int = 20,
                     min_duration_ms: float | None = None, operation: str | None = None,
                     errors_only: bool = False) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "service": service, "start": int(start * 1e6), "end": int(end * 1e6), "limit": int(limit),
        }
        if min_duration_ms:
            params["minDuration"] = f"{int(min_duration_ms)}ms"
        if operation:
            params["operation"] = operation
        if errors_only:
            params["tags"] = json.dumps({"error": "true"})
        body = await self._get("/api/traces", params)
        return (body or {}).get("data") or []

    async def trace(self, trace_id: str) -> dict[str, Any] | None:
        body = await self._get(f"/api/traces/{quote(trace_id)}")
        data = (body or {}).get("data") or []
        return data[0] if data else None

    def trace_link(self, trace_id: str) -> str:
        return f"{self.base}/trace/{trace_id}"


def _tags(items: list[dict[str, Any]] | None) -> dict[str, Any]:
    return {t.get("key"): t.get("value") for t in items or []}


def summarize_trace(trace: dict[str, Any], max_errors: int = 5) -> dict[str, Any]:
    """Collapse a Jaeger trace into duration, per-service self time and errors."""
    processes = {pid: p.get("serviceName", "?") for pid, p in (trace.get("processes") or {}).items()}
    spans = trace.get("spans") or []
    if not spans:
        return {"trace_id": trace.get("traceID"), "span_count": 0}
    by_id = {s["spanID"]: s for s in spans}
    children: dict[str, list[dict]] = {}
    root = None
    for span in spans:
        parent = next((r["spanID"] for r in span.get("references") or [] if r.get("refType") == "CHILD_OF"), None)
        if parent and parent in by_id:
            children.setdefault(parent, []).append(span)
        elif root is None or span["startTime"] < root["startTime"]:
            root = span

    def covered(span: dict) -> float:
        # Union of child intervals, clipped to the parent, so concurrent children are not double-counted.
        lo, hi = span["startTime"], span["startTime"] + span["duration"]
        intervals = sorted((max(lo, c["startTime"]), min(hi, c["startTime"] + c["duration"]))
                           for c in children.get(span["spanID"], []))
        total, cur_s, cur_e = 0.0, None, None
        for s, e in intervals:
            if e <= s:
                continue
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    total += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            total += cur_e - cur_s
        return total

    self_time: dict[str, float] = {}
    errors = []
    for span in spans:
        service = processes.get(span.get("processID"), "?")
        self_time[service] = self_time.get(service, 0.0) + max(0.0, span["duration"] - covered(span)) / 1000
        tags = _tags(span.get("tags"))
        if tags.get("error") in (True, "true") or tags.get("otel.status_code") == "ERROR":
            message = tags.get("otel.status_description") or tags.get("rpc.grpc.status_code") or ""
            for log in span.get("logs") or []:
                fields = _tags(log.get("fields"))
                message = fields.get("exception.message") or fields.get("message") or message
            if len(errors) < max_errors:
                errors.append({"service": service, "operation": span.get("operationName"), "message": str(message)[:300]})
    slowest = sorted(spans, key=lambda s: s["duration"], reverse=True)[:6]
    return {
        "trace_id": trace.get("traceID"),
        "root": f"{processes.get(root.get('processID'), '?')}:{root.get('operationName')}" if root else None,
        "duration_ms": round(root["duration"] / 1000, 1) if root else None,
        "span_count": len(spans),
        "self_time_ms_by_service": {k: round(v, 1) for k, v in sorted(self_time.items(), key=lambda kv: -kv[1])},
        "slowest_spans": [{"service": processes.get(s.get("processID"), "?"), "operation": s.get("operationName"),
                           "duration_ms": round(s["duration"] / 1000, 1)} for s in slowest],
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# OpenSearch logs
# ---------------------------------------------------------------------------
class Logs:
    # Field names written by the collector's opensearch exporter (dynamic mapping with .keyword subfields).
    TIME = "@timestamp"
    SERVICE = "resource.service.name.keyword"
    SEVERITY_TEXT = "severity.text.keyword"
    SEVERITY_NUMBER = "severity.number"
    BODY = "body"
    # OpenTelemetry severity numbers: WARN 13-16, ERROR 17-20, FATAL 21-24.
    MIN_SEVERITY = {"WARN": 13, "ERROR": 17}
    # The collector's own debug exporter logs; never application evidence.
    EXCLUDED_SERVICES = ["otelcol-contrib"]

    def __init__(self, base_url: str, client: httpx.AsyncClient):
        self.base = base_url.rstrip("/")
        self.client = client

    async def search(self, start: float, end: float, *, service: str | None = None,
                     min_severity: str | None = None, text: str | None = None, limit: int = 30) -> dict[str, Any]:
        filters: list[dict[str, Any]] = [{"range": {self.TIME: {
            "gte": int(start * 1000), "lte": int(end * 1000), "format": "epoch_millis"}}}]
        if service:
            filters.append({"term": {self.SERVICE: service}})
        if min_severity:
            filters.append({"range": {self.SEVERITY_NUMBER: {"gte": self.MIN_SEVERITY.get(min_severity.upper(), 13)}}})
        query: dict[str, Any] = {"bool": {"filter": filters,
                                          "must_not": [{"terms": {self.SERVICE: self.EXCLUDED_SERVICES}}]}}
        if text:
            query["bool"]["must"] = [{"match_phrase": {self.BODY: text}}]
        body = {
            "size": min(int(limit), 100), "sort": [{self.TIME: "desc"}], "query": query,
            "_source": [self.TIME, "resource.service.name", "severity", self.BODY, "traceId"],
            "aggs": {
                "by_service": {"terms": {"field": self.SERVICE, "size": 15}},
                "by_severity": {"terms": {"field": self.SEVERITY_TEXT, "size": 10}},
            },
        }
        try:
            response = await self.client.post(f"{self.base}/{LOG_INDEX}/_search", json=body)
        except httpx.HTTPError as exc:
            raise TelemetryUnavailable(f"OpenSearch unreachable: {type(exc).__name__}") from exc
        if response.status_code == 404:
            return {"total": 0, "by_service": {}, "by_severity": {}, "entries": []}
        if response.status_code >= 400:
            raise TelemetryUnavailable(f"OpenSearch HTTP {response.status_code}: {response.text[:300]}")
        data = response.json()
        hits = data.get("hits", {})
        total = hits.get("total", {})
        aggs = data.get("aggregations", {})
        return {
            "total": total.get("value", 0) if isinstance(total, dict) else total,
            "by_service": {b["key"]: b["doc_count"] for b in aggs.get("by_service", {}).get("buckets", [])},
            "by_severity": {b["key"]: b["doc_count"] for b in aggs.get("by_severity", {}).get("buckets", [])},
            "entries": [h.get("_source", {}) for h in hits.get("hits", [])],
        }


def dedupe_log_entries(entries: list[dict[str, Any]], limit: int = 15) -> list[dict[str, Any]]:
    """Group near-identical log lines so a noisy loop does not crowd out the rest."""
    groups: dict[tuple, dict[str, Any]] = {}
    for entry in entries:
        resource = entry.get("resource") or {}
        service = resource.get("service.name") or (resource.get("service") or {}).get("name") or "?"
        severity = (entry.get("severity") or {}).get("text") if isinstance(entry.get("severity"), dict) else None
        body = entry.get("body")
        text = body if isinstance(body, str) else json.dumps(body, default=str)
        text = " ".join(str(text).split())[:240]
        key = (service, severity, text[:120])
        group = groups.setdefault(key, {"service": service, "severity": severity,
                                        "message": text, "count": 0, "last_seen": entry.get("@timestamp"),
                                        "trace_id": entry.get("traceId")})
        group["count"] += 1
    return sorted(groups.values(), key=lambda g: -g["count"])[:limit]


# ---------------------------------------------------------------------------
# Docker engine (read-only use)
# ---------------------------------------------------------------------------
class DockerInspector:
    """Reads container configuration, state and lifecycle events. Never mutates."""

    WATCHED_EVENTS = {"die", "oom", "kill", "restart", "start", "stop", "update", "create", "destroy", "health_status"}

    def __init__(self, base_url: str):
        self.base_url = base_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import docker  # imported lazily so tests and the UI do not require a daemon

            try:
                self._client = docker.DockerClient(base_url=self.base_url, timeout=10)
            except Exception as exc:  # docker raises DockerException subclasses
                raise TelemetryUnavailable(f"Docker engine unreachable: {exc}") from exc
        return self._client

    def snapshot(self, name: str) -> dict[str, Any]:
        try:
            attrs = self.client.containers.get(name).attrs
        except TelemetryUnavailable:
            raise
        except Exception as exc:
            raise TelemetryUnavailable(f"Container {name} unavailable: {type(exc).__name__}") from exc
        return config_snapshot(attrs)

    def snapshots(self, names: list[str]) -> dict[str, dict[str, Any]]:
        out = {}
        for name in names:
            try:
                out[name] = self.snapshot(name)
            except TelemetryUnavailable:
                continue
        return out

    def events(self, start: float, end: float, names: list[str] | None = None, limit: int = 60) -> list[dict[str, Any]]:
        until = min(end, time.time())
        if until <= start:
            return []
        try:
            stream = self.client.events(since=int(start), until=int(math.ceil(until)), decode=True,
                                        filters={"type": "container"})
            rows = []
            for item in stream:
                action = (item.get("Action") or "").split(":")[0]
                attributes = (item.get("Actor") or {}).get("Attributes") or {}
                name = attributes.get("name")
                if action not in self.WATCHED_EVENTS or (names and name not in names):
                    continue
                if action == "health_status":
                    action = item.get("Action", "").replace(" ", "")
                rows.append({"time": item.get("timeNano", 0) / 1e9 or item.get("time"), "container": name,
                             "action": action, "exit_code": attributes.get("exitCode")})
        except TelemetryUnavailable:
            raise
        except Exception as exc:
            raise TelemetryUnavailable(f"Docker events unavailable: {type(exc).__name__}: {exc}") from exc
        return rows[-limit:]


def config_snapshot(attrs: dict[str, Any]) -> dict[str, Any]:
    """The configuration and state fields that matter for incident diagnosis."""
    host = attrs.get("HostConfig") or {}
    state = attrs.get("State") or {}
    config = attrs.get("Config") or {}
    nano = host.get("NanoCpus") or 0
    quota, period = host.get("CpuQuota") or 0, host.get("CpuPeriod") or 0
    cpus = nano / 1e9 if nano else (quota / period if quota > 0 and period else None)
    return {
        "image": config.get("Image"),
        "cpu_limit_cpus": round(cpus, 3) if cpus else None,
        "memory_limit_mb": round(host["Memory"] / 2**20) if host.get("Memory") else None,
        "restart_policy": (host.get("RestartPolicy") or {}).get("Name"),
        "status": state.get("Status"),
        "health": (state.get("Health") or {}).get("Status"),
        "restart_count": attrs.get("RestartCount", 0),
        "oom_killed": state.get("OOMKilled", False),
        "started_at": state.get("StartedAt"),
    }

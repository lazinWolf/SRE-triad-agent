"""Thin clients for the demo's observability backends and Docker.

Prometheus (span metrics and container stats), Jaeger (traces), OpenSearch (logs),
the Docker Engine API (container config, events, updates) and the flagd flag file.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from .config import Settings

SPAN_CALLS = "traces_span_metrics_calls_total"
SPAN_BUCKET = "traces_span_metrics_duration_milliseconds_bucket"
CPU_USAGE = "container_cpu_usage_nanoseconds_total"
CPU_PERIODS = "container_cpu_throttling_data_periods_total"
CPU_THROTTLED = "container_cpu_throttling_data_throttled_periods_total"
MEMORY_USAGE = "container_memory_usage_total_bytes"
MEMORY_LIMIT = "container_memory_usage_limit_bytes"


class Unavailable(RuntimeError):
    """A backend could not answer."""


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class Prometheus:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def query(self, expr: str, at: float) -> float | None:
        try:
            r = httpx.get(f"{self.url}/api/v1/query", params={"query": expr, "time": at}, timeout=10)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise Unavailable(f"Prometheus: {exc}") from exc
        result = r.json()["data"]["result"]
        return _num(result[0]["value"][1]) if result else None

    def service_red(self, service: str, start: float, end: float) -> dict[str, float | None]:
        """Request rate, error ratio and latency of a service's server spans over [start, end]."""
        w = f"{max(30, int(end - start))}s"
        sel = f'service_name="{service}",span_kind="SPAN_KIND_SERVER"'
        exprs = {
            "requests_per_s": f"sum(rate({SPAN_CALLS}{{{sel}}}[{w}]))",
            "error_ratio": f'(sum(rate({SPAN_CALLS}{{{sel},status_code="STATUS_CODE_ERROR"}}[{w}])) or vector(0))'
                           f" / sum(rate({SPAN_CALLS}{{{sel}}}[{w}]))",
            "p50_ms": f"histogram_quantile(0.5, sum by (le) (rate({SPAN_BUCKET}{{{sel}}}[{w}])))",
            "p95_ms": f"histogram_quantile(0.95, sum by (le) (rate({SPAN_BUCKET}{{{sel}}}[{w}])))",
        }
        return {k: _round(self.query(e, end)) for k, e in exprs.items()}

    def container(self, name: str, start: float, end: float) -> dict[str, float | None]:
        w = f"{max(30, int(end - start))}s"
        sel = f'container_name="{name}"'
        exprs = {
            "cpu_cores_used": f"sum(rate({CPU_USAGE}{{{sel}}}[{w}])) / 1e9",
            "cpu_throttled_ratio": f"sum(increase({CPU_THROTTLED}{{{sel}}}[{w}])) / sum(increase({CPU_PERIODS}{{{sel}}}[{w}]))",
            "memory_mb": f"max(max_over_time({MEMORY_USAGE}{{{sel}}}[{w}])) / 1048576",
            "memory_limit_mb": f"max(last_over_time({MEMORY_LIMIT}{{{sel}}}[{w}])) / 1048576",
        }
        return {k: _round(self.query(e, end)) for k, e in exprs.items()}


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4 if abs(value) < 1 else 1)


class Jaeger:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def _get(self, path: str, params: dict | None = None) -> Any:
        try:
            r = httpx.get(f"{self.url}{path}", params=params, timeout=15)
        except httpx.HTTPError as exc:
            raise Unavailable(f"Jaeger: {exc}") from exc
        if r.status_code == 404:
            return {}
        if r.status_code >= 400:
            raise Unavailable(f"Jaeger HTTP {r.status_code}")
        return r.json()

    def search(self, service: str, start: float, end: float, *, limit: int = 20,
               min_ms: float | None = None, errors: bool = False) -> list[dict]:
        params: dict[str, Any] = {"service": service, "start": int(start * 1e6), "end": int(end * 1e6), "limit": limit}
        if min_ms:
            params["minDuration"] = f"{int(min_ms)}ms"
        if errors:
            params["tags"] = json.dumps({"error": "true"})
        return self._get("/api/traces", params).get("data") or []


def _tags(items: list[dict] | None) -> dict[str, Any]:
    return {t.get("key"): t.get("value") for t in items or []}


def summarize_trace(trace: dict) -> dict[str, Any]:
    """Duration, time spent in each service (self time) and errors of one Jaeger trace."""
    procs = {pid: p.get("serviceName", "?") for pid, p in (trace.get("processes") or {}).items()}
    spans = trace.get("spans") or []
    children: dict[str, list[dict]] = {}
    ids = {s["spanID"] for s in spans}
    root = None
    for s in spans:
        parent = next((r["spanID"] for r in s.get("references") or [] if r.get("refType") == "CHILD_OF"), None)
        if parent in ids:
            children.setdefault(parent, []).append(s)
        elif root is None or s["startTime"] < root["startTime"]:
            root = s
    self_ms: dict[str, float] = {}
    errors = []
    for s in spans:
        lo, hi = s["startTime"], s["startTime"] + s["duration"]
        covered, cur_s, cur_e = 0.0, None, None  # union of child intervals
        for a, b in sorted((max(lo, c["startTime"]), min(hi, c["startTime"] + c["duration"])) for c in children.get(s["spanID"], [])):
            if b <= a:
                continue
            if cur_e is None or a > cur_e:
                covered += (cur_e - cur_s) if cur_e is not None else 0
                cur_s, cur_e = a, b
            else:
                cur_e = max(cur_e, b)
        covered += (cur_e - cur_s) if cur_e is not None else 0
        svc = procs.get(s.get("processID"), "?")
        self_ms[svc] = self_ms.get(svc, 0) + max(0.0, s["duration"] - covered) / 1000
        tags = _tags(s.get("tags"))
        if tags.get("error") in (True, "true") or tags.get("otel.status_code") == "ERROR":
            msg = tags.get("otel.status_description") or ""
            for log in s.get("logs") or []:
                msg = _tags(log.get("fields")).get("exception.message") or msg
            errors.append({"service": svc, "operation": s.get("operationName"), "message": str(msg)[:200]})
    return {
        "trace_id": trace.get("traceID"),
        "root": f"{procs.get(root.get('processID'), '?')} {root.get('operationName')}" if root else None,
        "duration_ms": round(root["duration"] / 1000, 1) if root else None,
        "self_time_ms": {k: round(v, 1) for k, v in sorted(self_ms.items(), key=lambda kv: -kv[1])[:6]},
        "errors": errors[:4],
    }


class Logs:
    SERVICE = "resource.service.name.keyword"

    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def search(self, start: float, end: float, *, service: str | None = None, min_severity: int = 13,
               text: str | None = None, limit: int = 50) -> dict[str, Any]:
        """Warning-or-worse log lines (OTel severity >= 13), grouped by message."""
        filters: list[dict] = [{"range": {"@timestamp": {"gte": int(start * 1000), "lte": int(end * 1000),
                                                         "format": "epoch_millis"}}},
                               {"range": {"severity.number": {"gte": min_severity}}}]
        if service:
            filters.append({"term": {self.SERVICE: service}})
        query: dict = {"bool": {"filter": filters, "must_not": [{"term": {self.SERVICE: "otelcol-contrib"}}]}}
        if text:
            query["bool"]["must"] = [{"match_phrase": {"body": text}}]
        body = {"size": limit, "sort": [{"@timestamp": "desc"}], "query": query,
                "aggs": {"by_service": {"terms": {"field": self.SERVICE, "size": 10}}}}
        try:
            r = httpx.post(f"{self.url}/otel-logs-*/_search", json=body, timeout=15)
        except httpx.HTTPError as exc:
            raise Unavailable(f"OpenSearch: {exc}") from exc
        if r.status_code == 404:
            return {"total": 0, "by_service": {}, "messages": []}
        if r.status_code >= 400:
            raise Unavailable(f"OpenSearch HTTP {r.status_code}")
        data = r.json()
        groups: dict[tuple, dict] = {}
        for hit in data["hits"]["hits"]:
            src = hit["_source"]
            svc = (src.get("resource") or {}).get("service.name") or "?"
            text_ = " ".join(str(src.get("body")).split())[:200]
            g = groups.setdefault((svc, text_[:100]), {"service": svc, "message": text_, "count": 0})
            g["count"] += 1
        total = data["hits"]["total"]
        return {"total": total.get("value", 0) if isinstance(total, dict) else total,
                "by_service": {b["key"]: b["doc_count"] for b in data["aggregations"]["by_service"]["buckets"]},
                "messages": sorted(groups.values(), key=lambda g: -g["count"])[:8]}


class Docker:
    """Docker Engine API over the unix socket: inspect, events, CPU updates and restarts."""

    def __init__(self, socket: str):
        self.socket = socket

    def _request(self, method: str, path: str, timeout: float = 30, **kw) -> httpx.Response:
        try:
            with httpx.Client(transport=httpx.HTTPTransport(uds=self.socket), base_url="http://docker", timeout=timeout) as c:
                r = c.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise Unavailable(f"Docker: {exc}") from exc
        if r.status_code >= 400:
            raise Unavailable(f"Docker {path}: HTTP {r.status_code} {r.text[:200]}")
        return r

    def inspect(self, name: str) -> dict[str, Any]:
        attrs = self._request("GET", f"/containers/{quote(name)}/json").json()
        host, state = attrs.get("HostConfig") or {}, attrs.get("State") or {}
        return {"cpu_limit": round((host.get("NanoCpus") or 0) / 1e9, 3) or None,
                "memory_limit_mb": round(host["Memory"] / 2**20) if host.get("Memory") else None,
                "status": state.get("Status"), "restart_count": attrs.get("RestartCount", 0),
                "oom_killed": state.get("OOMKilled", False), "started_at": state.get("StartedAt")}

    def events(self, since: float, until: float, names: list[str] | None = None) -> list[dict[str, Any]]:
        until = min(until, time.time() - 1)
        if until <= since:
            return []
        filters = json.dumps({"type": ["container"], "event": ["update", "restart", "die", "oom", "start"]})
        r = self._request("GET", "/events", params={"since": int(since), "until": int(until), "filters": filters})
        rows = []
        for line in r.text.splitlines():
            event = json.loads(line)
            name = (event.get("Actor") or {}).get("Attributes", {}).get("name")
            if not names or name in names:
                rows.append({"time": event.get("time"), "container": name, "action": event.get("Action")})
        return rows

    def set_cpus(self, name: str, cpus: float) -> None:
        self._request("POST", f"/containers/{quote(name)}/update", json={"NanoCpus": int(cpus * 1e9)})

    def restart(self, name: str) -> None:
        self._request("POST", f"/containers/{quote(name)}/restart", params={"t": 10}, timeout=90)


def read_flags(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    return {k: {"variant": f.get("defaultVariant"), "variants": sorted(f.get("variants", {}))}
            for k, f in data.get("flags", {}).items()}


def write_flag(path: Path, flag: str, variant: str) -> str:
    """Set a flag's default variant; returns the previous one. Edits in place (flagd watches the file)."""
    data = json.loads(path.read_text())
    entry = data["flags"].get(flag)
    if entry is None or variant not in entry.get("variants", {}):
        raise ValueError(f"unknown flag or variant: {flag}={variant}")
    previous, entry["defaultVariant"] = entry["defaultVariant"], variant
    with open(path, "r+") as handle:
        handle.write(json.dumps(data, indent=2) + "\n")
        handle.truncate()
    return previous


class Backends:
    def __init__(self, settings: Settings):
        self.prometheus = Prometheus(settings.prometheus_url)
        self.jaeger = Jaeger(settings.jaeger_url)
        self.logs = Logs(settings.opensearch_url)
        self.docker = Docker(settings.docker_socket)
        self.flagd_file = settings.flagd_file

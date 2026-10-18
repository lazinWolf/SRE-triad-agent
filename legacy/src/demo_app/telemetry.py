"""Bounded local telemetry, exported from real requests and OpenTelemetry spans."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any, Iterable

from fastapi import FastAPI, HTTPException, Query, Response
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest


def percentile(values: Iterable[float], quantile: float = 0.95) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    return round(ordered[max(0, math.ceil(len(ordered) * quantile) - 1)], 3)


class RecordStore:
    """A count- and age-bounded stream with explicit availability metadata."""

    def __init__(self, started_at: float, max_records: int = 10000, retention_seconds: int = 3600):
        self.started_at = started_at
        self.max_records = max_records
        self.retention_seconds = retention_seconds
        self.records: deque[dict[str, Any]] = deque()
        self.lock = threading.Lock()
        self.dropped_through = started_at

    def _prune(self, now: float) -> None:
        cutoff = now - self.retention_seconds
        while self.records and self.records[0]["timestamp"] < cutoff:
            self.dropped_through = max(self.dropped_through, self.records.popleft()["timestamp"])
        while len(self.records) > self.max_records:
            self.dropped_through = max(self.dropped_through, self.records.popleft()["timestamp"])

    def append(self, record: dict[str, Any]) -> None:
        with self.lock:
            self.records.append(record)
            self._prune(time.time())

    def select(self, start: float, end: float, limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        with self.lock:
            now = time.time()
            self._prune(now)
            rows = [dict(row) for row in self.records if start <= row["timestamp"] <= end]
            available = max(self.started_at, self.dropped_through, now - self.retention_seconds)
            selected = rows[-limit:] if limit is not None else rows
            metadata = {
                "retention_seconds": self.retention_seconds,
                "max_records": self.max_records,
                "telemetry_started_epoch": self.started_at,
                "oldest_available_epoch": available,
                "window_complete": start >= available and end <= now,
                "matched_count": len(rows),
                "returned_count": len(selected),
                "truncated": len(selected) < len(rows),
            }
            return selected, metadata


class RingSpanExporter(SpanExporter):
    def __init__(self, store: RecordStore):
        self.store = store

    def export(self, spans: Any) -> SpanExportResult:
        for span in spans:
            self.store.append({
                "timestamp": span.end_time / 1e9,
                "service": span.resource.attributes.get("service.name"),
                "trace_id": format(span.context.trace_id, "032x"),
                "span_id": format(span.context.span_id, "016x"),
                "parent_span_id": format(span.parent.span_id, "016x") if span.parent else None,
                "name": span.name,
                "kind": span.kind.name,
                "start_time": span.start_time / 1e9,
                "end_time": span.end_time / 1e9,
                "duration_ms": round((span.end_time - span.start_time) / 1e6, 3),
                "status": span.status.status_code.name,
                "attributes": dict(span.attributes or {}),
            })
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


class Telemetry:
    def __init__(self, service: str, max_records: int = 10000, retention_seconds: int = 3600):
        self.service = service
        self.started_at = time.time()
        self.requests = RecordStore(self.started_at, max_records, retention_seconds)
        self.spans = RecordStore(self.started_at, max_records, retention_seconds)
        self.logs = RecordStore(self.started_at, max_records, retention_seconds)
        self.changes = RecordStore(self.started_at, max_records, retention_seconds)
        self.provider = TracerProvider(resource=Resource.create({"service.name": service, "deployment.environment.name": "local"}))
        self.provider.add_span_processor(SimpleSpanProcessor(RingSpanExporter(self.spans)))
        self.tracer = self.provider.get_tracer("sre-demo")
        self.registry = CollectorRegistry()
        self.request_counter = Counter("demo_requests_total", "Completed requests", ["service", "status"], registry=self.registry)
        self.request_latency = Histogram("demo_request_duration_seconds", "Request duration", ["service"], buckets=(0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 5), registry=self.registry)
        self.queue_latency = Histogram("demo_queue_wait_seconds", "Inventory queue wait", ["service"], buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 5), registry=self.registry)
        self.active_gauge = Gauge("demo_active_workers", "Active inventory workers", ["service"], registry=self.registry)
        self.slots_gauge = Gauge("demo_worker_slots", "Configured inventory worker slots", ["service"], registry=self.registry)

    def log(self, severity: str, message: str, trace_id: str | None = None, **attributes: Any) -> None:
        self.logs.append({"timestamp": time.time(), "service": self.service, "severity": severity, "message": message, "trace_id": trace_id, "attributes": attributes})

    def record_request(self, started: float, status: int, queue_wait_ms: float = 0, **attributes: Any) -> None:
        now = time.time()
        self.requests.append({"timestamp": now, "start_time": started, "service": self.service, "latency_ms": (now - started) * 1000, "queue_wait_ms": queue_wait_ms, "status": status, **attributes})
        self.request_counter.labels(self.service, str(status)).inc()
        self.request_latency.labels(self.service).observe(now - started)
        self.queue_latency.labels(self.service).observe(queue_wait_ms / 1000)


def validate_window(start: float, end: float) -> None:
    if not math.isfinite(start) or not math.isfinite(end) or start > end:
        raise HTTPException(status_code=422, detail="start and end must be finite epoch seconds with start <= end")
    if end - start > 3600:
        raise HTTPException(status_code=422, detail="query window cannot exceed 3600 seconds")


def add_telemetry_routes(app: FastAPI, telemetry: Telemetry, context: dict[str, Any], runtime: Any) -> None:
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": telemetry.service}

    @app.get("/metrics")
    async def metrics_export() -> Response:
        return Response(generate_latest(telemetry.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/telemetry/context")
    async def service_context() -> dict[str, Any]:
        return {"service": telemetry.service, "environment": "local", **context}

    @app.get("/telemetry/runtime")
    async def runtime_state() -> dict[str, Any]:
        return {"service": telemetry.service, "environment": "local", "observed_at": time.time(), **runtime()}

    @app.get("/telemetry/metrics")
    async def metrics_window(start: float, end: float) -> dict[str, Any]:
        validate_window(start, end)
        rows, metadata = telemetry.requests.select(start, end)
        response = {
            "service": telemetry.service,
            "window_start": start,
            "window_end": end,
            "sample_count": len(rows),
            "request_count": len(rows),
            "error_rate": sum(row["status"] >= 400 for row in rows) / len(rows) if rows else None,
            "p95_latency_ms": percentile(row["latency_ms"] for row in rows),
            "p95_queue_wait_ms": percentile(row["queue_wait_ms"] for row in rows),
            "request_rate_per_second": round(len(rows) / (end - start), 3) if end > start else None,
            "metadata": metadata,
        }
        current = runtime()["current_state"]
        if "worker_slots" in current:
            response.update({"worker_slots": current["worker_slots"], "active_workers": current["active_workers"], "resource_observed_at": time.time()})
            response["observed_worker_slots"] = sorted({row["worker_slots"] for row in rows if "worker_slots" in row})
        return response

    def register_stream(path: str, key: str, store: RecordStore) -> None:
        async def stream(start: float, end: float, limit: int = Query(100, ge=1, le=200)) -> dict[str, Any]:
            validate_window(start, end)
            rows, metadata = store.select(start, end, limit)
            return {"service": telemetry.service, "window_start": start, "window_end": end, key: rows, "metadata": metadata}
        app.add_api_route(path, stream, methods=["GET"])

    register_stream("/telemetry/traces", "traces", telemetry.spans)
    register_stream("/telemetry/logs", "logs", telemetry.logs)
    register_stream("/telemetry/changes", "changes", telemetry.changes)

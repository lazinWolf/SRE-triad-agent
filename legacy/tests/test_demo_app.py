"""Check capacity-induced latency, cross-service tracing, and evidence bounds."""

import asyncio
import time

import httpx
import pytest

from demo_app.server import create_checkout_app, create_inventory_app
from demo_app.telemetry import RecordStore


@pytest.mark.asyncio
async def test_capacity_reduction_produces_queue_wait_and_audited_change():
    app = create_inventory_app(worker_slots=4, processing_ms=25)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://inventory") as client:
        started = time.time()
        baseline = await asyncio.gather(*(client.post("/reserve", json={}) for _ in range(4)))
        assert all(result.status_code == 200 for result in baseline)
        assert max(result.json()["queue_wait_ms"] for result in baseline) < 15
        changed = await client.post("/admin/config", json={"worker_slots": 1})
        assert changed.status_code == 200
        degraded = await asyncio.gather(*(client.post("/reserve", json={}) for _ in range(4)))
        assert max(result.json()["queue_wait_ms"] for result in degraded) > 50
        end = time.time()
        metrics = (await client.get("/telemetry/metrics", params={"start": started, "end": end})).json()
        assert metrics["request_count"] == 8
        assert metrics["p95_queue_wait_ms"] > 50
        assert metrics["active_workers"] == 0
        assert metrics["observed_worker_slots"] == [1, 4]
        changes = (await client.get("/telemetry/changes", params={"start": started, "end": end})).json()["changes"]
        assert len(changes) == 1
        assert changes[0]["before"] == {"worker_slots": 4}
        assert changes[0]["after"] == {"worker_slots": 1}
        logs = (await client.get("/telemetry/logs", params={"start": started, "end": end})).json()["logs"]
        assert any(row["severity"] == "WARNING" and row["attributes"]["queue_wait_ms"] > 50 for row in logs)
        exported = await client.get("/metrics")
        assert "demo_requests_total" in exported.text


@pytest.mark.asyncio
async def test_checkout_propagates_trace_context_to_inventory():
    inventory = create_inventory_app(processing_ms=5)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=inventory), base_url="http://inventory") as downstream:
        checkout = create_checkout_app("http://inventory", downstream)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=checkout), base_url="http://checkout") as client:
            start = time.time()
            response = await client.post("/checkout", json={})
            assert response.status_code == 200
            assert response.json()["trace_id"] == response.json()["inventory"]["trace_id"]
            end = time.time()
            checkout_spans = (await client.get("/telemetry/traces", params={"start": start, "end": end})).json()["traces"]
            inventory_spans = (await downstream.get("/telemetry/traces", params={"start": start, "end": end})).json()["traces"]
            assert len(checkout_spans) == 2 and len(inventory_spans) == 1
            client_span = next(span for span in checkout_spans if span["kind"] == "CLIENT")
            assert inventory_spans[0]["parent_span_id"] == client_span["span_id"]
            assert inventory_spans[0]["trace_id"] == client_span["trace_id"]
            assert inventory_spans[0]["attributes"]["inventory.processing_ms"] == 5
            context = (await client.get("/telemetry/context")).json()
            assert context["dependencies"][0]["service"] == "inventory"


@pytest.mark.asyncio
async def test_query_limits_and_missing_data_are_explicit():
    app = create_inventory_app(processing_ms=1)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://inventory") as client:
        now = time.time()
        assert (await client.get("/telemetry/metrics", params={"start": now, "end": now - 1})).status_code == 422
        assert (await client.get("/telemetry/metrics", params={"start": now - 3601, "end": now})).status_code == 422
        assert (await client.get("/telemetry/logs", params={"start": now, "end": now, "limit": 201})).status_code == 422
        metrics = (await client.get("/telemetry/metrics", params={"start": now, "end": now})).json()
        assert metrics["request_count"] == 0
        assert metrics["p95_latency_ms"] is None
        assert metrics["error_rate"] is None
        assert (await client.post("/admin/config", json={"worker_slots": 0})).status_code == 422


def test_record_store_eviction_and_collection_truncation():
    start = time.time()
    store = RecordStore(start, max_records=2)
    for index in range(3):
        store.append({"timestamp": start + index * 0.0001, "index": index})
    rows, metadata = store.select(start, start + 0.001, limit=1)
    assert rows[0]["index"] == 2
    assert metadata["matched_count"] == 2
    assert metadata["truncated"] is True
    assert metadata["window_complete"] is False

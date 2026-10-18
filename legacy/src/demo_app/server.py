"""Two HTTP services whose latency changes when real inventory capacity changes.

Run inventory_app on port 8102 and checkout_app on port 8101. Administrative
configuration endpoints belong to the operator and are not agent tools.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from opentelemetry import propagate, trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from pydantic import BaseModel, ConfigDict, Field

from demo_app.telemetry import Telemetry, add_telemetry_routes


class OrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sku: str = Field(default="demo-item", min_length=1, max_length=100)
    quantity: int = Field(default=1, ge=1, le=10)


class CapacityUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    worker_slots: int = Field(ge=1, le=16)


class WorkerPool:
    def __init__(self, worker_slots: int, telemetry: Telemetry):
        self.worker_slots = worker_slots
        self.active_workers = 0
        self.condition = asyncio.Condition()
        self.telemetry = telemetry
        telemetry.slots_gauge.labels(telemetry.service).set(worker_slots)

    async def acquire(self) -> None:
        async with self.condition:
            await self.condition.wait_for(lambda: self.active_workers < self.worker_slots)
            self.active_workers += 1
            self.telemetry.active_gauge.labels(self.telemetry.service).set(self.active_workers)

    async def release(self) -> None:
        async with self.condition:
            self.active_workers -= 1
            self.telemetry.active_gauge.labels(self.telemetry.service).set(self.active_workers)
            self.condition.notify_all()

    async def configure(self, worker_slots: int) -> dict[str, Any]:
        async with self.condition:
            before = {"worker_slots": self.worker_slots}
            self.worker_slots = worker_slots
            self.telemetry.slots_gauge.labels(self.telemetry.service).set(worker_slots)
            change = {"timestamp": time.time(), "service": self.telemetry.service, "change_id": uuid.uuid4().hex, "actor": "local-operator", "kind": "runtime_configuration", "before": before, "after": {"worker_slots": worker_slots}}
            self.telemetry.changes.append(change)
            self.telemetry.log("INFO", "Worker capacity configuration updated", before=before, after=change["after"], change_id=change["change_id"])
            self.condition.notify_all()
            return change


def current_trace_id() -> str:
    return format(trace.get_current_span().get_span_context().trace_id, "032x")


def create_inventory_app(worker_slots: int = 4, processing_ms: float = 80) -> FastAPI:
    if not 1 <= worker_slots <= 16 or processing_ms <= 0:
        raise ValueError("worker_slots must be 1..16 and processing_ms must be positive")
    app = FastAPI(title="Local Inventory")
    telemetry = Telemetry("inventory")
    pool = WorkerPool(worker_slots, telemetry)
    app.state.telemetry = telemetry
    app.state.pool = pool

    def runtime() -> dict[str, Any]:
        return {"desired_config": {"worker_slots": pool.worker_slots, "processing_ms": processing_ms}, "current_state": {"worker_slots": pool.worker_slots, "active_workers": pool.active_workers, "process_started_at": telemetry.started_at}}

    add_telemetry_routes(app, telemetry, {"aliases": ["inventory", "inventory-api"], "dependencies": [], "expected_behavior": {"processing_ms": processing_ms, "default_worker_slots": worker_slots, "healthy_queue_wait_ms": "near zero when request rate is below processing capacity"}, "routes": ["POST /reserve"]}, runtime)

    @app.post("/reserve")
    async def reserve(order: OrderRequest, request: Request) -> dict[str, Any]:
        started = time.time()
        status = 200
        acquired = False
        queue_ms = 0.0
        with telemetry.tracer.start_as_current_span("POST /reserve", context=propagate.extract(dict(request.headers)), kind=SpanKind.SERVER) as span:
            span.set_attributes({"http.request.method": "POST", "http.route": "/reserve", "order.sku": order.sku, "order.quantity": order.quantity})
            try:
                await pool.acquire()
                acquired = True
                queue_ms = (time.time() - started) * 1000
                span.set_attributes({"inventory.queue_wait_ms": queue_ms, "inventory.worker_slots": pool.worker_slots, "inventory.active_workers": pool.active_workers, "inventory.processing_ms": processing_ms})
                if queue_ms > processing_ms:
                    telemetry.log("WARNING", "Reservation waited for an available worker", current_trace_id(), queue_wait_ms=round(queue_ms, 3), worker_slots=pool.worker_slots)
                await asyncio.sleep(processing_ms / 1000)
                return {"reserved": True, "sku": order.sku, "quantity": order.quantity, "queue_wait_ms": round(queue_ms, 3), "processing_ms": processing_ms, "trace_id": current_trace_id()}
            except BaseException:
                status = 500
                span.set_status(Status(StatusCode.ERROR, "Reservation interrupted"))
                raise
            finally:
                if acquired:
                    await pool.release()
                span.set_attribute("http.response.status_code", status)
                telemetry.record_request(started, status, queue_ms, worker_slots=pool.worker_slots)

    @app.post("/admin/config")
    async def configure(update: CapacityUpdate) -> dict[str, Any]:
        return {"change": await pool.configure(update.worker_slots), **runtime()}

    return app


def create_checkout_app(inventory_url: str = "http://127.0.0.1:8102", inventory_client: httpx.AsyncClient | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if inventory_client is not None:
            yield
        else:
            async with httpx.AsyncClient(timeout=10) as client:
                app.state.inventory_client = client
                yield
                app.state.inventory_client = None

    app = FastAPI(title="Local Checkout", lifespan=lifespan)
    app.state.inventory_client = inventory_client
    telemetry = Telemetry("checkout")
    app.state.telemetry = telemetry

    def runtime() -> dict[str, Any]:
        return {"desired_config": {"inventory_url": inventory_url, "dependency_timeout_seconds": 10}, "current_state": {"process_started_at": telemetry.started_at}}

    add_telemetry_routes(app, telemetry, {"aliases": ["checkout", "checkout-api"], "dependencies": [{"service": "inventory", "url": inventory_url, "operation": "POST /reserve"}], "expected_behavior": {"latency_ms": "approximately inventory processing time when capacity is available"}, "routes": ["POST /checkout"]}, runtime)

    @app.post("/checkout")
    async def checkout(order: OrderRequest, request: Request) -> dict[str, Any]:
        started = time.time()
        status = 200
        queue_ms = 0.0
        with telemetry.tracer.start_as_current_span("POST /checkout", context=propagate.extract(dict(request.headers)), kind=SpanKind.SERVER) as span:
            span.set_attributes({"http.request.method": "POST", "http.route": "/checkout", "order.sku": order.sku})
            try:
                client = app.state.inventory_client
                if client is None:
                    raise HTTPException(status_code=503, detail="Checkout service client is not initialized")
                with telemetry.tracer.start_as_current_span("POST inventory /reserve", kind=SpanKind.CLIENT) as downstream:
                    downstream.set_attributes({"server.address": inventory_url, "peer.service": "inventory", "http.request.method": "POST"})
                    headers: dict[str, str] = {}
                    propagate.inject(headers)
                    response = await client.post(f"{inventory_url}/reserve", json=order.model_dump(), headers=headers)
                    downstream.set_attribute("http.response.status_code", response.status_code)
                    if response.is_error:
                        downstream.set_status(Status(StatusCode.ERROR, "Inventory request failed"))
                    response.raise_for_status()
                    result = response.json()
                queue_ms = float(result["queue_wait_ms"])
                span.set_attribute("dependency.inventory.queue_wait_ms", queue_ms)
                elapsed_ms = (time.time() - started) * 1000
                if elapsed_ms > 250:
                    telemetry.log("WARNING", "Checkout dependency request was slow", current_trace_id(), dependency="inventory", latency_ms=round(elapsed_ms, 3))
                return {"order_id": uuid.uuid4().hex, "status": "confirmed", "inventory": result, "trace_id": current_trace_id()}
            except httpx.HTTPError as exc:
                status = 502
                span.set_status(Status(StatusCode.ERROR, "Inventory unavailable"))
                telemetry.log("ERROR", "Inventory request failed", current_trace_id(), dependency="inventory", error_type=type(exc).__name__)
                raise HTTPException(status_code=502, detail="Inventory request failed") from exc
            except HTTPException as exc:
                status = exc.status_code
                span.set_status(Status(StatusCode.ERROR, exc.detail))
                raise
            finally:
                span.set_attribute("http.response.status_code", status)
                telemetry.record_request(started, status, queue_ms)

    return app


inventory_app = create_inventory_app()
checkout_app = create_checkout_app(os.environ.get("DEMO_INVENTORY_URL", "http://127.0.0.1:8102"))

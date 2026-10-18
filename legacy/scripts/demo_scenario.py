#!/usr/bin/env python3
"""Generate real traffic, change inventory capacity, and save an incident alert.

Operator-only scenario controls and answer keys are written outside agent tools.
The application never exports scenario names or expected diagnoses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path

import httpx


def p95(values: list[float]) -> float | None:
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)], 3) if ordered else None


async def traffic(client: httpx.AsyncClient, checkout_url: str, duration: float, interval: float, concurrency: int) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    observations: list[dict] = []
    tasks: list[asyncio.Task] = []
    start = time.time()
    deadline = time.monotonic() + duration

    async def request_once():
        began = time.monotonic()
        try:
            response = await client.post(f"{checkout_url}/checkout", json={"sku": "demo-item", "quantity": 1})
            observations.append({"latency_ms": (time.monotonic() - began) * 1000, "status": response.status_code})
        except httpx.HTTPError:
            observations.append({"latency_ms": (time.monotonic() - began) * 1000, "status": 599})
        finally:
            semaphore.release()

    while time.monotonic() < deadline:
        await semaphore.acquire()
        if time.monotonic() >= deadline:
            semaphore.release()
            break
        tasks.append(asyncio.create_task(request_once()))
        await asyncio.sleep(interval)
    await asyncio.gather(*tasks)
    return {"start": start, "end": time.time(), "request_count": len(observations), "error_count": sum(row["status"] >= 400 for row in observations), "p95_latency_ms": p95([row["latency_ms"] for row in observations])}


async def run(args: argparse.Namespace) -> None:
    checkout_url = args.checkout_url.rstrip("/")
    inventory_url = args.inventory_url.rstrip("/")
    async with httpx.AsyncClient(timeout=15) as client:
        for url in (checkout_url, inventory_url):
            health = await client.get(f"{url}/health")
            health.raise_for_status()
        response = await client.post(f"{inventory_url}/admin/config", json={"worker_slots": args.baseline_slots})
        response.raise_for_status()
        baseline = await traffic(client, checkout_url, args.baseline_duration, args.request_interval, args.concurrency)
        response = await client.post(f"{inventory_url}/admin/config", json={"worker_slots": args.incident_slots})
        response.raise_for_status()
        change = response.json()["change"]
        incident = await traffic(client, checkout_url, args.incident_duration, args.request_interval, args.concurrency)

    # incident_start identifies symptom onset for comparison windows, not a cause.
    alert = {"service": "checkout", "environment": "local", "summary": "Checkout latency increased during sustained request traffic", "start": baseline["start"], "end": incident["end"], "incident_start": incident["start"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(alert, indent=2) + "\n")
    args.oracle_output.parent.mkdir(parents=True, exist_ok=True)
    args.oracle_output.write_text(json.dumps({"scenario": "inventory_worker_capacity_reduction", "change": change, "baseline": baseline, "incident": incident, "expected_evidence": ["increased inventory queue wait", "inventory capacity configuration change", "inventory spans on slow checkout traces"], "expected_remediation_proposal": "Restore the prior worker_slots value after operator review"}, indent=2) + "\n")
    print(json.dumps({"alert_file": str(args.output), "baseline": baseline, "incident": incident}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout-url", default="http://127.0.0.1:8101")
    parser.add_argument("--inventory-url", default="http://127.0.0.1:8102")
    parser.add_argument("--baseline-duration", type=float, default=4)
    parser.add_argument("--incident-duration", type=float, default=4)
    parser.add_argument("--request-interval", type=float, default=0.04)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--baseline-slots", type=int, default=4)
    parser.add_argument("--incident-slots", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("data/demo-alert.json"))
    parser.add_argument("--oracle-output", type=Path, default=Path("data/evaluation/demo-oracle.json"))
    args = parser.parse_args()
    if min(args.baseline_duration, args.incident_duration, args.request_interval) <= 0 or args.concurrency < 1:
        parser.error("durations, request interval, and concurrency must be positive")
    if not (1 <= args.baseline_slots <= 16 and 1 <= args.incident_slots <= 16):
        parser.error("worker slots must be between 1 and 16")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

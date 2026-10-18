"""ADK tools bound to one case, with a shared budget and durable evidence."""

import asyncio
import json
import math

from google.adk.tools import ToolContext

from .storage import CaseStore
from .telemetry import TelemetryGateway


class CaseTools:
    def __init__(self, store: CaseStore, case: dict, gateway: TelemetryGateway, max_calls=24):
        self.store, self.case, self.gateway = store, case, gateway
        self.max_calls = max_calls
        self.calls = 0
        self._lock = asyncio.Lock()
        self._cache = {}

    async def collect(self, role, operation, **query):
        key = json.dumps([operation, query], sort_keys=True)
        async with self._lock:
            if key in self._cache:
                return {**self._cache[key], "cached": True}
            if self.calls >= self.max_calls:
                return {"status": "budget_exhausted", "error": "Shared telemetry budget exhausted", "data": None}
            self.calls += 1
        self.store.add_event(self.case["id"], role, f"{operation}: {query.get('service', '')}", "tool")
        try:
            if "start" in query:
                start, end = query["start"], query["end"]
                if not all(math.isfinite(x) for x in (start, end)) or not (
                    self.case["start"] <= start < end <= self.case["end"]
                ):
                    raise ValueError("Query must remain inside this case's investigation window")
            result = await getattr(self.gateway, operation)(**query)
        except ValueError as exc:
            result = {"status": "invalid_query", "error": str(exc), "data": None}
        evidence = self.store.add_evidence(self.case["id"], role, operation, query, result)
        response = {"evidence_id": evidence["id"], **result}
        async with self._lock:
            self._cache[key] = response
        return response

    def for_role(self, role):
        async def query_metrics(service: str, start: float, end: float, tool_context: ToolContext) -> dict:
            """Read measured latency, traffic and queue statistics for a service and bounded epoch interval."""
            return await self.collect(role, "query_metrics", service=service, start=start, end=end)

        async def find_traces(service: str, start: float, end: float, tool_context: ToolContext) -> dict:
            """Read up to twenty actual request spans; missing samples do not prove health."""
            return await self.collect(role, "find_traces", service=service, start=start, end=end, limit=20)

        async def search_logs(service: str, start: float, end: float, tool_context: ToolContext) -> dict:
            """Read up to fifty structured log entries for the incident window."""
            return await self.collect(role, "search_logs", service=service, start=start, end=end, limit=50)

        async def inspect_runtime(service: str, tool_context: ToolContext) -> dict:
            """Read current runtime configuration; this is a present snapshot, not historical state."""
            return await self.collect(role, "inspect_runtime", service=service)

        async def get_change_history(service: str, start: float, end: float, tool_context: ToolContext) -> dict:
            """Read audited configuration changes in the specified epoch interval."""
            return await self.collect(role, "get_change_history", service=service, start=start, end=end)

        async def get_service_context(service: str, tool_context: ToolContext) -> dict:
            """Read service identity, known dependencies and instrumentation context."""
            return await self.collect(role, "get_service_context", service=service)

        permitted = {
            "metrics": [query_metrics, get_service_context],
            "traces": [find_traces, get_service_context],
            "changes": [search_logs, get_change_history, inspect_runtime, get_service_context],
        }
        return permitted[role]

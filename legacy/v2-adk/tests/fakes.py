"""Test doubles: telemetry backends served by httpx.MockTransport, a fake Docker inspector,
and a scripted ADK model that drives real agents through real tool calls."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator, Callable
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import PrivateAttr

from sre_triage.telemetry import DockerInspector, Jaeger, Logs, Prometheus
from sre_triage.tools import Backends

T0 = 1_800_000_000.0  # incident start used by tests


class World:
    """A scripted incident: cart CPU limited from T0; everything else healthy."""

    def __init__(self, fault_at: float = T0, prometheus_up: bool = True):
        self.fault_at = fault_at
        self.prometheus_up = prometheus_up
        self.queries: list[str] = []

    def prom_value(self, query: str, at: float) -> list[dict[str, Any]]:
        faulty = at > self.fault_at + 30
        service = re.search(r'(?:service_name|container_name)="([^"]+)"', query)
        svc = service.group(1) if service else None
        if "sum by (service_name)" in query or "sum by (service_name, le)" in query:
            rows = []
            for name in ("frontend", "checkout", "cart", "product-catalog"):
                if "histogram_quantile" in query:
                    v = (900.0 if name in ("checkout", "cart", "frontend") else 20.0) if faulty else 20.0
                elif "STATUS_CODE_ERROR" in query:
                    v = 0.0
                else:
                    v = 2.0
                rows.append({"metric": {"service_name": name}, "value": [at, str(v)]})
            return rows
        if "sum by (span_name" in query:
            ops = {"cart": ["oteldemo.CartService/GetCart", "valkey-cart:6379"], "checkout": ["oteldemo.CartService/GetCart"]}
            rows = []
            for op in ops.get(svc, ["op"]):
                if "histogram_quantile" in query:
                    v = 3.0 if "valkey" in op else (850.0 if faulty else 15.0)
                elif "STATUS_CODE_ERROR" in query:
                    v = 0.0
                else:
                    v = 1.0
                rows.append({"metric": {"span_name": op}, "value": [at, str(v)]})
            return rows
        if "container_cpu_throttling" in query:
            v = 0.8 if (faulty and svc == "cart") else 0.0
        elif "container_cpu_limit" in query:
            v = 0.05 if (faulty and svc == "cart") else 1.0
        elif "container_cpu_usage" in query:
            v = 0.02 if svc == "cart" else 0.01
        elif "container_memory" in query:
            v = 80e6
        elif "container_restarts" in query:
            v = 0.0
        elif "redis_keyspace_hits" in query:
            v = 0.9
        elif query.startswith("sum(rate(redis") or "redis" in query:
            v = 5.0
        elif "histogram_quantile" in query:
            v = 900.0 if faulty and svc in ("checkout", "cart") else 20.0
        elif "STATUS_CODE_ERROR" in query:
            v = 0.0
        elif "traces_span_metrics_calls_total" in query:
            v = 2.0
        else:
            return []
        return [{"metric": {}, "value": [at, str(v)]}]

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = urlparse(str(request.url))
        params = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.port == 9090:
            if not self.prometheus_up:
                raise httpx.ConnectError("refused", request=request)
            query = params.get("query", "")
            self.queries.append(query)
            if url.path.endswith("/query_range"):
                start, end = float(params["start"]), float(params["end"])
                points = [[start + i * 60, "900" if start + i * 60 > self.fault_at else "20"] for i in range(int((end - start) // 60) + 1)]
                return httpx.Response(200, json={"status": "success", "data": {"result": [{"metric": {}, "values": points}]}})
            if url.path.endswith("/label/__name__/values"):
                return httpx.Response(200, json={"status": "success", "data": []})
            rows = self.prom_value(query, float(params.get("time", T0)))
            return httpx.Response(200, json={"status": "success", "data": {"resultType": "vector", "result": rows}})
        if url.port == 16686:
            if url.path.endswith("/api/services"):
                return httpx.Response(200, json={"data": ["checkout", "cart"]})
            start = int(params.get("start", 0)) / 1e6
            return httpx.Response(200, json={"data": [trace("a1b2c3", slow=start >= self.fault_at - 61)]})
        if url.port == 9200:
            return httpx.Response(200, json={"hits": {"total": {"value": 0}, "hits": []}, "aggregations": {}})
        return httpx.Response(404)


def trace(trace_id: str, slow: bool) -> dict[str, Any]:
    cart_ms = 800_000 if slow else 2_000
    return {
        "traceID": trace_id,
        "processes": {"p1": {"serviceName": "checkout"}, "p2": {"serviceName": "cart"}, "p3": {"serviceName": "valkey-cart"}},
        "spans": [
            {"spanID": "s1", "processID": "p1", "operationName": "PlaceOrder", "startTime": 0, "duration": cart_ms + 10_000, "references": [], "tags": []},
            {"spanID": "s2", "processID": "p2", "operationName": "GetCart", "startTime": 1_000, "duration": cart_ms,
             "references": [{"refType": "CHILD_OF", "spanID": "s1"}], "tags": []},
            {"spanID": "s3", "processID": "p3", "operationName": "HGET", "startTime": 2_000, "duration": 500,
             "references": [{"refType": "CHILD_OF", "spanID": "s2"}], "tags": []},
        ],
    }


class FakeDocker(DockerInspector):
    def __init__(self, configs: dict[str, dict[str, Any]] | None = None, events: list[dict] | None = None):
        super().__init__("unix:///nonexistent")
        self.configs = configs or {"cart": {"image": "cart:1", "cpu_limit_cpus": 0.05, "memory_limit_mb": 160,
                                            "restart_policy": "unless-stopped", "status": "running", "restart_count": 0}}
        self.event_rows = events or []

    def snapshot(self, name: str) -> dict[str, Any]:
        from sre_triage.telemetry import TelemetryUnavailable

        if name not in self.configs:
            raise TelemetryUnavailable(f"Container {name} unavailable")
        return dict(self.configs[name])

    def events(self, start, end, names=None, limit=60):
        return [e for e in self.event_rows if start <= e["time"] <= end and (not names or e["container"] in names)]


def fake_backends(world: World | None = None, docker: FakeDocker | None = None) -> Backends:
    world = world or World()
    http = httpx.AsyncClient(transport=httpx.MockTransport(world.handler))
    return Backends(Prometheus("http://prom:9090", http), Jaeger("http://jaeger:16686/jaeger/ui", http),
                    Logs("http://os:9200", http), docker or FakeDocker(), http)


# --------------------------------------------------------------------------- scripted model
Script = Callable[[str, LlmRequest, int], list[types.Part]]


class ScriptedModel(BaseLlm):
    """Deterministic stand-in for an LLM. Each agent role follows a script that can read
    tool results from the request, so real ADK tool execution and validation are exercised."""

    role: str
    _script: Script = PrivateAttr()
    _turn: int = PrivateAttr(default=0)

    def __init__(self, role: str, script: Script):
        super().__init__(model="scripted", role=role)
        self._script = script

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        parts = self._script(self.role, llm_request, self._turn)
        self._turn += 1
        yield LlmResponse(content=types.Content(role="model", parts=parts),
                          usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=100, candidates_token_count=20))


def call(name: str, **args: Any) -> types.Part:
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


def tool_results(request: LlmRequest) -> list[dict[str, Any]]:
    out = []
    for content in request.contents or []:
        for part in content.parts or []:
            if part.function_response is not None:
                out.append({"name": part.function_response.name, **(part.function_response.response or {})})
    return out


def system_text(request: LlmRequest) -> str:
    instruction = request.config.system_instruction if request.config else None
    if isinstance(instruction, str):
        return instruction
    if instruction is None:
        return ""
    return "".join(p.text or "" for p in getattr(instruction, "parts", []) or [])


def ledger_ids(request: LlmRequest, tool: str | None = None, status: str = "ok") -> list[str]:
    ids = []
    for line in system_text(request).splitlines():
        m = re.match(r"(ev_[0-9a-f]+): \S+ (\w+)\(.*\) -> (\w+)$", line)
        if m and (tool is None or m.group(2) == tool) and m.group(3) == status:
            ids.append(m.group(1))
    return ids


def model_factory(script: Script) -> Callable[[str], ScriptedModel]:
    return lambda role: ScriptedModel(role, script)


def dumps(obj: Any) -> str:
    return json.dumps(obj)

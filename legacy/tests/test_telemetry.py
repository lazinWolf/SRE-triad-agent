"""Contract and safety checks for the read-only telemetry adapter."""

import math
import time

import httpx
import pytest

from incident_agent.telemetry import MAX_RECORDS, TelemetryGateway


SERVICES = {"checkout": "http://127.0.0.1:8101", "inventory": "http://127.0.0.1:8102"}


@pytest.mark.asyncio
async def test_queries_capture_provenance_and_explicit_coverage_limits():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "service": "inventory",
                "sample_count": 3,
                "error_rate": 0.0,
                "p95_latency_ms": 40,
                "p95_queue_wait_ms": 10,
                "worker_slots": 1,
                "metadata": {"retention_seconds": 3600, "window_complete": False},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = TelemetryGateway(SERVICES, client=client)
        result = await gateway.query_metrics("inventory", 100, 200)

    assert result["status"] == "ok"
    assert result["service"] == "inventory"
    assert result["source"] == "http://127.0.0.1:8102/telemetry/metrics"
    assert result["query"] == {"start": 100.0, "end": 200.0}
    assert math.isfinite(result["collected_at"])
    assert result["data"]["worker_slots"] == 1
    assert requests[0].url.params == httpx.QueryParams({"start": 100.0, "end": 200.0})
    assert any("incomplete coverage" in text for text in result["limitations"])
    assert any("sampling behavior" in text for text in result["limitations"])


@pytest.mark.asyncio
async def test_all_tools_use_only_fixed_read_only_endpoints_and_enforce_limits():
    requests = []

    def handler(request):
        requests.append(request)
        endpoint = request.url.path.rsplit("/", 1)[-1]
        if endpoint in {"traces", "logs", "changes"}:
            return httpx.Response(200, json={endpoint: [{"timestamp": 110}] * 110})
        if endpoint == "runtime":
            return httpx.Response(
                200, json={"service": "checkout", "current_state": {}, "desired_config": {}}
            )
        return httpx.Response(200, json={"service": "checkout", "dependencies": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = TelemetryGateway(SERVICES, client=client)
        traces = await gateway.find_traces("checkout", 100, 200, limit=2)
        logs = await gateway.search_logs("checkout", 100, 200, limit=3)
        changes = await gateway.get_change_history("checkout", 100, 200)
        runtime = await gateway.inspect_runtime("checkout")
        context = await gateway.get_service_context("checkout")

    assert len(traces["data"]["traces"]) == 2
    assert len(logs["data"]["logs"]) == 3
    assert len(changes["data"]["changes"]) == MAX_RECORDS
    assert runtime["status"] == context["status"] == "ok"
    assert all(request.method == "GET" for request in requests)
    assert [request.url.path for request in requests] == [
        "/telemetry/traces",
        "/telemetry/logs",
        "/telemetry/changes",
        "/telemetry/runtime",
        "/telemetry/context",
    ]
    assert all("admin" not in str(request.url) for request in requests)


@pytest.mark.asyncio
async def test_empty_observations_are_missing_and_never_healthy():
    def handler(request):
        if request.url.path.endswith("metrics"):
            return httpx.Response(200, json={"sample_count": 0, "error_rate": 0})
        return httpx.Response(200, json={"logs": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = TelemetryGateway(SERVICES, client=client)
        metrics = await gateway.query_metrics("checkout", 100, 200)
        logs = await gateway.search_logs("checkout", 100, 200)

    assert metrics["status"] == logs["status"] == "missing"
    assert metrics["data"]["sample_count"] == 0
    assert logs["data"]["logs"] == []
    assert any("not health evidence" in text for text in metrics["limitations"])
    assert any("retention bounds" in text for text in logs["limitations"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service", "start", "end", "limit"),
    [
        ("http://evil.example", 100, 200, 20),
        ("../admin", 100, 200, 20),
        ("checkout", float("nan"), 200, 20),
        ("checkout", 100, float("inf"), 20),
        ("checkout", True, 200, 20),
        ("checkout", "100", 200, 20),
        ("checkout", 200, 100, 20),
        ("checkout", 0, 3601, 20),
        ("checkout", 100, 200, 101),
        ("checkout", 100, 200, 0),
        ("checkout", 100, 200, True),
        ("checkout", 100, 200, 2.5),
    ],
)
async def test_invalid_arguments_never_make_network_requests(service, start, end, limit):
    def handler(request):
        pytest.fail("invalid query reached the network")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = TelemetryGateway(SERVICES, client=client)
        with pytest.raises(ValueError):
            await gateway.find_traces(service, start, end, limit)


@pytest.mark.asyncio
async def test_network_failure_is_unavailable_without_exception_details():
    def handler(request):
        raise httpx.ConnectError("private diagnostic and credentials", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await TelemetryGateway(SERVICES, client=client).query_metrics("checkout", 100, 200)

    assert result["status"] == "unavailable"
    assert result["data"] is None
    assert "ConnectError" in result["error"]
    assert "private diagnostic" not in result["error"]


@pytest.mark.asyncio
async def test_redirects_cannot_escape_service_allowlist_even_with_redirect_enabled_client():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "http://other.example/admin/oracle"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        result = await TelemetryGateway(SERVICES, client=client).inspect_runtime("checkout")

    assert len(requests) == 1
    assert result["status"] == "tool_error"
    assert result["data"] is None
    assert "other.example" not in str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_status"),
    [(404, "unsupported"), (405, "unsupported"), (501, "unsupported"),
     (503, "unavailable"), (429, "unavailable"), (403, "tool_error"), (204, "missing")],
)
async def test_source_failures_are_distinct_from_missing_observations(status_code, expected_status):
    def handler(request):
        return httpx.Response(status_code, json={"internal_detail": "must not be forwarded"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await TelemetryGateway(SERVICES, client=client).inspect_runtime("checkout")

    assert result["status"] == expected_status
    assert result["data"] is None
    assert "internal_detail" not in str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [[], {"traces": "not a list"}, {"traces": ["not an object"]},
     {"traces": [], "service": "inventory"}, {"traces": [], "metadata": []}],
)
async def test_malformed_data_cannot_become_evidence(body):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    ) as client:
        result = await TelemetryGateway(SERVICES, client=client).find_traces("checkout", 100, 200)

    assert result["status"] == "tool_error"
    assert result["data"] is None


@pytest.mark.parametrize(
    "service_urls",
    [{}, {"bad name": "http://localhost"}, {"checkout": "file:///tmp/oracle"},
     {"checkout": "http://user:password@localhost"}, {"checkout": "http://localhost/admin"},
     {"checkout": "http://localhost?token=secret"}, {"checkout": "http://localhost#fragment"}],
)
def test_service_configuration_requires_plain_http_origins(service_urls):
    with pytest.raises(ValueError):
        TelemetryGateway(service_urls)


@pytest.mark.asyncio
async def test_gateway_does_not_close_injected_client():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"service": "checkout", "dependencies": []})
        )
    ) as client:
        async with TelemetryGateway(SERVICES, client=client) as gateway:
            await gateway.get_service_context("checkout")
        assert not client.is_closed


@pytest.mark.asyncio
async def test_adapter_reads_actual_demo_evidence_without_diagnostic_shortcuts():
    from demo_app.server import create_inventory_app

    app = create_inventory_app(processing_ms=1)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://inventory"
    ) as client:
        start = time.time()
        response = await client.post("/reserve", json={})
        assert response.status_code == 200
        end = time.time()
        gateway = TelemetryGateway({"inventory": "http://inventory"}, client=client)
        metrics = await gateway.query_metrics("inventory", start, end)
        traces = await gateway.find_traces("inventory", start, end)
        logs = await gateway.search_logs("inventory", start, end)
        changes = await gateway.get_change_history("inventory", start, end)
        runtime = await gateway.inspect_runtime("inventory")
        context = await gateway.get_service_context("inventory")

    assert metrics["status"] == traces["status"] == runtime["status"] == context["status"] == "ok"
    assert metrics["data"]["sample_count"] == 1
    assert traces["data"]["traces"][0]["trace_id"] == response.json()["trace_id"]
    assert logs["status"] in {"ok", "missing"}
    assert changes["status"] == "missing"
    assert runtime["data"]["current_state"]["worker_slots"] == 4
    assert context["data"]["dependencies"] == []

"""Bounded, read-only adapters for the demo application's telemetry APIs.

The gateway knows endpoint paths, not incident answers. Service URLs are supplied
by application configuration; agent-supplied arguments can only select one of
those configured services and a bounded query window.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping
from typing import Any

import httpx


MAX_WINDOW_SECONDS = 3600
MAX_RECORDS = 100
MAX_RESPONSE_BYTES = 2_000_000
_SERVICE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*\Z")


class TelemetryGateway:
    """Read telemetry using configured service origins and fixed endpoint paths.

    Invalid query arguments raise ``ValueError``. Valid requests return JSON
    dictionaries with ``ok``, ``missing``, ``unsupported``, ``unavailable``, or
    ``tool_error`` status. ``missing`` means a source returned no observations;
    it is never an assertion that the service is healthy.

    A supplied client is owned by its caller. Otherwise the gateway owns an
    asynchronous HTTP client and should be used as an async context manager or
    closed with :meth:`aclose`.
    """

    def __init__(
        self,
        service_urls: dict[str, str],
        timeout: float = 5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.timeout = _finite_number(timeout, "timeout")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if not isinstance(service_urls, Mapping) or not service_urls:
            raise ValueError("at least one configured service is required")

        self._service_urls: dict[str, str] = {}
        for service, origin in service_urls.items():
            if not isinstance(service, str) or not _SERVICE_NAME.fullmatch(service):
                raise ValueError("configured service names must be simple identifiers")
            if not isinstance(origin, str):
                raise ValueError("service URL must be an HTTP or HTTPS origin")
            try:
                url = httpx.URL(origin)
            except httpx.InvalidURL as exc:
                raise ValueError("service URL must be an HTTP or HTTPS origin") from exc
            if (
                url.scheme not in {"http", "https"}
                or not url.host
                or url.userinfo
                or url.query
                or url.fragment
                or url.path not in {"", "/"}
            ):
                raise ValueError("service URL must be an HTTP or HTTPS origin")
            self._service_urls[service] = str(url).rstrip("/")

        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=self.timeout, follow_redirects=False, trust_env=False
        )

    async def __aenter__(self) -> TelemetryGateway:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def query_metrics(
        self, service: str, start: float, end: float
    ) -> dict[str, Any]:
        return await self._request(
            service, "metrics", _window(start, end), shape="metrics"
        )

    async def find_traces(
        self, service: str, start: float, end: float, limit: int = 20
    ) -> dict[str, Any]:
        query = {**_window(start, end), "limit": _limit(limit)}
        return await self._request(service, "traces", query, shape="traces")

    async def search_logs(
        self, service: str, start: float, end: float, limit: int = 50
    ) -> dict[str, Any]:
        query = {**_window(start, end), "limit": _limit(limit)}
        return await self._request(service, "logs", query, shape="logs")

    async def inspect_runtime(self, service: str) -> dict[str, Any]:
        return await self._request(service, "runtime", {}, shape="runtime")

    async def get_change_history(
        self, service: str, start: float, end: float
    ) -> dict[str, Any]:
        query = {**_window(start, end), "limit": MAX_RECORDS}
        return await self._request(service, "changes", query, shape="changes")

    async def get_service_context(self, service: str) -> dict[str, Any]:
        return await self._request(service, "context", {}, shape="context")

    async def _request(
        self,
        service: str,
        endpoint: str,
        query: dict[str, Any],
        *,
        shape: str,
    ) -> dict[str, Any]:
        if not isinstance(service, str) or service not in self._service_urls:
            raise ValueError("service must be one of the configured service names")
        source = f"{self._service_urls[service]}/telemetry/{endpoint}"
        result: dict[str, Any] = {
            "status": "ok",
            "service": service,
            "source": source,
            "query": query,
            "collected_at": time.time(),
            "data": None,
            "limitations": [],
        }
        try:
            # Enforce this even when the caller provides a redirect-enabled client.
            response = await self._client.get(
                source, params=query, timeout=self.timeout, follow_redirects=False
            )
        except httpx.RequestError as exc:
            return _failure(
                result,
                "unavailable",
                f"Telemetry request failed ({type(exc).__name__}).",
            )

        result["collected_at"] = time.time()
        if response.status_code in {404, 405, 501}:
            return _failure(
                result, "unsupported", f"Endpoint is unsupported (HTTP {response.status_code})."
            )
        if response.status_code in {408, 429} or response.status_code >= 500:
            return _failure(
                result, "unavailable", f"Telemetry source is unavailable (HTTP {response.status_code})."
            )
        if 300 <= response.status_code < 400:
            return _failure(result, "tool_error", "Telemetry redirects are not allowed.")
        if response.status_code == 204:
            return _failure(result, "missing", "Telemetry source returned no data.")
        if not 200 <= response.status_code < 300:
            return _failure(
                result, "tool_error", f"Telemetry source rejected the request (HTTP {response.status_code})."
            )
        if len(response.content) > MAX_RESPONSE_BYTES:
            return _failure(result, "tool_error", "Telemetry response exceeds the size limit.")

        try:
            body = response.json()
        except ValueError:
            return _failure(result, "tool_error", "Telemetry response is not valid JSON.")
        issue = _shape_issue(body, shape, service)
        if issue:
            return _failure(result, "tool_error", issue)

        limitations = _limitations(body, query, shape)
        if shape in {"traces", "logs", "changes"}:
            records = body[shape]
            cap = query["limit"]
            if len(records) > cap:
                body = {**body, shape: records[:cap]}
                limitations.append(
                    f"Source returned more than the requested limit; output was capped at {cap} records."
                )
            if not body[shape]:
                result["status"] = "missing"
                limitations.append(
                    "No matching records were returned; this does not establish that the service is healthy."
                )
        elif shape == "metrics" and body["sample_count"] == 0:
            result["status"] = "missing"
            limitations.append(
                "No metric observations were returned for this window; summary values are not health evidence."
            )

        result["data"] = body
        result["limitations"] = limitations
        return result


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _window(start: Any, end: Any) -> dict[str, float]:
    first = _finite_number(start, "start")
    last = _finite_number(end, "end")
    if last < first:
        raise ValueError("end must be greater than or equal to start")
    if last - first > MAX_WINDOW_SECONDS:
        raise ValueError(f"query window must not exceed {MAX_WINDOW_SECONDS} seconds")
    return {"start": first, "end": last}


def _limit(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_RECORDS:
        raise ValueError(f"limit must be an integer between 1 and {MAX_RECORDS}")
    return value


def _failure(result: dict[str, Any], status: str, error: str) -> dict[str, Any]:
    result.update(status=status, error=error)
    result["limitations"] = ["This tool result contains no usable telemetry evidence."]
    return result


def _shape_issue(body: Any, shape: str, service: str) -> str | None:
    if not isinstance(body, dict):
        return "Telemetry response must be a JSON object."
    if "service" in body and body["service"] != service:
        return "Telemetry response service does not match the requested service."
    if "metadata" in body and not isinstance(body["metadata"], dict):
        return "Telemetry metadata must be a JSON object."
    if shape in {"traces", "logs", "changes"}:
        if not isinstance(body.get(shape), list) or any(
            not isinstance(record, dict) for record in body.get(shape, [])
        ):
            return f"Telemetry response must contain a {shape} array of objects."
    elif shape == "metrics":
        count = body.get("sample_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return "Metrics response must contain a nonnegative integer sample_count."
        for key in ("error_rate", "p95_latency_ms", "p95_queue_wait_ms"):
            value = body.get(key)
            if value is not None:
                try:
                    _finite_number(value, key)
                except ValueError:
                    return f"Metrics response {key} must be finite or null."
    elif shape == "runtime":
        if not isinstance(body.get("current_state"), dict) or not isinstance(
            body.get("desired_config"), dict
        ):
            return "Runtime response must contain current_state and desired_config objects."
    elif shape == "context":
        if not isinstance(body.get("dependencies"), list):
            return "Context response must contain a dependencies array."
    return None


def _limitations(body: dict[str, Any], query: dict[str, Any], shape: str) -> list[str]:
    metadata = body.get("metadata", {})
    limitations: list[str] = []
    if "start" in query:
        retention = metadata.get("retention_seconds")
        if retention is None:
            limitations.append("The source does not report telemetry retention bounds.")
        else:
            limitations.append(f"Source retention is bounded to {retention} seconds.")
        if metadata.get("window_complete") is False:
            limitations.append("The source reports incomplete coverage of the requested time window.")
        if metadata.get("window_complete") is None:
            limitations.append("The source does not report whether this time window is complete.")
        if "sampling" not in metadata and "sample_rate" not in metadata:
            limitations.append(
                "The source does not report sampling behavior; missing observations do not prove absent events."
            )
        else:
            sampling = metadata.get("sampling", metadata.get("sample_rate"))
            limitations.append(f"Source sampling metadata: {sampling}.")
        if metadata.get("truncated") is True or metadata.get("has_more") is True:
            limitations.append("The source reports that additional matching records were omitted.")
    elif shape in {"runtime", "context"}:
        limitations.append("This is a current snapshot and does not establish historical state.")
    return limitations

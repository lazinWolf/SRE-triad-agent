"""Case-bound, read-only agent tools with a shared budget and durable evidence.

Every call is validated against the case's time frame, charged to the run's
shared budget, recorded in the evidence ledger, and returned with an
``evidence_id`` that reports must cite. Missing data is reported as such and
never as health. No tool here can change the environment.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx

from . import analysis
from .config import APP_SERVICES, DATA_STORES, Settings
from .ledger import Ledger
from .telemetry import (
    CLIENT_KIND, SERVER_KIND, DockerInspector, Jaeger, Logs, Prometheus, TelemetryUnavailable,
    cache_queries, container_queries, dedupe_log_entries, operation_queries, red_queries, summarize_trace,
)

# Static call graph of the OpenTelemetry Demo (core deployment). Every service also reads flagd.
DEPENDENCIES: dict[str, list[str]] = {
    "frontend-proxy": ["frontend", "image-provider"],
    "frontend": ["ad", "cart", "checkout", "currency", "product-catalog", "recommendation", "shipping"],
    "checkout": ["cart", "currency", "email", "payment", "product-catalog", "shipping"],
    "cart": ["valkey-cart"],
    "recommendation": ["product-catalog"],
    "product-catalog": ["astronomy-db"],
    "shipping": ["quote"],
    "ad": [], "currency": [], "email": [], "payment": [], "quote": [], "image-provider": [],
    "valkey-cart": [], "astronomy-db": [], "flagd": [],
}
KNOWN = set(APP_SERVICES) | set(DATA_STORES)
_RELATIVE_SPAN = re.compile(r"t?\s*([+-]?\d+(?:\.\d+)?)\s*s?\s*(?:\.\.|to|,)\s*t?\s*([+-]?\d+(?:\.\d+)?)\s*s?")


@dataclass
class Backends:
    prometheus: Prometheus
    jaeger: Jaeger
    logs: Logs
    docker: DockerInspector
    http: httpx.AsyncClient | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> "Backends":
        http = httpx.AsyncClient(timeout=settings.http_timeout, trust_env=False, follow_redirects=False)
        return cls(Prometheus(settings.prometheus_url, http), Jaeger(settings.jaeger_url, http),
                   Logs(settings.opensearch_url, http), DockerInspector(settings.docker_base_url), http)

    async def aclose(self) -> None:
        if self.http is not None:
            await self.http.aclose()


@dataclass
class RunBudget:
    max_tool_calls: int
    tool_calls: int = 0
    cached_calls: int = 0
    by_agent: dict[str, int] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class InvalidQuery(ValueError):
    pass


class CaseToolkit:
    def __init__(self, ledger: Ledger, case: dict[str, Any], run_id: str, backends: Backends,
                 budget: RunBudget, *, change_lookback_s: float = 3600, use_memory: bool = True,
                 memory_search: Callable[[str, str], list[dict]] | None = None):
        self.ledger, self.case, self.run_id, self.b, self.budget = ledger, case, run_id, backends, budget
        self.t0 = case["incident_start"]
        self.frames = {
            "baseline": (case["baseline_start"], case["baseline_end"]),
            "incident": (case["incident_start"], case["incident_end"]),
            "full": (case["baseline_start"], case["incident_end"]),
        }
        self.change_lookback_s = change_lookback_s
        self.use_memory = use_memory
        self.memory_search = memory_search
        self._cache: dict[str, dict[str, Any]] = {}
        self._inflight: dict[str, asyncio.Future] = {}

    # ---- time frames ---------------------------------------------------------
    def rel(self, ts: float | None) -> str | None:
        if ts is None:
            return None
        return f"t{ts - self.t0:+.0f}s"

    def resolve(self, window: str, start_offset_s: float = 0, end_offset_s: float = 0) -> tuple[float, float, str]:
        window = (window or "incident").strip().lower()
        # Accept the relative notation used in prompts, e.g. "t-120s..t+60s" or "custom,t+0s..t+240s".
        span = _RELATIVE_SPAN.search(window)
        if span:
            window, start_offset_s, end_offset_s = "custom", float(span.group(1)), float(span.group(2))
        if window in self.frames:
            start, end = self.frames[window]
            return start, end, window
        if window != "custom":
            raise InvalidQuery("window must be 'incident', 'baseline', 'full', or a relative span such as 't-120s..t+60s'")
        start, end = self.t0 + float(start_offset_s), self.t0 + float(end_offset_s)
        lo, hi = self.frames["full"]
        if not (lo - 1 <= start < end <= hi + 1):
            raise InvalidQuery(
                f"custom window must satisfy {lo - self.t0:+.0f} <= start_offset_s < end_offset_s <= {hi - self.t0:+.0f} "
                "(seconds relative to incident start)")
        if end - start < 30:
            raise InvalidQuery("custom windows must be at least 30 seconds long")
        return start, end, f"custom[{self.rel(start)},{self.rel(end)}]"

    def _service(self, service: str, allow_empty: bool = False) -> str:
        service = (service or "").strip()
        if allow_empty and not service:
            return ""
        if service not in KNOWN:
            raise InvalidQuery(f"Unknown service {service!r}. Known: {', '.join(sorted(KNOWN))}")
        return service

    # ---- execution wrapper ---------------------------------------------------------
    async def run(self, agent: str, tool: str, args: dict[str, Any],
                  body: Callable[[], Awaitable[tuple[str, Any, list[dict[str, str]]]]]) -> dict[str, Any]:
        key = json.dumps([tool, args], sort_keys=True, default=str)
        async with self.budget.lock:
            if key in self._cache:
                self.budget.cached_calls += 1
                return {**self._cache[key], "cached": True}
            pending = self._inflight.get(key)
            if pending is None:
                if self.budget.tool_calls >= self.budget.max_tool_calls:
                    return {"status": "budget_exhausted",
                            "message": "The shared tool budget for this investigation is spent. Report with the evidence you have."}
                self.budget.tool_calls += 1
                self.budget.by_agent[agent] = self.budget.by_agent.get(agent, 0) + 1
                pending = asyncio.get_running_loop().create_future()
                self._inflight[key] = pending
                owner = True
            else:
                owner = False
                self.budget.cached_calls += 1
        if not owner:
            # Same query already running for another agent: share its evidence.
            return {**(await asyncio.shield(pending)), "cached": True}
        try:
            self.ledger.event(self.case["id"], self.run_id, agent, "tool_call", f"{tool}({_brief(args)})", {"args": args})
            try:
                status, data, links = await body()
            except InvalidQuery as exc:
                status, data, links = "invalid", {"error": str(exc)}, []
            except TelemetryUnavailable as exc:
                status, data, links = "unavailable", {
                    "error": str(exc), "note": "The source could not answer; this says nothing about service health."}, []
            except Exception as exc:  # a tool bug must not crash the investigation
                status, data, links = "error", {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}, []
            evidence_id = self.ledger.add_evidence(self.case["id"], self.run_id, agent, tool, args, status, data, links)
            response = {"evidence_id": evidence_id, "status": status, "data": data}
            async with self.budget.lock:
                self._cache[key] = response
            pending.set_result(response)
            return response
        except BaseException as exc:
            if not pending.done():
                pending.set_exception(exc if isinstance(exc, Exception) else RuntimeError("tool call cancelled"))
                pending.exception()  # mark retrieved so an unawaited future does not warn
            raise
        finally:
            async with self.budget.lock:
                self._inflight.pop(key, None)

    # ---- metrics ----------------------------------------------------------------------
    async def _red(self, service: str, start: float, end: float, kind: str = SERVER_KIND) -> dict[str, float | None]:
        queries = red_queries(service, end - start, kind)
        values = await asyncio.gather(*(self.b.prometheus.scalar(q, end) for q in queries.values()))
        out = dict(zip(queries, values))
        if out.get("p95_latency_ms") is not None:
            out["p95_latency_ms"] = round(out["p95_latency_ms"], 2)
        return out

    async def service_overview(self, agent: str, window: str = "incident") -> dict[str, Any]:
        args = {"window": window}

        async def body():
            start, end, label = self.resolve(window)
            b_start, b_end = self.frames["baseline"]

            async def grouped(start: float, end: float) -> dict[str, dict[str, float]]:
                w = f"{max(30, int(end - start))}s"
                exprs = {
                    "request_rate_per_s": f'sum by (service_name) (rate(traces_span_metrics_calls_total{{span_kind="{SERVER_KIND}"}}[{w}]))',
                    "errors_per_s": f'sum by (service_name) (rate(traces_span_metrics_calls_total{{span_kind="{SERVER_KIND}",status_code="STATUS_CODE_ERROR"}}[{w}]))',
                    "p95_latency_ms": f'histogram_quantile(0.95, sum by (service_name, le) (rate(traces_span_metrics_duration_milliseconds_bucket{{span_kind="{SERVER_KIND}"}}[{w}])))',
                }
                results = await asyncio.gather(*(self.b.prometheus.instant(e, end) for e in exprs.values()))
                table: dict[str, dict[str, float]] = {}
                for name, rows in zip(exprs, results):
                    for row in rows:
                        svc = row["labels"].get("service_name")
                        if svc:
                            table.setdefault(svc, {})[name] = row["value"]
                for row in table.values():
                    rate = row.get("request_rate_per_s")
                    row["error_ratio"] = (row.get("errors_per_s", 0.0) / rate) if rate else None
                    row.pop("errors_per_s", None)
                return table

            current, baseline = await asyncio.gather(grouped(start, end), grouped(b_start, b_end))
            if not current:
                return "empty", {"window": label, "note": "No span metrics in this window."}, []
            ranked = []
            for svc, cur in current.items():
                comparison = analysis.compare_signals(baseline.get(svc, {}), cur) if label != "baseline" else {}
                worse = [k for k, v in comparison.items() if v.get("change") == "worse"]
                ranked.append({"service": svc, **{k: _round(v) for k, v in cur.items()}, "worse_than_baseline": worse,
                               "p95_ratio_vs_baseline": comparison.get("p95_latency_ms", {}).get("ratio")})
            ranked.sort(key=lambda r: (-len(r["worse_than_baseline"]), -(_num(r["p95_ratio_vs_baseline"]))))
            return "ok", {"window": label, "services": ranked[:16],
                          "note": "Server-side RED metrics from span metrics; compared with the case baseline."}, []

        return await self.run(agent, "service_overview", args, body)

    async def service_red(self, agent: str, service: str, window: str = "incident",
                          start_offset_s: float = 0, end_offset_s: float = 0) -> dict[str, Any]:
        args = {"service": service, "window": window, "start_offset_s": start_offset_s, "end_offset_s": end_offset_s}

        async def body():
            svc = self._service(service)
            start, end, label = self.resolve(window, start_offset_s, end_offset_s)
            b_start, b_end = self.frames["baseline"]
            current, baseline = await asyncio.gather(self._red(svc, start, end), self._red(svc, b_start, b_end))
            p95_expr = red_queries(svc, 60)["p95_latency_ms"]
            trend = await self.b.prometheus.series(p95_expr, start, end, points=10)
            points = [(self.rel(t), round(v, 1)) for t, v in (trend[0]["values"] if trend else [])]
            if all(v is None for v in current.values()):
                return "empty", {"service": svc, "window": label,
                                 "note": "No server spans recorded for this service in the window."}, []
            data = {"service": svc, "window": label, "current": {k: _round(v) for k, v in current.items()},
                    "p95_trend_1m": points}
            if label != "baseline":
                data["vs_baseline"] = analysis.compare_signals(baseline, current)
            links = [{"label": f"{svc} p95 latency", "url": self.b.prometheus.graph_link(p95_expr, b_start, end)}]
            return "ok", data, links

        return await self.run(agent, "service_red", args, body)

    async def operation_latency(self, agent: str, service: str, span_kind: str = "server",
                                window: str = "incident") -> dict[str, Any]:
        args = {"service": service, "span_kind": span_kind, "window": window}

        async def body():
            svc = self._service(service)
            kind = {"server": SERVER_KIND, "client": CLIENT_KIND}.get(span_kind.lower())
            if kind is None:
                raise InvalidQuery("span_kind must be 'server' (handled requests) or 'client' (outgoing calls)")
            start, end, label = self.resolve(window)
            b_start, b_end = self.frames["baseline"]

            async def table(start: float, end: float) -> dict[str, dict[str, float]]:
                queries = operation_queries(svc, end - start, kind)
                results = await asyncio.gather(*(self.b.prometheus.instant(q, end) for q in queries.values()))
                out: dict[str, dict[str, float]] = {}
                for name, rows in zip(queries, results):
                    for row in rows:
                        out.setdefault(row["labels"].get("span_name", "?"), {})[name] = row["value"]
                return out

            current, baseline = await asyncio.gather(table(start, end), table(b_start, b_end))
            if not current:
                return "empty", {"service": svc, "span_kind": span_kind, "window": label,
                                 "note": "No spans of this kind for the service in the window."}, []
            rows = []
            for op, cur in current.items():
                base = baseline.get(op, {})
                rows.append({"operation": op, "rate_per_s": _round(cur.get("rate")),
                             "error_ratio": _round((cur.get("errors", 0) / cur["rate"]) if cur.get("rate") else None),
                             "p95_ms": _round(cur.get("p95")), "baseline_p95_ms": _round(base.get("p95")),
                             "p95_change": analysis.compare(base.get("p95"), cur.get("p95"), min_abs_delta=20)["change"]})
            rows.sort(key=lambda r: -(_num(r["p95_ms"])))
            note = ("client spans are calls this service makes to its dependencies; their latency includes the "
                    "dependency and network time") if kind == CLIENT_KIND else "server spans are requests this service handled"
            return "ok", {"service": svc, "span_kind": span_kind, "window": label, "operations": rows[:12], "note": note}, []

        return await self.run(agent, "operation_latency", args, body)

    async def container_resources(self, agent: str, service: str, window: str = "incident") -> dict[str, Any]:
        args = {"service": service, "window": window}

        async def body():
            svc = self._service(service)
            start, end, label = self.resolve(window)
            b_start, b_end = self.frames["baseline"]

            async def values(start: float, end: float) -> dict[str, float | None]:
                queries = container_queries(svc, end - start)
                results = await asyncio.gather(*(self.b.prometheus.scalar(q, end) for q in queries.values()))
                return dict(zip(queries, results))

            current, baseline = await asyncio.gather(values(start, end), values(b_start, b_end))
            if all(v is None for v in current.values()):
                return "empty", {"service": svc, "window": label, "note": "No container metrics for this service."}, []
            data = {"service": svc, "window": label,
                    "current": {k: _round(v) for k, v in current.items()},
                    "vs_baseline": analysis.compare_signals(baseline, current) if label != "baseline" else None,
                    "note": ("CPU is in cores (1.0 = one full CPU); compare with cpu_limit_cpus. cpu_throttled_ratio is the "
                             "share of 100 ms scheduler periods in which the container hit its limit; bursty services can be "
                             "throttled even when their average use is far below the limit.")}
            expr = container_queries(svc, 60)["cpu_throttled_ratio"]
            return "ok", data, [{"label": f"{svc} CPU throttling", "url": self.b.prometheus.graph_link(expr, b_start, end)}]

        return await self.run(agent, "container_resources", args, body)

    async def cache_health(self, agent: str, window: str = "incident") -> dict[str, Any]:
        args = {"window": window}

        async def body():
            start, end, label = self.resolve(window)
            b_start, b_end = self.frames["baseline"]

            async def values(start: float, end: float) -> dict[str, float | None]:
                queries = cache_queries(end - start)
                results = await asyncio.gather(*(self.b.prometheus.scalar(q, end) for q in queries.values()))
                return dict(zip(queries, results))

            current, baseline = await asyncio.gather(values(start, end), values(b_start, b_end))
            if all(v is None for v in current.values()):
                return "empty", {"window": label, "note": "No cache (valkey-cart) metrics in this window."}, []
            return "ok", {"cache": "valkey-cart (used by cart)", "window": label,
                          "current": {k: _round(v) for k, v in current.items()},
                          "vs_baseline": analysis.compare_signals(baseline, current) if label != "baseline" else None,
                          "note": "For cache call latency as seen by cart, use operation_latency(service='cart', span_kind='client')."}, []

        return await self.run(agent, "cache_health", args, body)

    # ---- traces ------------------------------------------------------------------------
    async def find_traces(self, agent: str, service: str, window: str = "incident", min_duration_ms: float = 0,
                          errors_only: bool = False, operation: str = "", limit: int = 10) -> dict[str, Any]:
        args = {"service": service, "window": window, "min_duration_ms": min_duration_ms,
                "errors_only": errors_only, "operation": operation, "limit": limit}

        async def body():
            svc = self._service(service)
            start, end, label = self.resolve(window)
            n = max(1, min(int(limit), 20))
            traces = await self.b.jaeger.search(svc, start, end, limit=n, min_duration_ms=min_duration_ms or None,
                                                operation=operation or None, errors_only=errors_only)
            if not traces:
                return "empty", {"service": svc, "window": label,
                                 "note": "No matching traces were returned; sampling or filters may exclude them."}, []
            summaries = [summarize_trace(t) for t in traces]
            profile: dict[str, float] = {}
            for s in summaries:
                for name, ms in (s.get("self_time_ms_by_service") or {}).items():
                    profile[name] = profile.get(name, 0.0) + ms
            total = sum(profile.values()) or 1.0
            share = {k: round(v / total, 3) for k, v in sorted(profile.items(), key=lambda kv: -kv[1])[:8]}
            compact = [{"trace_id": s["trace_id"], "root": s.get("root"), "duration_ms": s.get("duration_ms"),
                        "top_self_time": dict(list((s.get("self_time_ms_by_service") or {}).items())[:3]),
                        "errors": s.get("errors")[:2]} for s in summaries]
            links = [{"label": f"trace {s['trace_id'][:12]}", "url": self.b.jaeger.trace_link(s["trace_id"])}
                     for s in summaries[:5]]
            return "ok", {"service": svc, "window": label, "matched": len(summaries),
                          "self_time_share_by_service": share, "traces": compact,
                          "note": "self time = span time not covered by child spans; it localizes where time is spent."}, links

        return await self.run(agent, "find_traces", args, body)

    async def trace_detail(self, agent: str, trace_id: str) -> dict[str, Any]:
        args = {"trace_id": trace_id}

        async def body():
            if not trace_id or not all(c in "0123456789abcdefABCDEF" for c in trace_id) or len(trace_id) > 32:
                raise InvalidQuery("trace_id must be a hex trace ID returned by find_traces")
            trace = await self.b.jaeger.trace(trace_id)
            if trace is None:
                return "empty", {"trace_id": trace_id, "note": "Trace not found (it may have aged out)."}, []
            return "ok", summarize_trace(trace, max_errors=8), [{"label": "Jaeger trace", "url": self.b.jaeger.trace_link(trace_id)}]

        return await self.run(agent, "trace_detail", args, body)

    async def compare_trace_profiles(self, agent: str, service: str, operation: str = "") -> dict[str, Any]:
        args = {"service": service, "operation": operation}

        async def body():
            svc = self._service(service)

            async def profile(window: str) -> tuple[dict[str, float], int, float | None]:
                start, end = self.frames[window]
                traces = await self.b.jaeger.search(svc, start, end, limit=20, operation=operation or None)
                totals: dict[str, float] = {}
                durations = []
                for t in traces:
                    s = summarize_trace(t)
                    durations.append(s.get("duration_ms") or 0)
                    for name, ms in (s.get("self_time_ms_by_service") or {}).items():
                        totals[name] = totals.get(name, 0.0) + ms
                n = len(traces)
                avg = {k: round(v / n, 1) for k, v in totals.items()} if n else {}
                median = sorted(durations)[len(durations) // 2] if durations else None
                return avg, n, median

            (inc, n_inc, med_inc), (base, n_base, med_base) = await asyncio.gather(profile("incident"), profile("baseline"))
            if not n_inc:
                return "empty", {"service": svc, "note": "No incident traces found."}, []
            rows = []
            for name in sorted(set(inc) | set(base), key=lambda k: -(inc.get(k, 0) - base.get(k, 0))):
                rows.append({"service": name, "baseline_avg_self_ms": base.get(name), "incident_avg_self_ms": inc.get(name),
                             "change": analysis.compare(base.get(name), inc.get(name), min_abs_delta=10)["change"]})
            return "ok", {"service": svc, "operation": operation or "(all)", "traces_compared": {"baseline": n_base, "incident": n_inc},
                          "median_trace_ms": {"baseline": med_base, "incident": med_inc}, "per_service_self_time": rows[:10],
                          "note": "Average per-trace self time by service, incident vs baseline samples (up to 20 traces each)."}, []

        return await self.run(agent, "compare_trace_profiles", args, body)

    # ---- logs and changes -------------------------------------------------------------------
    async def search_logs(self, agent: str, service: str = "", window: str = "incident", min_severity: str = "WARN",
                          text: str = "", limit: int = 30) -> dict[str, Any]:
        args = {"service": service, "window": window, "min_severity": min_severity, "text": text, "limit": limit}

        async def body():
            svc = self._service(service, allow_empty=True)
            start, end, label = self.resolve(window)
            severity = (min_severity or "").upper()
            if severity not in ("", "ANY", "WARN", "ERROR"):
                raise InvalidQuery("min_severity must be ANY, WARN or ERROR")
            result = await self.b.logs.search(start, end, service=svc or None,
                                              min_severity=None if severity in ("", "ANY") else severity,
                                              text=text[:200] or None, limit=max(1, min(int(limit), 60)))
            if not result["total"]:
                return "empty", {"service": svc or "(all)", "window": label, "min_severity": severity or "ANY",
                                 "note": "No matching log records; absence of logs is not evidence of health."}, []
            return "ok", {"service": svc or "(all)", "window": label, "matched": result["total"],
                          "by_service": result["by_service"], "by_severity": result["by_severity"],
                          "messages": dedupe_log_entries(result["entries"])}, []

        return await self.run(agent, "search_logs", args, body)

    async def config_changes(self, agent: str, target: str = "", lookback_s: float = 1800) -> dict[str, Any]:
        args = {"target": target, "lookback_s": lookback_s}

        async def body():
            lookback = max(60.0, min(float(lookback_s), self.change_lookback_s))
            start, end = self.t0 - lookback, self.case["incident_end"]
            rows = self.ledger.changes(start, end)
            if target:
                rows = [r for r in rows if r["target"] == target or r["target"] == f"flag:{target}"]
            if not rows:
                return "empty", {"from": self.rel(start), "to": self.rel(end),
                                 "note": "No recorded configuration or feature-flag changes in this period."}, []
            changes = [{"change_id": r["id"], "at": self.rel(r["ts"]), "target": r["target"], "kind": r["kind"],
                        "before": r["before_state"], "after": r["after_state"], "description": r["description"]}
                       for r in rows[-40:]]
            return "ok", {"from": self.rel(start), "to": self.rel(end), "changes": changes,
                          "note": "Times are relative to incident start (t+0s). A change is a lead, not proof of cause."}, []

        return await self.run(agent, "config_changes", args, body)

    async def container_events(self, agent: str, service: str = "", window: str = "full") -> dict[str, Any]:
        args = {"service": service, "window": window}

        async def body():
            svc = self._service(service, allow_empty=True)
            start, end, label = self.resolve(window)
            rows = await asyncio.to_thread(self.b.docker.events, start, end, [svc] if svc else None)
            interesting = [r for r in rows if r["action"] not in ("exec_create", "exec_start")]
            if not interesting:
                return "empty", {"service": svc or "(all)", "window": label, "note": "No container lifecycle events."}, []
            return "ok", {"service": svc or "(all)", "window": label,
                          "events": [{"at": self.rel(r["time"]), "container": r["container"], "action": r["action"],
                                      "exit_code": r["exit_code"]} for r in interesting]}, []

        return await self.run(agent, "container_events", args, body)

    async def runtime_config(self, agent: str, service: str) -> dict[str, Any]:
        args = {"service": service}

        async def body():
            svc = self._service(service)
            snap = await asyncio.to_thread(self.b.docker.snapshot, svc)
            return "ok", {"service": svc, "observed_at": "NOW (not historical)", "config": snap,
                          "note": "This is the present configuration. Use config_changes for what changed and when."}, []

        return await self.run(agent, "runtime_config", args, body)

    # ---- shared ---------------------------------------------------------------------------------
    async def service_dependencies(self, agent: str, service: str) -> dict[str, Any]:
        args = {"service": service}

        async def body():
            svc = self._service(service)
            callers = sorted(k for k, deps in DEPENDENCIES.items() if svc in deps)
            return "ok", {"service": svc, "calls": DEPENDENCIES.get(svc, []), "called_by": callers,
                          "note": "Static architecture map of the demo; every service also reads feature flags from flagd."}, []

        return await self.run(agent, "service_dependencies", args, body)

    async def search_incident_memory(self, agent: str, query: str, service: str = "") -> dict[str, Any]:
        args = {"query": query, "service": service}

        async def body():
            if not self.use_memory or self.memory_search is None:
                return "empty", {"note": "Incident memory is disabled for this investigation."}, []
            hits = self.memory_search(query, service)
            if not hits:
                return "empty", {"note": "No similar reviewed incidents."}, []
            return "ok", {"incidents": hits,
                          "note": "Past reviewed incidents are leads to verify with current evidence, never proof."}, []

        return await self.run(agent, "search_incident_memory", args, body)

    def stats(self) -> dict[str, Any]:
        return {"tool_calls": self.budget.tool_calls, "cached_calls": self.budget.cached_calls,
                "tool_calls_by_agent": dict(self.budget.by_agent)}


def _round(value: Any) -> Any:
    return round(value, 4) if isinstance(value, float) else value


def _num(value: Any) -> float:
    if value == "inf":
        return 1e9
    return float(value) if isinstance(value, (int, float)) else 0.0


def _brief(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items() if v not in ("", 0, None, False))[:200]


def now() -> float:
    return time.time()

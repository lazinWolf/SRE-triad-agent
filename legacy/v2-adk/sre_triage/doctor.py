"""Environment checks: backends reachable, expected metrics present, model configured."""

from __future__ import annotations

import time

from . import telemetry as t
from .changes import read_flags
from .config import REMEDIABLE_SERVICES, Settings
from .tools import Backends

REQUIRED_METRICS = [t.SPAN_CALLS, t.SPAN_DURATION_BUCKET, t.CPU_USAGE, t.CPU_LIMIT, t.CPU_PERIODS,
                    t.CPU_THROTTLED_PERIODS, t.MEMORY_USAGE, t.REDIS_HITS]


async def run_checks(settings: Settings) -> bool:
    ok = True

    def report(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"[{'ok' if passed else 'FAIL'}] {name}{': ' + detail if detail else ''}")

    backends = Backends.from_settings(settings)
    try:
        try:
            names = set(await backends.prometheus.metric_names())
            missing = [m for m in REQUIRED_METRICS if m not in names]
            report("prometheus metrics", not missing, f"missing {missing}" if missing else f"{len(names)} metrics")
            rows = await backends.prometheus.instant(f"count by (container_name) ({t.CPU_LIMIT})", time.time())
            seen = {r["labels"].get("container_name") for r in rows}
            absent = [s for s in REMEDIABLE_SERVICES if s not in seen]
            report("container metrics coverage", not absent,
                   f"no CPU metrics for {absent}; run `docker restart otel-collector`" if absent else "all app containers")
        except t.TelemetryUnavailable as exc:
            report("prometheus", False, str(exc))
        try:
            services = await backends.jaeger.services()
            report("jaeger", bool(services), f"{len(services)} services")
        except t.TelemetryUnavailable as exc:
            report("jaeger", False, str(exc))
        try:
            result = await backends.logs.search(time.time() - 900, time.time(), limit=1)
            report("opensearch logs", result["total"] > 0, f"{result['total']} app log records in 15 min")
        except t.TelemetryUnavailable as exc:
            report("opensearch logs", False, str(exc))
        try:
            snap = backends.docker.snapshot("cart")
            report("docker engine", snap.get("status") == "running", f"cart cpu limit {snap.get('cpu_limit_cpus')}")
        except t.TelemetryUnavailable as exc:
            report("docker engine", False, str(exc))
        try:
            flags = read_flags(settings.flagd_file)
            on = {k: v["defaultVariant"] for k, v in flags.items() if v["defaultVariant"] != "off"}
            report("feature flags", True, f"{len(flags)} flags; non-off: {on or 'none'}")
        except (OSError, ValueError) as exc:
            report("feature flags", False, str(exc))
        report("model", bool(settings.model), settings.model or "SRE_MODEL is not set (live investigations disabled)")
    finally:
        await backends.aclose()
    return ok

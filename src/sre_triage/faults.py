"""Fault scenarios for demos and evaluation (operator only; agents never see this module)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import BASELINE_CPUS, SERVICES
from .telemetry import Backends, Unavailable, read_flags, write_flag


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    symptom: str
    changes: tuple[tuple[str, str, Any], ...]   # ("cpus", container, value) or ("flag", name, variant)
    service: str | None                         # expected root-cause service (None = healthy control)
    category: str | None
    remediation: dict | None
    decoys: tuple[tuple[str, str, Any], ...] = ()


SCENARIOS = {s.id: s for s in [
    Scenario("cart_cpu_limit", "Cart CPU limit cut to 0.02", "Checkout latency is elevated",
             (("cpus", "cart", 0.02),), "cart", "cpu_limit", {"action": "set_cpu_limit", "service": "cart"}),
    Scenario("cart_cpu_limit_decoys", "Cart CPU limit cut, plus two unrelated changes", "Checkout latency is elevated",
             (("cpus", "cart", 0.02),), "cart", "cpu_limit", {"action": "set_cpu_limit", "service": "cart"},
             decoys=(("flag", "emitRawPii", "on"), ("cpus", "recommendation", 0.75))),
    Scenario("payment_failure", "Payment charges failing (flag paymentFailure 50%)", "Checkout requests are failing",
             (("flag", "paymentFailure", "50%"),), "payment", "dependency_errors",
             {"action": "set_feature_flag", "flag": "paymentFailure"}),
    Scenario("catalog_failure", "Product catalog errors (flag productCatalogFailure)", "Some storefront requests fail",
             (("flag", "productCatalogFailure", "on"),), "product-catalog", "dependency_errors",
             {"action": "set_feature_flag", "flag": "productCatalogFailure"}),
    Scenario("healthy_control", "No fault (control case)", "A user said checkout felt slow",
             (), None, None, None),
]}


def _apply(change: tuple[str, str, Any], backends: Backends) -> None:
    kind, target, value = change
    if kind == "cpus":
        backends.docker.set_cpus(target, float(value))
    else:
        write_flag(backends.flagd_file, target, str(value))


def inject(scenario: Scenario, backends: Backends) -> None:
    for change in scenario.decoys + scenario.changes:
        _apply(change, backends)


def reset(backends: Backends) -> list[str]:
    """Every flag off and every app container back to the standard CPU limit."""
    done = []
    for flag, info in read_flags(backends.flagd_file).items():
        if info["variant"] != "off" and "off" in info["variants"]:
            write_flag(backends.flagd_file, flag, "off")
            done.append(f"{flag} -> off")
    for service in SERVICES:
        try:
            if backends.docker.inspect(service)["cpu_limit"] != BASELINE_CPUS:
                backends.docker.set_cpus(service, BASELINE_CPUS)
                done.append(f"{service} cpus -> {BASELINE_CPUS}")
        except Unavailable:
            continue
    return done


def score(scenario: Scenario, report: dict | None) -> dict[str, bool]:
    report = report or {}
    diagnosed = report.get("status") == "diagnosed"
    if scenario.service is None:
        return {"correct": not diagnosed, "service": not diagnosed, "category": not diagnosed,
                "remediation": not report.get("remediation"), "false_alarm": diagnosed}
    service_ok = diagnosed and report.get("root_cause_service") == scenario.service
    category_ok = diagnosed and report.get("category") == scenario.category
    fix = report.get("remediation") or {}
    fix_ok = all(fix.get(k) == v for k, v in (scenario.remediation or {}).items()) and bool(fix)
    return {"correct": service_ok and category_ok, "service": service_ok, "category": category_ok,
            "remediation": fix_ok, "false_alarm": diagnosed and not service_ok}

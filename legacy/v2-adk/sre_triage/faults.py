"""Controlled fault scenarios and their ground truth. OPERATOR/EVALUATION ONLY.

Nothing in the agent toolkit imports this module (enforced by a test). Faults are
applied through the same mechanisms a real operator would use - container
resource updates and feature flags - so they appear in the configuration audit
like any other change. Decoy changes add plausible but irrelevant leads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .changes import read_flags, write_flag
from .config import REMEDIABLE_SERVICES, get_settings
from .remediation import DockerControl

BASELINE_CPUS = 1.0


@dataclass(frozen=True)
class Step:
    kind: str                 # "cpus" | "flag"
    target: str               # container name or flag key
    value: Any                # cpus number or variant name


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    difficulty: str
    alert_service: str
    symptom: str
    steps: tuple[Step, ...]
    expected_service: str | None
    expected_category: str | None
    expected_remediation: dict[str, Any] | None
    decoys: tuple[Step, ...] = ()
    incident_seconds: float = 180
    notes: str = ""
    also_accept_services: tuple[str, ...] = ()  # equally valid localisations (e.g. a service's own database)
    tags: tuple[str, ...] = field(default_factory=tuple)


SCENARIOS: dict[str, Scenario] = {s.id: s for s in [
    Scenario(
        "cart_cpu_limit", "Cart CPU limit reduced", "medium", "checkout",
        "Checkout latency is elevated", (Step("cpus", "cart", 0.02),),
        "cart", "cpu_limit", {"action": "set_cpu_limit", "service": "cart"},
        notes="The synopsis scenario: a downstream CPU-limit reduction slows checkout; the valkey cache is a plausible red herring."),
    Scenario(
        "cart_cpu_limit_decoys", "Cart CPU limit reduced, with unrelated changes", "hard", "checkout",
        "Checkout latency is elevated", (Step("cpus", "cart", 0.02),),
        "cart", "cpu_limit", {"action": "set_cpu_limit", "service": "cart"},
        decoys=(Step("flag", "emitRawPii", "on"), Step("cpus", "recommendation", 0.75)),
        notes="Two plausible recent changes (a checkout/payment flag and another CPU limit) must be rejected."),
    Scenario(
        "currency_cpu_limit", "Currency CPU limit reduced", "medium", "frontend",
        "Storefront latency is elevated", (Step("cpus", "currency", 0.02),),
        "currency", "cpu_limit", {"action": "set_cpu_limit", "service": "currency"}),
    Scenario(
        "payment_failure", "Payment charge failures", "easy", "checkout",
        "Checkout requests are failing", (Step("flag", "paymentFailure", "50%"),),
        "payment", "dependency_errors", {"action": "set_feature_flag", "flag": "paymentFailure", "variant": "off"}),
    Scenario(
        "payment_unreachable", "Payment service unreachable", "easy", "checkout",
        "Checkout requests are failing", (Step("flag", "paymentUnreachable", "on"),),
        "payment", "dependency_unavailable", {"action": "set_feature_flag", "flag": "paymentUnreachable", "variant": "off"}),
    Scenario(
        "catalog_failure", "Product catalog errors", "easy", "frontend",
        "Some storefront requests are failing", (Step("flag", "productCatalogFailure", "on"),),
        "product-catalog", "dependency_errors", {"action": "set_feature_flag", "flag": "productCatalogFailure", "variant": "off"}),
    Scenario(
        "catalog_lock_contention", "Product catalog lock contention", "medium", "frontend",
        "Storefront latency is elevated", (Step("flag", "productCatalogLockContention", "on"),),
        "product-catalog", "lock_contention",
        {"action": "set_feature_flag", "flag": "productCatalogLockContention", "variant": "off"},
        also_accept_services=("astronomy-db",)),
    Scenario(
        "ad_high_cpu", "Ad service CPU saturation", "medium", "frontend",
        "Storefront latency is elevated", (Step("flag", "adHighCpu", "on"),),
        "ad", "cpu_saturation", {"action": "set_feature_flag", "flag": "adHighCpu", "variant": "off"}),
    Scenario(
        "email_memory_leak", "Email memory leak", "hard", "checkout",
        "Checkout latency spikes intermittently", (Step("flag", "emailMemoryLeak", "1000x"),),
        "email", "memory_exhaustion", {"action": "set_feature_flag", "flag": "emailMemoryLeak", "variant": "off"},
        incident_seconds=300),
    Scenario(
        "healthy_control", "No fault (control)", "control", "checkout",
        "A user reported that checkout felt slow", (),
        None, None, None,
        notes="Nothing is injected. A correct result is an explicit insufficient-evidence report with no remediation."),
]}


def _docker_cpus(container: str, cpus: float) -> None:
    DockerControl(get_settings().docker_base_url).set_cpus(container, cpus)


def apply_step(step: Step, flagd_file: Path) -> dict[str, Any]:
    if step.kind == "cpus":
        _docker_cpus(step.target, float(step.value))
        return {"container": step.target, "cpus": step.value}
    if step.kind == "flag":
        return write_flag(flagd_file, step.target, str(step.value))
    raise ValueError(f"unknown step kind {step.kind}")


def inject(scenario: Scenario, flagd_file: Path, *, decoy_lead_s: float = 45,
           sleep=time.sleep) -> dict[str, Any]:
    """Apply decoys, wait, then the fault. Returns the oracle record (never shown to agents)."""
    applied = []
    if scenario.decoys:
        for step in scenario.decoys:
            applied.append({"decoy": True, **apply_step(step, flagd_file), "at": time.time()})
        sleep(decoy_lead_s)
    injected_at = time.time()
    for step in scenario.steps:
        applied.append({"decoy": False, **apply_step(step, flagd_file), "at": time.time()})
    return {"scenario": scenario.id, "injected_at": injected_at, "applied": applied,
            "expected": {"service": scenario.expected_service, "category": scenario.expected_category,
                         "remediation": scenario.expected_remediation}}


def reset(flagd_file: Path) -> dict[str, Any]:
    """Return every flag to 'off' (or its first variant) and every app container to the CPU baseline."""
    restored_flags = {}
    for flag, info in read_flags(flagd_file).items():
        target = "off" if "off" in info["variants"] else None
        if target and info["defaultVariant"] != target:
            write_flag(flagd_file, flag, target)
            restored_flags[flag] = target
    restored_cpus = {}
    for container in REMEDIABLE_SERVICES:
        try:
            _docker_cpus(container, BASELINE_CPUS)
            restored_cpus[container] = BASELINE_CPUS
        except Exception:
            continue
    return {"flags": restored_flags, "cpus": restored_cpus}


def score(scenario: Scenario, report: dict[str, Any] | None) -> dict[str, Any]:
    report = report or {}
    root = report.get("root_cause") or {}
    diagnosed = report.get("diagnosis_status") == "diagnosed"
    remediation = report.get("remediation") or None
    if scenario.expected_service is None:
        correct = not diagnosed and remediation is None
        return {"diagnosed": diagnosed, "service_correct": not diagnosed, "category_correct": not diagnosed,
                "diagnosis_correct": correct, "remediation_correct": remediation is None,
                "false_diagnosis": diagnosed}
    service_ok = diagnosed and root.get("service") in (scenario.expected_service, *scenario.also_accept_services)
    category_ok = diagnosed and root.get("category") == scenario.expected_category
    expected = scenario.expected_remediation or {}
    remediation_ok = bool(remediation) and all(
        remediation.get(k) == v for k, v in expected.items()) and (
        expected.get("action") != "set_cpu_limit" or (remediation.get("cpus") or 0) >= 0.5)
    return {"diagnosed": diagnosed, "service_correct": service_ok, "category_correct": category_ok,
            "diagnosis_correct": service_ok and category_ok, "remediation_correct": remediation_ok,
            "false_diagnosis": diagnosed and not service_ok}

"""Guardrails and execution for the three allowlisted remediation actions."""

from __future__ import annotations

from typing import Any

from .config import BASELINE_CPUS, SERVICES
from .telemetry import Backends, read_flags, write_flag


def validate(proposal: dict[str, Any] | None, backends: Backends) -> list[str]:
    """Problems with a proposed action; empty means it may be shown to the operator."""
    if not proposal:
        return []
    action = proposal.get("action")
    if action in ("set_cpu_limit", "restart_service") and proposal.get("service") not in SERVICES:
        return [f"{action} needs one of the app services, got {proposal.get('service')!r}"]
    if action == "set_cpu_limit":
        cpus = proposal.get("cpus") or BASELINE_CPUS
        if not 0.25 <= float(cpus) <= 2.0:
            return ["cpus must be between 0.25 and 2.0"]
        return []
    if action == "set_feature_flag":
        flags = read_flags(backends.flagd_file)
        if proposal.get("flag") not in flags:
            return [f"unknown feature flag {proposal.get('flag')!r}; known: {', '.join(flags)}"]
        if proposal.get("variant") != "off":
            return ["remediation may only turn a flag 'off'"]
        return []
    if action == "restart_service":
        return []
    return [f"unknown action {action!r}"]


def describe(p: dict[str, Any]) -> str:
    if p["action"] == "set_cpu_limit":
        return f"Set CPU limit of {p['service']} to {p.get('cpus') or BASELINE_CPUS} CPUs"
    if p["action"] == "set_feature_flag":
        return f"Turn feature flag {p['flag']} off"
    return f"Restart {p['service']}"


def execute(p: dict[str, Any], backends: Backends) -> dict[str, Any]:
    """Apply the action; returns how to undo it."""
    if p["action"] == "set_cpu_limit":
        before = backends.docker.inspect(p["service"])["cpu_limit"]
        backends.docker.set_cpus(p["service"], float(p.get("cpus") or BASELINE_CPUS))
        return {"action": "set_cpu_limit", "service": p["service"], "cpus": before}
    if p["action"] == "set_feature_flag":
        before = write_flag(backends.flagd_file, p["flag"], "off")
        return {"action": "set_feature_flag", "flag": p["flag"], "variant": before}
    backends.docker.restart(p["service"])
    return {}


def rollback(undo: dict[str, Any], backends: Backends) -> None:
    if undo.get("action") == "set_cpu_limit" and undo.get("cpus"):
        backends.docker.set_cpus(undo["service"], float(undo["cpus"]))
    elif undo.get("action") == "set_feature_flag":
        write_flag(backends.flagd_file, undo["flag"], undo["variant"])

"""Guardrails and execution for the allowlisted remediation actions.

Agents only *propose* an action. It is validated here, shown to the operator with its risk, executed after
approval, and undone if the system does not recover.
"""

from __future__ import annotations

import json
from typing import Any

from .config import BASELINE_CPUS, SERVICES, STORES
from .telemetry import Backends, read_flags, write_flag_entry

RISK = {"set_feature_flag": "low", "set_cpu_limit": "low", "set_memory_limit": "low", "unpause_service": "low",
        "start_service": "low", "restart_service": "medium"}  # a restart drops in-flight requests
ON_CONTAINER = ("restart_service", "start_service", "unpause_service")


def _standard(backends: Backends, service: str) -> dict:
    return (backends.standard.get("services") or {}).get(service) or {"cpus": BASELINE_CPUS, "memory_mb": None}


def _flag_off(backends: Backends, flag: str) -> dict:
    """The flag's healthy definition: the pristine one if known, else the current one with the default set to off."""
    pristine = ((backends.standard.get("flags") or {}).get("flags") or {}).get(flag)
    if pristine:
        return pristine
    entry = json.loads(backends.flagd_file.read_text())["flags"][flag]
    return {**{k: v for k, v in entry.items() if k != "targeting"}, "defaultVariant": "off"}


def validate(proposal: dict[str, Any] | None, backends: Backends) -> list[str]:
    """Problems with a proposed action; empty means it may be shown to the operator."""
    if not proposal:
        return []
    action, service = proposal.get("action"), proposal.get("service")
    if action in ON_CONTAINER:
        return [] if service in SERVICES + STORES else [f"{action} needs a known service or data store, got {service!r}"]
    if action in ("set_cpu_limit", "set_memory_limit") and service not in SERVICES:
        return [f"{action} needs one of the app services, got {service!r}"]
    if action == "set_cpu_limit":
        cpus = proposal.get("cpus") or _standard(backends, service)["cpus"] or BASELINE_CPUS
        return [] if 0.25 <= float(cpus) <= 2.0 else ["cpus must be between 0.25 and 2.0"]
    if action == "set_memory_limit":
        standard = _standard(backends, service)["memory_mb"]
        if not standard:
            return [f"no standard memory limit is known for {service}"]
        mb = float(proposal.get("memory_mb") or standard)
        return [] if standard <= mb <= 4 * standard else [f"memory_mb must be between {standard} (standard) and {4 * standard}"]
    if action == "set_feature_flag":
        flags = read_flags(backends.flagd_file)
        if proposal.get("flag") not in flags:
            return [f"unknown feature flag {proposal.get('flag')!r}; known: {', '.join(flags)}"]
        if proposal.get("variant") not in (None, "off"):
            return ["remediation may only turn a flag 'off'"]
        if proposal.get("restart_after") and service not in SERVICES:
            return ["restart_after needs `service`: the app service to restart after the flag is off"]
        return []
    return [f"unknown action {action!r}"]


def risk(p: dict[str, Any]) -> str:
    return "medium" if p.get("restart_after") else RISK.get(p["action"], "high")


def describe(p: dict[str, Any], backends: Backends | None = None) -> str:
    standard = _standard(backends, p["service"]) if backends and p.get("service") else {}
    if p["action"] == "set_cpu_limit":
        return f"Set CPU limit of {p['service']} to {p.get('cpus') or standard.get('cpus') or BASELINE_CPUS} CPUs"
    if p["action"] == "set_memory_limit":
        return (f"Set memory limit of {p['service']} to {p.get('memory_mb') or standard.get('memory_mb')} MB"
                " (and start it if it was killed)")
    if p["action"] == "set_feature_flag":
        return f"Turn feature flag {p['flag']} off" + (f", then restart {p['service']}" if p.get("restart_after") else "")
    return {"restart_service": "Restart", "start_service": "Start", "unpause_service": "Unpause"}[p["action"]] + f" {p['service']}"


def execute(p: dict[str, Any], backends: Backends) -> dict[str, Any]:
    """Apply the action; returns how to undo it ({} when there is nothing to undo)."""
    action, service = p["action"], p.get("service")
    if action == "set_cpu_limit":
        before = backends.docker.inspect(service)["cpu_limit"]
        backends.docker.set_cpus(service, float(p.get("cpus") or _standard(backends, service)["cpus"] or BASELINE_CPUS))
        return {"action": action, "service": service, "cpus": before}
    if action == "set_memory_limit":
        state = backends.docker.inspect(service)
        before = state["memory_limit_mb"]
        backends.docker.set_memory(service, float(p.get("memory_mb") or _standard(backends, service)["memory_mb"]))
        if state["status"] != "running":  # the kernel killed it at the old limit: bring it back under the new one
            backends.docker.start(service)
        return {"action": action, "service": service, "memory_mb": before}
    if action == "set_feature_flag":
        before = write_flag_entry(backends.flagd_file, p["flag"], _flag_off(backends, p["flag"]))
        if p.get("restart_after"):
            backends.docker.restart(service)
        return {"action": action, "flag": p["flag"], "entry": before}
    {"restart_service": backends.docker.restart, "start_service": backends.docker.start,
     "unpause_service": backends.docker.unpause}[action](service)
    return {}


def rollback(undo: dict[str, Any], backends: Backends) -> None:
    if undo.get("action") == "set_cpu_limit" and undo.get("cpus"):
        backends.docker.set_cpus(undo["service"], float(undo["cpus"]))
    elif undo.get("action") == "set_memory_limit" and undo.get("memory_mb"):
        backends.docker.set_memory(undo["service"], float(undo["memory_mb"]))
    elif undo.get("action") == "set_feature_flag":
        write_flag_entry(backends.flagd_file, undo["flag"], undo["entry"])

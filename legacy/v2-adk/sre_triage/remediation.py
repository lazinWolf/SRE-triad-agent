"""Allowlisted remediation: validation, approval-gated execution and rollback.

Agents can only *propose* one of these actions. Execution happens here, after an
operator approves the exact proposal, and the prior state is captured so the
action can be rolled back.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import httpx
from typing import Any

from .changes import read_flags, write_flag
from .config import REMEDIABLE_SERVICES
from .telemetry import DockerInspector

CPU_RANGE = (0.25, 2.0)


class RemediationError(RuntimeError):
    pass


def validate_proposal(proposal: dict[str, Any], flagd_file: Path, *,
                      allowed_flag_variants: dict[str, set[str]] | None = None) -> list[str]:
    """Return problems with a proposal; an empty list means it is executable."""
    errors: list[str] = []
    action = proposal.get("action")
    service = proposal.get("service")
    if action == "set_cpu_limit":
        if service not in REMEDIABLE_SERVICES:
            errors.append(f"set_cpu_limit is not allowed for {service!r}; allowed: {', '.join(REMEDIABLE_SERVICES)}")
        cpus = proposal.get("cpus")
        if not isinstance(cpus, (int, float)) or not CPU_RANGE[0] <= float(cpus) <= CPU_RANGE[1]:
            errors.append(f"cpus must be a number between {CPU_RANGE[0]} and {CPU_RANGE[1]}")
    elif action == "set_feature_flag":
        flag, variant = proposal.get("flag"), proposal.get("variant")
        try:
            flags = read_flags(flagd_file)
        except (OSError, ValueError) as exc:
            return [f"feature flags unreadable: {exc}"]
        if flag not in flags:
            errors.append(f"unknown feature flag {flag!r}")
        elif variant not in flags[flag]["variants"]:
            errors.append(f"flag {flag!r} has no variant {variant!r}; options: {sorted(flags[flag]['variants'])}")
        else:
            allowed = {"off"} | (allowed_flag_variants or {}).get(flag, set())
            if variant not in allowed:
                errors.append(f"variant {variant!r} is neither 'off' nor a previously recorded value of {flag!r}")
    elif action == "restart_service":
        if service not in REMEDIABLE_SERVICES:
            errors.append(f"restart_service is not allowed for {service!r}")
    else:
        errors.append(f"unknown action {action!r}; allowed: set_cpu_limit, set_feature_flag, restart_service")
    return errors


def describe(proposal: dict[str, Any]) -> str:
    action = proposal.get("action")
    if action == "set_cpu_limit":
        return f"Set CPU limit of {proposal.get('service')} to {proposal.get('cpus')} CPUs"
    if action == "set_feature_flag":
        return f"Set feature flag {proposal.get('flag')} to '{proposal.get('variant')}'"
    if action == "restart_service":
        return f"Restart container {proposal.get('service')}"
    return str(proposal)


class DockerControl:
    """The only code that changes containers: CPU limit updates and restarts via the Engine API."""

    def __init__(self, base_url: str = "unix:///var/run/docker.sock"):
        if base_url.startswith("unix://"):
            self.transport = httpx.HTTPTransport(uds=base_url.removeprefix("unix://"))
            self.base = "http://docker"
        else:
            self.transport = httpx.HTTPTransport()
            self.base = base_url.replace("tcp://", "http://").rstrip("/")

    def _post(self, path: str, body: dict | None = None, timeout: float = 60) -> None:
        with httpx.Client(transport=self.transport, timeout=timeout) as client:
            response = client.post(f"{self.base}{path}", json=body)
        if response.status_code >= 400:
            raise RemediationError(f"Docker API {path} failed: HTTP {response.status_code} {response.text[:300]}")

    def set_cpus(self, container: str, cpus: float) -> None:
        self._post(f"/containers/{quote(container)}/update", {"NanoCpus": int(round(float(cpus) * 1e9))})

    def restart(self, container: str) -> None:
        self._post(f"/containers/{quote(container)}/restart?t=10", timeout=90)


class Remediator:
    def __init__(self, inspector: DockerInspector, flagd_file: Path, control: DockerControl | None = None):
        self.inspector = inspector
        self.flagd_file = flagd_file
        self.control = control or DockerControl(inspector.base_url)

    def current_state(self, proposal: dict[str, Any]) -> dict[str, Any]:
        action = proposal["action"]
        if action == "set_cpu_limit":
            return {"service": proposal["service"],
                    "cpus": self.inspector.snapshot(proposal["service"]).get("cpu_limit_cpus")}
        if action == "set_feature_flag":
            flags = read_flags(self.flagd_file)
            return {"flag": proposal["flag"], "variant": flags[proposal["flag"]]["defaultVariant"]}
        return {"service": proposal["service"], **self.inspector.snapshot(proposal["service"])}

    def rollback_plan(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """Deterministic rollback for an approved action, computed before execution."""
        try:
            before = self.current_state(proposal)
        except Exception as exc:
            return {"available": False, "reason": f"current state unreadable: {exc}"}
        action = proposal["action"]
        if action == "set_cpu_limit" and before.get("cpus"):
            return {"available": True, "action": "set_cpu_limit", "service": proposal["service"],
                    "cpus": before["cpus"], "description": f"Restore CPU limit of {proposal['service']} to {before['cpus']}"}
        if action == "set_feature_flag":
            return {"available": True, "action": "set_feature_flag", "flag": proposal["flag"],
                    "variant": before["variant"], "description": f"Restore flag {proposal['flag']} to '{before['variant']}'"}
        if action == "restart_service":
            return {"available": False, "reason": "A restart is not reversible; it only resets process state."}
        return {"available": False, "reason": "No prior CPU limit recorded"}

    def execute(self, proposal: dict[str, Any]) -> dict[str, Any]:
        action = proposal["action"]
        if action == "set_cpu_limit":
            self.control.set_cpus(proposal["service"], float(proposal["cpus"]))
        elif action == "set_feature_flag":
            write_flag(self.flagd_file, proposal["flag"], proposal["variant"])
        elif action == "restart_service":
            self.control.restart(proposal["service"])
        else:
            raise RemediationError(f"unsupported action {action}")
        return self.current_state(proposal)


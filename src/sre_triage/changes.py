"""Configuration audit: feature flags and container settings, recorded as timestamped diffs.

This plays the role of a deployment/config history system. It records every
change it sees, whoever made it (operator, fault injection, remediation), so
agents see a realistic mix of relevant and irrelevant changes.
"""

from __future__ import annotations

import fcntl
import json
import logging
from pathlib import Path
from typing import Any

from .config import APP_SERVICES, DATA_STORES
from .ledger import Ledger
from .telemetry import DockerInspector, TelemetryUnavailable

log = logging.getLogger(__name__)

WATCHED_CONTAINERS = list(APP_SERVICES) + list(DATA_STORES)
CONFIG_FIELDS = ("image", "cpu_limit_cpus", "memory_limit_mb", "restart_policy")


def read_flags(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text())
    return {key: {"defaultVariant": flag.get("defaultVariant"), "variants": flag.get("variants", {}),
                  "state": flag.get("state")} for key, flag in data.get("flags", {}).items()}


def write_flag(path: Path, flag: str, variant: str) -> dict[str, Any]:
    """Set a flag's default variant in the flagd file; returns the previous variant."""
    data = json.loads(path.read_text())
    flags = data.get("flags", {})
    if flag not in flags:
        raise ValueError(f"Unknown feature flag {flag!r}")
    if variant not in flags[flag].get("variants", {}):
        raise ValueError(f"Flag {flag!r} has no variant {variant!r}; options: {sorted(flags[flag]['variants'])}")
    previous = flags[flag].get("defaultVariant")
    flags[flag]["defaultVariant"] = variant
    text = json.dumps(data, indent=2) + "\n"
    # Rewrite in place: flagd watches this path inside a bind-mounted directory.
    with open(path, "r+") as handle:
        handle.seek(0)
        handle.write(text)
        handle.truncate()
    return {"flag": flag, "previous": previous, "current": variant}


class ChangeWatcher:
    def __init__(self, ledger: Ledger, inspector: DockerInspector, flagd_file: Path, state_file: Path):
        self.ledger = ledger
        self.inspector = inspector
        self.flagd_file = flagd_file
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_file.read_text())
        except (OSError, ValueError):
            return {}

    def _save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=1))

    def current(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {"containers": {}, "flags": {}}
        try:
            for name, snap in self.inspector.snapshots(WATCHED_CONTAINERS).items():
                snapshot["containers"][name] = {k: snap.get(k) for k in CONFIG_FIELDS}
        except TelemetryUnavailable as exc:
            log.warning("container snapshot failed: %s", exc)
            snapshot["containers"] = None
        try:
            snapshot["flags"] = {k: v["defaultVariant"] for k, v in read_flags(self.flagd_file).items()}
        except (OSError, ValueError) as exc:
            log.warning("flag snapshot failed: %s", exc)
            snapshot["flags"] = None
        return snapshot

    def poll(self) -> list[str]:
        """Record differences since the last poll. Returns new change IDs.

        A file lock serialises pollers (worker and evaluation runner), and the
        state is re-read inside it, so each change is recorded exactly once.
        """
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                self.state = self._load_state()
                return self._poll_locked()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _poll_locked(self) -> list[str]:
        now = self.current()
        recorded = []
        previous = self.state
        if previous:
            if now["containers"] is not None and previous.get("containers") is not None:
                for name, config in now["containers"].items():
                    before = previous["containers"].get(name)
                    if before is None:
                        continue  # newly observed container; not a configuration change
                    diff_before = {k: before.get(k) for k in CONFIG_FIELDS if before.get(k) != config.get(k)}
                    if diff_before:
                        diff_after = {k: config.get(k) for k in diff_before}
                        desc = "; ".join(f"{k}: {diff_before[k]} -> {diff_after[k]}" for k in diff_before)
                        recorded.append(self.ledger.add_change(name, "container_config", diff_before, diff_after,
                                                               f"Container {name} configuration changed ({desc})"))
            if now["flags"] is not None and previous.get("flags") is not None:
                for flag, variant in now["flags"].items():
                    before = previous["flags"].get(flag)
                    if before is not None and before != variant:
                        recorded.append(self.ledger.add_change(
                            f"flag:{flag}", "feature_flag", {"variant": before}, {"variant": variant},
                            f"Feature flag {flag} default variant changed: {before} -> {variant}"))
        # Keep the last known config of containers that are briefly absent (e.g. restarting).
        containers = dict(previous.get("containers") or {})
        containers.update(now["containers"] or {})
        merged = {
            "containers": containers,
            "flags": now["flags"] if now["flags"] is not None else previous.get("flags"),
        }
        self.state = merged
        self._save_state()
        return recorded

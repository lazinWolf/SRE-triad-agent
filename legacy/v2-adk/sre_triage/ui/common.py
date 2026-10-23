"""Shared UI helpers: ledger access, status badges, time formatting."""

from __future__ import annotations

import time
from functools import lru_cache

import streamlit as st

from sre_triage.config import get_settings
from sre_triage.ledger import Ledger

# Status always renders as icon + label; color is never the only cue.
CASE_STATUS = {
    "open": ("gray", ":material/radio_button_unchecked:", "Open"),
    "investigating": ("blue", ":material/manage_search:", "Investigating"),
    "needs_review": ("violet", ":material/rate_review:", "Needs review"),
    "awaiting_approval": ("orange", ":material/approval:", "Awaiting approval"),
    "remediating": ("blue", ":material/build:", "Remediating"),
    "verifying": ("blue", ":material/monitor_heart:", "Verifying recovery"),
    "recovered": ("green", ":material/check_circle:", "Recovered"),
    "reopened": ("red", ":material/replay:", "Reopened"),
    "closed": ("gray", ":material/inventory_2:", "Closed"),
}
HYPOTHESIS_STATUS = {
    "candidate": ("gray", ":material/help:", "Candidate"),
    "supported": ("green", ":material/verified:", "Supported"),
    "weakened": ("orange", ":material/trending_down:", "Weakened"),
    "rejected": ("red", ":material/cancel:", "Rejected"),
}
EVIDENCE_STATUS = {"ok": "✅ ok", "empty": "◻️ empty", "unavailable": "⚠️ unavailable", "error": "❌ error",
                   "invalid": "🚫 invalid", "budget_exhausted": "⏹ budget"}
MODE_COLORS = {"multi_agent": "#2a78d6", "single_agent": "#eb6834"}  # categorical slots 1 and 2
MODE_LABELS = {"multi_agent": "Multi-agent", "single_agent": "Single-agent"}


@lru_cache(maxsize=1)
def ledger() -> Ledger:
    return Ledger(get_settings().db_path)


def badge(mapping: dict, status: str) -> None:
    color, icon, label = mapping.get(status, ("gray", ":material/circle:", status))
    st.badge(label, icon=icon, color=color)


def status_text(mapping: dict, status: str) -> str:
    color, icon, label = mapping.get(status, ("gray", "", status))
    return f":{color}-badge[{icon} {label}]"


def ago(ts: float | None) -> str:
    if not ts:
        return "-"
    delta = time.time() - ts
    if delta < 90:
        return f"{delta:.0f}s ago"
    if delta < 5400:
        return f"{delta / 60:.0f}m ago"
    if delta < 172800:
        return f"{delta / 3600:.1f}h ago"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def clock(ts: float | None) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "-"


def worker_alive() -> bool:
    beat = get_settings().data_dir / "worker.heartbeat"
    try:
        return time.time() - beat.stat().st_mtime < 30
    except OSError:
        return False

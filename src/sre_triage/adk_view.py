"""Read an ADK session back as demonstrable views: Runner event stream, state changes, ReAct traces and
Pydantic validation outcomes. Pure functions over ``session.events``, used by the dashboard's ADK page."""

from __future__ import annotations

import json
from typing import Any

from .schemas import Critique, Finding, Plan, Report

SCHEMAS = {"Plan (coordinator → submit_plan)": Plan, "Finding (specialists → submit_finding)": Finding,
           "Report (coordinator → submit_report)": Report, "Critique (critic → submit_review / A2A)": Critique}


def _path(event) -> str:
    info = getattr(event, "node_info", None)
    return getattr(info, "path", "") or ""


def _short(path: str) -> str:
    """incident_triage@1/specialists@1/specialists@2/traces@traces-1 -> specialists/traces"""
    parts = [p.split("@")[0] for p in path.split("/") if p]
    out = [p for i, p in enumerate(parts) if i == 0 or p != parts[i - 1]]
    return "/".join(out[1:]) or (out[0] if out else "")


def _text(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def event_rows(events: list) -> list[dict[str, Any]]:
    """One row per Runner event: who produced it, where in the workflow, and what it carried."""
    rows = []
    for e in events:
        kinds = []
        for p in (e.content.parts if e.content else None) or []:
            if p.function_call:
                kinds.append(f"call {p.function_call.name}")
            elif p.function_response:
                kinds.append(f"response {p.function_response.name}")
            elif p.text:
                kinds.append("thought" if p.thought else "text")
        if getattr(e, "output", None) is not None:
            kinds.append("node output")
        if e.actions and e.actions.state_delta:
            kinds.append("state Δ " + ", ".join(e.actions.state_delta))
        rows.append({"time": e.timestamp, "author": e.author, "node": _short(_path(e)),
                     "branch": e.branch or "", "carries": "; ".join(kinds) or "—"})
    return rows


def state_timeline(events: list) -> list[dict[str, Any]]:
    """Which workflow node wrote which session-state keys, in order."""
    return [{"time": e.timestamp, "node": _short(_path(e)) or e.author, "keys": ", ".join(e.actions.state_delta)}
            for e in events if e.actions and e.actions.state_delta]


def react_traces(events: list) -> dict[str, list[dict[str, str]]]:
    """Per agent run: Thought (model text) → Action (tool call) → Observation (tool result)."""
    traces: dict[str, list[dict[str, str]]] = {}
    for e in events:
        if e.author in ("user",) or not e.content:
            continue
        path = _path(e)
        if not path or "@" not in path.split("/")[-1]:
            continue
        key = f"{e.author} · {_short(path)}"
        for p in e.content.parts or []:
            if p.function_call:
                step = {"step": "Action", "detail": f"{p.function_call.name}({_text(dict(p.function_call.args or {}), 200)})"}
            elif p.function_response:
                step = {"step": "Observation", "detail": _text(p.function_response.response or {}, 300)}
            elif p.text and p.text.strip():
                step = {"step": "Thought", "detail": _text(p.text.strip(), 400)}
            else:
                continue
            traces.setdefault(key, []).append(step)
    return {k: v for k, v in traces.items() if any(s["step"] == "Action" for s in v)}


def validations(events: list) -> list[dict[str, Any]]:
    """Every submit_* answer and whether Pydantic/code validation accepted it."""
    rows = []
    for e in events:
        for p in (e.content.parts if e.content else None) or []:
            r = p.function_response
            if r and r.name and r.name.startswith("submit_"):
                body = r.response or {}
                rows.append({"agent": e.author, "tool": r.name, "status": body.get("status", "?"),
                             "errors": "; ".join(body.get("errors") or [])})
    return rows


def a2a_events(events: list) -> list[dict[str, Any]]:
    """Messages exchanged with the remote (A2A) critic."""
    return [{"time": e.timestamp, "text": _text(" ".join(p.text for p in e.content.parts if p.text), 500)}
            for e in events if e.author == "remote_critic" and e.content and any(p.text for p in e.content.parts or [])]

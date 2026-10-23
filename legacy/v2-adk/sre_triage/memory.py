"""Incident memory: operator-reviewed outcomes and seeded historical cases.

Only explicit operator reviews (or curated seeds) become memory. Retrieval is a
transparent keyword/service score so it is easy to inspect why a lead appeared.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .ledger import Ledger
from .models import ReviewInput

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "was", "with", "by", "at", "from", "after"}


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def search_memory(ledger: Ledger, query: str, service: str = "", *, exclude_case_id: str | None = None,
                  limit: int = 3) -> list[dict[str, Any]]:
    wanted = _tokens(query)
    scored = []
    for entry in ledger.memories():
        if exclude_case_id and entry.get("case_id") == exclude_case_id:
            continue
        body = entry["body"]
        haystack = " ".join(str(body.get(k, "")) for k in ("symptom", "confirmed_cause", "notes", "lessons"))
        haystack += f" {entry['title']} {entry['category']} {entry['service']}"
        overlap = len(wanted & _tokens(haystack))
        score = overlap + (3 if service and service in (entry["service"], *body.get("affected_services", [])) else 0)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda pair: -pair[0])
    return [{
        "memory_id": e["id"], "origin": e["origin"], "title": e["title"], "service": e["service"],
        "category": e["category"], "symptom": e["body"].get("symptom"), "confirmed_cause": e["body"].get("confirmed_cause"),
        "outcome": e["body"].get("outcome"), "remediation": e["body"].get("remediation"),
        "lessons": e["body"].get("lessons"), "match_score": score,
    } for score, e in scored[:limit]]


def review_case(ledger: Ledger, case_id: str, review: ReviewInput) -> str:
    """Record the operator's reviewed outcome and retain it as incident memory."""
    case = ledger.case(case_id)
    if case["status"] == "investigating":
        raise ValueError("Wait for the investigation to finish before reviewing it")
    report = case.get("report") or {}
    executed = [a for a in ledger.actions(case_id) if a["status"] == "executed"]
    checks = ledger.recovery_checks(case_id)
    root = report.get("root_cause") or {}
    body = {
        "symptom": case["symptom"], "outcome": review.outcome,
        "confirmed_cause": review.confirmed_cause or "unknown",
        "notes": review.notes, "agent_summary": report.get("summary"),
        "agent_root_cause": root, "remediation": [a["proposal"] for a in executed],
        "recovery": [c["status"] for c in checks],
        "lessons": review.notes, "affected_services": [case["service"]],
    }
    service = review.confirmed_service or root.get("service") or case["service"]
    category = review.confirmed_category or root.get("category") or "other"
    title = review.confirmed_cause[:120] if review.confirmed_cause else f"{case['service']}: {case['symptom'][:100]}"
    memory_id = ledger.add_memory(case_id=case_id, origin="review", service=service, category=category,
                                  title=title, body=body, memory_id=f"mem_{case_id}")
    ledger.set_case(case_id, status="closed")
    ledger.event(case_id, None, "operator", "review", f"Reviewed outcome '{review.outcome}' saved to incident memory")
    return memory_id


def load_seeds(ledger: Ledger, path: Path) -> int:
    seeds = json.loads(path.read_text())
    for seed in seeds:
        ledger.add_memory(case_id=None, origin="seed", service=seed["service"], category=seed["category"],
                          title=seed["title"], body=seed["body"], memory_id=seed["id"])
    return len(seeds)

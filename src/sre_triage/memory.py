"""Long-term incident memory: past incidents are recalled for new ones and every case is written back.

Operator feedback (correct / incorrect) is stored with the incident and changes how much it is trusted.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT

_lock = threading.Lock()
WORD = re.compile(r"[a-z][a-z0-9_-]+")
STOP = {"the", "and", "for", "with", "was", "are", "from", "that", "this", "into", "than", "rose", "high"}


def _words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 2}


class Memory:
    def __init__(self, data_dir: Path):
        self.path = data_dir / "memory.json"

    def all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            seeds = json.loads((PROJECT_ROOT / "seeds" / "incidents.json").read_text())
            self._save(seeds)
        return json.loads(self.path.read_text())

    def _save(self, items: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=2))
        tmp.replace(self.path)

    def recall(self, text: str, services: list[str], k: int = 3) -> list[dict[str, Any]]:
        """Most similar past incidents by word overlap and shared services; wrong diagnoses rank lower."""
        query = _words(text) | set(services)
        scored = []
        for item in self.all():
            doc = _words(" ".join(str(item.get(f, "")) for f in ("title", "symptom", "mechanism", "lessons", "category")))
            doc |= set(item.get("services") or []) | {item.get("root_cause_service") or ""}
            score = len(query & doc) / (len(query) ** 0.5 + 1)
            score += 0.5 * len(set(services) & set(item.get("services") or []))
            if item.get("feedback") == "incorrect":
                score *= 0.3
            elif item.get("feedback") == "correct":
                score *= 1.3
            if score > 0.5:
                scored.append((score, item))
        scored.sort(key=lambda s: -s[0])
        return [{**item, "similarity": round(score, 2)} for score, item in scored[:k]]

    def remember(self, entry: dict[str, Any]) -> None:
        with _lock:
            items = [i for i in self.all() if i["id"] != entry["id"]]
            items.append({**entry, "created": time.time()})
            self._save(items)

    def rate(self, incident_id: str, correct: bool, note: str = "") -> bool:
        with _lock:
            items = self.all()
            for item in items:
                if item["id"] == incident_id:
                    item["feedback"] = "correct" if correct else "incorrect"
                    item["feedback_note"] = note
                    self._save(items)
                    return True
        return False

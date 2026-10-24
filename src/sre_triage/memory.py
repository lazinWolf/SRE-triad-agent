"""Long-term incident memory on ADK's ``SqliteMemoryService`` (full-text search, persisted in SQLite).

Each incident is one memory event: readable text for the agents plus a JSON record for the UI.
The workflow recalls similar incidents at the start of a case, the coordinator can search memory itself
with ADK's ``load_memory`` tool, and every diagnosed case is written back. Operator feedback
(correct / incorrect) is added to the incident's memory text, so later agents see it.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from google.adk.events import Event
from google.adk.memory import SqliteMemoryService
from google.genai import types

from .config import PROJECT_ROOT

APP = "sre_triage"   # must match the Runner's app and user, so load_memory searches the same scope
USER = "operator"
RECORD = "record: "


def incident_text(r: dict[str, Any]) -> str:
    fix = r.get("remediation") or {}
    lines = [f"Past incident {r['id']}: {r.get('title', '')}",
             f"Symptom: {r.get('symptom', '')}",
             f"Root cause: {r.get('root_cause_service')} ({r.get('category')}). {r.get('mechanism', '')}",
             f"Fix: {fix.get('action', 'none')} {fix.get('service') or fix.get('flag') or ''}".rstrip(),
             f"Outcome: {r.get('outcome', '')}. Lessons: {r.get('lessons', '')}"]
    if r.get("feedback"):
        lines.append(f"Operator feedback: diagnosis was {r['feedback'].upper()}. {r.get('feedback_note', '')}")
    return "\n".join(lines)


class Memory:
    def __init__(self, data_dir: Path):
        self.path = data_dir / "memory.sqlite"
        self.service = SqliteMemoryService(self.path)

    # ------------------------------------------------------------------ async API (used by the workflow)
    async def add(self, record: dict[str, Any], author: str = "triage") -> None:
        event = Event(id="incident", author=author, timestamp=record.get("created") or time.time(),
                      content=types.Content(role="model", parts=[
                          types.Part(text=incident_text(record)),
                          types.Part(text=RECORD + json.dumps(record, default=str))]))
        await self.service.add_events_to_memory(app_name=APP, user_id=USER, events=[event], session_id=record["id"])

    async def seed(self) -> None:
        if self.all():
            return
        for record in json.loads((PROJECT_ROOT / "seeds" / "incidents.json").read_text()):
            await self.add(record, author="runbook")

    async def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        await self.seed()
        response = await self.service.search_memory(app_name=APP, user_id=USER, query=query)
        out, seen = [], set()
        for entry in response.memories:
            record = _record(entry.content)
            if record and record["id"] not in seen:
                seen.add(record["id"])
                out.append(record)
        return out[:k]

    # ------------------------------------------------------------------ sync API (CLI / dashboard)
    def all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as db:
            try:
                rows = db.execute("SELECT content_json FROM memory_events WHERE app_name=? AND user_id=? "
                                  "ORDER BY timestamp", (APP, USER)).fetchall()
            except sqlite3.OperationalError:  # not created yet
                return []
        return [r for r in (_record(types.Content.model_validate_json(row[0])) for row in rows) if r]

    def remember(self, record: dict[str, Any]) -> None:
        asyncio.run(self.add({**record, "created": time.time()}))

    def rate(self, incident_id: str, correct: bool, note: str = "") -> bool:
        record = next((r for r in self.all() if r["id"] == incident_id), None)
        if record is None:
            return False
        record.update(feedback="correct" if correct else "incorrect", feedback_note=note)
        asyncio.run(self.add(record, author="operator"))
        return True


def _record(content: types.Content) -> dict[str, Any] | None:
    for part in content.parts or []:
        if part.text and part.text.startswith(RECORD):
            return json.loads(part.text[len(RECORD):])
    return None

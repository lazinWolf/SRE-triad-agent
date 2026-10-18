"""Local durable case ledger and explicitly reviewed incident memory."""

import json
import sqlite3
import time
import uuid
from pathlib import Path

from .models import CaseInput, ReviewInput


class CaseBusy(ValueError):
    pass


class CaseStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, body TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS memory (id TEXT PRIMARY KEY, case_id TEXT NOT NULL, revision INTEGER NOT NULL, body TEXT NOT NULL, UNIQUE(case_id, revision))")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def create(self, item: CaseInput):
        case = {
            **item.model_dump(), "id": uuid.uuid4().hex[:12], "created_at": time.time(),
            "status": "draft", "revision": 0, "run_id": None, "report": None,
            "evidence": [], "events": [], "feedback": [], "error": None,
        }
        with self._connect() as db:
            db.execute("INSERT INTO cases VALUES (?, ?)", (case["id"], json.dumps(case)))
        return case

    def get(self, case_id: str):
        with self._connect() as db:
            row = db.execute("SELECT body FROM cases WHERE id=?", (case_id,)).fetchone()
        if row is None:
            raise KeyError(case_id)
        return json.loads(row[0])

    def list(self):
        with self._connect() as db:
            rows = db.execute("SELECT body FROM cases ORDER BY rowid DESC LIMIT 100").fetchall()
        return [json.loads(row[0]) for row in rows]

    def mutate(self, case_id, change):
        # Serializes read-modify-write updates from parallel tool completions.
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM cases WHERE id=?", (case_id,)).fetchone()
            if row is None:
                raise KeyError(case_id)
            case = json.loads(row[0])
            change(case)
            db.execute("UPDATE cases SET body=? WHERE id=?", (json.dumps(case), case_id))
        return case

    def begin_run(self, case_id):
        def change(case):
            if case["status"] == "investigating":
                raise CaseBusy("This case is already being investigated")
            case.update(status="investigating", report=None, error=None, run_id=uuid.uuid4().hex)
        return self.mutate(case_id, change)

    def add_event(self, case_id, agent, message, kind="progress"):
        event = {"timestamp": time.time(), "agent": agent, "message": message, "kind": kind}
        self.mutate(case_id, lambda case: case["events"].append(event))

    def add_evidence(self, case_id, agent, tool, query, result):
        evidence = {
            "id": "ev_" + uuid.uuid4().hex[:10], "agent": agent, "tool": tool,
            "query": query, "result": result, "collected_at": time.time(),
        }
        def change(case):
            evidence["run_id"] = case["run_id"]
            case["evidence"].append(evidence)
        self.mutate(case_id, change)
        return evidence

    def finish_run(self, case_id, report, error=None):
        return self.mutate(case_id, lambda case: case.update(
            status="needs_review" if report and report["diagnosis_status"] == "supported" else "unresolved",
            report=report, error=error,
        ))

    def feedback(self, case_id, text):
        def change(case):
            if case["status"] == "investigating":
                raise CaseBusy("Wait for this investigation to finish before correcting it")
            case["feedback"].append({"text": text, "timestamp": time.time()})
            case["revision"] += 1
            case.update(status="draft", report=None, error=None)
            case["events"].append({"timestamp": time.time(), "agent": "operator", "message": text, "kind": "feedback"})
        return self.mutate(case_id, change)

    def review(self, case_id, review: ReviewInput):
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM cases WHERE id=?", (case_id,)).fetchone()
            if row is None:
                raise KeyError(case_id)
            case = json.loads(row[0])
            if case["status"] == "investigating":
                raise CaseBusy("Wait for the investigation to finish before reviewing it")
            if case["report"] is None:
                raise ValueError("Investigate this case before recording a reviewed outcome")
            previous = db.execute("SELECT id FROM memory WHERE case_id=? AND revision=?", (case_id, case["revision"])).fetchone()
            entry = {
                **review.model_dump(), "id": previous[0] if previous else "mem_" + uuid.uuid4().hex[:10],
                "case_id": case_id, "revision": case["revision"], "service": case["service"],
                "environment": case["environment"], "summary": case["summary"],
                "reviewed_at": time.time(), "origin_mode": case["mode"],
                "evidence_ids": [e["id"] for e in case["evidence"] if e["run_id"] == case["run_id"]],
            }
            db.execute("INSERT INTO memory VALUES (?,?,?,?) ON CONFLICT(case_id, revision) DO UPDATE SET body=excluded.body", (entry["id"], case_id, case["revision"], json.dumps(entry)))
            case["events"].append({"timestamp": time.time(), "agent": "operator", "message": "Reviewed outcome saved to incident memory", "kind": "review"})
            db.execute("UPDATE cases SET body=? WHERE id=?", (json.dumps(case), case_id))
        return entry

    def memory(self, service=None, exclude_case_id=None, limit=5):
        with self._connect() as db:
            rows = db.execute("SELECT body FROM memory ORDER BY rowid DESC LIMIT 100").fetchall()
        entries = [json.loads(row[0]) for row in rows]
        return [e for e in entries if (service is None or e["service"] == service) and e["case_id"] != exclude_case_id][:limit]

    def interrupt_stale_runs(self):
        for case in self.list():
            if case["status"] == "investigating":
                self.mutate(case["id"], lambda c: c.update(status="unresolved", error="Server restarted during investigation; run again"))

"""SQLite case ledger: cases, runs, evidence, hypotheses, actions, memory, change audit and jobs.

Every write is a short transaction on a fresh connection, so the worker, the UI
and the CLI can share one database file safely (WAL mode).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .models import CaseInput

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
  id TEXT PRIMARY KEY, created_at REAL, updated_at REAL, status TEXT, service TEXT,
  symptom TEXT, severity TEXT, source TEXT, mode TEXT, use_memory INTEGER,
  incident_start REAL, incident_end REAL, baseline_start REAL, baseline_end REAL,
  revision INTEGER DEFAULT 0, run_id TEXT, report TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, case_id TEXT, revision INTEGER, mode TEXT, model TEXT,
  started_at REAL, finished_at REAL, status TEXT, stats TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, run_id TEXT, ts REAL,
  actor TEXT, kind TEXT, message TEXT, data TEXT
);
CREATE INDEX IF NOT EXISTS events_case ON events(case_id, seq);
CREATE TABLE IF NOT EXISTS evidence (
  id TEXT PRIMARY KEY, case_id TEXT, run_id TEXT, agent TEXT, tool TEXT, args TEXT,
  status TEXT, summary TEXT, links TEXT, collected_at REAL
);
CREATE INDEX IF NOT EXISTS evidence_case ON evidence(case_id, run_id);
CREATE TABLE IF NOT EXISTS hypotheses (
  case_id TEXT, run_id TEXT, round INTEGER, body TEXT, recorded_at REAL
);
CREATE TABLE IF NOT EXISTS findings (
  case_id TEXT, run_id TEXT, round INTEGER, agent TEXT, task TEXT, body TEXT, recorded_at REAL
);
CREATE TABLE IF NOT EXISTS feedback (
  id TEXT PRIMARY KEY, case_id TEXT, ts REAL, revision INTEGER, text TEXT
);
CREATE TABLE IF NOT EXISTS actions (
  id TEXT PRIMARY KEY, case_id TEXT, run_id TEXT, proposal TEXT, status TEXT,
  created_at REAL, decided_at REAL, decided_by TEXT, decision_note TEXT,
  executed_at REAL, before_state TEXT, after_state TEXT, rollback TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS recovery_checks (
  id TEXT PRIMARY KEY, case_id TEXT, action_id TEXT, created_at REAL, finished_at REAL,
  status TEXT, result TEXT
);
CREATE TABLE IF NOT EXISTS memory (
  id TEXT PRIMARY KEY, case_id TEXT, created_at REAL, origin TEXT, service TEXT,
  category TEXT, title TEXT, body TEXT
);
CREATE TABLE IF NOT EXISTS changes (
  id TEXT PRIMARY KEY, ts REAL, target TEXT, kind TEXT, before_state TEXT,
  after_state TEXT, description TEXT
);
CREATE INDEX IF NOT EXISTS changes_ts ON changes(ts);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, kind TEXT, case_id TEXT, payload TEXT, status TEXT,
  created_at REAL, started_at REAL, finished_at REAL, error TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
  id TEXT PRIMARY KEY, rule TEXT, service TEXT, fired_at REAL, case_id TEXT, details TEXT
);
"""

_JSON_COLUMNS = {
    "report", "stats", "data", "args", "summary", "links", "body", "proposal",
    "before_state", "after_state", "rollback", "result", "payload", "details",
}


def new_id(prefix: str, size: int = 10) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:size]}"


def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out = dict(row)
    for key, value in out.items():
        if key in _JSON_COLUMNS and isinstance(value, str):
            out[key] = json.loads(value)
    return out


def _encode(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return value


class CaseBusy(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            columns = {row[1] for row in db.execute("PRAGMA table_info(cases)")}
            if "model" not in columns:  # added after the first schema version
                db.execute("ALTER TABLE cases ADD COLUMN model TEXT DEFAULT ''")
        finally:
            db.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def _insert(self, table: str, row: dict[str, Any]) -> None:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self._tx() as db:
            db.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", [_encode(v) for v in row.values()])

    def _update(self, table: str, key: str, row: dict[str, Any]) -> None:
        sets = ", ".join(f"{col}=?" for col in row)
        with self._tx() as db:
            db.execute(f"UPDATE {table} SET {sets} WHERE id=?", [_encode(v) for v in row.values()] + [key])

    def _all(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        with self._tx() as db:
            return [_decode(r) for r in db.execute(sql, args).fetchall()]

    def _one(self, sql: str, args: tuple = ()) -> dict[str, Any] | None:
        with self._tx() as db:
            return _decode(db.execute(sql, args).fetchone())

    # ---- cases -----------------------------------------------------------
    def create_case(self, item: CaseInput) -> dict[str, Any]:
        now = time.time()
        case = {
            "id": new_id("case", 8), "created_at": now, "updated_at": now, "status": "open",
            "service": item.service, "symptom": item.symptom, "severity": item.severity,
            "source": item.source, "mode": item.mode, "use_memory": int(item.use_memory),
            "incident_start": item.incident_start, "incident_end": item.incident_end,
            "baseline_start": item.incident_start - item.baseline_seconds,
            "baseline_end": item.incident_start, "revision": 0, "run_id": None,
            "report": None, "error": None, "model": item.model,
        }
        self._insert("cases", case)
        self.event(case["id"], None, "operator" if item.source == "operator" else "alerting",
                   "case_opened", f"Case opened for {item.service}: {item.symptom}")
        return self.case(case["id"])

    def case(self, case_id: str) -> dict[str, Any]:
        case = self._one("SELECT * FROM cases WHERE id=?", (case_id,))
        if case is None:
            raise KeyError(case_id)
        return case

    def cases(self, limit: int = 200) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM cases ORDER BY created_at DESC LIMIT ?", (limit,))

    def set_case(self, case_id: str, **fields: Any) -> None:
        fields["updated_at"] = time.time()
        self._update("cases", case_id, fields)

    def open_case_for(self, service: str) -> dict[str, Any] | None:
        return self._one(
            "SELECT * FROM cases WHERE service=? AND status NOT IN ('closed','recovered') "
            "ORDER BY created_at DESC LIMIT 1", (service,))

    # ---- runs --------------------------------------------------------------
    def begin_run(self, case_id: str, mode: str, model: str) -> dict[str, Any]:
        run_id = new_id("run", 10)
        with self._tx() as db:
            row = _decode(db.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone())
            if row is None:
                raise KeyError(case_id)
            if row["status"] == "investigating":
                raise CaseBusy("This case is already being investigated")
            db.execute("UPDATE cases SET status='investigating', run_id=?, report=NULL, error=NULL, updated_at=? WHERE id=?",
                       (run_id, time.time(), case_id))
            db.execute("INSERT INTO runs (id, case_id, revision, mode, model, started_at, status) VALUES (?,?,?,?,?,?,?)",
                       (run_id, case_id, row["revision"], mode, model, time.time(), "running"))
        return {"id": run_id, "case_id": case_id, "revision": row["revision"]}

    def finish_run(self, run_id: str, status: str, stats: dict[str, Any], error: str | None = None) -> None:
        self._update("runs", run_id, {"finished_at": time.time(), "status": status, "stats": stats, "error": error})

    def runs(self, case_id: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM runs WHERE case_id=? ORDER BY started_at", (case_id,))

    def run(self, run_id: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM runs WHERE id=?", (run_id,))

    # ---- events / evidence / board ----------------------------------------
    def event(self, case_id: str, run_id: str | None, actor: str, kind: str, message: str,
              data: dict[str, Any] | None = None) -> None:
        self._insert("events", {"case_id": case_id, "run_id": run_id, "ts": time.time(), "actor": actor,
                                "kind": kind, "message": message[:8000], "data": data or {}})

    def events(self, case_id: str, after_seq: int = 0) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM events WHERE case_id=? AND seq>? ORDER BY seq", (case_id, after_seq))

    def add_evidence(self, case_id: str, run_id: str, agent: str, tool: str, args: dict[str, Any],
                     status: str, summary: Any, links: list[dict[str, str]] | None = None) -> str:
        evidence_id = new_id("ev", 8)
        self._insert("evidence", {
            "id": evidence_id, "case_id": case_id, "run_id": run_id, "agent": agent, "tool": tool,
            "args": args, "status": status, "summary": summary, "links": links or [],
            "collected_at": time.time(),
        })
        return evidence_id

    def evidence(self, case_id: str, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id:
            return self._all("SELECT * FROM evidence WHERE case_id=? AND run_id=? ORDER BY collected_at", (case_id, run_id))
        return self._all("SELECT * FROM evidence WHERE case_id=? ORDER BY collected_at", (case_id,))

    def record_hypotheses(self, case_id: str, run_id: str, round_number: int, hypotheses: list[dict]) -> None:
        self._insert("hypotheses", {"case_id": case_id, "run_id": run_id, "round": round_number,
                                    "body": hypotheses, "recorded_at": time.time()})

    def hypothesis_history(self, case_id: str, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id:
            return self._all("SELECT * FROM hypotheses WHERE case_id=? AND run_id=? ORDER BY recorded_at", (case_id, run_id))
        return self._all("SELECT * FROM hypotheses WHERE case_id=? ORDER BY recorded_at", (case_id,))

    def record_finding(self, case_id: str, run_id: str, round_number: int, agent: str, task: str, finding: dict) -> None:
        self._insert("findings", {"case_id": case_id, "run_id": run_id, "round": round_number, "agent": agent,
                                  "task": task, "body": finding, "recorded_at": time.time()})

    def findings(self, case_id: str, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id:
            return self._all("SELECT * FROM findings WHERE case_id=? AND run_id=? ORDER BY recorded_at", (case_id, run_id))
        return self._all("SELECT * FROM findings WHERE case_id=? ORDER BY recorded_at", (case_id,))

    # ---- feedback ------------------------------------------------------------
    def add_feedback(self, case_id: str, text: str) -> dict[str, Any]:
        with self._tx() as db:
            row = _decode(db.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone())
            if row is None:
                raise KeyError(case_id)
            if row["status"] == "investigating":
                raise CaseBusy("Wait for the running investigation to finish before correcting it")
            revision = row["revision"] + 1
            db.execute("INSERT INTO feedback (id, case_id, ts, revision, text) VALUES (?,?,?,?,?)",
                       (new_id("fb", 8), case_id, time.time(), revision, text))
            db.execute("UPDATE cases SET revision=?, status='open', report=NULL, updated_at=? WHERE id=?",
                       (revision, time.time(), case_id))
        self.event(case_id, None, "operator", "feedback", text)
        return self.case(case_id)

    def feedback(self, case_id: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM feedback WHERE case_id=? ORDER BY ts", (case_id,))

    # ---- actions & recovery ------------------------------------------------
    def propose_action(self, case_id: str, run_id: str, proposal: dict, rollback: dict) -> str:
        action_id = new_id("act", 8)
        with self._tx() as db:
            # A new proposal supersedes any undecided one for the case.
            db.execute("UPDATE actions SET status='superseded' WHERE case_id=? AND status='proposed'", (case_id,))
            db.execute("INSERT INTO actions (id, case_id, run_id, proposal, status, created_at, rollback) VALUES (?,?,?,?,?,?,?)",
                       (action_id, case_id, run_id, json.dumps(proposal), "proposed", time.time(), json.dumps(rollback)))
        return action_id

    def action(self, action_id: str) -> dict[str, Any]:
        row = self._one("SELECT * FROM actions WHERE id=?", (action_id,))
        if row is None:
            raise KeyError(action_id)
        return row

    def actions(self, case_id: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM actions WHERE case_id=? ORDER BY created_at", (case_id,))

    def set_action(self, action_id: str, **fields: Any) -> None:
        self._update("actions", action_id, fields)

    def decide_action(self, action_id: str, approved: bool, who: str, note: str = "") -> dict[str, Any]:
        with self._tx() as db:
            row = _decode(db.execute("SELECT * FROM actions WHERE id=?", (action_id,)).fetchone())
            if row is None:
                raise KeyError(action_id)
            if row["status"] != "proposed":
                raise ValueError(f"Action is {row['status']}, not awaiting a decision")
            db.execute("UPDATE actions SET status=?, decided_at=?, decided_by=?, decision_note=? WHERE id=?",
                       ("approved" if approved else "rejected", time.time(), who, note, action_id))
        self.event(row["case_id"], None, who, "action_decision",
                   f"{'Approved' if approved else 'Rejected'} by {who}" + (f": {note}" if note else ""))
        return self.action(action_id)

    def add_recovery_check(self, case_id: str, action_id: str | None) -> str:
        check_id = new_id("rc", 8)
        self._insert("recovery_checks", {"id": check_id, "case_id": case_id, "action_id": action_id,
                                         "created_at": time.time(), "status": "pending", "result": {}})
        return check_id

    def set_recovery_check(self, check_id: str, status: str, result: dict) -> None:
        self._update("recovery_checks", check_id, {"status": status, "result": result, "finished_at": time.time()})

    def recovery_checks(self, case_id: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM recovery_checks WHERE case_id=? ORDER BY created_at", (case_id,))

    # ---- memory --------------------------------------------------------------
    def add_memory(self, *, case_id: str | None, origin: str, service: str, category: str,
                   title: str, body: dict[str, Any], memory_id: str | None = None) -> str:
        memory_id = memory_id or new_id("mem", 8)
        with self._tx() as db:
            db.execute("INSERT OR REPLACE INTO memory (id, case_id, created_at, origin, service, category, title, body) "
                       "VALUES (?,?,?,?,?,?,?,?)",
                       (memory_id, case_id, time.time(), origin, service, category, title, json.dumps(body)))
        return memory_id

    def memories(self) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM memory ORDER BY created_at DESC")

    def delete_memory(self, memory_id: str) -> None:
        with self._tx() as db:
            db.execute("DELETE FROM memory WHERE id=?", (memory_id,))

    # ---- configuration change audit -----------------------------------------
    def add_change(self, target: str, kind: str, before: dict, after: dict, description: str,
                   ts: float | None = None) -> str:
        change_id = new_id("chg", 8)
        self._insert("changes", {"id": change_id, "ts": ts or time.time(), "target": target, "kind": kind,
                                 "before_state": before, "after_state": after, "description": description})
        return change_id

    def changes(self, start: float, end: float, target: str | None = None) -> list[dict[str, Any]]:
        if target:
            return self._all("SELECT * FROM changes WHERE ts BETWEEN ? AND ? AND target=? ORDER BY ts", (start, end, target))
        return self._all("SELECT * FROM changes WHERE ts BETWEEN ? AND ? ORDER BY ts", (start, end))

    def change(self, change_id: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM changes WHERE id=?", (change_id,))

    # ---- jobs ------------------------------------------------------------------
    def enqueue(self, kind: str, case_id: str | None = None, payload: dict | None = None) -> str:
        job_id = new_id("job", 8)
        self._insert("jobs", {"id": job_id, "kind": kind, "case_id": case_id, "payload": payload or {},
                              "status": "queued", "created_at": time.time()})
        return job_id

    def claim_job(self) -> dict[str, Any] | None:
        with self._tx() as db:
            row = _decode(db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone())
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?", (time.time(), row["id"]))
        return row

    def finish_job(self, job_id: str, error: str | None = None) -> None:
        self._update("jobs", job_id, {"status": "failed" if error else "done", "finished_at": time.time(), "error": error})

    def jobs(self, case_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if case_id:
            return self._all("SELECT * FROM jobs WHERE case_id=? ORDER BY created_at DESC LIMIT ?", (case_id, limit))
        return self._all("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))

    def recover_interrupted(self) -> int:
        """Mark work interrupted by a worker restart; nothing silently resumes."""
        with self._tx() as db:
            n = db.execute("UPDATE jobs SET status='failed', error='worker restarted' WHERE status='running'").rowcount
            db.execute("UPDATE runs SET status='interrupted', finished_at=? WHERE status='running'", (time.time(),))
            db.execute("UPDATE cases SET status='open', error='Worker restarted during investigation' WHERE status='investigating'")
            db.execute("UPDATE recovery_checks SET status='error', result='{\"error\": \"worker restarted\"}' WHERE status='pending'")
        return n

    # ---- alerts ------------------------------------------------------------------
    def add_alert(self, rule: str, service: str, case_id: str | None, details: dict) -> str:
        alert_id = new_id("al", 8)
        self._insert("alerts", {"id": alert_id, "rule": rule, "service": service, "fired_at": time.time(),
                                "case_id": case_id, "details": details})
        return alert_id

    def alerts(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM alerts ORDER BY fired_at DESC LIMIT ?", (limit,))

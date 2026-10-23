"""Case records and their live activity log (data/cases/<id>/), shared by the CLI and the dashboard."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_lock = threading.Lock()


class Cases:
    def __init__(self, data_dir: Path):
        self.root = data_dir / "cases"
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, **meta: Any) -> str:
        case_id = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        (self.root / case_id).mkdir()
        self._write(case_id, {"id": case_id, "created": time.time(), "status": "investigating", **meta})
        return case_id

    def _write(self, case_id: str, meta: dict) -> None:
        path = self.root / case_id / "case.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2, default=str))
        tmp.replace(path)

    def get(self, case_id: str) -> dict[str, Any]:
        return json.loads((self.root / case_id / "case.json").read_text())

    def update(self, case_id: str, **fields: Any) -> None:
        with _lock:
            self._write(case_id, {**self.get(case_id), **fields})

    def list(self) -> list[dict[str, Any]]:
        out = []
        for path in self.root.glob("*/case.json"):
            try:
                out.append(json.loads(path.read_text()))
            except ValueError:
                continue
        return sorted(out, key=lambda c: -c["created"])

    def emit(self, case_id: str, kind: str, agent: str, text: str) -> None:
        line = json.dumps({"t": time.time(), "kind": kind, "agent": agent, "text": text}, default=str)
        with _lock, open(self.root / case_id / "events.jsonl", "a") as handle:
            handle.write(line + "\n")

    def events(self, case_id: str) -> list[dict[str, Any]]:
        path = self.root / case_id / "events.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

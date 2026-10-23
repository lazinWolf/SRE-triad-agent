"""Background worker: configuration audit, alert rules and the job queue.

The UI and CLI only write intents (jobs, approvals) to the ledger; this process
performs investigations, approved actions and recovery checks, so long-running
work survives page reloads and runs one step at a time per case.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from .alerts import AlertRules, open_cases_for_alerts
from .changes import ChangeWatcher
from .config import Settings
from .ledger import Ledger
from .orchestrator import investigate
from .recovery import execute_action, rollback_action, verify_recovery
from .remediation import Remediator
from .telemetry import DockerInspector, TelemetryUnavailable
from .tools import Backends

log = logging.getLogger(__name__)


class Worker:
    def __init__(self, settings: Settings, *, auto_investigate: bool | None = None, alerts: bool = True,
                 max_parallel: int = 2):
        self.settings = settings
        self.ledger = Ledger(settings.db_path)
        self.inspector = DockerInspector(settings.docker_base_url)
        self.watcher = ChangeWatcher(self.ledger, self.inspector, settings.flagd_file,
                                     settings.data_dir / "watch_state.json")
        self.remediator = Remediator(self.inspector, settings.flagd_file)
        self.rules = AlertRules()
        self.alerts_enabled = alerts
        if auto_investigate is None:
            auto_investigate = os.getenv("SRE_AUTO_INVESTIGATE", "1") not in ("0", "false", "no") and bool(settings.model)
        self.auto_investigate = auto_investigate
        self.slots = asyncio.Semaphore(max_parallel)
        self.tasks: set[asyncio.Task] = set()

    async def run(self) -> None:
        interrupted = self.ledger.recover_interrupted()
        if interrupted:
            log.warning("marked %d interrupted jobs as failed", interrupted)
        backends = Backends.from_settings(self.settings)
        try:
            await asyncio.gather(self._watch_loop(), self._alert_loop(backends), self._job_loop(backends))
        finally:
            for task in self.tasks:
                task.cancel()
            await backends.aclose()

    async def _watch_loop(self) -> None:
        heartbeat = self.settings.data_dir / "worker.heartbeat"
        while True:
            heartbeat.touch()
            try:
                changes = await asyncio.to_thread(self.watcher.poll)
                if changes:
                    log.info("recorded %d configuration change(s)", len(changes))
            except Exception:
                log.exception("change watcher poll failed")
            await asyncio.sleep(5)

    async def _alert_loop(self, backends: Backends) -> None:
        while self.alerts_enabled:
            try:
                firing = await self.rules.evaluate(backends.prometheus)
                opened = open_cases_for_alerts(self.ledger, firing, self.rules, self.settings.baseline_seconds)
                for case_id in opened:
                    log.info("alert opened case %s", case_id)
                    if self.auto_investigate:
                        self.ledger.enqueue("investigate", case_id, {"reason": "alert"})
            except TelemetryUnavailable as exc:
                log.warning("alert evaluation skipped: %s", exc)
            except Exception:
                log.exception("alert evaluation failed")
            await asyncio.sleep(30)

    async def _job_loop(self, backends: Backends) -> None:
        while True:
            await self.slots.acquire()
            job = self.ledger.claim_job()
            if job is None:
                self.slots.release()
                await asyncio.sleep(1)
                continue
            task = asyncio.create_task(self._run_job(job, backends))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

    async def _run_job(self, job: dict[str, Any], backends: Backends) -> None:
        try:
            await self._dispatch(job, backends)
            self.ledger.finish_job(job["id"])
        except asyncio.CancelledError:
            self.ledger.finish_job(job["id"], "cancelled")
            raise
        except Exception as exc:
            log.exception("job %s failed", job["id"])
            self.ledger.finish_job(job["id"], f"{type(exc).__name__}: {exc}"[:500])
        finally:
            self.slots.release()

    async def _dispatch(self, job: dict[str, Any], backends: Backends) -> None:
        kind, payload = job["kind"], job["payload"] or {}
        if kind == "investigate":
            await investigate(self.ledger, self.settings, job["case_id"], backends=backends)
        elif kind == "execute_action":
            action = await asyncio.to_thread(execute_action, self.ledger, self.settings, self.remediator,
                                             payload["action_id"])
            await asyncio.to_thread(self.watcher.poll)  # record the remediation in the change audit promptly
            if action["status"] == "executed":
                self.ledger.enqueue("verify_recovery", action["case_id"], {"action_id": action["id"]})
        elif kind == "verify_recovery":
            result = await verify_recovery(self.ledger, self.settings, backends.prometheus, job["case_id"],
                                           payload.get("action_id"))
            if result["verdict"] == "not_recovered" and self.settings.model:
                self.ledger.enqueue("investigate", job["case_id"], {"reason": "recovery_failed"})
        elif kind == "rollback":
            await asyncio.to_thread(rollback_action, self.ledger, self.remediator, payload["action_id"])
            await asyncio.to_thread(self.watcher.poll)
        else:
            raise ValueError(f"unknown job kind {kind}")

"""Scenario evaluation: multi-agent vs single-agent on the same incidents and budget.

For each scenario: reset the environment, let a healthy baseline accumulate,
apply decoys and the fault, let the incident develop, then investigate the same
case window once per mode. Reports are scored against the scenario's hidden
ground truth on diagnosis, unsupported claims, remediation, time and tool use.
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from .changes import ChangeWatcher
from .config import Settings
from .faults import SCENARIOS, Scenario, inject, reset, score
from .ledger import Ledger, new_id
from .models import CaseInput
from .orchestrator import investigate
from .telemetry import DockerInspector
from .tools import Backends
from .validation import unsupported_claims

log = logging.getLogger(__name__)


async def _watch(watcher: ChangeWatcher, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.to_thread(watcher.poll)
        except Exception:
            log.exception("watcher poll failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=5)
        except TimeoutError:
            pass


async def run_evaluation(settings: Settings, scenario_ids: list[str], *, modes: tuple[str, ...] = ("multi_agent", "single_agent"),
                         baseline_seconds: float = 300, pre_window_s: float = 60, use_memory: bool = True,
                         repeats: int = 1, model_override: Any = None, out_dir: Path | None = None,
                         progress: Callable[[str], None] = print) -> dict[str, Any]:
    ledger = Ledger(settings.db_path)
    run_id = new_id("eval", 8)
    out_dir = out_dir or settings.data_dir / "eval" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    inspector = DockerInspector(settings.docker_base_url)
    watcher = ChangeWatcher(ledger, inspector, settings.flagd_file, settings.data_dir / "watch_state.json")
    stop = asyncio.Event()
    watch_task = asyncio.create_task(_watch(watcher, stop))
    backends = Backends.from_settings(settings)
    rows: list[dict[str, Any]] = []
    try:
        for repeat in range(repeats):
            for sid in scenario_ids:
                scenario = SCENARIOS[sid]
                progress(f"[{sid}] reset; accumulating {baseline_seconds:.0f}s of healthy baseline")
                await asyncio.to_thread(reset, settings.flagd_file)
                await asyncio.sleep(baseline_seconds + pre_window_s)
                oracle = await asyncio.to_thread(inject, scenario, settings.flagd_file)
                progress(f"[{sid}] fault applied; incident develops for {scenario.incident_seconds:.0f}s")
                await asyncio.sleep(scenario.incident_seconds)
                # The case window starts shortly before injection, as an alert lookback would;
                # the exact injection time is never given to the agents.
                incident_start = oracle["injected_at"] - pre_window_s
                incident_end = time.time()
                for mode in modes:
                    case = ledger.create_case(CaseInput(
                        service=scenario.alert_service, symptom=scenario.symptom, severity="warning",
                        source="evaluation", mode=mode, use_memory=use_memory, incident_start=incident_start,
                        incident_end=incident_end, baseline_seconds=baseline_seconds))
                    progress(f"[{sid}] investigating case {case['id']} in {mode} mode")
                    result = await investigate(ledger, settings, case["id"], backends=backends,
                                               model_override=model_override)
                    evidence = ledger.evidence(case["id"], result["run_id"])
                    report = result["report"]
                    row = {
                        "eval_id": run_id, "repeat": repeat, "scenario": sid, "difficulty": scenario.difficulty,
                        "mode": mode, "case_id": case["id"], "run_id": result["run_id"], "status": result["status"],
                        "error": result["error"], **score(scenario, report),
                        "unsupported_claims": unsupported_claims(report, report.get("hypotheses", []), evidence),
                        "rejected_or_weakened": sum(h.get("status") in ("rejected", "weakened")
                                                    for h in report.get("hypotheses", [])),
                        "reported_service": (report.get("root_cause") or {}).get("service"),
                        "reported_category": (report.get("root_cause") or {}).get("category"),
                        "expected_service": scenario.expected_service, "expected_category": scenario.expected_category,
                        **{k: result["stats"].get(k) for k in ("tool_calls", "llm_calls", "prompt_tokens",
                                                               "completion_tokens", "rounds", "duration_s")},
                        "oracle": oracle,
                    }
                    rows.append(row)
                    with open(out_dir / "results.jsonl", "a") as handle:
                        handle.write(json.dumps(row, default=str) + "\n")
                    progress(f"[{sid}] {mode}: correct={row['diagnosis_correct']} reported="
                             f"{row['reported_service']}/{row['reported_category']} unsupported={row['unsupported_claims']} "
                             f"tools={row['tool_calls']} llm={row['llm_calls']} {row['duration_s']}s")
        await asyncio.to_thread(reset, settings.flagd_file)
    finally:
        stop.set()
        await watch_task
        await backends.aclose()
    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps({"eval_id": run_id, "summary": summary,
                                                      "scenarios": scenario_ids, "modes": modes,
                                                      "use_memory": use_memory, "model": settings.model}, indent=2))
    return {"eval_id": run_id, "out_dir": str(out_dir), "summary": summary, "rows": rows}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        sel = [r for r in rows if r["mode"] == mode]
        faults = [r for r in sel if r["expected_service"] is not None]
        controls = [r for r in sel if r["expected_service"] is None]

        def rate(items, key):
            return round(sum(bool(r[key]) for r in items) / len(items), 3) if items else None

        def mean(key):
            values = [r[key] for r in sel if isinstance(r.get(key), (int, float))]
            return round(statistics.mean(values), 2) if values else None

        out[mode] = {
            "runs": len(sel), "diagnosis_accuracy": rate(faults, "diagnosis_correct"),
            "service_accuracy": rate(faults, "service_correct"), "remediation_accuracy": rate(faults, "remediation_correct"),
            "control_correct": rate(controls, "diagnosis_correct"), "false_diagnosis_rate": rate(sel, "false_diagnosis"),
            "unsupported_claims_per_run": mean("unsupported_claims"), "mean_duration_s": mean("duration_s"),
            "mean_tool_calls": mean("tool_calls"), "mean_llm_calls": mean("llm_calls"),
            "mean_prompt_tokens": mean("prompt_tokens"), "mean_rejected_hypotheses": mean("rejected_or_weakened"),
        }
    return out


def load_results(data_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((data_dir / "eval").glob("*/results.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def scenario_list() -> list[Scenario]:
    return list(SCENARIOS.values())

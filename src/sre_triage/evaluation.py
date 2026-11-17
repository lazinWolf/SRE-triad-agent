"""End-to-end evaluation: inject a fault, wait for the alert, let the agents investigate, score the result.

One run measures performance (right service / category / fix, recovery), cost (tokens, list-price USD) and running
time (detect, diagnose, recover, total). A suite is a list of (scenario, variant) runs written to
data/eval/<run-id>/results.jsonl one line at a time, so an interrupted suite can be resumed.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import simulator, tracing
from .config import Settings
from .simulator import CATALOG, Scenario
from .telemetry import Unavailable

TIMING_KEYS = ("detect_s", "diagnose_s", "recover_s", "total_s")
NO_SCORE = {"correct": False, "service": False, "category": False, "remediation": False, "false_alarm": False}
ANSWER = {"answer": "No additional information; decide from the evidence."}
APPROVE = {"approved": True, "feedback": "", "choice": 0}
COST_NOTE = ("Cost is the list-price equivalent from LiteLLM's price table: calls served by a free tier cost nothing "
             "here, and models without a price entry count as 0.")


@dataclass(frozen=True)
class Variant:
    """One configuration under test. Memory is off by default so runs do not leak answers into each other."""
    name: str
    mode: str = "multi"      # "multi" | "single"
    memory: bool = False
    reviewer: bool = True    # False: no A2A review board (the local critic still runs)


VARIANTS = {v.name: v for v in (Variant("multi"), Variant("single", mode="single"),
                                Variant("multi+memory", memory=True), Variant("multi-no-review", reviewer=False))}


def variant_settings(settings: Settings, variant: Variant, out_dir: Path) -> Settings:
    """The settings a variant runs with; memory variants share one memory file inside the run directory."""
    return settings.model_copy(update={
        "memory": variant.memory, "memory_file": out_dir / "memory.sqlite" if variant.memory else None,
        "reviewer_a2a_url": settings.reviewer_a2a_url if variant.reviewer else ""})


def _suites() -> dict[str, list[tuple[str, str]]]:
    quick = ["cart_cpu_limit", "payment_failure", "payment_memory_limit", "currency_paused", "valkey_cart_stopped"]
    ablated = ["cart_cpu_limit", "payment_failure", "payment_memory_limit", "currency_paused"]
    return {
        "smoke": [("cart_cpu_limit", "multi"), ("healthy_control", "multi")],
        "quick": [(s, "multi") for s in quick] + [("healthy_control", "multi")] + [(s, "single") for s in quick[:3]],
        "full": [(s, v) for v in ("multi", "single") for s in CATALOG]
                + [(s, "multi+memory") for s in ablated for _ in (1, 2)] + [(s, "multi-no-review") for s in ablated]}


SUITES = _suites()


# ---------------------------------------------------------------------------- one run
def _blank(scenario: Scenario, variant: Variant, error: str | None = None) -> dict:
    truth = scenario.truth
    return {"case": None, "scenario": scenario.id, "variant": variant.name, "difficulty": scenario.difficulty,
            "truth": {"service": truth.service, "category": truth.category} if truth else None, "status": "error",
            "score": dict(NO_SCORE), "recovered": False, "timings": dict.fromkeys(TIMING_KEYS), "stats": {},
            "rounds": 0, "revisions": 0, "asked_operator": False, "fix_retried": False, "review_via": None,
            "guardrail_hits": 0, "evidence_count": 0, "report": None, "alerts": [], "error": error, "at": time.time()}


def _detect(backends: Any, control: bool, injected: float, wait: float, poll: float,
            sleep: Callable[[float], None], before: set[tuple] = frozenset()) -> tuple[list[dict], float | None]:
    """Poll for alerts that fire after the injection (`before` holds the ones already there). Faults: wait at least
    60 s so telemetry accumulates, then until an alert fires or `wait`. Controls: min(wait, 90)."""
    floor = min(wait, 90) if control else 60
    limit = floor if control else max(wait, 60)
    waited, seen, detect_s = 0.0, [], None
    while True:
        if new := simulator.new_firing(backends, before):
            seen = new
            detect_s = detect_s if detect_s is not None else round(time.time() - injected, 1)
        if waited >= floor and (seen or waited >= limit):
            return seen, detect_s
        step = max(poll, 1)
        sleep(step)
        waited += step


def _timings(events: list[dict], created: float, detect_s: float | None, total_s: float) -> dict:
    """Diagnose: case creation to the first approval request (or the end). Recover: fix executed to the verify result."""
    end = events[-1]["t"] if events else created
    waiting = next((e["t"] for e in events if e["kind"] == "approval" and e["text"].startswith("Waiting for approval")), end)
    acted = next((e["t"] for e in events if e["kind"] == "action" and e["text"].startswith("Executed:")), None)
    verified = next((e["t"] for e in events if acted is not None and e["kind"] == "result" and e["agent"] == "verify"
                     and e["t"] >= acted and e["text"].startswith(("Recovered", "Not recovered"))), None)
    return {"detect_s": detect_s, "diagnose_s": round(max(waiting - created, 0), 1),
            "recover_s": round(max(verified - acted, 0), 1) if verified is not None else None, "total_s": round(total_s, 1)}


def _score_trace(triage: Any, meta: dict, score: dict, recovered: bool) -> None:
    values = {"correct": score["correct"], "service_correct": score["service"], "category_correct": score["category"],
              "remediation_correct": score["remediation"], "recovered": recovered}
    for trace_id in meta.get("traces") or []:
        for name, value in values.items():
            tracing.score(trace_id, name, int(value))


def run_scenario(triage: Any, scenario: Scenario, variant: Variant, *, wait: float = 120, poll: float = 10,
                 settle: float = 300,
                 run: Callable[[str, dict | None], Any] | None = None, sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], Any] = print) -> dict:
    """One end-to-end run with nobody at the keyboard; the demo is always reset afterwards."""
    run = run or (lambda case_id, resume: triage.run(case_id, resume))
    backends, data_dir = triage.backends, triage.settings.data_dir
    started, case_id, error, alerts, detect_s = time.time(), None, None, [], None
    try:
        simulator.reset(backends)
        before = simulator.wait_clear(backends, settle, poll, sleep)  # alerts of the previous run must not count
        injection = simulator.inject(scenario, backends)
        simulator.save_active(data_dir, injection)
        log(f"Injected {scenario.id}; waiting for a firing alert"
            + (f" (still active from before: {', '.join(sorted(f'{n}:{s}' for n, s in before))})" if before else ""))
        alerts, detect_s = _detect(backends, scenario.truth is None, injection["at"], wait, poll, sleep, before)
        symptom = (alerts[0].get("summary") if alerts else None) or scenario.symptom
        case_id = triage.open_case(symptom, start=injection["at"], mode=variant.mode, scenario=scenario.id, source="eval")
        run(case_id, None)
        for _ in range(6):  # a question gets a neutral answer, an approval is granted
            request = triage.pending(case_id)
            if not request:
                break
            run(case_id, ANSWER if request["kind"] == "question" else APPROVE)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            simulator.reset(backends)
        except Exception as exc:
            error = error or f"reset failed: {type(exc).__name__}: {exc}"
        simulator.clear_active(data_dir)

    record = _blank(scenario, variant, error)
    record.update(alerts=list(dict.fromkeys(a["name"] for a in alerts)), at=started)
    record["timings"]["detect_s"] = detect_s
    record["timings"]["total_s"] = round(time.time() - started, 1)
    if case_id is None:
        return record
    meta, events = triage.cases.get(case_id), triage.cases.events(case_id)
    try:
        state = triage.state(case_id)
    except Exception:
        state = {}
    status, report = meta["status"], meta.get("report")
    score = dict(NO_SCORE) if status == "failed" else simulator.score(scenario, report)
    recovered = status == "resolved"
    critiques = state.get("critiques") or []
    triage.cases.update(case_id, score=score)
    _score_trace(triage, meta, score, recovered)
    record.update(
        case=case_id, status=status, score=score, recovered=recovered, stats=meta.get("stats") or {}, report=report,
        timings=_timings(events, meta["created"], detect_s, time.time() - started), rounds=state.get("round", 0),
        revisions=state.get("revisions", 0), asked_operator=bool(state.get("questions")),
        fix_retried=bool(state.get("failed_fixes")), evidence_count=len(state.get("evidence") or []),
        review_via=(critiques[-1].get("via") or "local") if critiques else None,
        guardrail_hits=(meta.get("stats") or {}).get("guardrail_hits", 0), error=error or meta.get("error"))
    log(f"{scenario.id} / {variant.name}: {status}, correct={score['correct']}, {record['timings']['total_s']} s")
    return record


# ---------------------------------------------------------------------------- suites
def new_run_dir(data_dir: Path, name: str) -> Path:
    path = Path(data_dir) / "eval" / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:  # a half-written last line of an interrupted suite
            continue
    return out


def run_suite(make_triage: Callable[[Variant], Any], runs: list[tuple[Scenario, Variant]], out_dir: Path, *,
              wait: float = 120, resume: bool = True, run_factory: Callable[[Any], Callable] | None = None,
              sleep: Callable[[float], None] = time.sleep, log: Callable[[str], Any] = print,
              guardrails: dict | None = None) -> list[dict]:
    """Run every (scenario, variant), appending one JSON line per run; with `resume`, finished runs are skipped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file, suite_file = out_dir / "results.jsonl", out_dir / "suite.json"
    started = time.time()
    if resume and suite_file.exists():
        started = json.loads(suite_file.read_text()).get("started", started)
    suite_file.write_text(json.dumps({"started": started, "runs": [[s.id, v.name] for s, v in runs], "wait": wait}, indent=2))
    results = _read_results(results_file) if resume else []
    done = Counter((r["scenario"], r["variant"]) for r in results)  # a pair may be listed twice
    triages: dict[str, Any] = {}
    for i, (scenario, variant) in enumerate(runs, 1):
        if resume and done[(scenario.id, variant.name)] > 0:
            done[(scenario.id, variant.name)] -= 1
            log(f"[{i}/{len(runs)}] {scenario.id} / {variant.name}: already done, skipped")
            continue
        log(f"[{i}/{len(runs)}] {scenario.id} / {variant.name}")
        try:
            if variant.name not in triages:
                triages[variant.name] = make_triage(variant)
            triage = triages[variant.name]
            record = run_scenario(triage, scenario, variant, wait=wait, sleep=sleep, log=log,
                                  run=run_factory(triage) if run_factory else None)
        except Exception as exc:  # one broken run must not stop the suite
            record = _blank(scenario, variant, f"{type(exc).__name__}: {exc}")
            log(f"  failed: {record['error']}")
        with open(results_file, "a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        results.append(record)
    summary = summarize(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "report.md").write_text(report_markdown(summary, results, guardrails))
    return results


def load_runs(data_dir: Path) -> list[dict]:
    """Every evaluation run under data_dir/eval that has results, newest first (the dashboard reads this)."""
    root = Path(data_dir) / "eval"
    out = []
    for path in sorted(root.iterdir(), reverse=True) if root.is_dir() else []:
        # rows written by earlier versions of the project have no "variant" and cannot be summarized
        results = [r for r in _read_results(path / "results.jsonl") if "variant" in r]
        if not results:
            continue
        try:
            config = json.loads((path / "suite.json").read_text())
        except (OSError, ValueError):
            config = {}
        out.append({"id": path.name, "path": str(path), "config": config, "results": results, "summary": summarize(results)})
    return out


# ---------------------------------------------------------------------------- summary
def _rate(items: list[dict], test: Callable[[dict], Any]) -> float | None:
    return round(sum(1 for r in items if test(r)) / len(items), 3) if items else None


def _clean(values: list) -> list:
    return [v for v in values if v is not None]


def _mean(values: list, digits: int = 1) -> float | None:
    values = _clean(values)
    return round(statistics.fmean(values), digits) if values else None


def _median(values: list) -> float | None:
    values = _clean(values)
    return round(statistics.median(values), 1) if values else None


def _block(rs: list[dict]) -> dict:
    faults, controls = [r for r in rs if r.get("truth")], [r for r in rs if not r.get("truth")]
    ok = lambda key: (lambda r: (r.get("score") or {}).get(key))  # noqa: E731
    out = {"n": len(rs), "faults": len(faults), "controls": len(controls), "correct": _rate(faults, ok("correct")),
           "service": _rate(faults, ok("service")), "category": _rate(faults, ok("category")),
           "remediation": _rate(faults, ok("remediation")), "recovered": _rate(faults, lambda r: r.get("recovered")),
           "detected": _rate(faults, lambda r: r.get("alerts")), "false_alarm": _rate(rs, ok("false_alarm")),
           "control_clean": _rate(controls, ok("correct"))}
    for key in TIMING_KEYS:
        times = [(r.get("timings") or {}).get(key) for r in rs]
        out[f"{key}_mean"], out[f"{key}_median"] = _mean(times), _median(times)
    stats = [r["stats"] for r in rs if r.get("stats")]
    for key in ("llm_calls", "tokens", "prompt_tokens", "completion_tokens", "cost_usd"):
        out[f"{key}_mean"] = _mean([s.get(key) for s in stats], 5 if key == "cost_usd" else 1)
    out["cost_usd_total"] = round(sum(s.get("cost_usd") or 0 for s in stats), 5)
    out["rounds_mean"] = _mean([r.get("rounds") for r in rs])
    calls = Counter()
    for s in stats:
        for agent, info in (s.get("agents") or {}).items():
            calls[agent] += info.get("calls", 0)
    out["agent_calls"] = {agent: round(n / len(stats), 1) for agent, n in sorted(calls.items())}
    return out


def _group(results: list[dict], key: Callable[[dict], str]) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(key(r), []).append(r)
    return {name: _block(rs) for name, rs in groups.items()}


def summarize(results: list[dict]) -> dict:
    return {"overall": _block(results), "by_variant": _group(results, lambda r: r["variant"]),
            "by_difficulty": _group(results, lambda r: r.get("difficulty") or "unknown"),
            "by_category": _group(results, lambda r: (r.get("truth") or {}).get("category", "control"))}


# ---------------------------------------------------------------------------- report
def _table(head: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]) + "\n"


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def _num(x: float | None, digits: int = 1) -> str:
    return "-" if x is None else f"{x:,.{digits}f}"


def _yn(x: Any) -> str:
    return "yes" if x else "no"


def _fix(report: dict | None) -> str:
    fix = (report or {}).get("remediation")
    return f"{fix.get('action')} {fix.get('service') or fix.get('flag') or ''}".strip() if fix else "-"


def _pair(a: dict, b: dict, group: str, key: str, digits: int) -> str:
    return " / ".join(_num((r.get(group) or {}).get(key), digits) for r in (a, b))


def _first_by_scenario(results: list[dict], variant: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in results:
        if r["variant"] == variant:
            out.setdefault(r["scenario"], r)
    return out


def _compare_row(label: str, rs: list[dict]) -> list:
    b = _block(rs)
    return [label, b["n"], _pct(b["correct"]), _pct(b["recovered"]), _num(b["llm_calls_mean"]), _num(b["tokens_mean"], 0),
            _num(b["cost_usd_mean"], 4), _num(b["total_s_mean"])]


def report_markdown(summary: dict, results: list[dict], guardrails: dict | None = None) -> str:
    overall, variants = summary["overall"], summary["by_variant"]
    md = ["# SRE triage: end-to-end evaluation\n",
          f"{overall['n']} runs ({overall['faults']} faults, {overall['controls']} healthy controls). "
          f"Total list-price cost {_num(overall['cost_usd_total'], 4)} USD.\n", "## Performance\n",
          _table(["Variant", "Runs", "Correct", "Service", "Category", "Fix", "Recovered", "Detected", "False alarms",
                  "Controls clean"],
                 [[v, b["n"], _pct(b["correct"]), _pct(b["service"]), _pct(b["category"]), _pct(b["remediation"]),
                   _pct(b["recovered"]), _pct(b["detected"]), _pct(b["false_alarm"]), _pct(b["control_clean"])]
                  for v, b in variants.items()]),
          "Rates over fault runs, except false alarms (all runs) and controls clean (control runs).\n", "## Cost\n",
          _table(["Variant", "Tokens in", "Tokens out", "LLM calls", "USD / incident", "USD total", "Calls per agent"],
                 [[v, _num(b["prompt_tokens_mean"], 0), _num(b["completion_tokens_mean"], 0), _num(b["llm_calls_mean"]),
                   _num(b["cost_usd_mean"], 4), _num(b["cost_usd_total"], 4),
                   ", ".join(f"{a} {n}" for a, n in b["agent_calls"].items()) or "-"] for v, b in variants.items()]),
          f"Note: {COST_NOTE}\n", "## Running time (seconds, mean / median)\n",
          _table(["Variant", "Detect", "Diagnose", "Recover", "Total"],
                 [[v] + [f"{_num(b[f'{k}_mean'])} / {_num(b[f'{k}_median'])}" for k in TIMING_KEYS]
                  for v, b in variants.items()]), "## Per scenario\n"]
    rows = []
    for r in results:
        truth, report = r.get("truth"), r.get("report") or {}
        expected = f"{truth['service']} / {truth['category']}" if truth else "no incident"
        found = f"{report.get('root_cause_service')} / {report.get('category')}" if report.get("status") == "diagnosed" \
            else report.get("status") or "-"
        rows.append([r["scenario"], r["variant"], expected, "ERROR" if r.get("error") and not report else found,
                     _fix(report) + (" (ok)" if r["score"].get("remediation") else ""),
                     _yn(r["recovered"]) if truth else "-", _num(r["timings"].get("total_s")),
                     (r.get("stats") or {}).get("llm_calls", "-")])
    md += [_table(["Scenario", "Variant", "Expected", "Diagnosed", "Fix", "Recovered", "Seconds", "Calls"], rows)]

    multi, single = _first_by_scenario(results, "multi"), _first_by_scenario(results, "single")
    common = [s for s in multi if s in single]
    if common:
        md += ["## Multi-agent vs single agent (scenarios both ran)\n",
               _table(["Scenario", "Multi correct", "Single correct", "Calls m / s", "USD m / s", "Seconds m / s"],
                      [[s, _yn(multi[s]["score"]["correct"]), _yn(single[s]["score"]["correct"]),
                        _pair(multi[s], single[s], "stats", "llm_calls", 0), _pair(multi[s], single[s], "stats", "cost_usd", 4),
                        _pair(multi[s], single[s], "timings", "total_s", 1)] for s in common]),
               _table(["Variant", "Runs", "Correct", "Recovered", "Calls", "Tokens", "USD", "Seconds"],
                      [_compare_row("multi", [multi[s] for s in common]), _compare_row("single", [single[s] for s in common])])]
    ablations = [(v, [r for r in results if r["variant"] == v]) for v in ("multi+memory", "multi-no-review")]
    ablations = [(v, rs) for v, rs in ablations if rs]
    if ablations:
        rows = []
        for v, rs in ablations:
            scenarios = {r["scenario"] for r in rs}
            rows.append(_compare_row("multi (baseline)", [r for r in results if r["variant"] == "multi" and r["scenario"] in scenarios]))
            if v == "multi+memory":  # the second run of a scenario can recall the first
                seen: Counter = Counter()
                runs = {1: [], 2: []}
                for r in rs:
                    seen[r["scenario"]] += 1
                    runs[min(seen[r["scenario"]], 2)].append(r)
                rows += [_compare_row("multi+memory, 1st run", runs[1]), _compare_row("multi+memory, 2nd run (recalls)", runs[2])]
            else:
                rows.append(_compare_row(v, rs))
        md += ["## Ablations\n", _table(["Variant", "Runs", "Correct", "Recovered", "Calls", "Tokens", "USD", "Seconds"], rows)]
    if guardrails:
        md += ["## Guardrails (input screen on labelled prompts)\n",
               f"{guardrails.get('n')} prompts: precision {guardrails.get('precision')}, recall {guardrails.get('recall')}, "
               f"false positives {guardrails.get('false_positives')}.\n",
               _table(["Label", "Total", "Correct", "Recall"],
                      [[k, v["total"], v["correct"], v["recall"]] for k, v in (guardrails.get("per_category") or {}).items()])]
    return "\n".join(md)

"""The evaluation harness, offline: fake world + scripted models, and hand-made records for the maths."""

from __future__ import annotations

import json

import pytest

from fakes import ScriptedModel, World, call, fake_backends, settings_for, submit, write_flags
from sre_triage import evaluation, simulator
from sre_triage.evaluation import CATALOG, SUITES, VARIANTS, load_runs, new_run_dir, report_markdown, run_scenario, run_suite, summarize
from sre_triage.workflow import Triage

ALERT = {"name": "HighLatency", "service": "checkout", "kind": "latency", "severity": "warning", "state": "firing",
         "since": 0, "summary": "checkout p95 latency is high", "value": 0.4}
NO_INCIDENT = {"plan": submit("submit_report", {"status": "no_incident", "summary": "nothing abnormal"})}
FAST = dict(wait=3, poll=10, sleep=lambda s: None, log=lambda m: None)


@pytest.fixture
def world(tmp_path):
    return World(write_flags(tmp_path / "flags.json"), faulty=False)  # the simulator injects the fault


def make(tmp_path, world, scripts=None, alerts=(ALERT,)) -> Triage:
    t = Triage(settings_for(tmp_path), fake_backends(world), model_factory=lambda name: ScriptedModel(scripts),
               db_path=str(tmp_path / "sessions.db"))
    t.backends.prometheus.firing = list(alerts)
    return t


def rec(scenario="a", variant="multi", difficulty="easy", fault=True, correct=True, recovered=True, alerts=("X",),
        false_alarm=False, timings=(70, 20, 80, 200), calls=10, tokens=1000, cost=0.01, rounds=2) -> dict:
    return {"scenario": scenario, "variant": variant, "difficulty": difficulty, "status": "resolved", "error": None,
            "truth": {"service": "cart", "category": "cpu_limit"} if fault else None,
            "score": {"correct": correct, "service": correct, "category": correct, "remediation": correct,
                      "false_alarm": false_alarm},
            "recovered": recovered, "alerts": list(alerts), "rounds": rounds, "report": {"status": "diagnosed"},
            "timings": dict(zip(evaluation.TIMING_KEYS, timings)),
            "stats": {"llm_calls": calls, "tokens": tokens, "prompt_tokens": tokens - 100, "completion_tokens": 100,
                      "cost_usd": cost, "agents": {"coordinator": {"calls": 2, "tokens": 10}}}}


# ---------------------------------------------------------------------------- one run
def test_run_scenario_end_to_end(tmp_path, world):
    t = make(tmp_path, world)
    r = run_scenario(t, CATALOG["cart_cpu_limit"], VARIANTS["multi"], **FAST)
    assert r["status"] == "resolved" and r["recovered"] and r["error"] is None
    assert r["score"]["correct"] and r["score"]["remediation"] and not r["score"]["false_alarm"]
    assert r["truth"] == {"service": "cart", "category": "cpu_limit"} and r["alerts"] == ["HighLatency"]
    assert all(r["timings"][k] is not None and r["timings"][k] >= 0 for k in evaluation.TIMING_KEYS)
    assert r["stats"]["llm_calls"] >= 5 and r["stats"]["tokens"] > 0
    assert r["rounds"] >= 1 and r["evidence_count"] >= 1 and r["review_via"] == "local" and not r["asked_operator"]
    assert r["report"]["root_cause_service"] == "cart"
    assert world.cpu["cart"] == 1.0, "the world is reset afterwards"
    assert simulator.load_active(tmp_path) is None
    meta = t.cases.get(r["case"])
    assert meta["scenario"] == "cart_cpu_limit" and meta["source"] == "eval" and meta["score"]["correct"]
    assert meta["symptom"] == ALERT["summary"], "the case starts from the alert text"
    json.dumps(r)


def test_healthy_control_is_not_diagnosed(tmp_path, world):
    t = make(tmp_path, world, NO_INCIDENT, alerts=())
    r = run_scenario(t, CATALOG["healthy_control"], VARIANTS["multi"], **FAST)
    assert r["status"] == "no_incident" and r["truth"] is None and not r["recovered"]
    assert r["score"]["correct"] and not r["score"]["false_alarm"]
    assert r["timings"]["detect_s"] is None and r["timings"]["recover_s"] is None and r["alerts"] == []
    assert t.cases.get(r["case"])["symptom"] == CATALOG["healthy_control"].symptom


def test_single_mode_is_passed_to_the_case(tmp_path, world):
    t = make(tmp_path, world)
    r = run_scenario(t, CATALOG["cart_cpu_limit"], VARIANTS["single"], **FAST)
    assert t.cases.get(r["case"])["mode"] == "single" and r["variant"] == "single" and r["review_via"] is None


def test_pending_loop_answers_a_question_then_approves(tmp_path, world):
    asked = []

    def plan_or_ask(request):
        if not asked:
            asked.append(1)
            return call("submit_question", question="Was anything changed on cart?")
        return call("submit_plan", hypotheses=["cart CPU limit", "payment errors"],
                    tasks=[{"agent": "metrics", "question": "Is cart resource-bound?"}])

    t = make(tmp_path, world, {"plan": plan_or_ask})
    resumes = []
    r = run_scenario(t, CATALOG["cart_cpu_limit"], VARIANTS["multi"],
                     run=lambda case_id, resume: resumes.append(resume) or t.run(case_id, resume), **FAST)
    assert resumes == [None, evaluation.ANSWER, evaluation.APPROVE]
    assert r["asked_operator"] and r["status"] == "resolved"


def test_world_is_reset_even_when_the_run_raises(tmp_path, world):
    t = make(tmp_path, world)
    seen = []

    def boom(case_id, resume):
        seen.append(world.cpu["cart"])
        raise RuntimeError("model exploded")

    r = run_scenario(t, CATALOG["cart_cpu_limit"], VARIANTS["multi"], run=boom, **FAST)
    assert seen == [0.02], "the fault was injected while the run happened"
    assert world.cpu["cart"] == 1.0 and simulator.load_active(tmp_path) is None
    assert "model exploded" in r["error"] and r["case"] and not r["recovered"]
    assert not r["score"]["correct"], "a failed run never scores as correct, not even on a control"


def test_a_failure_before_the_case_exists_still_gives_a_record(tmp_path, world):
    t = make(tmp_path, world)
    t.open_case = lambda *a, **k: (_ for _ in ()).throw(ValueError("no case"))
    r = run_scenario(t, CATALOG["cart_cpu_limit"], VARIANTS["multi"], **FAST)
    assert r["case"] is None and r["status"] == "error" and "no case" in r["error"] and world.cpu["cart"] == 1.0


def test_detection_waits_at_least_a_minute_for_faults_but_not_for_controls(tmp_path, world):
    t = make(tmp_path, world)
    naps = []
    run_scenario(t, CATALOG["cart_cpu_limit"], VARIANTS["multi"], wait=5, poll=10, sleep=naps.append, log=lambda m: None)
    assert sum(naps) == 60
    naps.clear()
    run_scenario(make(tmp_path, world, NO_INCIDENT, alerts=()), CATALOG["healthy_control"], VARIANTS["multi"],
                 wait=500, poll=10, sleep=naps.append, log=lambda m: None)
    assert sum(naps) == 90


# ---------------------------------------------------------------------------- suites
def fake_runner(calls: list, fail: tuple = ()):
    def run(triage, scenario, variant, **kw):
        calls.append((scenario.id, variant.name))
        if scenario.id in fail:
            raise RuntimeError("docker is gone")
        return {**rec(scenario.id, variant.name, fault=scenario.truth is not None), "case": f"c{len(calls)}"}
    return run


def pairs(*ids):
    return [(CATALOG[s], VARIANTS[v]) for s, v in ids]


def test_run_suite_appends_resumes_and_writes_files(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(evaluation, "run_scenario", fake_runner(calls))
    runs = pairs(("cart_cpu_limit", "multi"), ("cart_cpu_limit", "multi"), ("payment_failure", "single"))
    made = []
    out = tmp_path / "suite"
    results = run_suite(lambda v: made.append(v.name) or object(), runs, out, log=lambda m: None)
    assert len(results) == 3 and len(calls) == 3 and made == ["multi", "single"], "one Triage per variant"
    lines = (out / "results.jsonl").read_text().splitlines()
    assert [json.loads(x)["scenario"] for x in lines] == ["cart_cpu_limit", "cart_cpu_limit", "payment_failure"]
    config = json.loads((out / "suite.json").read_text())
    assert config["runs"] == [["cart_cpu_limit", "multi"]] * 2 + [["payment_failure", "single"]] and "started" in config
    assert json.loads((out / "summary.json").read_text())["overall"]["n"] == 3
    assert "## Performance" in (out / "report.md").read_text()

    # an interrupted suite: only the first of the two identical runs finished
    (out / "results.jsonl").write_text(lines[0] + "\n")
    calls.clear()
    results = run_suite(lambda v: object(), runs, out, log=lambda m: None)
    assert calls == [("cart_cpu_limit", "multi"), ("payment_failure", "single")]
    assert len(results) == 3 and len((out / "results.jsonl").read_text().splitlines()) == 3
    calls.clear()
    run_suite(lambda v: object(), runs, out, log=lambda m: None)
    assert calls == [], "everything finished: nothing is re-run"
    run_suite(lambda v: object(), runs, out, resume=False, log=lambda m: None)
    assert len(calls) == 3 and len((out / "results.jsonl").read_text().splitlines()) == 3 + 3


def test_run_suite_continues_after_a_failing_run(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(evaluation, "run_scenario", fake_runner(calls, fail=("payment_failure",)))
    runs = pairs(("cart_cpu_limit", "multi"), ("payment_failure", "multi"), ("healthy_control", "multi"))
    results = run_suite(lambda v: object(), runs, tmp_path, log=lambda m: None)
    assert [r["error"] is None for r in results] == [True, False, True]
    assert "docker is gone" in results[1]["error"] and results[1]["status"] == "error" and not results[1]["score"]["correct"]
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 3


def test_run_suite_for_real_with_the_fakes(tmp_path, world):
    t = make(tmp_path, world)
    out = tmp_path / "eval-run"
    results = run_suite(lambda v: t, pairs(("cart_cpu_limit", "multi")), out, wait=3, sleep=lambda s: None,
                        log=lambda m: None, guardrails={"n": 4, "precision": 1.0, "recall": 0.5, "false_positives": 0,
                                                        "per_category": {"toxic": {"total": 2, "correct": 1, "recall": 0.5}}})
    assert results[0]["status"] == "resolved" and results[0]["score"]["correct"]
    assert "## Guardrails" in (out / "report.md").read_text()


# ---------------------------------------------------------------------------- summary and report
def test_summarize_rates_none_handling_and_groups():
    results = [
        rec("a", "multi", "easy", timings=(60, 10, 70, 100), calls=10, tokens=1000, cost=0.01),
        rec("b", "multi", "hard", correct=False, recovered=False, alerts=(), false_alarm=True,
            timings=(None, 30, None, 300), calls=20, tokens=3000, cost=0.03),
        rec("c", "multi", "easy", fault=False, alerts=(), recovered=False, timings=(None, 5, None, 100), calls=6),
        rec("a", "single", "easy", timings=(80, 20, 90, 200), calls=4, tokens=500, cost=0.0),
    ]
    s = summarize(results)
    o = s["overall"]
    assert (o["n"], o["faults"], o["controls"]) == (4, 3, 1)
    assert o["correct"] == pytest.approx(2 / 3, abs=1e-3) and o["recovered"] == pytest.approx(2 / 3, abs=1e-3)
    assert o["detected"] == pytest.approx(2 / 3, abs=1e-3) and o["false_alarm"] == 0.25 and o["control_clean"] == 1.0
    assert o["detect_s_mean"] == 70.0 and o["detect_s_median"] == 70.0, "None timings are ignored"
    assert o["diagnose_s_mean"] == 16.2 and o["diagnose_s_median"] == 15.0
    assert o["recover_s_mean"] == 80.0 and o["total_s_median"] == 150.0
    assert o["llm_calls_mean"] == 10.0 and o["tokens_mean"] == 1375.0 and o["cost_usd_total"] == 0.05
    assert o["cost_usd_mean"] == 0.0125 and o["rounds_mean"] == 2.0 and o["agent_calls"] == {"coordinator": 2.0}
    assert set(s["by_variant"]) == {"multi", "single"} and s["by_variant"]["single"]["n"] == 1
    assert s["by_variant"]["multi"]["correct"] == 0.5 and s["by_variant"]["single"]["correct"] == 1.0
    assert s["by_difficulty"]["hard"]["correct"] == 0.0 and s["by_difficulty"]["easy"]["n"] == 3
    assert set(s["by_category"]) == {"cpu_limit", "control"}


def test_summarize_rates_are_none_without_a_denominator():
    only_control = summarize([rec(fault=False, alerts=(), recovered=False)])["overall"]
    assert only_control["correct"] is None and only_control["detected"] is None and only_control["control_clean"] == 1.0
    empty = summarize([])["overall"]
    assert empty["n"] == 0 and empty["correct"] is None and empty["detect_s_mean"] is None and empty["cost_usd_total"] == 0


def test_report_markdown_has_the_tables_and_the_cost_note():
    results = [rec("a", "multi"), rec("a", "single", correct=False, calls=4), rec("c", "multi", fault=False),
               rec("a", "multi+memory"), rec("a", "multi+memory", calls=7), rec("a", "multi-no-review")]
    md = report_markdown(summarize(results), results)
    for heading in ("## Performance", "## Cost", "## Running time", "## Per scenario",
                    "## Multi-agent vs single agent", "## Ablations"):
        assert heading in md
    assert "LiteLLM" in md and "without a price entry count as 0" in md and "Tokens in" in md and "Calls per agent" in md
    assert "multi+memory, 2nd run (recalls)" in md and "cart / cpu_limit" in md and "## Guardrails" not in md
    assert "## Guardrails" in report_markdown(summarize(results), results, {"n": 3, "precision": 1.0, "recall": 1.0})


def test_load_runs_returns_the_contract_shape_newest_first(tmp_path):
    root = tmp_path / "eval"
    for name, config in (("20260101-100000-quick", {"wait": 60}), ("20260102-100000-full", None)):
        (root / name).mkdir(parents=True)
        (root / name / "results.jsonl").write_text(json.dumps(rec()) + "\n" + json.dumps(rec(variant="single")) + "\n{broken")
        if config is not None:
            (root / name / "suite.json").write_text(json.dumps(config))
    (root / "20260103-empty").mkdir()
    (root / "eval_old").mkdir()   # a run from an earlier version of the project: other fields, no "variant"
    (root / "eval_old" / "results.jsonl").write_text(json.dumps({"scenario": "cart_cpu_limit", "mode": "multi_agent"}))
    runs = load_runs(tmp_path)
    assert [r["id"] for r in runs] == ["20260102-100000-full", "20260101-100000-quick"]
    assert set(runs[0]) == {"id", "path", "config", "results", "summary"}
    assert runs[0]["config"] == {} and runs[1]["config"] == {"wait": 60}
    assert len(runs[0]["results"]) == 2 and runs[0]["summary"]["overall"]["n"] == 2 and runs[0]["path"] == str(root / runs[0]["id"])
    assert load_runs(tmp_path / "nothing") == []


def test_new_run_dir_and_variant_settings(tmp_path):
    path = new_run_dir(tmp_path, "quick")
    assert path.is_dir() and path.parent == tmp_path / "eval" and path.name.endswith("-quick")
    base = settings_for(tmp_path).model_copy(update={"reviewer_a2a_url": "http://x:8002"})
    plain = evaluation.variant_settings(base, VARIANTS["multi"], path)
    assert not plain.memory and plain.memory_file is None and plain.reviewer_a2a_url == "http://x:8002"
    remembering = evaluation.variant_settings(base, VARIANTS["multi+memory"], path)
    assert remembering.memory and remembering.memory_file == path / "memory.sqlite"
    assert evaluation.variant_settings(base, VARIANTS["multi-no-review"], path).reviewer_a2a_url == ""


def test_suites_only_name_known_scenarios_and_variants():
    for name, runs in SUITES.items():
        assert runs, name
        assert all(s in CATALOG and v in VARIANTS for s, v in runs), name
    assert len(SUITES["smoke"]) == 2 and len(SUITES["quick"]) == 9
    quick = SUITES["quick"]
    assert ("healthy_control", "multi") in quick and sum(v == "single" for _, v in quick) == 3
    assert len({s for s, v in quick if v == "multi"}) == 6
    full = SUITES["full"]
    assert {s for s, v in full if v == "multi"} == set(CATALOG) == {s for s, v in full if v == "single"}
    assert sum(v == "multi+memory" for _, v in full) == 8 and sum(v == "multi-no-review" for _, v in full) == 4
    memory = [s for s, v in full if v == "multi+memory"]
    assert memory[0] == memory[1], "each memory scenario runs twice in a row"

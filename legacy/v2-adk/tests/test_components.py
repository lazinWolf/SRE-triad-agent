"""Tools, validation, remediation/recovery, change audit, alerts, scoring and isolation."""

import ast
import json
from pathlib import Path

import pytest

from fakes import T0, FakeDocker, World, fake_backends
from sre_triage import analysis
from sre_triage.alerts import AlertRules, open_cases_for_alerts
from sre_triage.changes import ChangeWatcher, read_flags, write_flag
from sre_triage.config import Settings
from sre_triage.faults import SCENARIOS, score
from sre_triage.ledger import CaseBusy, Ledger
from sre_triage.models import CaseInput, ReviewInput
from sre_triage.memory import load_seeds, review_case, search_memory
from sre_triage.recovery import execute_action, verify_recovery
from sre_triage.remediation import Remediator, validate_proposal
from sre_triage.telemetry import summarize_trace
from sre_triage.tools import CaseToolkit, RunBudget
from sre_triage.validation import unsupported_claims

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "t.sqlite")


@pytest.fixture
def flagd(tmp_path):
    path = tmp_path / "demo.flagd.json"
    path.write_text(json.dumps({"flags": {
        "paymentFailure": {"state": "ENABLED", "defaultVariant": "off", "variants": {"off": 0, "50%": 0.5}},
        "adHighCpu": {"state": "ENABLED", "defaultVariant": "off", "variants": {"off": False, "on": True}}}}))
    return path


def make_case(ledger, **kw):
    args = dict(service="checkout", symptom="slow checkout", incident_start=T0, incident_end=T0 + 240, baseline_seconds=300)
    args.update(kw)
    return ledger.create_case(CaseInput(**args))


def toolkit(ledger, world=None, max_calls=10, docker=None):
    case = make_case(ledger)
    run = ledger.begin_run(case["id"], "multi_agent", "test")
    return CaseToolkit(ledger, ledger.case(case["id"]), run["id"], fake_backends(world, docker), RunBudget(max_calls))


# --------------------------------------------------------------------------- tools
async def test_tools_compare_against_baseline_and_record_evidence(ledger):
    tk = toolkit(ledger)
    res = await tk.container_resources("metrics", "cart")
    assert res["status"] == "ok"
    assert res["data"]["vs_baseline"]["cpu_throttled_ratio"]["change"] == "worse"
    assert res["data"]["vs_baseline"]["cpu_limit_cpus"]["change"] == "worse"
    [row] = ledger.evidence(tk.case["id"], tk.run_id)
    assert row["id"] == res["evidence_id"] and row["tool"] == "container_resources"
    await tk.b.aclose()


async def test_trace_profile_localizes_self_time(ledger):
    tk = toolkit(ledger)
    res = await tk.compare_trace_profiles("traces", "checkout")
    top = res["data"]["per_service_self_time"][0]
    assert top["service"] == "cart" and top["change"] == "worse"
    await tk.b.aclose()


async def test_window_bounds_unknown_services_and_cache(ledger):
    tk = toolkit(ledger)
    bad = await tk.service_red("metrics", "cart", "custom", -5000, 10)
    assert bad["status"] == "invalid"
    unknown = await tk.service_red("metrics", "mainframe")
    assert unknown["status"] == "invalid" and "Unknown service" in unknown["data"]["error"]
    a = await tk.cache_health("metrics")
    b = await tk.cache_health("traces")
    assert b["cached"] and a["evidence_id"] == b["evidence_id"]
    assert tk.budget.tool_calls == 3
    await tk.b.aclose()


async def test_budget_exhaustion_and_unavailable_backend(ledger):
    tk = toolkit(ledger, World(prometheus_up=False), max_calls=1)
    first = await tk.service_red("metrics", "cart")
    assert first["status"] == "unavailable" and "says nothing about service health" in first["data"]["note"]
    second = await tk.service_red("metrics", "checkout")
    assert second["status"] == "budget_exhausted"
    await tk.b.aclose()


async def test_container_events_and_runtime_snapshot(ledger):
    docker = FakeDocker(events=[{"time": T0 + 10, "container": "email", "action": "oom", "exit_code": None},
                                {"time": T0 + 11, "container": "email", "action": "die", "exit_code": "137"}])
    tk = toolkit(ledger, docker=docker)
    ev = await tk.container_events("logs_changes", "email")
    assert [e["action"] for e in ev["data"]["events"]] == ["oom", "die"] and ev["data"]["events"][0]["at"] == "t+10s"
    snap = await tk.runtime_config("logs_changes", "cart")
    assert snap["data"]["observed_at"].startswith("NOW")
    await tk.b.aclose()


def test_summarize_trace_unions_concurrent_children():
    trace = {"traceID": "t", "processes": {"a": {"serviceName": "front"}, "b": {"serviceName": "back"}},
             "spans": [{"spanID": "1", "processID": "a", "operationName": "root", "startTime": 0, "duration": 100_000},
                       {"spanID": "2", "processID": "b", "operationName": "x", "startTime": 10_000, "duration": 50_000,
                        "references": [{"refType": "CHILD_OF", "spanID": "1"}]},
                       {"spanID": "3", "processID": "b", "operationName": "y", "startTime": 20_000, "duration": 50_000,
                        "references": [{"refType": "CHILD_OF", "spanID": "1"}]}]}
    s = summarize_trace(trace)
    assert s["self_time_ms_by_service"]["front"] == 40.0  # 100 - union(10..70)
    assert s["self_time_ms_by_service"]["back"] == 100.0


# --------------------------------------------------------------------------- analysis / validation
def test_compare_and_recovery_rules():
    assert analysis.compare(20, 900, min_abs_delta=20)["change"] == "worse"
    assert analysis.compare(20, 22, min_abs_delta=20)["change"] == "unchanged"
    assert analysis.compare(None, 5)["change"] == "unknown"
    ok = analysis.recovered({"p95_latency_ms": 100, "error_ratio": 0.0}, {"p95_latency_ms": 130, "error_ratio": 0.0, "request_rate_per_s": 2},
                            latency_ratio=1.5, latency_slack_ms=25, error_rate_slack=0.01)
    assert ok["verdict"] == "recovered"
    bad = analysis.recovered({"p95_latency_ms": 100, "error_ratio": 0.0}, {"p95_latency_ms": 900, "error_ratio": 0.0, "request_rate_per_s": 2},
                             latency_ratio=1.5, latency_slack_ms=25, error_rate_slack=0.01)
    assert bad["verdict"] == "not_recovered"
    unknown = analysis.recovered({"p95_latency_ms": None}, {"p95_latency_ms": None}, latency_ratio=1.5, latency_slack_ms=25, error_rate_slack=0.01)
    assert unknown["verdict"] == "inconclusive"


def test_remediation_allowlist(flagd):
    assert validate_proposal({"action": "set_cpu_limit", "service": "cart", "cpus": 1.0}, flagd) == []
    assert validate_proposal({"action": "set_cpu_limit", "service": "prometheus", "cpus": 1.0}, flagd)
    assert validate_proposal({"action": "set_cpu_limit", "service": "cart", "cpus": 64}, flagd)
    assert validate_proposal({"action": "set_feature_flag", "flag": "paymentFailure", "variant": "off"}, flagd) == []
    assert validate_proposal({"action": "set_feature_flag", "flag": "paymentFailure", "variant": "50%"}, flagd)
    assert validate_proposal({"action": "delete_database", "service": "cart"}, flagd)


def test_unsupported_claims_counts_bad_citations():
    evidence = [{"id": "ev_1", "tool": "a", "status": "ok"}, {"id": "ev_2", "tool": "a", "status": "ok"}]
    hyps = [{"status": "supported", "supporting_evidence": ["ev_1", "ev_2"], "contradicting_evidence": []}]
    report = {"diagnosis_status": "diagnosed", "root_cause": {"evidence_ids": ["ev_9"]}}
    assert unsupported_claims(report, hyps, evidence) == 3  # one tool only, unknown id, unusable root cause


# --------------------------------------------------------------------------- ledger, memory
def test_ledger_busy_feedback_and_jobs(ledger):
    case = make_case(ledger)
    ledger.begin_run(case["id"], "multi_agent", "m")
    with pytest.raises(CaseBusy):
        ledger.begin_run(case["id"], "multi_agent", "m")
    with pytest.raises(CaseBusy):
        ledger.add_feedback(case["id"], "too early")
    assert ledger.recover_interrupted() == 0
    assert ledger.case(case["id"])["status"] == "open"
    updated = ledger.add_feedback(case["id"], "Look at payment too")
    assert updated["revision"] == 1 and updated["report"] is None
    job = ledger.enqueue("investigate", case["id"])
    assert ledger.claim_job()["id"] == job and ledger.claim_job() is None


def test_memory_review_and_retrieval(ledger):
    assert load_seeds(ledger, ROOT / "seeds" / "incidents.json") == 5
    case = make_case(ledger, service="cart")
    ledger.set_case(case["id"], report={"summary": "cart throttled", "root_cause": {"service": "cart", "category": "cpu_limit"}})
    review_case(ledger, case["id"], ReviewInput(outcome="resolved", confirmed_cause="cart CPU limit reduced to 0.05",
                                                notes="Throttling ratio was the giveaway"))
    hits = search_memory(ledger, "checkout latency cart cpu limit throttling", "cart")
    assert hits[0]["memory_id"] == f"mem_{case['id']}"
    assert search_memory(ledger, "cart cpu", "cart", exclude_case_id=case["id"])[0]["memory_id"] != f"mem_{case['id']}"
    assert ledger.case(case["id"])["status"] == "closed"


# --------------------------------------------------------------------------- change audit
def test_change_watcher_records_diffs_once(ledger, flagd, tmp_path):
    docker = FakeDocker({"cart": {"image": "i", "cpu_limit_cpus": 1.0, "memory_limit_mb": 160, "restart_policy": "x"}})
    watcher = ChangeWatcher(ledger, docker, flagd, tmp_path / "state.json")
    other = ChangeWatcher(ledger, docker, flagd, tmp_path / "state.json")
    assert watcher.poll() == []  # first poll establishes the baseline silently
    docker.configs["cart"]["cpu_limit_cpus"] = 0.05
    write_flag(flagd, "paymentFailure", "50%")
    assert len(watcher.poll()) == 2
    assert other.poll() == []  # a second poller sharing state does not duplicate
    changes = ledger.changes(0, 4e9)
    assert {c["kind"] for c in changes} == {"container_config", "feature_flag"}
    assert read_flags(flagd)["paymentFailure"]["defaultVariant"] == "50%"


# --------------------------------------------------------------------------- remediation and recovery
async def test_approved_action_executes_and_failed_recovery_reopens(ledger, flagd, tmp_path):
    settings = Settings(data_dir=tmp_path, demo_dir=tmp_path / "demo")
    settings.recovery.settle_seconds = 0
    settings.recovery.observe_seconds = 60
    (tmp_path / "demo" / "src" / "flagd").mkdir(parents=True)
    settings.flagd_file.write_text(flagd.read_text())
    write_flag(settings.flagd_file, "paymentFailure", "50%")
    case = make_case(ledger)
    action_id = ledger.propose_action(case["id"], "run", {"action": "set_feature_flag", "flag": "paymentFailure", "variant": "off",
                                                         "rationale": "r", "evidence_ids": ["ev_1"]}, {})
    with pytest.raises(ValueError):
        execute_action(ledger, settings, Remediator(FakeDocker(), settings.flagd_file), action_id)  # not approved yet
    ledger.decide_action(action_id, True, "tester")
    done = execute_action(ledger, settings, Remediator(FakeDocker(), settings.flagd_file), action_id)
    assert done["status"] == "executed" and done["after_state"]["variant"] == "off"
    assert read_flags(settings.flagd_file)["paymentFailure"]["defaultVariant"] == "off"

    ledger.set_action(action_id, executed_at=T0 + 600)  # place the action inside the simulated incident
    backends = fake_backends(World(fault_at=T0))  # symptoms persist after the action
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    result = await verify_recovery(ledger, settings, backends.prometheus, case["id"], action_id, sleep=fake_sleep,
                                   clock=lambda: T0 + 1000)
    await backends.aclose()
    assert result["verdict"] == "not_recovered"
    assert ledger.case(case["id"])["status"] == "reopened"


# --------------------------------------------------------------------------- alerts
async def test_alerts_group_onto_one_case(ledger):
    world = World(fault_at=T0)
    backends = fake_backends(world)
    rules = AlertRules(consecutive=1)
    firing = await rules.evaluate(backends.prometheus, now=T0 + 300)
    await backends.aclose()
    services = {f["service"] for f in firing}
    assert {"checkout", "cart"} <= services
    opened = open_cases_for_alerts(ledger, firing, rules, 600, now=T0 + 300)
    assert len(opened) == 1
    assert len(ledger.alerts()) == len(firing)


# --------------------------------------------------------------------------- scoring and isolation
def test_scoring():
    s = SCENARIOS["cart_cpu_limit"]
    good = {"diagnosis_status": "diagnosed", "root_cause": {"service": "cart", "category": "cpu_limit"},
            "remediation": {"action": "set_cpu_limit", "service": "cart", "cpus": 1.0}}
    assert score(s, good)["diagnosis_correct"] and score(s, good)["remediation_correct"]
    wrong = {"diagnosis_status": "diagnosed", "root_cause": {"service": "valkey-cart", "category": "cache_failure"}}
    assert score(s, wrong)["false_diagnosis"]
    control = SCENARIOS["healthy_control"]
    assert score(control, {"diagnosis_status": "insufficient_evidence"})["diagnosis_correct"]
    assert not score(control, good)["diagnosis_correct"]


def test_agent_code_cannot_reach_fault_injection_or_remediation_execution():
    """Ground truth and environment-changing code must be unreachable from agent tools."""
    forbidden = {"sre_triage.faults", "sre_triage.evaluation", "sre_triage.recovery"}
    for module in ("tools.py", "telemetry.py", "agents/team.py", "agents/runtime.py", "agents/prompts.py", "validation.py"):
        tree = ast.parse((ROOT / "src" / "sre_triage" / module).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = "sre_triage" + ("." if node.level else "") + (node.module or "") if node.level else (node.module or "")
                imported.add(base.replace("sre_triage..", "sre_triage."))
                imported.update(f"sre_triage.{a.name}" for a in node.names)
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
        assert not imported & forbidden, f"{module} imports {imported & forbidden}"
        text = (ROOT / "src" / "sre_triage" / module).read_text()
        assert "write_flag" not in text and "docker update" not in text and ".execute(" not in text


def test_relative_window_notation(ledger):
    tk = toolkit(ledger)
    assert tk.resolve("t-120s..t+60s")[:2] == (T0 - 120, T0 + 60)
    assert tk.resolve("custom,t+0s..t+240s")[:2] == (T0, T0 + 240)
    with pytest.raises(Exception):
        tk.resolve("t-99999s..t+60s")


def test_submission_coercion_accepts_common_model_shapes():
    from sre_triage.agents.team import coerce
    from sre_triage.models import Finding

    body = {"summary": "s", "observations": [{"statement": "x", "evidence_ids": ["ev_1"]}]}
    for shape in (body, json.dumps(body), json.dumps([body]), [body]):
        value, errors = coerce(Finding, shape)
        assert not errors and value.summary == "s"
    value, errors = coerce(Finding, {"observations": "nope"})
    assert value is None and "summary" in errors[0]


class _FakeInner:
    def __init__(self, model, error=None):
        self.model, self.error, self.calls = model, error, 0

    async def generate_content_async(self, request, stream=False):
        from google.adk.models.llm_response import LlmResponse
        from google.genai import types

        self.calls += 1
        if self.error:
            raise self.error
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=self.model)]))


async def test_rotating_llm_skips_exhausted_models(tmp_path):
    from google.adk.models.llm_request import LlmRequest

    from sre_triage import llm

    settings = Settings(data_dir=tmp_path)
    rot = llm.RotatingLlm(["gemini/a", "gemini/b", "groq/c"], settings)
    daily = RuntimeError('429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier')
    rot._models = [_FakeInner("gemini/a", daily), _FakeInner("gemini/b", RuntimeError("429 rate limit")), _FakeInner("groq/c")]
    out = [r async for r in rot.generate_content_async(LlmRequest())]
    assert out[0].content.parts[0].text == "groq/c"
    cool = llm.Cooldowns(tmp_path / "model_cooldowns.json").active()
    assert cool["gemini/a"] - cool["gemini/b"] > 3600  # daily quota cools far longer than a rate limit
    out = [r async for r in rot.generate_content_async(LlmRequest())]
    assert rot._models[0].calls == 1 and rot._models[1].calls == 1  # cooled models are not retried
    assert any("daily_quota" in n for n in llm.NOTICES)


async def test_rotating_llm_lets_tool_call_errors_reach_recovery(tmp_path):
    from google.adk.models.llm_request import LlmRequest

    from sre_triage import llm

    rot = llm.RotatingLlm(["groq/a", "groq/b"], Settings(data_dir=tmp_path))
    rot._models = [_FakeInner("groq/a", RuntimeError("tool_use_failed: did not match schema")), _FakeInner("groq/b")]
    with pytest.raises(RuntimeError, match="tool_use_failed"):
        [r async for r in rot.generate_content_async(LlmRequest())]

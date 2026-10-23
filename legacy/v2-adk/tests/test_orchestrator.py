"""End-to-end investigations with real ADK agents, real tools and validation, scripted models."""

import json

import pytest

import re

from fakes import T0, World, call, fake_backends, ledger_ids, model_factory, system_text, tool_results
from sre_triage.config import Budget, Settings
from sre_triage.ledger import Ledger
from sre_triage.models import CaseInput
from sre_triage.orchestrator import investigate


@pytest.fixture
def env(tmp_path):
    flags = tmp_path / "demo" / "src" / "flagd"
    flags.mkdir(parents=True)
    (flags / "demo.flagd.json").write_text(json.dumps({"flags": {
        "paymentFailure": {"state": "ENABLED", "defaultVariant": "off", "variants": {"off": 0, "50%": 0.5}}}}))
    settings = Settings(data_dir=tmp_path / "data", demo_dir=tmp_path / "demo", model="")
    ledger = Ledger(settings.db_path)
    ledger.add_change("cart", "container_config", {"cpu_limit_cpus": 1.0}, {"cpu_limit_cpus": 0.05},
                      "Container cart configuration changed (cpu_limit_cpus: 1.0 -> 0.05)", ts=T0 + 5)
    ledger.add_change("flag:emitRawPii", "feature_flag", {"variant": "off"}, {"variant": "on"},
                      "Feature flag emitRawPii default variant changed: off -> on", ts=T0 - 40)
    return settings, ledger


def new_case(ledger, mode="multi_agent"):
    return ledger.create_case(CaseInput(service="checkout", symptom="Checkout latency is elevated",
                                        incident_start=T0, incident_end=T0 + 240, baseline_seconds=300, mode=mode))


def ok_ids(request, name):
    return [r["evidence_id"] for r in tool_results(request) if r.get("name") == name and r.get("status") == "ok"]


def team_script(role, request, turn):
    """Metrics finds cart throttling, traces localizes time to cart, logs_changes finds the change."""
    if role == "critic":
        return [call("submit_critique", critique={"verdict": "accept", "summary": "Timing and magnitude check out."})]
    if role == "coordinator":
        text = system_text(request)
        all_ok = ledger_ids(request)
        if "PHASE: PLAN" in text:
            return [call("submit_decision", decision={
                "reasoning": "Checkout latency; test cart CPU, cache and recent changes.",
                "hypotheses": [
                    {"id": "H1", "statement": "Cart CPU limit throttles cart", "service": "cart", "category": "cpu_limit", "status": "candidate"},
                    {"id": "H2", "statement": "Valkey cache is slow", "service": "valkey-cart", "category": "cache_failure", "status": "candidate"}],
                "next_step": "investigate",
                "tasks": [{"agent": "metrics", "question": "Is cart CPU throttled and is the cache healthy?"},
                          {"agent": "traces", "question": "Where is checkout time spent during the incident?"},
                          {"agent": "logs_changes", "question": "Which configuration changes precede the incident?"}]})]
        cpu = ledger_ids(request, "container_resources")
        traces = ledger_ids(request, "compare_trace_profiles")
        cache = ledger_ids(request, "cache_health")
        changes = ledger_ids(request, "config_changes")
        change_id = re.search(r"change (chg_[0-9a-f]+) on cart", text)
        return [call("submit_decision", decision={
            "reasoning": "Throttling in cart aligns with the change; cache healthy.",
            "hypotheses": [
                {"id": "H1", "statement": "Cart CPU limit throttles cart", "service": "cart", "category": "cpu_limit",
                 "status": "supported", "supporting_evidence": cpu + traces + changes},
                {"id": "H2", "statement": "Valkey cache is slow", "service": "valkey-cart", "category": "cache_failure",
                 "status": "rejected", "contradicting_evidence": cache}],
            "next_step": "conclude",
            "report": {"summary": "Cart CPU limit reduction throttles cart and slows checkout.", "diagnosis_status": "diagnosed",
                       "confidence": "high",
                       "root_cause": {"service": "cart", "category": "cpu_limit", "description": "CPU limit lowered to 0.05",
                                      "trigger_change_id": change_id.group(1) if change_id else None, "evidence_ids": cpu + traces},
                       "remediation": {"action": "set_cpu_limit", "service": "cart", "cpus": 1.0,
                                       "rationale": "Revert the limit", "evidence_ids": cpu}}})]
    if turn == 0:
        return {
            "metrics": [call("container_resources", service="cart"), call("cache_health")],
            "traces": [call("compare_trace_profiles", service="checkout")],
            "logs_changes": [call("config_changes")],
        }[role]
    results = tool_results(request)
    ids = [r["evidence_id"] for r in results if r.get("evidence_id")]
    changes = [c for r in results for c in ((r.get("data") or {}).get("changes") or [])]
    statement = "; ".join(f"change {c['change_id']} on {c['target']}" for c in changes) or "observed"
    return [call("submit_findings", findings={"summary": f"{role} done", "observations": [
        {"statement": statement, "kind": "observation", "evidence_ids": ids}]})]



async def test_multi_agent_diagnoses_and_proposes_remediation(env):
    settings, ledger = env
    case = new_case(ledger)
    backends = fake_backends(World())
    result = await investigate(ledger, settings, case["id"], backends=backends, model_override=model_factory(team_script))
    await backends.aclose()
    report = result["report"]
    assert result["status"] == "completed", result["error"]
    assert report["diagnosis_status"] == "diagnosed"
    assert report["root_cause"]["service"] == "cart"
    assert report["root_cause"]["trigger_change_id"], "the cart change should be identified"
    statuses = {h["id"]: h["status"] for h in report["hypotheses"]}
    assert statuses == {"H1": "supported", "H2": "rejected"}
    case = ledger.case(case["id"])
    assert case["status"] == "awaiting_approval"
    [action] = ledger.actions(case["id"])
    assert action["proposal"]["action"] == "set_cpu_limit" and action["status"] == "proposed"
    assert action["rollback"]["available"] and action["rollback"]["cpus"] == 0.05
    findings = ledger.findings(case["id"], result["run_id"])
    assert {f["agent"] for f in findings} == {"metrics", "traces", "logs_changes", "critic"}
    assert result["stats"]["tool_calls"] == 4 and result["stats"]["llm_calls"] >= 9
    assert len(ledger.hypothesis_history(case["id"], result["run_id"])) == 2
    assert report["critiques"][0]["verdict"] == "accept"
    assert result["stats"]["llm_calls_by_agent"]["critic"] == 1


def lazy_coordinator(role, request, turn):
    """Claims 'supported' with a single tool and invents an evidence ID, then corrects itself."""
    if role != "coordinator":
        return team_script(role, request, turn)
    text = system_text(request)
    if "PHASE: PLAN" in text:
        return team_script(role, request, turn)
    if turn == 0:
        return [call("submit_decision", decision={
            "reasoning": "Cart.", "next_step": "conclude",
            "hypotheses": [{"id": "H1", "statement": "Cart CPU", "service": "cart", "category": "cpu_limit",
                            "status": "supported", "supporting_evidence": ["ev_madeup"]}],
            "report": {"summary": "Cart", "diagnosis_status": "diagnosed",
                       "root_cause": {"service": "cart", "category": "cpu_limit", "description": "x", "evidence_ids": ["ev_madeup"]}}})]
    rejection = [r for r in tool_results(request) if r.get("status") == "rejected"]
    assert rejection and any("ev_madeup" in e for e in rejection[0]["errors"])
    return [call("submit_decision", decision={
        "reasoning": "Not enough independent evidence.", "next_step": "conclude",
        "hypotheses": [{"id": "H1", "statement": "Cart CPU", "service": "cart", "category": "cpu_limit", "status": "candidate"}],
        "report": {"summary": "Inconclusive", "diagnosis_status": "insufficient_evidence", "next_checks": ["Compare traces"]}})]


async def test_invalid_citations_are_rejected_and_corrected(env):
    settings, ledger = env
    case = new_case(ledger)
    backends = fake_backends()
    result = await investigate(ledger, settings, case["id"], backends=backends, model_override=model_factory(lazy_coordinator))
    await backends.aclose()
    assert result["report"]["diagnosis_status"] == "insufficient_evidence"
    assert ledger.case(case["id"])["status"] == "needs_review"
    assert not ledger.actions(case["id"])


def single_script(role, request, turn):
    if turn == 0:
        return [call("service_overview"), call("container_resources", service="cart"), call("config_changes")]
    if turn == 1:
        return [call("compare_trace_profiles", service="checkout")]
    ids = {r["name"]: r["evidence_id"] for r in tool_results(request) if r.get("status") == "ok"}
    return [call("submit_report", hypotheses=[
        {"id": "H1", "statement": "Cart CPU limit", "service": "cart", "category": "cpu_limit", "status": "supported",
         "supporting_evidence": [ids["container_resources"], ids["compare_trace_profiles"]]}],
        report={"summary": "Cart throttled", "diagnosis_status": "diagnosed", "root_cause": {
            "service": "cart", "category": "cpu_limit", "description": "limit", "evidence_ids": [ids["container_resources"]]}})]


async def test_single_agent_baseline_uses_same_tools(env):
    settings, ledger = env
    case = new_case(ledger, mode="single_agent")
    backends = fake_backends()
    result = await investigate(ledger, settings, case["id"], backends=backends, model_override=model_factory(single_script))
    await backends.aclose()
    assert result["report"]["root_cause"]["service"] == "cart"
    assert result["stats"]["tool_calls"] == 4
    assert result["stats"]["tool_calls_by_agent"] == {"single_agent": 4}
    assert not ledger.findings(case["id"])


def greedy_script(role, request, turn):
    """Specialists that never stop querying; the shared budget must stop them."""
    if role == "coordinator":
        if "PHASE: PLAN" in system_text(request):
            return team_script(role, request, turn)
        return [call("submit_decision", decision={
            "reasoning": "Out of budget.", "next_step": "conclude", "hypotheses": [],
            "report": {"summary": "Budget exhausted", "diagnosis_status": "insufficient_evidence"}})]
    return [call("service_red", service="cart", window="custom", start_offset_s=-200 + turn, end_offset_s=100)]


async def test_shared_tool_budget_bounds_the_run(env):
    settings, ledger = env
    settings = settings.model_copy(update={"budget": Budget(max_tool_calls=5, max_llm_calls=40, max_rounds=1)})
    case = new_case(ledger)
    backends = fake_backends()
    result = await investigate(ledger, settings, case["id"], backends=backends, model_override=model_factory(greedy_script))
    await backends.aclose()
    assert result["stats"]["tool_calls"] == 5
    assert result["report"]["diagnosis_status"] == "insufficient_evidence"
    assert result["status"] == "completed"


async def test_missing_model_fails_explicitly(env):
    settings, ledger = env
    case = new_case(ledger)
    result = await investigate(ledger, settings, case["id"], backends=fake_backends())
    assert result["status"] == "failed" and "SRE_MODEL" in result["error"]
    assert result["report"]["diagnosis_status"] == "insufficient_evidence"
    assert ledger.case(case["id"])["status"] == "needs_review"


async def test_feedback_reaches_rerun_and_old_evidence_is_not_citable(env):
    settings, ledger = env
    case = new_case(ledger)
    backends = fake_backends()
    first = await investigate(ledger, settings, case["id"], backends=backends, model_override=model_factory(team_script))
    ledger.add_feedback(case["id"], "The cache team confirms valkey is healthy; focus elsewhere.")
    assert ledger.case(case["id"])["report"] is None
    seen = {}

    def spy(role, request, turn):
        if role == "coordinator" and turn == 0:
            seen["instruction"] = system_text(request)
        return team_script(role, request, turn)

    second = await investigate(ledger, settings, case["id"], backends=backends, model_override=model_factory(spy))
    await backends.aclose()
    assert "valkey is healthy" in seen["instruction"]
    old_ids = {e["id"] for e in ledger.evidence(case["id"], first["run_id"])}
    cited = set(second["report"]["root_cause"]["evidence_ids"])
    assert cited and not cited & old_ids


def endless_specialists(role, request, turn):
    """Specialists keep querying until only the submit tool is offered (forced finalisation)."""
    if role == "coordinator":
        return greedy_script(role, request, turn)
    names = [d.name for t in request.config.tools or [] for d in t.function_declarations or []]
    if names == ["submit_findings"]:
        ids = [r["evidence_id"] for r in tool_results(request) if r.get("evidence_id")]
        return [call("submit_findings", findings={"summary": "forced", "observations": [
            {"statement": "partial", "evidence_ids": ids[:2]}], "limitations": ["step limit"]})]
    window = f"t-{200 - turn}s..t+100s"
    return [{"metrics": call("service_red", service="cart", window=window),
             "traces": call("find_traces", service="cart", window=window),
             "logs_changes": call("search_logs", service="cart", window=window)}[role]]


async def test_step_limit_forces_specialists_to_submit(env):
    settings, ledger = env
    settings = settings.model_copy(update={"budget": Budget(max_tool_calls=60, max_llm_calls=60, max_rounds=1)})
    case = new_case(ledger)
    backends = fake_backends()
    result = await investigate(ledger, settings, case["id"], backends=backends, model_override=model_factory(endless_specialists))
    await backends.aclose()
    findings = ledger.findings(case["id"], result["run_id"])
    assert len(findings) == 3 and all(f["body"]["summary"] == "forced" for f in findings)
    assert result["stats"]["tool_calls_by_agent"] == {"metrics": 6, "traces": 6, "logs_changes": 6}


def challenging_critic(role, request, turn):
    """The critic challenges the first diagnosis and asks for a cache check; the coordinator revises."""
    text = system_text(request)
    if role == "critic":
        return [call("submit_critique", critique={
            "verdict": "challenge", "summary": "Cache latency was never checked directly.",
            "issues": ["valkey client latency from cart not examined"],
            "requested_checks": [{"agent": "metrics", "question": "What is cart's client-side latency to valkey-cart?"}]})]
    if role == "coordinator" and "PHASE: REVISE" in text:
        assert "CRITIC REVIEW" in text and "Cache latency was never checked" in text
        return team_script(role, request, turn)
    if role == "metrics" and "client-side latency to valkey-cart" in text and turn == 0:
        return [call("operation_latency", service="cart", span_kind="client")]
    return team_script(role, request, turn)


async def test_critic_challenge_triggers_targeted_follow_up(env):
    settings, ledger = env
    case = new_case(ledger)
    backends = fake_backends()
    result = await investigate(ledger, settings, case["id"], backends=backends,
                               model_override=model_factory(challenging_critic))
    await backends.aclose()
    report = result["report"]
    assert report["critiques"][0]["verdict"] == "challenge"
    assert len(report["critiques"]) == 1  # one review per run
    rounds = {f["round"] for f in ledger.findings(case["id"], result["run_id"]) if f["agent"] == "metrics"}
    assert rounds == {1, 2}, "the critic's requested check ran as a second round"
    tools = [e["tool"] for e in ledger.evidence(case["id"], result["run_id"])]
    assert "operation_latency" in tools
    assert result["stats"]["rounds"] == 2


def hasty_coordinator(role, request, turn):
    """Tries to conclude during planning; must be told to delegate first."""
    if role == "coordinator" and "PHASE: PLAN" in system_text(request) and turn == 0:
        return [call("submit_decision", decision={"reasoning": "Looks like nothing.", "next_step": "conclude", "hypotheses": [],
                                                  "report": {"summary": "x", "diagnosis_status": "insufficient_evidence"}})]
    return team_script(role, request, turn)


async def test_planning_phase_must_delegate(env):
    settings, ledger = env
    case = new_case(ledger)
    backends = fake_backends()
    result = await investigate(ledger, settings, case["id"], backends=backends, model_override=model_factory(hasty_coordinator))
    await backends.aclose()
    assert result["report"]["diagnosis_status"] == "diagnosed"
    assert any(e["kind"] == "validation" and "planning phase" in e["message"] for e in ledger.events(case["id"]))


def test_hypothesis_accepts_statement_aliases():
    from sre_triage.models import Hypothesis
    h = Hypothesis.model_validate({"id": "H1", "description": "cart cpu", "service": "cart", "category": "cpu_limit", "status": "candidate"})
    assert h.statement == "cart cpu" and h.model_dump()["statement"] == "cart cpu"

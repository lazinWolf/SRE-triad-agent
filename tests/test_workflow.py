"""The whole ADK workflow, offline: fake telemetry + scripted ADK models (tools really execute)."""

from __future__ import annotations

import pytest

from fakes import ScriptedModel, World, call, fake_backends, settings_for, submit, write_flags
from sre_triage.workflow import Triage


@pytest.fixture
def world(tmp_path):
    return World(write_flags(tmp_path / "flags.json"))


def make(tmp_path, world, model: ScriptedModel) -> Triage:
    return Triage(settings_for(tmp_path), fake_backends(world), model_factory=lambda name: model,
                  db_path=str(tmp_path / "sessions.db"))


def nodes(t: Triage, case_id: str) -> list[str]:
    return [e["agent"] for e in t.cases.events(case_id) if e["kind"] == "node"]


def test_full_flow_waits_for_approval_then_fixes_verifies_and_learns(tmp_path, world):
    t = make(tmp_path, world, ScriptedModel())
    case = t.open_case("Checkout latency is elevated")
    state = t.run(case)

    assert t.cases.get(case)["status"] == "awaiting_approval"
    assert t.pending(case)["proposal"]["action"] == "set_cpu_limit"
    assert world.cpu["cart"] == 0.02, "nothing may change before approval"
    assert any("checkout" in a for a in state["anomalies"])
    assert {f["agent"] for f in state["findings"]} == {"metrics", "logs_changes"}  # only delegated agents ran
    assert {e["id"] for e in state["evidence"]} == {"M1", "L1"}

    # a fresh Triage (e.g. after a restart) resumes the case from the SQLite session
    t2 = make(tmp_path, world, ScriptedModel())
    state = t2.decide_case(case, approved=True)
    assert world.cpu["cart"] == 1.0
    assert state["recovery"]["recovered"] is True
    assert t2.cases.get(case)["status"] == "resolved"
    assert nodes(t2, case)[-4:] == ["decide", "remediate", "verify", "learn"]
    remembered = {m["id"]: m for m in t2.memory.all()}[case]
    assert remembered["root_cause_service"] == "cart" and remembered["outcome"] == "resolved"


def test_critic_sends_team_back_once(tmp_path, world):
    verdicts = iter([{"verdict": "revise", "issues": ["traces not checked"],
                      "follow_up": [{"agent": "traces", "question": "Where does checkout time go?"}]},
                     {"verdict": "accept", "issues": [], "follow_up": []}])
    t = make(tmp_path, world, ScriptedModel({"critic": submit("submit_review", lambda r: next(verdicts))}))
    case = t.open_case("Checkout latency is elevated")
    state = t.run(case)
    assert state["revisions"] == 1
    assert [c["verdict"] for c in state["critiques"]] == ["revise", "accept"]
    assert [f["agent"] for f in state["findings"] if f["round"] == 2] == ["traces"]
    assert nodes(t, case).count("synthesize") == 2
    assert t.cases.get(case)["status"] == "awaiting_approval"


def test_rejection_with_feedback_reinvestigates(tmp_path, world):
    model = ScriptedModel()
    t = make(tmp_path, world, model)
    case = t.open_case("Checkout latency is elevated")
    t.run(case)
    state = t.decide_case(case, approved=False, feedback="Check the payment service too")
    assert state["operator_feedback"] == ["Check the payment service too"]
    assert any("REJECTED" in p for role, p in model.prompts if role == "plan")
    assert t.pending(case) is not None, "a new proposal awaits approval after re-investigation"
    t.decide_case(case, approved=False)
    assert t.cases.get(case)["status"] == "rejected"
    assert world.cpu["cart"] == 0.02


def test_invalid_submissions_are_returned_to_the_model(tmp_path, world):
    attempts = []

    def bad_then_good(request):
        attempts.append(1)
        if len(attempts) == 1:  # cites evidence that does not exist and proposes an unknown flag
            return call("submit_report", status="diagnosed", summary="payment", root_cause_service="payment",
                        evidence_ids=["X9"], remediation_action={"action": "set_feature_flag",
                                                                  "flag": "noSuchFlag", "variant": "off"})
        from fakes import good_report
        return call("submit_report", **good_report(request))

    model = ScriptedModel({"report": bad_then_good})
    t = make(tmp_path, world, model)
    case = t.open_case("Checkout latency is elevated")
    state = t.run(case)
    assert len(attempts) == 2
    assert any(e["kind"] == "validation" or "rejected" in e["text"] for e in t.cases.events(case)) or len(attempts) == 2
    assert state["report"]["root_cause_service"] == "cart"
    assert state["checks"] == []


def test_guardrails_strip_unverifiable_remediation(tmp_path, world):
    bad = {"status": "diagnosed", "summary": "payment", "root_cause_service": "payment",
           "category": "dependency_errors", "evidence_ids": ["X9"],
           "remediation_action": {"action": "set_feature_flag", "flag": "noSuchFlag", "variant": "off"}}
    t = make(tmp_path, world, ScriptedModel({"report": submit("submit_report", bad)}))
    case = t.open_case("Checkout latency is elevated")
    state = t.run(case)
    assert any("does not exist" in c for c in state["checks"])
    assert any("unknown feature flag" in c for c in state["checks"])
    assert state["critiques"][0]["verdict"] == "revise"           # forced by the checks
    assert state["report"]["remediation"] is None                  # still failing after the revision
    assert t.pending(case) is None
    assert t.cases.get(case)["status"] == "diagnosed_no_action"


def test_single_agent_mode_skips_specialists_and_critic(tmp_path, world):
    t = make(tmp_path, world, ScriptedModel())
    case = t.open_case("Checkout latency is elevated", mode="single")
    state = t.run(case)
    assert [f["agent"] for f in state["findings"]] == ["solo"]
    assert "critic" not in nodes(t, case)
    assert t.pending(case) is not None


def test_inconclusive_report_ends_without_action(tmp_path, world):
    world.cpu["cart"] = 1.0
    t = make(tmp_path, world, ScriptedModel({"report": submit("submit_report", {
        "status": "inconclusive", "summary": "nothing abnormal",
        "remediation_action": {"action": "restart_service", "service": "cart"}})}))
    case = t.open_case("A user said checkout felt slow")
    state = t.run(case)
    assert state["anomalies"] == []
    assert state["report"]["remediation"] is None
    assert t.cases.get(case)["status"] == "inconclusive"
    assert world.restarts == []

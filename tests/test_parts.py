"""Tools as ADK function tools, submit-tool validation, memory, remediation guardrails and scoring."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from fakes import ScriptedModel, World, fake_backends, write_flags
from sre_triage import remediation
from sre_triage.a2a_critic import CARD_PATH, build_app, parse_review
from sre_triage.agents import Submission, Usage, specialist, telemetry_tools
from sre_triage.faults import SCENARIOS, score
from sre_triage.memory import Memory
from sre_triage.tools import AGENT_TOOLS, Toolbox, Window


@pytest.fixture
def box(tmp_path):
    world = World(write_flags(tmp_path / "flags.json"))
    now = time.time()
    return Toolbox(fake_backends(world), Window(now - 180, now, 300))


def test_tools_record_evidence_with_agent_prefixes(box):
    r1 = box.call("metrics", "container_resources", {"service": "cart"})
    r2 = box.call("logs_changes", "recent_changes", {})
    assert r1["evidence_id"] == "M1" and r2["evidence_id"] == "L1"
    assert "CPU limit 0.02 is below the standard 1.0" in r1["notes"]
    assert {"target": "cart", "change": "CPU limit 1.0 -> 0.02"} in r2["drift_from_standard_config"]
    assert [e["id"] for e in box.evidence] == ["M1", "L1"]


def test_bad_tool_calls_return_errors_not_evidence(box):
    assert "unknown service" in box.call("metrics", "service_health", {"service": "nope"})["error"]
    assert "unknown tool" in box.call("metrics", "rm_rf", {})["error"]
    assert "bad arguments" in box.call("metrics", "service_health", {"svc": "cart"})["error"]
    assert box.evidence == []


def test_telemetry_tools_become_adk_function_declarations(box):
    from google.adk.tools import FunctionTool

    for agent in AGENT_TOOLS:
        for fn in telemetry_tools(box, agent):
            decl = FunctionTool(fn)._get_declaration()
            assert decl.name == fn.__name__ and decl.description
    tool = {f.__name__: f for f in telemetry_tools(box, "metrics")}["container_resources"]
    assert tool(service="cart")["evidence_id"] == "M1"


class Ctx:
    def __init__(self):
        self.actions = SimpleNamespace(skip_summarization=False)


def test_submit_tool_rejects_unknown_evidence_then_accepts(box):
    sub = Submission()
    agent = specialist("metrics", ScriptedModel(), box, lambda *a: None, Usage(), 3, sub)
    submit = {getattr(t, "__name__", ""): t for t in agent.tools}["submit_finding"]
    ctx = Ctx()
    rejected = submit(summary="cart throttled", tool_context=ctx, suspect_service="cart", evidence_ids=["M7"])
    assert rejected["status"] == "rejected" and "M7" in rejected["errors"][0]
    box.call("metrics", "container_resources", {"service": "cart"})
    accepted = submit(summary="cart throttled", tool_context=ctx, suspect_service="cart",
                      category="cpu_limit", confidence="high", evidence_ids='["M1"]')
    assert accepted == {"status": "accepted"} and ctx.actions.skip_summarization
    assert sub.value["evidence_ids"] == ["M1"] and sub.value["confidence"] == 0.85
    assert submit(summary="x", tool_context=Ctx(), category="not-a-category")["status"] == "rejected"


def test_adk_memory_service_seeds_searches_learns_and_records_feedback(tmp_path):
    mem = Memory(tmp_path)
    assert mem.all() == []
    hits = asyncio.run(mem.search("checkout latency cart CPU throttling"))  # first search seeds the runbook
    assert len(mem.all()) == 5
    assert hits and hits[0]["root_cause_service"] == "cart"
    mem.remember({"id": "c1", "title": "valkey eviction storm", "symptom": "cart errors", "services": ["cart"],
                  "root_cause_service": "valkey-cart", "category": "resource_saturation", "mechanism": "evictions"})
    assert asyncio.run(mem.search("valkey eviction"))[0]["id"] == "c1"
    assert mem.rate("c1", correct=False, note="it was the network")
    records = {m["id"]: m for m in mem.all()}
    assert len(records) == 6 and records["c1"]["feedback"] == "incorrect"  # rating updates, not duplicates
    assert not mem.rate("missing", correct=True)
    # agents see the feedback in the memory text that load_memory returns
    found = asyncio.run(mem.service.search_memory(app_name="sre_triage", user_id="operator", query="valkey eviction"))
    assert any("INCORRECT" in (p.text or "") for m in found.memories for p in m.content.parts)


def test_a2a_answers_are_validated_as_critiques():
    assert parse_review('```json\n{"verdict": "revise", "issues": ["no trace evidence"]}\n```')["verdict"] == "revise"
    with pytest.raises(ValueError):
        parse_review("looks good to me")
    with pytest.raises(ValueError):
        parse_review('{"verdict": "maybe"}')


def test_a2a_critic_publishes_an_agent_card():
    from starlette.testclient import TestClient

    with TestClient(build_app(ScriptedModel(), port=8123)) as client:
        card = client.get(CARD_PATH).json()
    assert card["name"] == "critic" and card["supportedInterfaces"][0]["url"] == "http://127.0.0.1:8123"


def test_remediation_allowlist(tmp_path):
    backends = fake_backends(World(write_flags(tmp_path / "flags.json")))
    assert remediation.validate({"action": "set_cpu_limit", "service": "cart", "cpus": 1.0}, backends) == []
    assert remediation.validate({"action": "set_cpu_limit", "service": "prometheus"}, backends)
    assert remediation.validate({"action": "set_cpu_limit", "service": "cart", "cpus": 8}, backends)
    assert remediation.validate({"action": "set_feature_flag", "flag": "paymentFailure", "variant": "50%"}, backends)
    assert remediation.validate({"action": "delete_db"}, backends)
    undo = remediation.execute({"action": "set_feature_flag", "flag": "paymentFailure", "variant": "off"}, backends)
    assert undo == {"action": "set_feature_flag", "flag": "paymentFailure", "variant": "off"}


def test_scoring():
    cart = SCENARIOS["cart_cpu_limit"]
    good = {"status": "diagnosed", "root_cause_service": "cart", "category": "cpu_limit",
            "remediation": {"action": "set_cpu_limit", "service": "cart", "cpus": 1.0}}
    assert score(cart, good)["correct"] and score(cart, good)["remediation"]
    assert score(cart, {**good, "root_cause_service": "checkout"})["false_alarm"]
    control = SCENARIOS["healthy_control"]
    assert score(control, {"status": "inconclusive"})["correct"]
    assert score(control, good)["false_alarm"]


def test_fallback_skips_a_model_that_sends_an_invalid_tool_call(monkeypatch):
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    from sre_triage.config import Settings
    from sre_triage.llm import FallbackLlm

    async def fake(self, request, stream=False):
        if self.model == "groq/bad":
            raise RuntimeError("Tool call validation failed: parameters for tool submit_report did not match schema")
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="ok")]))

    monkeypatch.setattr(LiteLlm, "generate_content_async", fake)
    llm = FallbackLlm(["groq/bad", "groq/good"], Settings())
    request = LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="hi")])])

    async def run():
        return [r async for r in llm.generate_content_async(request)]

    [response] = asyncio.run(run())
    assert response.custom_metadata["served_by"] == "groq/good"

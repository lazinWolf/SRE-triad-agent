"""Real ADK replay integration checks; these do not evaluate LLM reasoning."""

import asyncio
import time

import httpx
import pytest
import pytest_asyncio

from demo_app.server import create_checkout_app, create_inventory_app
from incident_agent.models import CaseInput, ReviewInput
from incident_agent.storage import CaseBusy, CaseStore
from incident_agent.telemetry import TelemetryGateway
from incident_agent.workflow import investigate, validate_report


SERVICE_URLS = {"checkout": "http://checkout", "inventory": "http://inventory"}


@pytest_asyncio.fixture
async def observed_incident(tmp_path):
    """Create real checkout/inventory requests and an independently audited change."""
    inventory = create_inventory_app(worker_slots=6, processing_ms=25)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=inventory), base_url="http://inventory"
    ) as downstream:
        checkout = create_checkout_app("http://inventory", downstream)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: pytest.fail(f"Unexpected network target: {request.url}")
            ),
            mounts={
                "http://checkout": httpx.ASGITransport(app=checkout),
                "http://inventory": httpx.ASGITransport(app=inventory),
            },
        ) as client:
            start = time.time()
            baseline = await asyncio.gather(
                *(client.post("http://checkout/checkout", json={}) for _ in range(6))
            )
            assert all(response.status_code == 200 for response in baseline)
            # The test operator injects a fault; agent tools never receive this control.
            changed = await downstream.post("/admin/config", json={"worker_slots": 1})
            assert changed.status_code == 200
            degraded = await asyncio.gather(
                *(client.post("http://checkout/checkout", json={}) for _ in range(6))
            )
            assert all(response.status_code == 200 for response in degraded)
            end = time.time()
            store = CaseStore(tmp_path / "cases.sqlite3")
            case = store.create(
                CaseInput(
                    summary="Checkout requests are slower than the recent baseline",
                    start=start,
                    end=end,
                    mode="replay",
                )
            )
            gateway = TelemetryGateway(SERVICE_URLS, client=client)
            yield store, case, gateway


@pytest.mark.asyncio
async def test_replay_runs_actual_adk_specialists_and_targeted_follow_up(observed_incident):
    store, case, gateway = observed_incident
    finished = await investigate(store, case["id"], SERVICE_URLS, gateway=gateway)

    assert finished["error"] is None, finished["error"]
    assert finished["status"] == "needs_review"
    report = finished["report"]
    assert report["scripted_mode"] is True
    assert report["diagnosis_status"] == "supported"
    assert report["proposed_action"]["service"] == "inventory"
    assert any("scripted" in text.lower() for text in report["limitations"])
    assert any("recovery" in text.lower() for text in report["limitations"])
    assert {item["agent"] for item in finished["evidence"]} == {"metrics", "traces", "changes"}
    assert len([item for item in finished["evidence"] if item["tool"] == "query_metrics"]) == 6
    assert any(
        event["kind"] == "delegation" and "Follow-up for metrics" in event["message"]
        for event in finished["events"]
    )
    assert all("/telemetry/" in item["result"]["source"] for item in finished["evidence"])
    assert all("/admin" not in item["result"]["source"] for item in finished["evidence"])
    evidence_ids = {item["id"] for item in finished["evidence"]}
    assert set(report["proposed_action"]["evidence_ids"]) <= evidence_ids
    assert store.memory() == []  # Autonomous output is not reviewed incident memory.


class UnavailableGateway:
    """Tool transport failure only; no diagnosis, scenario, or answer fixture."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, operation):
        async def unavailable(**query):
            self.calls.append((operation, query))
            return {
                "status": "unavailable",
                "service": query["service"],
                "source": f"http://{query['service']}/telemetry/{operation}",
                "query": query,
                "collected_at": time.time(),
                "data": None,
                "error": "Telemetry transport is unavailable",
                "limitations": ["No usable observations were retrieved."],
            }
        return unavailable


def new_case(store):
    now = time.time()
    return store.create(CaseInput(summary="Checkout latency alert", start=now - 60, end=now, mode="replay"))


@pytest.mark.asyncio
async def test_missing_telemetry_stays_unresolved_without_action(tmp_path):
    store = CaseStore(tmp_path / "cases.sqlite3")
    case = new_case(store)
    gateway = UnavailableGateway()
    result = await investigate(store, case["id"], SERVICE_URLS, gateway=gateway)

    assert result["error"] is None, result["error"]
    assert result["status"] == "unresolved"
    assert result["report"]["diagnosis_status"] == "insufficient_evidence"
    assert result["report"]["proposed_action"] is None
    assert result["report"]["next_checks"]
    assert len(gateway.calls) == 6
    assert all(item["result"]["status"] == "unavailable" for item in result["evidence"])
    assert any("unavailable" in text for text in result["report"]["limitations"])


@pytest.mark.asyncio
async def test_parallel_specialists_share_one_tool_budget(tmp_path):
    store = CaseStore(tmp_path / "cases.sqlite3")
    case = new_case(store)
    gateway = UnavailableGateway()
    result = await investigate(store, case["id"], SERVICE_URLS, gateway=gateway, max_calls=2)

    assert result["error"] is None, result["error"]
    assert len(gateway.calls) == 2
    assert len(result["evidence"]) == 2
    assert result["report"]["diagnosis_status"] == "insufficient_evidence"
    assert result["report"]["proposed_action"] is None


@pytest.mark.asyncio
async def test_rerun_references_only_new_evidence_and_feedback_survives_restart(observed_incident):
    store, case, gateway = observed_incident
    first = await investigate(store, case["id"], SERVICE_URLS, gateway=gateway)
    assert first["error"] is None, first["error"]
    old_ids = {item["id"] for item in first["evidence"]}
    second = await investigate(store, case["id"], SERVICE_URLS, gateway=gateway)
    assert second["error"] is None, second["error"]
    assert second["report"]["diagnosis_status"] == "supported"
    assert second["run_id"] != first["run_id"]
    current_ids = {
        item["id"] for item in second["evidence"] if item["run_id"] == second["run_id"]
    }
    assert current_ids and not (current_ids & old_ids)
    for hypothesis in second["report"]["hypotheses"]:
        assert set(hypothesis["supporting_evidence"]) <= current_ids
        assert set(hypothesis["contradicting_evidence"]) <= current_ids
    if second["report"]["proposed_action"]:
        assert set(second["report"]["proposed_action"]["evidence_ids"]) <= current_ids
    feedback = "Please inspect the queueing mechanism before proposing a capacity change."
    corrected = store.feedback(case["id"], feedback)
    assert corrected["report"] is None
    assert corrected["status"] == "draft"
    reopened = CaseStore(store.path)
    assert reopened.get(case["id"])["feedback"][-1]["text"] == feedback
    assert reopened.get(case["id"])["revision"] == 1
    third = await investigate(reopened, case["id"], SERVICE_URLS, gateway=gateway)
    assert third["error"] is None, third["error"]
    assert third["report"]["diagnosis_status"] == "insufficient_evidence"
    assert third["report"]["proposed_action"] is None
    assert any("Operator corrections" in text for text in third["report"]["limitations"])
    assert reopened.memory() == []


def test_report_validation_rejects_fabricated_citations_and_unsupported_claims():
    evidence = [
        {"id": "ev_current", "tool": "query_metrics", "result": {"status": "ok"}}
    ]
    fabricated = {
        "summary": "Candidate explanation",
        "diagnosis_status": "insufficient_evidence",
        "hypotheses": [{
            "cause": "Queueing", "service": "inventory", "status": "candidate",
            "supporting_evidence": ["ev_fabricated"],
        }],
    }
    with pytest.raises(ValueError, match="absent"):
        validate_report(fabricated, evidence, "replay")
    unsupported = {
        "summary": "Candidate explanation",
        "diagnosis_status": "supported",
        "hypotheses": [{
            "cause": "Queueing", "service": "inventory", "status": "supported",
            "supporting_evidence": ["ev_current"],
        }],
    }
    with pytest.raises(ValueError, match="two kinds"):
        validate_report(unsupported, evidence, "replay")
    with pytest.raises(ValueError, match="supported hypothesis"):
        validate_report({"summary": "Unsupported", "diagnosis_status": "supported"}, evidence, "live")
    with pytest.raises(ValueError, match="available supporting evidence"):
        validate_report(
            {
                "summary": "No observations",
                "diagnosis_status": "insufficient_evidence",
                "proposed_action": {
                    "service": "inventory", "description": "Increase capacity",
                    "reason": "Missing measurements", "rollback": "Restore capacity",
                    "evidence_ids": ["ev_missing"],
                },
            },
            [{"id": "ev_missing", "tool": "query_metrics", "result": {"status": "unavailable"}}],
            "replay",
        )


def test_only_explicit_review_persists_memory_and_scopes_it_to_current_run(tmp_path):
    store = CaseStore(tmp_path / "cases.sqlite3")
    case = new_case(store)
    with pytest.raises(ValueError, match="Investigate"):
        store.review(case["id"], ReviewInput(confirmed_cause="Unknown", outcome="unresolved"))
    running = store.begin_run(case["id"])
    with pytest.raises(CaseBusy):
        store.begin_run(case["id"])
    with pytest.raises(CaseBusy):
        store.feedback(case["id"], "An operator correction")
    evidence = store.add_evidence(
        case["id"], "metrics", "query_metrics", {"service": "checkout"},
        {"status": "missing", "data": {"sample_count": 0}},
    )
    store.finish_run(
        case["id"], {"summary": "No observations", "diagnosis_status": "insufficient_evidence"}
    )
    assert store.memory() == []
    review = store.review(
        case["id"], ReviewInput(confirmed_cause="Cause remains unknown", outcome="unresolved")
    )
    assert review["evidence_ids"] == [evidence["id"]]
    assert review["origin_mode"] == "replay"
    reopened = CaseStore(store.path)
    assert reopened.memory("checkout")[0]["id"] == review["id"]
    assert reopened.memory("inventory") == []
    assert reopened.memory("checkout", exclude_case_id=case["id"]) == []
    assert reopened.get(case["id"])["run_id"] == running["run_id"]
    amended = reopened.review(
        case["id"], ReviewInput(confirmed_cause="Cause remains unknown", outcome="unresolved", notes="Amended review")
    )
    assert amended["id"] == review["id"]
    assert len(reopened.memory("checkout")) == 1
    assert reopened.memory("checkout")[0]["notes"] == "Amended review"
    reopened.feedback(case["id"], "The telemetry collector had stopped.")
    reopened.begin_run(case["id"])
    new_evidence = reopened.add_evidence(
        case["id"], "metrics", "query_metrics", {"service": "checkout"},
        {"status": "missing", "data": {"sample_count": 0}},
    )
    reopened.finish_run(
        case["id"], {"summary": "Missing collector", "diagnosis_status": "insufficient_evidence"}
    )
    next_review = reopened.review(
        case["id"], ReviewInput(confirmed_cause="Telemetry collector stopped", outcome="unresolved")
    )
    assert next_review["revision"] == 1
    assert next_review["evidence_ids"] == [new_evidence["id"]]
    assert next_review["id"] != review["id"]


@pytest.mark.asyncio
async def test_timeout_finishes_case_without_claiming_diagnosis(tmp_path):
    class SlowGateway(UnavailableGateway):
        def __getattr__(self, operation):
            inner = super().__getattr__(operation)

            async def delayed(**query):
                await asyncio.sleep(0.1)
                return await inner(**query)
            return delayed

    store = CaseStore(tmp_path / "cases.sqlite3")
    case = new_case(store)
    result = await investigate(
        store, case["id"], SERVICE_URLS, gateway=SlowGateway(), timeout=0.01
    )
    assert result["status"] == "unresolved"
    assert result["error"] == "Investigation time budget exhausted"
    assert result["report"]["diagnosis_status"] == "insufficient_evidence"
    assert result["report"]["proposed_action"] is None
    assert store.get(case["id"])["status"] != "investigating"

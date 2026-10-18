"""Operator API lifecycle checks without credentials, sockets, or LLM calls."""

import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest

from incident_agent import api


@asynccontextmanager
async def operator_client(app):
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://operator") as client:
            yield client


@pytest.fixture
def case_payload():
    return {
        "service": "checkout",
        "environment": "local",
        "summary": "Checkout latency increased",
        "start": 1000,
        "end": 1060,
        "incident_start": 1030,
        "mode": "replay",
    }


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/api/cases/unknown", None),
        ("POST", "/api/cases/unknown/investigate", None),
        ("POST", "/api/cases/unknown/feedback", {"text": "Check inventory"}),
        ("POST", "/api/cases/unknown/review", {"outcome": "unresolved"}),
    ],
)
async def test_missing_cases_return_404(tmp_path, method, path, body):
    async with operator_client(api.create_app(tmp_path)) as client:
        response = await client.request(method, path, json=body)
        assert response.status_code == 404
        assert response.json()["detail"] == "Case not found"


@pytest.mark.parametrize(
    "invalid_window",
    [
        {"start": 1060, "end": 1000},
        {"start": 1000, "end": 1000},
        {"start": 1000, "end": 4601},
        {"incident_start": 999},
        {"incident_start": 1060},
    ],
)
async def test_invalid_windows_do_not_create_cases(tmp_path, case_payload, invalid_window):
    async with operator_client(api.create_app(tmp_path)) as client:
        response = await client.post("/api/cases", json={**case_payload, **invalid_window})
        assert response.status_code == 422
        assert (await client.get("/api/cases")).json() == []


async def test_live_mode_requires_configuration_but_replay_can_be_created(tmp_path, case_payload, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    async with operator_client(api.create_app(tmp_path)) as client:
        assert (await client.get("/api/status")).json()["model_configured"] is False
        blocked = await client.post("/api/cases", json={**case_payload, "mode": "live"})
        assert blocked.status_code == 409
        assert "Configure" in blocked.json()["detail"]
        assert (await client.get("/api/cases")).json() == []
        created = await client.post("/api/cases", json=case_payload)
        assert created.status_code == 201
        assert created.json()["mode"] == "replay"
        assert created.json()["status"] == "draft"
        assert (await client.get(f"/api/cases/{created.json()['id']}")).json() == created.json()


async def test_feedback_invalidates_findings_and_unknown_cause_review_persists(tmp_path, case_payload):
    app = api.create_app(tmp_path)
    async with operator_client(app) as client:
        case_id = (await client.post("/api/cases", json=case_payload)).json()["id"]
        assert (await client.post(f"/api/cases/{case_id}/review", json={"outcome": "resolved"})).status_code == 409
        store = app.state.store
        store.begin_run(case_id)
        store.finish_run(case_id, {
            "summary": "A capacity change is a candidate",
            "diagnosis_status": "insufficient_evidence",
        })
        assert (await client.get("/api/memory")).json() == []
        correction = "The deployment affected staging; this incident is local."
        response = await client.post(f"/api/cases/{case_id}/feedback", json={"text": correction})
        assert response.status_code == 200
        revised = response.json()
        assert revised["report"] is None
        assert revised["status"] == "draft"
        assert revised["revision"] == 1
        assert revised["feedback"][-1]["text"] == correction
        assert (await client.post(f"/api/cases/{case_id}/review", json={"outcome": "resolved"})).status_code == 409

        store.begin_run(case_id)
        evidence = store.add_evidence(case_id, "metrics", "query_metrics", {"service": "checkout"}, {
            "status": "missing", "data": {"sample_count": 0},
        })
        store.finish_run(case_id, {
            "summary": "Service recovered; available observations do not establish a cause",
            "diagnosis_status": "insufficient_evidence",
        })
        review = await client.post(f"/api/cases/{case_id}/review", json={
            "confirmed_cause": "", "outcome": "resolved", "notes": "Operator observed recovery",
        })
        assert review.status_code == 200
        entry = review.json()
        assert entry["confirmed_cause"] == ""
        assert entry["outcome"] == "resolved"
        assert entry["revision"] == 1
        assert entry["origin_mode"] == "replay"
        assert entry["evidence_ids"] == [evidence["id"]]
        assert (await client.get("/api/memory")).json() == [entry]

    # A new application and database connection retain both operator facts.
    async with operator_client(api.create_app(tmp_path)) as reopened:
        retained = (await reopened.get(f"/api/cases/{case_id}")).json()
        assert retained["feedback"][-1]["text"] == correction
        assert (await reopened.get("/api/memory")).json() == [entry]


@pytest.mark.asyncio
async def test_duplicate_investigation_and_mutations_are_blocked_while_running(tmp_path, case_payload, monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def slow_investigation(store, case_id, service_urls, *, started):
        assert started is True
        assert store.get(case_id)["status"] == "investigating"
        entered.set()
        await release.wait()
        store.finish_run(case_id, {
            "summary": "Additional telemetry is needed",
            "diagnosis_status": "insufficient_evidence",
        })
        finished.set()

    monkeypatch.setattr(api, "investigate", slow_investigation)
    app = api.create_app(tmp_path)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://operator") as client:
            created = await client.post("/api/cases", json=case_payload)
            case_id = created.json()["id"]
            first = await client.post(f"/api/cases/{case_id}/investigate")
            assert first.status_code == 202
            assert first.json()["status"] == "investigating"
            await asyncio.wait_for(entered.wait(), timeout=1)
            assert (await client.post(f"/api/cases/{case_id}/investigate")).status_code == 409
            assert (await client.post(f"/api/cases/{case_id}/feedback", json={"text": "Corrected context"})).status_code == 409
            assert (await client.post(f"/api/cases/{case_id}/review", json={"outcome": "unresolved"})).status_code == 409
            release.set()
            await asyncio.wait_for(finished.wait(), timeout=1)
            completed = (await client.get(f"/api/cases/{case_id}")).json()
            assert completed["status"] == "unresolved"
            assert completed["report"]["diagnosis_status"] == "insufficient_evidence"
            assert (await client.post(f"/api/cases/{case_id}/feedback", json={"text": "Corrected context"})).status_code == 200


async def test_operator_api_exposes_no_action_execution_or_fault_controls(tmp_path, case_payload):
    async with operator_client(api.create_app(tmp_path)) as client:
        case_id = (await client.post("/api/cases", json=case_payload)).json()["id"]
        for path in (
            f"/api/cases/{case_id}/execute",
            f"/api/cases/{case_id}/action",
            f"/api/cases/{case_id}/approve",
            "/admin/config",
        ):
            assert (await client.post(path, json={"worker_slots": 4})).status_code == 404
        assert (await client.get(f"/api/cases/{case_id}")).json()["status"] == "draft"

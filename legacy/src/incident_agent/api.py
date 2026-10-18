"""Local operator API. Investigation agents have no write/action endpoint tools."""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import google.adk
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .models import CaseInput, FeedbackInput, ReviewInput
from .storage import CaseBusy, CaseStore
from .workflow import investigate, model_configured


def service_urls():
    return {
        "checkout": os.getenv("SRE_CHECKOUT_URL", "http://127.0.0.1:8101"),
        "inventory": os.getenv("SRE_INVENTORY_URL", "http://127.0.0.1:8102"),
    }


def create_app(data_dir=None):
    directory = Path(data_dir or os.getenv("SRE_DATA_DIR", "data"))
    store = CaseStore(directory / "cases.sqlite")
    jobs = set()

    @asynccontextmanager
    async def lifespan(app):
        store.interrupt_stale_runs()
        yield
        for task in jobs:
            task.cancel()
        if jobs:
            await asyncio.gather(*jobs, return_exceptions=True)

    app = FastAPI(title="SRE incident investigation", lifespan=lifespan)
    app.state.store = store
    static = Path(__file__).parent / "static"
    if static.exists():
        app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    async def index():
        return FileResponse(static / "index.html")

    @app.get("/api/status")
    async def status():
        return {"model_configured": model_configured(), "model": os.getenv("SRE_MODEL", "gemini-3.8-flash"),
                "demo_services": service_urls(), "stage": "1", "adk_version": google.adk.__version__}

    @app.get("/api/demo/alert")
    async def demo_alert():
        path = directory / "demo-alert.json"
        if not path.exists():
            raise HTTPException(404, "Generate demo traffic first: python scripts/demo_scenario.py")
        return json.loads(path.read_text())

    @app.get("/api/cases")
    async def list_cases():
        return store.list()

    @app.post("/api/cases", status_code=201)
    async def create_case(case: CaseInput):
        if case.mode == "live" and not model_configured():
            raise HTTPException(409, "Configure GOOGLE_API_KEY or Vertex AI before choosing live mode")
        return store.create(case)

    def get_case(case_id):
        try:
            return store.get(case_id)
        except KeyError:
            raise HTTPException(404, "Case not found") from None

    @app.get("/api/cases/{case_id}")
    async def read_case(case_id: str):
        return get_case(case_id)

    @app.post("/api/cases/{case_id}/investigate", status_code=202)
    async def start_investigation(case_id: str):
        get_case(case_id)
        if len(jobs) >= 4:
            raise HTTPException(429, "Four investigations are already active")
        try:
            case = store.begin_run(case_id)
        except CaseBusy as exc:
            raise HTTPException(409, str(exc)) from exc
        task = asyncio.create_task(investigate(store, case_id, service_urls(), started=True))
        jobs.add(task)
        task.add_done_callback(jobs.discard)
        return {"id": case_id, "status": case["status"]}

    @app.post("/api/cases/{case_id}/feedback")
    async def feedback(case_id: str, item: FeedbackInput):
        get_case(case_id)
        try:
            return store.feedback(case_id, item.text)
        except CaseBusy as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/cases/{case_id}/review")
    async def review(case_id: str, item: ReviewInput):
        get_case(case_id)
        try:
            return store.review(case_id, item)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/memory")
    async def memory():
        return store.memory(limit=100)

    return app


app = create_app()

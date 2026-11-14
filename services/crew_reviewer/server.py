"""A2A server for the CrewAI review board (a2a-sdk 0.3.x API, as installed by crewai[a2a])."""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message

import crew


class CrewReviewerExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        text = context.get_user_input()
        answer = await asyncio.to_thread(crew.review, text)  # errors propagate: the caller sees a failed call
        await event_queue.enqueue_event(
            new_agent_text_message(json.dumps(answer), context.context_id, context.task_id))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel is not supported")


def build_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="review_incident_report", name="Review incident report",
        description="Audits an incident root-cause report against its evidence and reviews the proposed "
                    "remediation. Replies with a JSON object {verdict, issues, follow_up, risk, risk_notes}.",
        tags=["sre", "incident", "review", "crewai"],
        examples=["Report: ... Automatic checks: passed. Findings: ... Evidence: ..."])
    return AgentCard(
        name="crew_reviewer",
        description="CrewAI review board (Evidence Auditor + Remediation Risk Reviewer) for SRE incident reports.",
        url=url, version="0.1.0", capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"], default_output_modes=["text/plain"], skills=[skill])


def build_app(url: str):
    handler = DefaultRequestHandler(agent_executor=CrewReviewerExecutor(), task_store=InMemoryTaskStore())
    return A2AStarletteApplication(agent_card=build_card(url), http_handler=handler).build()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    public_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    uvicorn.run(build_app(os.getenv("CREW_REVIEWER_URL", f"http://{public_host}:{args.port}/")),
                host=args.host, port=args.port, log_level="warning")

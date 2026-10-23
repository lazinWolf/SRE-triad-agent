"""Fake telemetry backends and a scripted LLM, so the whole graph runs offline."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import PrivateAttr

from sre_triage.config import Settings


class World:
    """A demo shop where cart's CPU limit was cut; checkout is slow until the limit is restored."""

    def __init__(self, flagd_file: Path, faulty: bool = True):
        self.flagd_file = flagd_file
        self.cpu = {"cart": 0.02 if faulty else 1.0}
        self.restarts: list[str] = []

    @property
    def broken(self) -> bool:
        return self.cpu.get("cart", 1.0) < 0.5


class FakePrometheus:
    def __init__(self, world: World):
        self.world = world

    def service_red(self, service, start, end):
        baseline = abs((end - start) - 300) < 1
        slow = self.world.broken and not baseline and service in ("checkout", "cart", "frontend")
        return {"requests_per_s": 5.0, "error_ratio": 0.0, "p50_ms": 30.0 if slow else 10.0,
                "p95_ms": 400.0 if slow else 50.0}

    def container(self, name, start, end):
        throttled = self.world.broken and name == "cart" and abs((end - start) - 300) >= 1
        return {"cpu_cores_used": 0.02 if throttled else 0.01, "cpu_throttled_ratio": 0.4 if throttled else 0.0,
                "memory_mb": 50.0, "memory_limit_mb": 160.0}


class FakeJaeger:
    def search(self, service, start, end, **kw):
        return []


class FakeLogs:
    def search(self, start, end, **kw):
        return {"total": 0, "by_service": {}, "messages": []}


class FakeDocker:
    def __init__(self, world: World):
        self.world = world

    def inspect(self, name):
        return {"cpu_limit": self.world.cpu.get(name, 1.0), "memory_limit_mb": None, "status": "running",
                "restart_count": 0, "oom_killed": False, "started_at": ""}

    def events(self, since, until, names=None):
        return [{"time": since + 290, "container": "cart", "action": "update"}] if self.world.broken else []

    def set_cpus(self, name, cpus):
        self.world.cpu[name] = cpus

    def restart(self, name):
        self.world.restarts.append(name)


def fake_backends(world: World):
    return SimpleNamespace(prometheus=FakePrometheus(world), jaeger=FakeJaeger(), logs=FakeLogs(),
                           docker=FakeDocker(world), flagd_file=world.flagd_file)


def write_flags(path: Path) -> Path:
    path.write_text(json.dumps({"flags": {
        "paymentFailure": {"defaultVariant": "off", "variants": {"off": 0, "50%": 0.5}, "state": "ENABLED"},
        "emitRawPii": {"defaultVariant": "off", "variants": {"off": False, "on": True}, "state": "ENABLED"}}}))
    return path


def settings_for(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, verify_wait_seconds=0, agent_steps=3)


# ---------------------------------------------------------------------------- scripted model
Script = Callable[[LlmRequest], list[types.Part]]


def call(name: str, **args) -> list[types.Part]:
    return [types.Part(function_call=types.FunctionCall(name=name, args=args))]


def role_of(request: LlmRequest) -> str:
    system = str(request.config.system_instruction or "") if request.config else ""
    for marker, role in (("Coordinator of a team", "plan"), ("Combine the specialists", "report"),
                         ("You are the Critic", "critic"), ("Metrics specialist", "metrics"),
                         ("Traces specialist", "traces"), ("Logs & Changes", "logs_changes")):
        if marker in system:
            return role
    return "solo"


def responses(request: LlmRequest) -> list[dict]:
    return [p.function_response.response for c in request.contents for p in c.parts or []
            if p.function_response and p.function_response.response]


def user_text(request: LlmRequest) -> str:
    return "\n".join(p.text for c in request.contents if c.role == "user" for p in c.parts or [] if p.text)


def cited(request: LlmRequest) -> list[str]:
    return sorted(set(re.findall(r"\b[MTLS]\d+\b", user_text(request))))


class ScriptedModel(BaseLlm):
    """Deterministic ADK model: each role follows a script; ADK really executes the tool calls."""

    _scripts: dict[str, Script] = PrivateAttr()
    _prompts: list = PrivateAttr(default_factory=list)

    def __init__(self, scripts: dict[str, Script] | None = None):
        super().__init__(model="scripted")
        self._scripts = {**DEFAULTS, **(scripts or {})}

    @property
    def prompts(self) -> list[tuple[str, str]]:
        return self._prompts

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False
                                     ) -> AsyncGenerator[LlmResponse, None]:
        role = role_of(llm_request)
        self._prompts.append((role, user_text(llm_request)))
        yield LlmResponse(content=types.Content(role="model", parts=self._scripts[role](llm_request)),
                          usage_metadata=types.GenerateContentResponseUsageMetadata(
                              prompt_token_count=100, candidates_token_count=20))


def specialist(tool: str, args: dict, finding: dict) -> Script:
    def script(request: LlmRequest) -> list[types.Part]:
        ids = [r["evidence_id"] for r in responses(request) if r.get("evidence_id")]
        if not ids:
            return [types.Part(text="Checking.")] + call(tool, **args)
        return call("submit_finding", **finding, evidence_ids=ids)
    return script


def submit(name: str, payload: dict | Callable[[LlmRequest], dict]) -> Script:
    return lambda request: call(name, **(payload(request) if callable(payload) else payload))


def good_report(request: LlmRequest) -> dict:
    ids = cited(request)
    return {"status": "diagnosed", "summary": "cart CPU limit cut to 0.02 throttles cart; checkout waits on cart",
            "root_cause_service": "cart", "category": "cpu_limit", "mechanism": "CFS throttling", "confidence": 0.9,
            "evidence_ids": ids[:2],
            "hypotheses": [{"statement": "cart CPU throttling", "status": "supported", "evidence_ids": ids[:1]},
                           {"statement": "payment errors", "status": "rejected", "evidence_ids": []}],
            "remediation_action": {"action": "set_cpu_limit", "service": "cart", "cpus": 1.0}}


DEFAULTS: dict[str, Script] = {
    "plan": submit("submit_plan", {"hypotheses": ["cart CPU limit", "payment errors"],
                                   "tasks": [{"agent": "metrics", "question": "Is cart resource-bound?"},
                                             {"agent": "logs_changes", "question": "What changed?"}]}),
    "metrics": specialist("container_resources", {"service": "cart"},
                          {"summary": "cart throttled 40%", "suspect_service": "cart", "category": "cpu_limit",
                           "confidence": 0.8}),
    "traces": specialist("slow_traces", {"service": "checkout"},
                         {"summary": "no traces", "suspect_service": "", "category": "unknown", "confidence": 0.2}),
    "logs_changes": specialist("recent_changes", {},
                               {"summary": "cart CPU limit 1.0 -> 0.02", "suspect_service": "cart",
                                "category": "cpu_limit", "confidence": 0.8}),
    "solo": specialist("container_resources", {"service": "cart"},
                       {"summary": "cart throttled", "suspect_service": "cart", "category": "cpu_limit", "confidence": 0.7}),
    "report": submit("submit_report", good_report),
    "critic": submit("submit_review", {"verdict": "accept", "issues": [], "follow_up": []}),
}

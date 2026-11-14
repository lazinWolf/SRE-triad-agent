"""CrewAI review board: Evidence Auditor -> Remediation Risk Reviewer (two sequential tasks, two LLM calls)."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Environment must be prepared before crewai is imported (telemetry switches are read at import time).
load_dotenv(Path(__file__).resolve().parent / "../../.env")  # the project's .env; absent inside the container
if os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
if not os.getenv("LANGFUSE_PUBLIC_KEY"):
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import LLM, Agent, Crew, Process, Task  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

DEFAULT_MODEL = "gemini/gemini-3.5-flash"
MODEL_SECONDS, REVIEW_SECONDS = 100, 240  # longest wait for one model, and for the whole review
PROVIDER_KEYS = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY",
                 "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def models() -> list[str]:
    """SRE_MODEL then SRE_FALLBACK_MODELS (the same chain as the ADK side), without models whose key is missing."""
    names = [os.getenv("SRE_MODEL") or DEFAULT_MODEL, *(os.getenv("SRE_FALLBACK_MODELS") or "").split(",")]
    names = list(dict.fromkeys(n.strip() for n in names if n.strip()))
    usable = [n for n in names if not (key := PROVIDER_KEYS.get(n.split("/", 1)[0])) or os.getenv(key)]
    return usable or names[:1]


class FollowUp(BaseModel):
    agent: Literal["metrics", "traces", "logs_changes", "infra"]
    question: str


class Review(BaseModel):
    """The contract the ADK caller validates."""
    verdict: Literal["accept", "revise"]
    issues: list[str] = Field(default_factory=list)
    follow_up: list[FollowUp] = Field(default_factory=list, max_length=2)
    risk: Literal["low", "medium", "high"] = "low"
    risk_notes: str = ""


def build_crew(text: str, model: str | None = None) -> Crew:
    model = model or models()[0]
    # A hung provider must not block the review: the caller falls back to the next model, then to the local critic.
    limits = {"timeout": 60, "max_retries": 0}
    llm = LLM(model=model, **limits) if model.startswith("gemini/") else LLM(model=model, temperature=0.1, **limits)

    auditor = Agent(
        role="Evidence Auditor",
        goal="Find concrete defects in an incident root-cause report by checking it against its own evidence.",
        backstory=("A skeptical senior SRE. You verify that every claim cites evidence IDs that really support it, "
                   "that the named root cause is the origin of the failure and not a victim of upstream "
                   "propagation, and that competing hypotheses and decoy changes were ruled out. An "
                   "'inconclusive' report on genuinely weak evidence is acceptable; you only flag concrete problems."),
        llm=llm, allow_delegation=False, max_iter=3, max_retry_limit=0, max_execution_time=120, verbose=False)
    risk_reviewer = Agent(
        role="Remediation Risk Reviewer",
        goal="Judge the proposed remediation and issue the final review verdict as JSON.",
        backstory=("A cautious change-advisory reviewer. You judge whether the remediation is the smallest fix for "
                   "the identified cause and how risky it is (low if none is proposed). You combine the auditor's "
                   "findings with your own judgement and ask for a revision only for a concrete problem."),
        llm=llm, allow_delegation=False, max_iter=3, max_retry_limit=0, max_execution_time=120, verbose=False)

    audit = Task(
        description=("Audit this incident report. List concrete problems only: claims without supporting evidence "
                     "IDs, a root cause that is a downstream victim rather than the origin, hypotheses or decoy "
                     "changes not ruled out. If everything is sound, say so.\n\n=== INPUT ===\n" + text),
        expected_output="A short bullet list of concrete audit findings, or 'no concrete problems'.",
        agent=auditor)
    final = Task(
        description=("Using the audit and the original input below, decide the final review. verdict is 'revise' "
                     "only for a concrete problem, otherwise 'accept'. issues lists the concrete problems. "
                     "follow_up has at most 2 items, each {agent: metrics|traces|logs_changes|infra, question} "
                     "naming which specialist should look again and what to check (empty if accept). risk is "
                     "low|medium|high for the proposed remediation (low if none). risk_notes is one sentence.\n\n"
                     "=== INPUT ===\n" + text),
        expected_output="The review as a JSON object matching the schema.",
        agent=risk_reviewer, context=[audit], output_pydantic=Review)
    return Crew(agents=[auditor, risk_reviewer], tasks=[audit, final], process=Process.sequential,
                verbose=False, memory=False, planning=False)


FAKE_REVIEW = {"verdict": "accept", "issues": [], "follow_up": [], "risk": "low",
               "risk_notes": "Canned response (CREW_REVIEWER_FAKE=1); no LLM was called."}


def review(text: str) -> dict:
    """Run the crew on one report. The answer also carries the crew's own LLM usage, so the caller can count it."""
    if os.getenv("CREW_REVIEWER_FAKE") == "1":
        print(f"[crew_reviewer] fake review of {len(text)} chars", flush=True)
        return dict(FAKE_REVIEW)
    result, model, last, deadline = None, "", None, time.time() + REVIEW_SECONDS
    for model in models():  # a rate-limited, failing or hung model is skipped: the next one in the chain reviews
        left = deadline - time.time()
        if left < 20:
            break
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            result = pool.submit(lambda m=model: build_crew(text, m).kickoff()).result(timeout=min(MODEL_SECONDS, left))
            break
        except Exception as exc:  # noqa: BLE001 - provider errors come in many types; a timeout is one of them
            last = exc
            print(f"[crew_reviewer] {model} failed: {type(exc).__name__} {str(exc)[:160]}", flush=True)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)  # a hung call is left behind, not waited for
    if result is None:
        raise RuntimeError(f"no model could review: {str(last)[:300]}")
    used = getattr(result, "token_usage", None)
    usage = {"model": model, "llm_requests": getattr(used, "successful_requests", 0) or 0,
             "prompt_tokens": getattr(used, "prompt_tokens", 0) or 0,
             "completion_tokens": getattr(used, "completion_tokens", 0) or 0}
    print(f"[crew_reviewer] input {len(text)} chars, usage {usage}", flush=True)
    parsed = result.pydantic or Review.model_validate_json(result.raw)
    if not isinstance(parsed, Review):
        parsed = Review.model_validate(parsed)
    return {**parsed.model_dump(), "usage": usage}

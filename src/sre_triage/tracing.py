"""LLM observability with Langfuse.

ADK already creates an OpenTelemetry span for every workflow node, agent, model call (with token usage) and
tool call. The Langfuse SDK exports them. This module adds only what ADK cannot know: one root observation per
run of a case, the case as the Langfuse session on every span (first run plus every resume after a human
decision), which model of the fallback chain answered and what it cost, and scores.

    LANGFUSE_PUBLIC_KEY=pk-...  LANGFUSE_SECRET_KEY=sk-...  LANGFUSE_BASE_URL=https://cloud.langfuse.com

Without the keys nothing is exported and the app works the same.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

_client = None


def enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def _keep(span) -> bool:
    """ADK's `call_llm` and tool spans carry the content and tokens; skip the copies and transport spans under them."""
    scope = span.instrumentation_scope
    return not span.name.startswith(("generate_content", "a2a.")) and (scope is None or scope.name != "mcp-python-sdk")


def client():
    """The Langfuse client (it reads the LANGFUSE_* variables itself); None when no keys are configured."""
    global _client
    if _client is None and enabled():
        from langfuse import Langfuse

        _client = Langfuse(should_export_span=_keep)
    return _client


@contextmanager
def case_span(case_id: str, meta: dict[str, Any], resumed: bool = False) -> Iterator[Any]:
    """Root observation of one run of a case (None when tracing is off); set its output with `.update`."""
    langfuse = client()
    if langfuse is None:
        yield None
        return
    from langfuse import propagate_attributes

    tags = [str(t) for t in (meta.get("mode"), meta.get("source"), meta.get("scenario")) if t]
    with propagate_attributes(session_id=case_id, user_id="operator", tags=tags, metadata={
            "model": str(meta.get("model") or ""), "resumed": str(resumed).lower()}):
        with langfuse.start_as_current_observation(name="incident_triage", input=meta.get("symptom", "")) as span:
            yield span


def model_call(model: str, cost: float) -> None:
    """On the current model-call span: the model that really answered and its list price."""
    if _client is not None:
        _client.update_current_generation(model=model, cost_details={"total": cost})


def remote_call(name: str, model: str, sent: int, got: int, cost: float, input: Any, output: Any) -> None:
    """Model work done in another process (the CrewAI board over A2A), with the token usage it reported."""
    if _client is not None:
        _client.start_observation(name=name, as_type="generation", model=model, input=input, output=output,
                                  usage_details={"input": sent, "output": got}, cost_details={"total": cost}).end()


def score(trace_id: str, name: str, value: float | str, comment: str = "") -> bool:
    """Attach a score (an evaluation result, the case outcome, the operator's rating) to a trace."""
    langfuse = client()
    if langfuse is None or not trace_id:
        return False
    langfuse.create_score(trace_id=trace_id, name=name, value=value, comment=comment or None,
                          data_type="CATEGORICAL" if isinstance(value, str) else "NUMERIC")
    return True


def flush() -> None:
    if _client is not None:
        _client.flush()


def trace_url(trace_id: str) -> str | None:
    langfuse = client()
    return langfuse.get_trace_url(trace_id=trace_id) if langfuse else None

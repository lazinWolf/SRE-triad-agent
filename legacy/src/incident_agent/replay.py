"""Credential-free scripted model for ADK integration checks, never an LLM evaluation."""

import json
from collections.abc import AsyncGenerator, Callable
from typing import Any

from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import PrivateAttr


class ScriptedReplayModel(BaseLlm):
    role: str
    _source: Callable = PrivateAttr()
    _turn: int = PrivateAttr(default=0)

    def __init__(self, role: str, source: Callable):
        super().__init__(model="scripted-adk-replay", role=role)
        self._source = source

    async def generate_content_async(self, llm_request: LlmRequest, stream=False) -> AsyncGenerator[LlmResponse, None]:
        case = self._source()
        evidence = [e for e in case["evidence"] if e["run_id"] == case["run_id"]]
        turn = self._turn
        self._turn += 1
        if self.role == "coordinator":
            decision = self._decision(case, evidence, turn)
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=json.dumps(decision))]))
            return

        calls = []
        window = {"start": case["start"], "end": case["end"]}
        if turn == 0:
            if self.role == "metrics":
                calls = [("query_metrics", {"service": s, **window}) for s in ("checkout", "inventory")]
            elif self.role == "traces":
                calls = [("find_traces", {"service": "checkout", **window})]
            else:
                calls = [("get_change_history", {"service": "inventory", **window}), ("inspect_runtime", {"service": "inventory"}), ("search_logs", {"service": "inventory", **window})]
        elif turn == 2 and self.role == "metrics":
            # The follow-up split comes from a retrieved audit timestamp, not the oracle.
            change = self._capacity_change(evidence)
            if change:
                point = float(change[1]["timestamp"])
                if case["start"] < point < case["end"]:
                    for service in ("checkout", "inventory"):
                        calls.extend([
                            ("query_metrics", {"service": service, "start": case["start"], "end": point}),
                            ("query_metrics", {"service": service, "start": point, "end": case["end"]}),
                        ])
        if calls:
            yield LlmResponse(content=types.Content(role="model", parts=[
                types.Part(function_call=types.FunctionCall(name=name, args=args)) for name, args in calls
            ]))
        else:
            own = [e["id"] for e in evidence if e["agent"] == self.role]
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=json.dumps({
                "mode": "scripted replay", "evidence_ids": own,
                "finding": "Collected measured telemetry; coordinator must check coverage and competing explanations.",
            }))]))

    @staticmethod
    def _capacity_change(evidence):
        for e in evidence:
            if e["tool"] != "get_change_history" or e["result"]["status"] != "ok":
                continue
            for change in e["result"]["data"].get("changes", []):
                before, after = change.get("before", {}), change.get("after", {})
                if isinstance(before, dict) and isinstance(after, dict) and "worker_slots" in before and "worker_slots" in after and after["worker_slots"] < before["worker_slots"]:
                    return e, change
        return None

    def _decision(self, case, evidence, turn):
        change = self._capacity_change(evidence)
        if turn == 0 and change and not case["feedback"]:
            return {"next_step": "follow_up", "reasoning": "An audited capacity reduction needs a before/after cross-check.",
                    "follow_ups": [{"agent": "metrics", "question": "Compare traffic, latency and queue wait before and after the inventory capacity change; check whether demand explains the delay."}], "report": None}
        ok = [e for e in evidence if e["result"]["status"] == "ok"]
        metrics = [e for e in ok if e["tool"] == "query_metrics" and e["query"]["service"] == "inventory"]
        before, after = None, None
        if change:
            point = float(change[1]["timestamp"])
            before = next((e for e in metrics if e["query"]["end"] == point), None)
            after = next((e for e in metrics if e["query"]["start"] == point), None)
        enough = bool(before and after and before["result"]["data"].get("sample_count", 0) >= 5 and after["result"]["data"].get("sample_count", 0) >= 5)
        supported = bool(enough and after["result"]["data"].get("p95_queue_wait_ms", 0) > max(30, before["result"]["data"].get("p95_queue_wait_ms", 0) * 2))
        traces = [e for e in ok if e["tool"] == "find_traces"]
        # Require real downstream timing evidence; a change plus metrics is insufficient for checkout localization.
        has_downstream = any(
            t.get("service") == "checkout" and "inventory" in str(t.get("name", ""))
            for e in traces for t in e["result"]["data"].get("traces", [])
        )
        supported = supported and (case["service"] == "inventory" or has_downstream)
        limitations = ["Scripted replay uses fixed tool choices and rules; this does not evaluate autonomous LLM reasoning.",
                       "No remediation was executed and recovery has not been verified.",
                       "Additional causes and changing demand have not been ruled out."]
        missing = [e["id"] for e in evidence if e["result"]["status"] != "ok"]
        if missing:
            limitations.append("Some telemetry is missing or unavailable: " + ", ".join(missing))
        if case["feedback"]:
            # A fixed script cannot interpret arbitrary operator corrections honestly.
            # Defer the corrected conclusion to live reasoning or operator review.
            supported = False
            limitations.append("Operator corrections were received. Scripted replay cannot interpret their meaning, so it withholds its prior conclusion; use live mode to investigate the corrected context.")
        refs = [change[0]["id"], before["id"], after["id"]] + [e["id"] for e in traces] if supported else []
        report = {
            "summary": "Measured inventory queueing increased after worker capacity was reduced. Review restoring the prior capacity." if supported else "The collected evidence does not establish a cause. Check telemetry coverage and compare affected requests with a healthy baseline.",
            "diagnosis_status": "supported" if supported else "insufficient_evidence",
            "hypotheses": [{"cause": "Reduced inventory worker capacity causes request queueing", "service": "inventory", "status": "supported" if supported else "candidate", "supporting_evidence": refs, "contradicting_evidence": []}],
            "next_checks": ["Compare offered request volume, processing duration, and queue wait under the same workload.", "Observe checkout latency after any operator-approved mitigation."],
            "proposed_action": {"service": "inventory", "description": f"Restore worker_slots from {change[1]['after']['worker_slots']} to {change[1]['before']['worker_slots']} through the operator configuration control.", "reason": "Audited capacity reduction aligns with increased measured queue delay.", "rollback": f"Return worker_slots to {change[1]['after']['worker_slots']} if the change introduces other problems.", "evidence_ids": refs} if supported else None,
            "limitations": limitations, "scripted_mode": True,
        }
        return {"next_step": "report", "reasoning": "Report measured findings with explicit uncertainty and replay limitations.", "follow_ups": [], "report": report}

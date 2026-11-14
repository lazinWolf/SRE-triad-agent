# crew_reviewer

A CrewAI review board (Evidence Auditor -> Remediation Risk Reviewer) served over the A2A protocol,
so the ADK incident-triage app can call it through `RemoteA2aAgent`. Separate uv project (CrewAI cannot share the main venv).

    uv sync && uv run python server.py --host 127.0.0.1 --port 8002     # card: /.well-known/agent-card.json
    CREW_REVIEWER_FAKE=1 uv run python server.py                          # canned answer, no LLM calls

Env: `SRE_MODEL` (LiteLLM id, default `gemini/gemini-3.5-flash`), keys from the repo-root `.env`,
`CREW_REVIEWER_URL` (public URL put in the card; set it when running in Docker).

Input: one text message (report, automatic checks, findings, evidence). Reply: one JSON text message:
`{"verdict": "accept"|"revise", "issues": [str], "follow_up": [{"agent": "metrics"|"traces"|"logs_changes"|"infra",
"question": str}] (max 2), "risk": "low"|"medium"|"high", "risk_notes": str}`. LLM errors fail the A2A call.

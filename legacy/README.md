# SRE incident investigation agent

A Google ADK prototype that investigates a running checkout/inventory application, records telemetry evidence, requests targeted follow-ups, and produces an operator-reviewable report. It supports case corrections and persistent, explicitly reviewed incident memory.

The project advances through **balanced stages**: application integration, agents, memory, operator feedback, and evaluation develop together. See [the project map](docs/PROJECT_MAP.md) for scope and exit conditions.

## Current status

Stage 1 has a working application, read-only telemetry tools, three ADK investigators, a coordinator, bounded follow-up, a persistent evidence ledger, operator UI, and reviewed memory. Real HTTP traffic and scripted ADK replay have been exercised end to end.

**Live LLM reasoning has not been evaluated yet.** It requires a configured model credential. Replay uses fixed tool choices and simple rules inside actual ADK agents; it validates orchestration and integration, not reasoning quality or diagnosis accuracy. Replay visibly identifies itself in the UI, reports, and saved memory.

Actions are recommendations. Operational execution, automated recovery verification, additional incident scenarios, and a measured single-agent comparison belong to later development.

## Run locally

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/). Docker is not required for this first local application.

From the project directory:

```bash
uv sync --extra dev --locked
uv run python scripts/run_local.py
```

Open [the operator interface](http://127.0.0.1:8080). The launcher also starts checkout on port 8101 and inventory on port 8102. All services bind to loopback. Stop the launcher with Ctrl+C to stop all three.

In another terminal, generate an incident:

```bash
uv run python scripts/demo_scenario.py
```

The scenario sends real HTTP checkout requests under a healthy inventory capacity, reduces `worker_slots` through the operator-only configuration endpoint, and sends more requests. Capacity reduction produces actual waiting on an asynchronous worker pool. This is worker exhaustion, not a simulation of CPU throttling. The processing work is an intentional timed delay in a small service, not a full commerce workload.

In the UI:

1. Select **Use latest demo alert**, then **Create case**.
2. Choose **Scripted ADK replay** for a credential-free integration walkthrough and select **Investigate**.
3. Inspect Findings, Evidence, and Activity. Evidence links open the exact returned tool record.
4. Add a correction in **Feedback & review**, then investigate again. The previous report is invalidated; evidence and events remain auditable.
5. Record an explicitly reviewed outcome to retain it in memory. A confirmed cause can remain unknown even if service has recovered.

Replay conservatively withholds its earlier conclusion after any operator correction because a fixed script cannot interpret arbitrary feedback. Live mode is responsible for reasoning over corrected context.

The scenario leaves inventory at reduced capacity so the incident remains inspectable. Running the scenario again resets the baseline first. To restore the local demo manually:

```bash
curl -X POST http://127.0.0.1:8102/admin/config \
  -H 'Content-Type: application/json' -d '{"worker_slots":4}'
```

This command changes only the local demo. Agents have no tool for that endpoint. Restoring capacity alone is not a recovery measurement; send new traffic and observe its latency in a fresh window.

## Enable live model investigation

Copy `.env.example` to `.env`, set `GOOGLE_API_KEY` locally, and restart the launcher. Credentials are never entered into the dashboard. `.env` is ignored by version control. The configured model is controlled by `SRE_MODEL`; the example uses `gemini-3.8-flash`. See Google's [model lifecycle guidance](https://ai.google.dev/gemini-api/docs/deprecations) when choosing a model available to your account.

Live mode uses the same tool layer and bounded workflow as replay, with LLM investigators choosing queries and the coordinator choosing follow-ups. Local demo telemetry, case summaries, operator feedback, and retrieved incident records are included in the selected model's requests. Model access and quota still need to be checked in a real run. Vertex AI configuration is also supported through ADK's environment configuration, but has not been tested here.

## CLI

```bash
uv run incident-agent run data/demo-alert.json --mode replay --output data/latest-case.json
uv run incident-agent list
uv run incident-agent feedback CASE_ID "The previous change affected a different environment."
uv run incident-agent investigate CASE_ID
uv run incident-agent review CASE_ID --outcome unresolved --notes "Need more telemetry."
uv run incident-agent memory
```

For a live case, use `--mode live` with configured credentials. `--data-dir PATH` before the subcommand changes the ledger location. A case's historical window stays fixed across reruns; rerunning does not observe current recovery.

## Implementation map

```text
src/demo_app/                  Checkout, inventory, OTel spans, metrics and bounded observations
src/incident_agent/workflow.py ADK agent tree and conditional follow-up loop
src/incident_agent/tools.py    Case-bound tool access, budgets, caching and evidence capture
src/incident_agent/telemetry.py Read-only HTTP adapters and explicit missing-data handling
src/incident_agent/storage.py  SQLite cases, evidence, corrections and reviewed memory
src/incident_agent/replay.py   Clearly labeled scripted model for integration checks
src/incident_agent/api.py      Local operator API
src/incident_agent/static/     Browser interface
scripts/                      Local launcher and operator-only scenario
tests/                        Application, tool, workflow and API checks
docs/                         Balanced stages and architecture decisions
```

The prototype queries bounded local telemetry endpoints. It exposes Prometheus-format `/metrics` and exports real OpenTelemetry spans to a local ring buffer. A standalone Prometheus server, trace backend, log backend, Kubernetes connector, and MCP server are not connected yet. Replacing the adapters can preserve the case/evidence contracts.

`data/cases.sqlite` holds persistent cases and reviewed memory. `data/demo-alert.json` is agent-visible intake. `data/evaluation/demo-oracle.json` contains evaluation ground truth and is never read by the agent workflow or tools. Model agents have neither filesystem access nor a generic HTTP client.

## Checks

```bash
uv run pytest -q
```

Checks exercise actual worker queueing, trace propagation, telemetry coverage, allowlisted tools, ADK parallel investigation and targeted follow-up, missing telemetry, tool budgets, timeouts, evidence citation validation, correction invalidation, review persistence, and operator API behavior.

These checks do not measure live model reasoning, generalization, or production incident performance. Pinning ADK makes the initial workflow reproducible. ADK 2.11 deprecates `ParallelAgent`; the project records a graph-workflow migration checkpoint in [the architecture notes](docs/ARCHITECTURE.md).

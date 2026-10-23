# Cloud-native SRE incident triage agent

A team of LLM agents investigates incidents in the [OpenTelemetry Demo](https://opentelemetry.io/docs/demo/)
shop, using live metrics, traces, logs and configuration. It proposes a fix, applies the fix only after a
human approves it, checks that the system recovered, and learns from the outcome.

Built on **Google ADK**: every agent is an ADK `LlmAgent`, and the flow is an ADK `Workflow` graph.
Models are reached through ADK's **LiteLLM** integration, so any model LiteLLM supports works (Gemini,
Groq, OpenRouter, Ollama, …). Case state is an ADK session stored in **SQLite**. The UI is **Streamlit**.

## The workflow graph

```
START → detect → recall → plan → specialists → synthesize ─multi─▶ critic ─accept─▶ approval ✋ ──▶ decide
                           ▲     (parallel ADK        │                 │             │ skip         │
                           │      LlmAgents)          └──single──────────┼────────────▶│              │
                           ├───────────── revise ───────────────────────┘             ▼              │
                           │                                                        learn ◀─reject───┤
                           └───────────── feedback (operator rejected, with a note) ──────────────────┤
                                                                   remediate ◀─approve──────────────┘
                                                                       └─▶ verify ─▶ learn
```

`uv run sre-triage graph` prints the graph from the ADK `Workflow` definition (Mermaid), and the dashboard draws it live.

| Node | What it does |
|---|---|
| `detect` | Deterministic anomaly scan: RED metrics for every service, incident window vs the healthy baseline just before it |
| `recall` | Pulls similar past incidents from long-term memory |
| `plan` | The **Coordinator** LLM writes competing hypotheses and delegates 1-3 focused tasks |
| `specialists` | An ADK parallel-worker node: one **specialist** `LlmAgent` per task, all running concurrently. Each runs a ReAct tool loop over its own tools and ends by calling `submit_finding`, citing evidence IDs |
| `synthesize` | The Coordinator merges the findings into a root-cause report and at most one remediation. Code then checks that the cited evidence exists and that the action is on the allowlist |
| `critic` | A **Critic** LLM reviews the report. A failed check or a concrete objection sends follow-up tasks back to the specialists (once) |
| `approval` → `decide` | **Human in the loop**: the workflow pauses with an ADK `RequestInput`; the session is stored in SQLite, so the case can be approved later from the CLI or the dashboard. A rejection with feedback re-plans |
| `remediate` | Executes the approved action (set a CPU limit, turn a feature flag off, or restart a service) and records how to undo it |
| `verify` | Waits, then compares the alerting services with the baseline. If they have not recovered, the action is rolled back |
| `learn` | Writes the incident to long-term memory. Operator ratings (👍/👎) later change how much it is trusted |

### Agentic patterns used

| Pattern | Where |
|---|---|
| Multi-agent (orchestrator–workers) | Coordinator + Metrics / Traces / Logs-and-Changes specialists + Critic, each an ADK `LlmAgent` (`agents.py`, `prompts.py`) |
| Graph flow with routes and loops | ADK `Workflow` with routed edges (`ctx.route`), a parallel-worker fan-out and agents run as dynamic child nodes (`workflow.py`) |
| Tool use / ReAct loop | Seven read-only ADK function tools over Prometheus, Jaeger, OpenSearch, Docker and flagd (`tools.py`, `agents.py`) |
| Shared memory (blackboard) | ADK session state: findings and evidence accumulate across rounds and every agent reads them |
| Long-term memory | `memory.py`: seeded runbook incidents plus every diagnosed case, recalled by similarity |
| Reflection | Critic → revise → specialists loop (bounded by `max_revisions`) |
| Self-correction | Every agent ends with a `submit_*` tool that validates the answer (schema, real evidence IDs, allowlisted action); errors go back to the model |
| Guardrails | Evidence-citation checks, a remediation allowlist, a per-agent step budget, rollback when recovery fails |
| Human feedback | Approve / reject with feedback (re-plans), and 👍/👎 ratings stored in memory |
| Evaluation | `sre-triage eval`: injected faults with ground truth, multi-agent vs single-agent baseline |
| Model resilience | `FallbackLlm` (an ADK `BaseLlm`) over ADK `LiteLlm` models; rate-limited or overloaded models are skipped for a while |

## Quick start

```bash
cp .env.example .env            # add GOOGLE_API_KEY (and optionally GROQ_API_KEY / OPENROUTER_API_KEY)
infra/sre/demo.sh up            # OpenTelemetry Demo + overlay (shop on http://localhost:8090)
uv sync --extra dev
uv run sre-triage doctor        # every check should say ok
uv run sre-triage ui            # dashboard on http://localhost:8501
```

The shop's host port is 8090 so it does not clash with other tools on 8080. Set `SRE_SHOP_PORT` to change it.

### Try it

On the dashboard, pick a scenario in the sidebar, click **Inject**, wait about 90 seconds, then click
**Investigate**. You can watch the graph light up node by node, follow each agent's tool calls in **Activity**,
and approve or reject the proposed fix. **Reset** restores the standard configuration.

From the terminal:

```bash
uv run sre-triage demo cart_cpu_limit          # inject → investigate → auto-approve → verify → reset
uv run sre-triage investigate "Checkout latency is elevated" --since 180   # asks you to approve
uv run sre-triage eval                         # all scenarios, multi vs single agent (≈ 1 h)
uv run sre-triage graph                        # Mermaid diagram of the graph
uv run sre-triage scenarios | memory | health | reset
```

| Scenario | Fault | Expected diagnosis |
|---|---|---|
| `cart_cpu_limit` | cart CPU limit 1.0 → 0.02 | cart / cpu_limit → restore CPU limit |
| `cart_cpu_limit_decoys` | same, plus two unrelated changes | cart / cpu_limit (decoys rejected) |
| `payment_failure` | flag `paymentFailure` = 50% | payment / dependency_errors → flag off |
| `catalog_failure` | flag `productCatalogFailure` = on | product-catalog / dependency_errors → flag off |
| `healthy_control` | nothing | inconclusive, no action |

The agents never see `faults.py`. Faults are applied the same way a real operator would make changes
(container updates and flag edits), and the agents find them as configuration drift.

## Code map

```
src/sre_triage/
  workflow.py     ADK Workflow: nodes, routes, parallel specialists, approval RequestInput, SQLite sessions
  agents.py       ADK LlmAgents, their tools and validating submit_* tools, step limits, usage callbacks
  llm.py          FallbackLlm: ADK LiteLlm models with a fallback chain
  tools.py        agent tools + evidence recording; deterministic health scan
  telemetry.py    Prometheus / Jaeger / OpenSearch / Docker / flagd clients
  prompts.py      coordinator, specialist and critic prompts
  schemas.py      Pydantic answers: Plan, Finding, Report, Critique
  memory.py       long-term incident memory with feedback
  remediation.py  allowlist, execute, rollback
  cases.py        case records + live activity log (data/cases/)
  faults.py       fault scenarios and scoring (operator only)
  cli.py, ui.py   terminal and Streamlit front ends
tests/            the whole graph offline with fake telemetry and scripted agents
infra/sre/        OpenTelemetry Demo overlay (ports, CPU baseline, collector extras)
legacy/           earlier prototypes (v1, and the larger v2)
```

`uv run pytest` runs the offline tests and needs no Docker or API keys.

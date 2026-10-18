# Cloud-native SRE incident triage agent

A multi-agent incident investigator for the [OpenTelemetry Demo](https://opentelemetry.io/docs/demo/) running in Docker Compose, built on **Google ADK** with any tool-capable model through **LiteLLM**.

An alert (or an operator) opens a case. An **Incident Coordinator** plans the investigation and delegates targeted questions to three specialists running in parallel:

| Agent | Looks at |
|---|---|
| Metrics Diagnostic | RED metrics against baseline and healthy services, per-operation latency, CPU limit / throttling / memory / restarts, cache health |
| Trace Investigation | Slow and failed traces, per-service self time (incident vs baseline), single-trace breakdowns |
| Logs and Changes | Error logs, container lifecycle events (OOM, restarts, updates), configuration and feature-flag change history |
| Critic | Independent review of a proposed diagnosis: timing, magnitude, localisation, alternatives, evidence quality |

Before a diagnosis is accepted, the **Critic** tries to falsify it with its own read-only checks; a material challenge sends targeted follow-ups back to the specialists and the coordinator must revise (reflection loop). The coordinator keeps a **hypothesis board** (candidate → supported / weakened / rejected, each with cited evidence), requests follow-ups until the evidence discriminates or the budget runs out, and concludes with either a diagnosis or an explicit **insufficient-evidence** result. A diagnosis may propose one **allowlisted, reversible remediation**. Nothing changes until an operator approves it. After execution the system **verifies recovery** against the original symptom, and **reopens** the investigation if symptoms persist. Reviewed outcomes become **incident memory** for later cases.

Every tool call is recorded in an **evidence ledger**; reports may cite only evidence IDs from the current run, and deterministic validation rejects unsupported claims (e.g. "supported" needs evidence from two different tools) before anything is shown.

## Architecture

```
OpenTelemetry Demo (Docker Compose)          sre-triage
┌──────────────────────────────┐   read   ┌─────────────────────────────────────────────┐
│ frontend, checkout, cart, …  │◀────────▶│ tools.py  (case-bound, budgeted, ledgered)  │
│ otel-collector ─▶ Prometheus │          │   ▲                                         │
│               ─▶ Jaeger      │          │ agents/  Coordinator + 3 specialists (ADK)  │
│               ─▶ OpenSearch  │          │          single-agent baseline              │
│ flagd (feature flags)        │          │ orchestrator.py  plan → parallel → reconcile│
│ Docker engine (limits/events)│          │ validation.py    citations, support rules   │
└──────────────────────────────┘          │ ledger.py (SQLite)  cases, evidence, board, │
        ▲            ▲                    │                     actions, memory, jobs   │
        │ approved   │ operator-only      │ worker.py  change audit · alert rules · jobs│
        │ actions    │ fault injection    │ recovery.py  execute → verify → reopen      │
        └────────────┴────────────────────│ ui/  Streamlit dashboard                    │
                                          └─────────────────────────────────────────────┘
```

- **Agents never change anything.** Their tools are read-only. Remediation runs in `remediation.py` only after an operator approves the exact proposal, with a rollback computed beforehand.
- **Ground truth is unreachable.** `faults.py` (scenarios and expected answers) is used only by the operator's fault lab and the evaluation runner; a test enforces that no agent module imports it.
- **Faults look like real changes.** They are applied as container CPU-limit updates and feature-flag changes, recorded by the change audit like any other change, alongside decoy changes in the harder scenarios.
- **Deterministic code does the arithmetic**: baseline comparisons, report validation, action validation, recovery verdicts and alert rules.

## Agentic design

| Concept | Implementation |
|---|---|
| Multi-agent system | Coordinator, three specialists and a Critic, each an ADK `LlmAgent` with its own role prompt and role-scoped tools (`agents/team.py`); single-agent baseline for comparison |
| Orchestrator–worker | The coordinator decomposes the incident into targeted questions and routes each to a specialist (`orchestrator.py`) |
| Parallelisation | Specialists of a round run concurrently with a shared tool budget and a shared query cache |
| ReAct tool loop | Each agent reasons, calls tools, reads results and repeats until it submits |
| Plan → act → reconcile loop | Up to three rounds of delegation and reconciliation of the hypothesis board |
| Reflection | The Critic tries to falsify a diagnosis; a challenge sends requested checks back and the coordinator revises |
| Self-correction | Submissions are validated in code (citations, support rules, schema, allowlist); errors are returned to the model for a retry. Malformed tool calls rejected by a provider are fed back too |
| Bounded autonomy | Tool-call, model-call, round and wall-clock budgets; a per-agent step limit forces a submission |
| Tool use | 14 read-only tools over Prometheus, Jaeger, OpenSearch, Docker and the change audit; every call is ledgered evidence |
| Memory | Working: hypothesis board and case history. Shared: evidence ledger (blackboard). Long-term: operator-reviewed incidents and seeds, retrieved as leads |
| Feedback loops | Operator corrections invalidate the report and re-run; recovery checks reopen failed fixes; reviews become memory |
| Human in the loop | Remediation is only proposed; an operator approves the exact action with its rollback |
| Model resilience | LiteLLM model rotation across providers: exhausted or rate-limited models are cooled down and skipped |

## Quick start

Requirements: Docker with Compose, [uv](https://docs.astral.sh/uv/), about 6 GB free RAM and 8 GB disk.

```bash
infra/sre/demo.sh up              # OpenTelemetry Demo 3.1.0 + overlay (first run pulls ~6 GB of images)
cp .env.example .env              # set SRE_MODEL and the provider key
uv sync --extra dev
uv run sre-triage doctor          # backends, metric names, container coverage, model
scripts/start.sh                  # seeds memory, starts the worker and the dashboard (http://localhost:8501)
```

Or run the worker and dashboard as containers next to the demo: `docker compose up --build -d`.

### Choosing a model (LiteLLM)

Models are LiteLLM strings (`<provider>/<model>`). Keys in `.env`: `GOOGLE_API_KEY` (or `GEMINI_API_KEY`), `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.

```bash
uv run sre-triage models            # catalog, and which providers have keys
uv run sre-triage models --probe    # real test: tool call + structured submission per model
```

| Setting | Meaning |
|---|---|
| `SRE_MODEL` | default model for all agents (recommended: `gemini/gemini-3.5-flash`) |
| `SRE_COORDINATOR_MODEL`, `SRE_SPECIALIST_MODEL` | optional per-role overrides |
| `SRE_FALLBACK_MODELS` | comma-separated LiteLLM fallbacks used when the primary fails or is rate-limited |
| per case | the dashboard and `sre-triage open --model …` can pick a model for one case |

Probe results on 2026-10-18 (free tiers): Gemini 3.5 Flash / 3.1 Flash-Lite, Groq GPT-OSS 120B/20B and Qwen 3.8 27B, and OpenRouter Nemotron 3 Super/Ultra all pass. Groq's free tier allows 8,000 tokens per minute, too little for full investigations (fallback only). OpenRouter's free tier allows few requests per day. `gemini-3.8-flash` (preview) was too slow at test time.

Provider quirks handled in `agents/runtime.py`: earlier reasoning text is stripped for Groq (it rejects `reasoning_content` on input); tool calls rejected server-side for schema mismatches are fed back to the model for a corrected retry; structured arguments sent as JSON strings are accepted.

## Demo walkthrough

1. **Fault lab** → inject `cart_cpu_limit` (the cart container's CPU limit drops to 0.02).
2. Within a few minutes the alert rules open a case for checkout latency, and the worker starts investigating (or open one from **Incidents**).
3. Follow the case: the coordinator's plan, parallel specialist work, the critic's review and the hypothesis board changing across rounds (**Agent flow** tab draws the run as a graph). In this scenario, a valkey-cache hypothesis should be weakened by normal cache timings, and seeded memory suggests a cache incident too.
4. Review the proposed remediation (restore cart's CPU limit) with its evidence and rollback, then approve it.
5. The worker executes it, observes a fresh window and marks the case **recovered**, or **reopens** it if symptoms persist.
6. Record the reviewed outcome. It becomes incident memory for the next case.

CLI equivalents: `sre-triage inject|reset|scenarios`, `open SERVICE "symptom" --investigate`, `cases`, `feedback`, `decide ACTION_ID --approve`, `review`.

## Evaluation

```bash
uv run sre-triage eval                                   # all scenarios, multi-agent and single-agent
uv run sre-triage eval cart_cpu_limit_decoys healthy_control --repeats 3
```

For each scenario the runner resets the environment, accumulates a healthy baseline, applies decoys and the fault, and then investigates the **same case window with the same tools, model and budget** in both modes. It records diagnosis accuracy (service and mechanism), remediation correctness, false diagnoses on the no-fault control, unsupported claims, rejected hypotheses, time, tool calls, model calls and tokens. Results go to `data/eval/<id>/` and the **Evaluation** page. Expect roughly 10 minutes per scenario with a hosted model.

### Results (2026-10-18, Gemini 3.5 Flash free tier, same tools/model/budget per mode)

| Scenario | Multi-agent | Single-agent |
|---|---|---|
| `cart_cpu_limit` (live demo run, end to end incl. approval and recovery) | ✅ 74 s, 22 tools, 15 model calls | – |
| `cart_cpu_limit_decoys` | ✅ 80 s, decoy changes not blamed, exact trigger change | ✅ 99 s |
| `payment_failure` | ✅ 122 s | ✅ 102 s |
| `catalog_lock_contention` | ❌ (planning bug, fixed); rerun with Critic: right service, mechanism labelled `other` (schema fixed) | ❌ right database, wrong mechanism, 389 s |
| `healthy_control` | ✅ abstained (twice, incl. with Critic) | ❌ false diagnosis `checkout/other` |

Small sample, single runs; treat as a demonstration, not a statistically meaningful comparison. Use `--repeats` for more.

| Scenario | Difficulty | Mechanism |
|---|---|---|
| `cart_cpu_limit` | medium | cart CPU limit → checkout latency; cache is a red herring |
| `cart_cpu_limit_decoys` | hard | same, plus an unrelated flag change and another CPU-limit change |
| `currency_cpu_limit` | medium | currency CPU limit → storefront latency |
| `payment_failure`, `payment_unreachable` | easy | checkout errors from payment |
| `catalog_failure`, `catalog_lock_contention` | easy/medium | product catalog errors / latency |
| `ad_high_cpu` | medium | CPU saturation in ad |
| `email_memory_leak` | hard | memory growth and OOM restarts |
| `healthy_control` | control | nothing injected; correct answer is "insufficient evidence" |

Feature-flag names in the demo are descriptive (e.g. `paymentFailure`), which makes flag scenarios easier than resource scenarios; the CPU-limit scenarios with decoys are the meaningful test of reasoning.

## Repository map

```
src/sre_triage/
  config.py        settings and budgets          telemetry.py   Prometheus/Jaeger/OpenSearch/Docker adapters
  tools.py         agent tools + evidence ledger  analysis.py    deterministic comparisons
  agents/          prompts, ADK runtime, team     orchestrator.py investigation loop (multi/single)
  validation.py    output validation              ledger.py      SQLite store
  remediation.py   allowlist, execution, rollback recovery.py    execute → verify → reopen
  changes.py       config/flag change audit       alerts.py      alert rules → cases
  memory.py        reviewed incidents, seeds      faults.py      scenarios + ground truth (operator only)
  evaluation.py    scenario runner and scoring    worker.py      background worker
  ui/              Streamlit dashboard            cli.py         command line
infra/sre/         demo overlay (ports, CPU baselines, collector metrics, Go memory fix), demo.sh
seeds/             seeded historical incidents
tests/             unit + end-to-end tests with scripted models (no network)
legacy/            the earlier prototype, kept for reference
```

## Environment notes

- The overlay raises checkout and product-catalog to `GOMEMLIMIT=64MiB` / 80 MB. With upstream's 16 MiB limit their Go GC ran continuously (about one full CPU each at idle) and checkout p50 was about 9 s, which made every "healthy" baseline meaningless.
- `demo.sh up` restarts the collector once after start: its docker_stats receiver can miss containers that start while it initialises.
- CPU is reported in cores from `container_cpu_usage_nanoseconds_total`; docker_stats' utilisation ratio is not comparable with the CPU limit.

## Tests

```bash
uv run pytest -q
```

The tests drive real ADK agents with scripted models against fake telemetry. They cover parallel delegation, citation validation and self-correction, budgets, single-agent mode, feedback reruns, tools, change audit, approval/execution/recovery/reopen, alerts, scoring, UI rendering, and the isolation of ground truth from agents. They do not measure model reasoning quality; that is what `sre-triage eval` is for.

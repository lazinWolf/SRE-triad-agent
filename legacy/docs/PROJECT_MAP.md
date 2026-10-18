# Project map

The project helps an engineer investigate an application incident through evidence, revise explanations when new information arrives, and retain reviewed lessons. Google ADK, multiple agents, agentic patterns, tool use, memory, and feedback are required. The application design, agent boundaries, model, storage, and operational integration can evolve.

The first application is a compact local checkout/inventory system, selected to make the request path and controlled incidents inspectable. The prototype recommends actions; an operator performs operational changes separately.

## Balanced stages

Each stage advances the complete system across the same five tracks. These are progressively stronger versions of the product, rather than a sequence that finishes tools before starting agents or adds evaluation at the end.

| Stage | Environment and integration | Agents and reasoning | State and memory | Operator feedback | Evaluation |
| --- | --- | --- | --- | --- | --- |
| **1. Minimal complete prototype — in progress** | Local checkout/inventory application, real telemetry, controlled capacity fault, small bounded query tools | ADK coordinator and investigators; tool calls, evidence sharing, targeted follow-up, a structured report | Persistent case state and evidence ledger; retrieval of human-reviewed incidents | Create a case, inspect queries, correct context, rerun, and review an outcome | Repeatable local incident, missing-data behavior, orchestration checks, and a live model run when configured |
| **2. Realistic investigation system — planned** | Multiple faults, relevant change history, comparative windows and service context | Concurrent investigation where useful, competing hypotheses, an independent challenge to the leading explanation | More selective retrieval; hypotheses and rejection reasons retained with provenance | Corrections during a run, clearer reasoning changes, useful handoff to an engineer | Misleading changes, multiple possible causes, unseen variations, and the initial single-agent comparison |
| **3. Reliable incident assistant — planned** | Tool timeouts, partial telemetry, healthy/unhealthy comparisons; optional approved action integration | Query and time budgets, explicit escalation, stopping conditions, recovery checks | Context applicability and freshness checks; reviewed outcomes stay distinct from hypotheses | Exact proposals, scoped approvals if execution is included, and recovery feedback | Ambiguous incidents, unhelpful remediation, telemetry loss, and unsupported-claim checks |
| **4. Evaluated and presentable system — planned** | Reproducible setup, documented integrations, repeatable scenario runner | Coordination refined using measured failures and cost | Inspectable records and consistent review workflow | Coherent demonstration and usable investigation history | Held-out scenarios, same-budget single-agent comparison, recorded limitations, and reproducible results |

### Stage 1 exit conditions

- The application produces real requests and telemetry that can be inspected independently of the agents.
- A controlled incident can be reproduced, and its expected explanation is kept outside the agents' accessible tools.
- A live ADK investigation delegates work, gathers evidence through tools, and returns a report with traceable evidence references.
- A finding can trigger a focused follow-up question, rather than always running a fixed sequence of identical queries.
- An operator correction is saved, invalidates the prior report, and influences a subsequent investigation.
- A reviewed incident is persisted and can be retrieved in a later case without becoming proof of the current cause.
- A case with insufficient evidence can finish with uncertainty and useful next checks.
- Scripted replay is clearly labeled and counted only as an orchestration/tool integration check. It does not satisfy the live reasoning criteria.
- Setup, repeatable checks, and the remaining limitations are documented.

### Stage 2 exit conditions

- The system distinguishes several causes that share the same user-facing symptom.
- A plausible but unrelated change can be rejected through contradictory evidence.
- An investigator's finding changes another investigator's task, including an observable change of hypothesis.
- Past incidents improve the investigation without overriding contradictory current observations.
- The first comparison with a single agent uses the same tools, model, incident inputs, and investigation budget.

### Stage 3 exit conditions

- Missing tools or data produce explicit limitations and escalation instead of invented observations.
- Investigation budgets stop an unproductive run while preserving useful findings.
- Recovery is assessed against the original symptom over an observation period; mitigation and confirmed cause have separate status.
- If execution is included, approval names one exact action and its rollback. A failed recovery returns to investigation.
- Reliability scenarios demonstrate the boundaries of supported conclusions.

### Stage 4 exit conditions

- A new operator can reproduce the environment, scenarios, and demonstration from the documentation.
- Evaluation records supported diagnosis quality, unsupported claims, evidence quality, tool and model cost, investigation duration, and outcome verification where applicable.
- Results include held-out variations and a comparable single-agent baseline.
- Demonstrated capabilities and remaining limitations are explicit; a successful scripted replay is never presented as evidence of model intelligence.

## Current map

Stage 1's local foundation is implemented and verified: a running application, read-only evidence tools, ADK investigators and coordinator, persistent case state and reviewed memory, and an operator interface. All 62 checks pass, and the live HTTP/browser walkthrough is recorded in [validation notes](VALIDATION.md). Live model reasoning remains pending model credentials and a live evaluation run; scripted replay does not close that requirement.

| Track | Stage 1 scope | Later development |
| --- | --- | --- |
| Application | Checkout calls inventory; a local workload generates requests | More dependencies and failure modes as justified by evaluation |
| Integration | Small HTTP tools over local telemetry, runtime state, and relevant changes | Adapters for Prometheus, tracing, logging, runtime, and deployment systems |
| Agents | Coordinator plus metrics, trace, and change investigators with concrete investigative objectives | Role boundaries and delegation refined by observed performance |
| Case state | Incident identity, environment, fixed query window, evidence, events, hypotheses, corrections, report | Richer hypothesis transitions and investigation checkpoints |
| Memory | Reviewed cause, outcome, notes, source case, and evidence references | Relevance and freshness policies, review revision, and measured retrieval benefit |
| Feedback | Operator correction followed by a new run; reviewed outcome stored separately and labeled as operator reported | Feedback while running and action/recovery interaction |
| Interface | Native browser dashboard showing actual cases, reports, tool evidence, activity, and memory | Improve based on operator walkthroughs |
| Actions | Evidence-backed recommendations only | Specific approved execution and recovery checks, if retained in scope |
| Evaluation | Deterministic checks of real telemetry and ADK orchestration; live runs require model configuration | Comparable baselines and held-out reliability evaluation |

## Investigation behavior

1. The operator or alert supplies the symptom, affected service, environment, and time window.
2. The coordinator retrieves service context and reviewed incident leads.
3. Investigators use bounded tools to establish impact, locate delay, and examine plausible mechanisms or changes.
4. Findings are recorded with their query, returned data, collector, and collection time. Interpretations cite those records.
5. The coordinator reconciles findings and selects a targeted next check when it can materially distinguish hypotheses.
6. The report identifies supported explanations, contradictions, limitations, useful next checks, and any justified action proposal.
7. Operator feedback revises the case. A fresh run evaluates the updated context and replaces the invalidated report.
8. Human-reviewed cause and outcome can become memory for future retrieval.

The time window in a case defines its historical investigation. A future recovery check needs fresh observation data and its own window; rerunning the same historical window does not demonstrate recovery.

## Stage 1 modes

**Live model investigation** uses the configured model through Google ADK to select queries and investigative follow-ups. Model credentials are supplied to the server environment, not entered in the browser.

**Scripted ADK replay / orchestration check** exercises the application, tools, persistence, agent event flow, feedback, and reviewed memory without model credentials. Its investigator behavior is deterministic and visibly labeled. It is useful for repeatable integration checks, but it cannot establish that an LLM reasons adaptively or generalizes across incidents.

## Flexible implementation choices

- Keep the compact local application until a richer environment answers an evaluation need.
- Start with a small number of agents; add roles only when their contribution is observable.
- Keep model and query budgets configurable.
- Use straightforward local persistence first; introduce a different memory store when retrieval needs justify it.
- Preserve the evidence and report contracts when replacing an operational backend.
- Reassess approved execution after investigation quality and operator review are established.

The next planning checkpoint is a Stage 1 walkthrough: reproduce an incident, inspect the evidence manually, run the agents, correct a finding, retain a reviewed outcome, and identify the weakest track before expanding scope.

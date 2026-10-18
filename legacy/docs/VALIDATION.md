# Stage 1 validation — 6 October 2026

## Automated checks

`uv run --locked pytest -q`: **62 passed**, in about two seconds.

The suite covers actual application queueing and trace propagation, bounded telemetry tools, missing evidence, actual ADK orchestration and tool execution, targeted follow-up, shared query budgets, timeouts, report citation validation, fresh-run evidence isolation, corrections, explicit memory review, and operator API status transitions. The installed project CLI and JavaScript syntax were also checked.

ADK emits a `ParallelAgent` deprecation warning and an experimental function-declaration warning. Dependency versions are locked; the workflow migration checkpoint is recorded in the architecture notes.

## Live local HTTP walkthrough

The local launcher served checkout, inventory, and the operator UI. The scenario produced:

| Window | Completed requests | Errors | Checkout p95 latency |
| --- | ---: | ---: | ---: |
| Healthy capacity | 98 | 0 | 86.425 ms |
| Reduced worker capacity | 65 | 0 | 1301.095 ms |

The generator uses the same configured request interval, with bounded concurrency. Under saturation, observed completions and offered requests can diverge; these counts are not proof of comparable demand. The workload exercises actual HTTP calls and queue waiting, with timed inventory processing.

A fresh scripted ADK run collected ten evidence records. The coordinator requested a targeted before/after metrics comparison based on retrieved audit history and returned a recommendation citing current-run evidence. No operational action was executed.

## Browser walkthrough

Verified case creation from the generated alert, starting an investigation, completed findings, evidence and activity views, saving an unresolved review with unknown cause, and correction invalidation. A corrected replay case withheld its former proposal and finished unresolved. Reviewed memory retained the replay-origin label.

The current interface preview is [prototype.jpg](prototype.jpg). Runtime data and the scenario's operator-only answer key are under ignored `data/`; the agent cannot read that key.

## What remains unverified

- Live model query selection, hypothesis revision, feedback interpretation and generalization. No model credential was configured.
- Whether multiple agents improve results against the same-budget single-agent baseline.
- Causal diagnosis accuracy across misleading changes, multiple faults or incomplete telemetry.
- Operational action execution and automated recovery verification.
- Integration with external Prometheus, trace/log backends, Kubernetes, or a production application.

The replay results establish a working integration path, not autonomous investigation quality. Stage 1 remains open until a credentialed live walkthrough demonstrates the model behavior and corrected-context reasoning required by the project map.

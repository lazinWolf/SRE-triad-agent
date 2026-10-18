#!/usr/bin/env bash
# Local development: start the demo (if needed), then the worker and dashboard on the host.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
docker ps --format '{{.Names}}' | grep -qx prometheus || infra/sre/demo.sh up
uv sync --extra dev --locked
uv run sre-triage seed-memory
uv run sre-triage worker &
worker=$!
trap 'kill $worker 2>/dev/null' EXIT
uv run sre-triage ui

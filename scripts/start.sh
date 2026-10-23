#!/usr/bin/env bash
# Start the demo (if needed) and the dashboard on http://localhost:8501.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
docker ps --format '{{.Names}}' | grep -qx prometheus || infra/sre/demo.sh up
uv sync --extra dev
uv run sre-triage doctor || true
exec uv run sre-triage ui

#!/usr/bin/env bash
# Start the demo (if needed), the MCP tool servers on :8011/:8012, the CrewAI review board on :8002
# and the dashboard on http://localhost:8501.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
docker ps --format '{{.Names}}' | grep -qx prometheus || infra/sre/demo.sh up
uv sync --extra dev
(cd services/crew_reviewer && uv sync)
uv run sre-triage mcp --port 8011 &
mcp=$!
(cd services/crew_reviewer && exec uv run python server.py --port 8002) &   # exec: the trap below must reach the server
reviewer=$!
trap 'kill $mcp $reviewer 2>/dev/null' EXIT
export SRE_MCP_URL=http://127.0.0.1:8011 SRE_REVIEWER_A2A_URL=http://127.0.0.1:8002
sleep 5
uv run sre-triage doctor || true
uv run sre-triage ui

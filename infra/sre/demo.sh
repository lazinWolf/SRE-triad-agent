#!/usr/bin/env bash
# Start, stop or inspect the OpenTelemetry Demo with the SRE triage overlay.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
demo="$here/../otel-demo"
ref="3.1.0"

if [ ! -d "$demo/.git" ]; then
  git clone --depth 1 --branch "$ref" https://github.com/open-telemetry/opentelemetry-demo.git "$demo"
fi

compose() {
  (cd "$demo" && docker compose \
    --env-file .env --env-file .env.override --env-file ../sre/sre.env \
    -f compose.yaml -f compose.observability.yaml -f compose.extras.yaml -f ../sre/compose.sre.yaml "$@")
}

case "${1:-}" in
  up)
    compose up --detach --remove-orphans
    # The collector's docker_stats receiver can miss containers that start while it
    # initialises; restarting it once everything runs gives complete container metrics.
    docker restart otel-collector >/dev/null
    port="${SRE_SHOP_PORT:-8090}"
    echo "Demo running: shop http://localhost:$port  Jaeger http://localhost:16686/jaeger/ui  Grafana http://localhost:$port/grafana"
    ;;
  pull)  compose pull ;;
  down)  compose down --remove-orphans --volumes ;;
  ps)    compose ps --format 'table {{.Name}}\t{{.Status}}' ;;
  logs)  shift; compose logs --tail 100 "$@" ;;
  config) compose config ;;
  *) echo "usage: $0 {up|pull|down|ps|logs [service]|config}" >&2; exit 2 ;;
esac

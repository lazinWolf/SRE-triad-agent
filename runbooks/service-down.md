# Service down (stopped, exited or paused)
Category: service_down
Symptoms: request rate to the service drops to zero, callers see connection refused or timeouts, active_alerts shows the service down, container_status shows exited or paused.
## How to confirm
- Run container_status on the service; exited means stopped, paused means frozen.
- Run metric_timeline for request rate; it drops to zero at one point in time.
- Run downstream_health from a caller such as frontend or checkout to see connection refused or timeouts.
- Run container_logs for the last lines before the container exited.
- Run active_alerts to see which services are reported down and when.
## What it is NOT
- dependency_unavailable: the container is running but callers cannot reach it. Check container_status; it shows running.
- datastore_down: valkey-cart or astronomy-db is stopped. Use datastore_health; the fix is the same action on the datastore.
## Fix
- start_service(service) when container_status shows exited.
- unpause_service(service) when container_status shows paused. Do not restart a paused container.
- restart_service(service) is the alternative when start_service fails on a container that is still running but unhealthy.
## After the fix
- container_status shows running and request rate returns to baseline within one minute.
- Callers stop reporting connection errors; the active alert clears after the next health check.

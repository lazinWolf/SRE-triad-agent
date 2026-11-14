# Memory limit too low (OOM kills)
Category: memory_limit
Symptoms: container restart count rising, container_status shows oom_killed, gaps in traffic to the service, errors in callers when the container is down, memory usage at the limit right after a limit change.
## How to confirm
- Run container_status on the service; look for oom_killed and a rising restart count.
- Run container_resources and compare the memory limit with the working set; the limit is below normal usage.
- Run recent_changes to see whether set_memory_limit lowered the limit recently.
- Run container_logs on the service to find the last lines before each kill.
- Run metric_timeline for request rate to see the gaps that match each restart.
## What it is NOT
- memory_exhaustion (memory leak): memory grows steadily over time with the limit unchanged. Check the trend in metric_timeline.
- service_down: the container stopped without an oom_killed status. Check container_status for exited or paused.
## Fix
- set_memory_limit(service, memory_mb) back to the standard value for that service. recent_changes shows the previous value.
- Prefer this over restart_service, because a restart with the same low limit will be killed again.
## After the fix
- The restart count stops rising and container_status shows the container running with no oom_killed flag.
- Traffic gaps end at once; request rate returns to baseline within a few minutes.

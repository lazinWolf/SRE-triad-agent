# CPU saturation
Category: cpu_saturation
Symptoms: one service uses close to its full CPU (1.0 CPU limit) with a busy loop, CPU usage high while request rate is flat, latency up in that service and its callers, no errors.
## How to confirm
- Run container_resources on the service; the CPU limit is the standard 1.0 and usage sits at that limit.
- Run list_feature_flags and look for adHighCpu set to on in the ad service.
- Run metric_timeline for CPU and request rate; CPU rises while request rate stays flat.
- Run container_logs on the service to see whether the process is stuck in a loop or logs nothing unusual.
- Run slow_traces on the service to confirm the time is spent inside the service itself.
## What it is NOT
- cpu_limit: the CPU limit is lowered below 1.0 and the container is throttled. Check the limit in container_resources first.
- traffic_spike: request rate is far above baseline and CPU is high in every service. Here only one service is hot.
## Fix
- set_feature_flag(adHighCpu, off) when list_feature_flags shows the flag on. This is the preferred fix.
- restart_service(service) only if no feature flag explains the CPU burn. Restart is a temporary fix; the cause may return.
## After the fix
- CPU usage in the service drops back to its normal level within a few minutes of turning the flag off.
- Latency in the service and its callers returns to baseline; a restart shows a brief gap in requests while the container starts.

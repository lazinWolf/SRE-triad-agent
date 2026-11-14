# Memory leak
Category: memory_exhaustion
Symptoms: memory usage in a service grows steadily over hours until the container is killed by OOM, sawtooth restarts, rising latency before each crash. Known flags: emailMemoryLeak (email), recommendationCacheFailure (recommendation).
## How to confirm
- Run metric_timeline for memory usage; a steady climb with no drop points to a leak.
- Run list_feature_flags and check whether emailMemoryLeak or recommendationCacheFailure is on.
- Run container_resources to compare current memory with the limit and the working set.
- Run container_logs for growing cache or repeated allocation warnings.
- Run container_status to check restart count and any oom_killed status.
## What it is NOT
- memory_limit (OOM from a low limit): memory is already at the limit right after a set_memory_limit change, not growing over time. Check recent_changes.
- cpu_saturation: CPU is the problem, not memory; the memory graph is flat.
## Fix
- set_feature_flag(emailMemoryLeak or recommendationCacheFailure, off) for the flag that is on in that service.
- restart_service(service) afterwards, because turning the flag off does not release memory already held. Both steps are needed.
## After the fix
- Memory usage stops climbing once the flag is off; after the restart it drops to the normal baseline at once.
- The memory trend stays flat over the next hour; the restart count stops rising.

# CPU limit throttling
Category: cpu_limit
Symptoms: CPU throttled periods high on a service container, CPU usage pinned at its limit while the limit is below 1.0 CPU, latency up in that service and its callers, slow spans with no errors.
## How to confirm
- Run container_resources on the suspect service and compare its CPU limit with the standard 1.0 CPU limit.
- Run metric_timeline for CPU usage; a flat line at the limit with the limit under 1.0 means throttling.
- Run recent_changes to see whether a set_cpu_limit change lowered the limit recently.
- Run slow_traces on the service; spans are long but show no error and no slow downstream call.
- Run service_map to confirm which callers see the latency increase.
## What it is NOT
- cpu_saturation: the limit is normal (1.0) and the process itself burns CPU. Check the container_resources limit first.
- Slow downstream service: if get_trace shows the time inside a downstream span, the cause is in that downstream service, not this limit.
## Fix
- set_cpu_limit(service, 1.0) restores the standard limit. Prefer this whenever recent_changes shows the limit was lowered.
- If the limit is already 1.0, this is not this runbook; see cpu-saturation.
## After the fix
- CPU throttled periods drop to near zero within one or two metric scrape intervals.
- Latency in the service and its callers returns to baseline within a few minutes.

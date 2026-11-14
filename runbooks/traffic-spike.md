# Traffic spike (request rate far above baseline)
Category: traffic_spike
Symptoms: request rate far above baseline at the frontend, latency up across frontend and the services behind it, CPU high in many services at once, no errors in a single deep service. Known flag: loadGeneratorFloodHomepage.
## How to confirm
- Run metric_timeline for request rate at the frontend; it is far above the normal baseline.
- Run list_feature_flags and check whether loadGeneratorFloodHomepage is on.
- Run container_resources on several services; CPU is high in many of them at the same time.
- Run error_traces and slow_traces; no single deep service is the source of errors or slowness.
- Run service_map to confirm the frontend is the entry point of the spike.
## What it is NOT
- cpu_saturation: only one service burns CPU and the request rate is flat. Here CPU is high in many services.
- dependency_errors or lock_contention: one downstream service is slow or erroring while the request rate stays normal.
## Fix
- set_feature_flag(loadGeneratorFloodHomepage, off). This is the preferred fix.
- Do not restart services to fix a spike; the load returns as soon as the flag is on again.
## After the fix
- Frontend request rate falls back to baseline within a few minutes once the flood stops.
- CPU across services and frontend latency return to normal over the next several minutes.

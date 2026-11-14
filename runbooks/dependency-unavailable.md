# Dependency unavailable (wrong address, container running)
Category: dependency_unavailable
Symptoms: callers cannot reach a service although its container runs, payment traffic drops while checkout errors rise, checkout logs show dial or connect errors to a bad address. Known flag: paymentUnreachable.
## How to confirm
- Run container_status on the target service; it shows running, so the container itself is fine.
- Run list_feature_flags and check whether paymentUnreachable is on.
- Run service_map and metric_timeline; request rate from checkout to payment drops while checkout errors rise.
- Run container_logs on checkout for dial or connection errors to the wrong address.
- Run downstream_health from checkout to see that payment is unreachable while its container runs.
## What it is NOT
- service_down: the container is exited or paused. Check container_status; this runbook needs running.
- dependency_errors: the service answers and returns error codes. Here the calls never arrive.
## Fix
- set_feature_flag(paymentUnreachable, off) to restore the correct address. This is the preferred fix.
- Do not restart payment; its container is healthy and the fault is in how checkout dials it.
## After the fix
- Payment request rate returns to baseline within a few minutes once checkout dials the right address.
- Checkout error ratio drops to baseline and downstream_health shows payment reachable again.

# Lock contention (database lock serialises requests)
Category: lock_contention
Symptoms: product-catalog latency very high, CPU NOT saturated, long astronomy-db client spans in traces, requests queue behind each other. Known flag: productCatalogLockContention.
## How to confirm
- Run slow_traces on product-catalog; the long time is in the astronomy-db client span.
- Run get_trace on one slow request to see the astronomy-db span waiting, not computing.
- Run container_resources on product-catalog; CPU is low or moderate, not at the limit.
- Run list_feature_flags and check whether productCatalogLockContention is on.
- Run datastore_health on astronomy-db to confirm queries succeed but are slow.
## What it is NOT
- cpu_saturation: CPU is at the limit in that case. Here CPU is not saturated.
- datastore_down: astronomy-db is stopped or paused and queries fail or hang completely. Check container_status.
## Fix
- set_feature_flag(productCatalogLockContention, off). This is the only fix; a restart does not clear the lock queue.
- Avoid restarting product-catalog, because it drops in-flight requests and the contention returns.
## After the fix
- product-catalog latency drops back to baseline within a few minutes; astronomy-db client spans shorten.
- Request queueing clears; frontend and recommendation latency returns to normal shortly after.

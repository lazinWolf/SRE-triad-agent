# Datastore down (valkey-cart or astronomy-db stopped or paused)
Category: service_down / dependency_unavailable
Symptoms: cart or product-catalog errors or hangs, error_traces point at a client span to valkey-cart or astronomy-db, datastore_health shows no client calls succeeding, container_status shows exited or paused on the datastore.
## How to confirm
- Run container_status on valkey-cart and astronomy-db; exited or paused means the datastore is down.
- Run datastore_health on the datastore; no client call succeeds.
- Run error_traces on cart or product-catalog; the failing span is the datastore client call.
- Run container_logs on the datastore for the last lines before it stopped.
- Run service_map to confirm which dependent services use that datastore (cart for valkey-cart, product-catalog for astronomy-db).
## What it is NOT
- service_down of an application service: the failing container is the app itself. Here the datastore is the failing container.
- lock_contention: queries succeed but are slow. datastore_health shows successful calls.
## Fix
- start_service(datastore) when container_status shows exited, for example valkey-cart or astronomy-db.
- unpause_service(datastore) when container_status shows paused. Do not restart a paused datastore.
## After the fix
- container_status shows the datastore running; datastore_health shows client calls succeeding again.
- Cart or product-catalog error ratio returns to baseline within a minute or two of the datastore accepting connections.

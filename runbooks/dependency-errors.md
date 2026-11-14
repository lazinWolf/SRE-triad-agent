# Dependency errors (service returns errors)
Category: dependency_errors
Symptoms: error ratio up in a service and in its callers, error_traces start in the deepest service, responses with error codes. Known flags: paymentFailure (payment), cartFailure (cart), productCatalogFailure (product-catalog).
## How to confirm
- Run error_traces and find the deepest service where the error first appears; that is the source.
- Run metric_timeline for error ratio in the source and its callers; both rise together.
- Run list_feature_flags and check whether paymentFailure, cartFailure or productCatalogFailure is on.
- Run get_trace on one failing request to confirm the error span is in the deepest service.
- Run search_logs for the error message to see the failure wording in the source service.
## What it is NOT
- dependency_unavailable: the callers get connection refused or dial errors, not error responses. Check container_logs for dial errors.
- traffic_spike: error ratio rises with the request rate; here the error ratio rises at a flat rate in one service.
## Fix
- set_feature_flag(flag, off) for the flag that is on in the source service. This is the preferred fix.
- If no flag is on, this is a real code or data fault; escalate rather than restarting, because a restart does not clear it.
## After the fix
- Error ratio in the source service and its callers drops to baseline within a few minutes.
- error_traces stop showing the source service; checkout and frontend success rates recover.

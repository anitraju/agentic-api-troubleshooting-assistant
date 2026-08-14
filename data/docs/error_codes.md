# Order Service Error Reference

| HTTP status | Error code | Typical cause | First investigation |
|---|---|---|---|
| 400 | `malformed_request` | Invalid JSON/request syntax | Inspect payload and content type |
| 401 | `invalid_token` | Expired, malformed, or invalid JWT | Inspect token-validation logs |
| 403 | `insufficient_scope` | Token lacks required scope | Compare granted and required scopes |
| 404 | `order_not_found` | Unknown order ID | Verify `order_id` |
| 409 | `duplicate_idempotency_key` | Key reused with another payload | Compare current and prior request |
| 422 | `schema_validation_failed` | Valid JSON violates API schema | Compare body with OpenAPI |
| 429 | `rate_limit_exceeded` | Client exceeded request quota | Inspect rate-limit event |
| 500 | `database_unavailable` | Database connectivity/persistence failure | Inspect database logs |
| 502 | `payment_dependency_failed` | Payment Service returned failure | Correlate downstream call |
| 503 | `service_unavailable` | Service/dependency saturation | Check health and saturation |
| 504 | `payment_timeout` | Payment authorization exceeded timeout | Inspect downstream latency |

## Evidence Priority

Prefer troubleshooting evidence in this order:

1. request-specific logs;
2. API/OpenAPI contract;
3. runbooks;
4. historical incidents;
5. general documentation.

HTTP status alone is not enough to prove root cause. If evidence conflicts or is missing,
the assistant should report uncertainty rather than invent certainty.

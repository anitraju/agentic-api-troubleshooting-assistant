# Runbook: Downstream Payment Failures and Timeouts

Order creation calls `payment-service` to authorize payment.

## 502 `payment_dependency_failed`

A `502` indicates that Order Service reached Payment Service but received a failed or
unusable downstream response.

Investigation:

1. capture the Order Service request ID;
2. locate the outbound payment call;
3. record downstream status or transport error;
4. check Payment Service health;
5. compare with known incidents.

Example evidence:

```text
payment-service returned 500
order-service mapped downstream failure to 502
```

## 504 `payment_timeout`

The sample payment authorization timeout is `3000 ms`.

Investigation:

1. locate `payment_authorization_started`;
2. locate `payment_timeout`;
3. compare elapsed time with `3000 ms`;
4. determine whether latency is isolated or widespread.

Do not increase timeouts without understanding downstream latency. Preserve request and
idempotency identifiers during any safe retry.

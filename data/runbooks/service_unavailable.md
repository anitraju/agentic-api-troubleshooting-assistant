# Runbook: Service Unavailable

## Applies To

- `503 service_unavailable`
- widespread infrastructure-related failures

## Investigation

1. Determine whether the problem is isolated or widespread.
2. Check Order Service health.
3. Check database and Payment Service health.
4. Look for saturation:
   - database connection-pool exhaustion;
   - worker/thread exhaustion;
   - memory pressure;
   - dependency connection failures.
5. Compare the signature with historical incidents.

## Known Signature: Database Connection Pool Exhaustion

Typical signals:

- `503 service_unavailable`;
- `db_pool_exhausted`;
- `available_connections=0`;
- elevated latency and waiting requests.

Mitigate abnormal connection consumption, restore pool availability, and verify latency and
error rate recover. Increasing pool size can mask a connection leak and is not the preferred
first response when a leak is suspected.

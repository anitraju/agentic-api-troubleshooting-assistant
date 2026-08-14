# Order Service Rate Limits

## Default Limits

The sample environment applies per-client limits:

- `POST /api/v1/orders`: 60 requests per minute.
- `GET /api/v1/orders/{order_id}`: 120 requests per minute.

When a client exceeds its quota, Order Service returns `429 Too Many Requests` and a
`Retry-After` header.

Example:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 30
```

```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Retry after the indicated delay."
}
```

## Troubleshooting

Check:

1. whether request volume recently increased;
2. whether a retry loop is creating a request storm;
3. whether the caller ignores `Retry-After`;
4. whether multiple workloads share the same client identity.

Clients should use bounded exponential backoff with jitter and honor `Retry-After`. Tight
retry loops can prolong throttling.

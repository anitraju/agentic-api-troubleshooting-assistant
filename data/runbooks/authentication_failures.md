# Runbook: Authentication and Authorization Failures

## 401 `invalid_token`

1. Capture the request ID.
2. Search Order Service logs for that request.
3. Look for `token_expired`, `jwt_validation_failed`, `invalid_audience`, or `invalid_issuer`.
4. Confirm the caller sent `Authorization: Bearer <token>`.
5. If the token expired, obtain a fresh access token and retry.
6. If a fresh token fails, verify issuer, audience, clock synchronization, and signing keys.

An expired token does not require an Order Service restart.

## 403 `insufficient_scope`

1. Confirm token validation succeeded.
2. Determine the scope required by the endpoint.
3. Inspect the token's granted scopes.
4. Compare granted scopes with the required scope.

Required scopes:

- `POST /api/v1/orders` -> `orders:write`
- `GET /api/v1/orders/{order_id}` -> `orders:read`

Resolve by issuing a token with the correct scope or correcting the OAuth client
configuration.

## Escalation

Escalate to the identity-platform team when multiple clients suddenly fail signature
validation, the issuer/JWKS endpoint is unavailable, or newly rotated signing keys are not
recognized.

# Order Service Authentication

## Overview

The Order Service uses OAuth 2.0 bearer access tokens encoded as JWTs.

Every protected request must include:

```http
Authorization: Bearer <access_token>
```

The service validates the JWT signature, issuer, audience, expiration time, and the OAuth
scope required by the endpoint.

## Token Requirements

A valid token must satisfy all of the following:

- `iss` is `https://auth.example.internal`.
- `aud` contains `order-service`.
- `exp` is later than the current server time.
- the signature validates against an active identity-provider signing key.
- the token contains the scope required by the requested operation.

## Required Scopes

| Operation | Required scope |
|---|---|
| `POST /api/v1/orders` | `orders:write` |
| `GET /api/v1/orders/{order_id}` | `orders:read` |

## 401 Unauthorized

A `401` means authentication failed.

Common causes:

- missing `Authorization` header;
- malformed bearer token;
- expired JWT;
- signature validation failure;
- invalid issuer;
- invalid audience.

Typical response:

```json
{
  "error": "invalid_token",
  "message": "The supplied access token is invalid or expired."
}
```

When `invalid_token` is returned, correlate the request ID with service logs. If the logs
show `token_expired`, obtain a fresh access token and retry. If a fresh token still fails,
verify issuer, audience, clock synchronization, and signing-key configuration.

## 403 Forbidden

A `403` means authentication succeeded but authorization failed.

Typical response:

```json
{
  "error": "insufficient_scope",
  "message": "Required scope orders:write is missing."
}
```

Do not treat `403 insufficient_scope` as a token-expiry issue. Obtain a token containing the
required scope or correct the OAuth client configuration.

## Useful Log Signals

- `token_expired`
- `jwt_validation_failed`
- `invalid_audience`
- `invalid_issuer`
- `missing_scope`

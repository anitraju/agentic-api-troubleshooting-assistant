# Orders API

## Service

- Service: `order-service`
- Base path: `/api/v1`
- Content type: `application/json`

## Create Order

```http
POST /api/v1/orders
```

Required headers:

```http
Authorization: Bearer <token>
Content-Type: application/json
Idempotency-Key: <unique-key>
```

The access token requires the `orders:write` scope.

Example request:

```json
{
  "customer_id": "cust-1001",
  "items": [
    {
      "sku": "SKU-RED-CHAIR",
      "quantity": 1
    }
  ],
  "payment_method": "pm-1001"
}
```

Validation rules:

- `customer_id` is required and non-empty.
- `items` is required and contains at least one item.
- every item requires `sku`.
- `quantity` is an integer greater than zero.
- `payment_method` is required.
- `Idempotency-Key` is required.

A caller retrying the same logical create-order request should reuse the same idempotency
key. A genuinely new order should use a new key.

## Get Order

```http
GET /api/v1/orders/{order_id}
```

The access token requires the `orders:read` scope.

## Failure Modes

### 400 `malformed_request`

The HTTP request cannot be parsed correctly. Typical causes are malformed JSON or an invalid
content type.

### 401 `invalid_token`

Authentication failed. See `authentication.md`.

### 403 `insufficient_scope`

The token is valid but lacks the required OAuth scope.

### 404 `order_not_found`

The supplied order ID does not exist.

### 409 `duplicate_idempotency_key`

The same idempotency key was reused with a different payload.

### 422 `schema_validation_failed`

The JSON is syntactically valid but violates the API schema or validation rules.

### 429 `rate_limit_exceeded`

The caller exceeded its configured request quota.

### 500 `database_unavailable`

An internal database connectivity or persistence operation failed.

### 502 `payment_dependency_failed`

Order Service contacted Payment Service but received a downstream failure.

### 503 `service_unavailable`

Order Service or a critical dependency is temporarily unavailable or saturated.

### 504 `payment_timeout`

Payment authorization exceeded the configured downstream timeout.

## Correlation

Responses propagate an `X-Request-ID` where possible. Use it to correlate the API response
with Order Service logs and downstream operations.

# Runbook: Order Creation Failures

## Endpoint

`POST /api/v1/orders`

## 400 `malformed_request`

Check that the body is valid JSON and `Content-Type` is `application/json`.

Resolution: correct the request syntax or JSON payload.

## 409 `duplicate_idempotency_key`

Order Service stores a payload fingerprint for each idempotency key. A conflict occurs when
the same key is reused with a different payload.

Investigation:

1. capture the idempotency key;
2. search for the original request using that key;
3. compare the stored and received payload fingerprints.

Reuse the same key only for the same logical request. Use a new key for a genuinely new
order. Do not blindly change keys when retrying an uncertain outcome because that can create
duplicate orders.

## 422 `schema_validation_failed`

Compare the exact validation message with the OpenAPI schema.

Common causes:

- missing `customer_id`;
- missing `payment_method`;
- empty `items`;
- missing item `sku`;
- `quantity <= 0`.

## 500 `database_unavailable`

Search request-specific logs for database evidence such as connection failures. Check
database health and known incidents. Do not classify every `500` as a database failure
without supporting evidence.

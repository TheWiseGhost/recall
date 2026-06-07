# Authentication

The API authenticates requests with a bearer token supplied in the
`Authorization` header. Tokens are issued by the auth service and are valid for
one hour, after which the client must exchange its refresh token for a new
access token.

## Token verification

Every request is verified in three steps: the signature is checked against the
service's public key, the expiry claim is compared against the current time,
and the token's scope is matched against the scope the endpoint requires. A
request that fails any of the three is rejected with `401 Unauthorized`, except
for a scope mismatch, which returns `403 Forbidden`.

## Rotating credentials

Signing keys rotate every 30 days. Both the current and the previous key are
accepted during a 24-hour overlap window so that tokens issued just before a
rotation remain valid until they expire naturally.

## Service-to-service calls

Internal services authenticate with mutual TLS rather than bearer tokens. Each
service holds a client certificate issued by the internal certificate
authority, and the mesh rejects any connection whose certificate is not signed
by that authority.

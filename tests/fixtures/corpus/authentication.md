# Authentication

The API authenticates requests with a bearer token supplied in the Authorization
header. Tokens are issued by the auth service and are valid for one hour.

## Token verification

Every request is verified in three steps: the signature is checked against the
service public key, the expiry claim is compared against the current time, and
the token scope is matched against the scope the endpoint requires.

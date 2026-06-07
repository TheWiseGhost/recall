# Observability

Every request is assigned a request identifier at the edge. The identifier is
carried through every log line, metric exemplar and trace span produced while
handling that request.

## Logging

Logs are structured. Each line is a JSON object with the request identifier,
the operation, the duration in milliseconds, and the outcome. Credentials are
redacted before a line is written.

## Metrics

Prometheus scrapes each instance every fifteen seconds. The metrics that matter
for capacity planning are request rate, error rate, and the p50, p95 and p99
latencies for each endpoint.

## Tracing

Traces are sampled at one percent under normal load and at one hundred percent
for requests that return a server error, so every failure has a full trace.

# Deployment

Deployments are rolling. The release pipeline builds a container image, pushes
it to the registry, and updates the deployment to the new tag.

## Rolling update

Replicas are replaced one at a time. The orchestrator waits for the new
replica's readiness probe to succeed before terminating the old one, so
capacity never drops below the configured minimum during a deploy.

## Readiness and liveness

The readiness probe checks that the service can reach both PostgreSQL and
Redis. The liveness probe only checks that the process is responding, so a
transient database outage does not cause a restart loop.

## Rollback

Rollback redeploys the previous image tag. Because database migrations are
applied separately and are written to be backwards compatible for one release,
a rollback does not require a schema change.

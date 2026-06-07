# Security policy

## Supported versions

Recall is pre-1.0. Only the latest release on `main` receives security fixes.

| Version | Supported |
|---|---|
| 0.1.x | ✅ |

## Reporting a vulnerability

**Please do not open a public issue.**

Report privately through GitHub's [private vulnerability reporting](https://github.com/TheWiseGhost/recall/security/advisories/new) on this repository. Include:

- what the issue is and where in the code,
- how to reproduce it,
- what an attacker could achieve.

You can expect an acknowledgement within a few days and an assessment shortly after. If a fix is warranted we will coordinate a release and credit you unless you would rather stay anonymous.

## Scope

Recall is a self-hosted framework, not a hosted service. The interesting classes of issue are:

- **Credential exposure** — API keys or database passwords appearing in logs, error messages, API responses, or the database.
- **SQL injection** — anything reaching the database without parameterisation. All queries go through SQLAlchemy; raw SQL with interpolated user input is a bug.
- **Path traversal** — the filesystem connector reading outside its configured root.
- **Deserialisation** — untrusted YAML or JSON reaching an unsafe parser. Configuration uses `yaml.safe_load`; anything else is a bug.
- **Denial of service through ingestion** — a crafted document exhausting memory or CPU.

Out of scope: findings that require an attacker to already have shell access or database credentials, and vulnerabilities in dependencies that do not affect Recall's use of them (report those upstream, and open a normal issue here so we can pin around it).

## Handling secrets

Recall's own rules, which are also what a report should test against:

- Credentials come from environment variables or configuration referencing them (`${OPENAI_API_KEY}`). They are never committed and never stored in the database.
- `recall.yaml` is designed to be safe to commit. `.env` is git-ignored; `.env.example` is not.
- API keys are held as `SecretStr`, so they do not appear in a `repr` or a validation error.
- The logging pipeline redacts keys named `api_key`, `token`, `password`, `secret`, `authorization` and `access_token` before any log line is written.

If you find a path where a secret escapes any of these, that is a vulnerability — please report it.

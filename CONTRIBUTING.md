# Contributing to Recall

Thanks for taking the time. Recall is a research framework, so the bar is a little different from a typical application: **a change that cannot be reproduced or verified is not finished.**

## Setup

```bash
git clone https://github.com/TheWiseGhost/recall.git
cd recall
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,pdf]"
docker compose up -d postgres
docker exec recall-postgres psql -U recall -d postgres -c "CREATE DATABASE recall_test OWNER recall;"
```

## The checks CI runs

```bash
ruff check .
```

```bash
ruff format --check .
```

```bash
mypy
```

```bash
pytest
```

```bash
pytest -m integration
```

All five must pass. Invoke `pytest` directly rather than `python -m pytest` —
the latter injects the working directory into `sys.path`, which can hide a
real import problem that CI will catch.

`mypy` runs in strict mode; new code is expected to type-check without `Any` escape hatches unless a third-party stub forces one.

## Testing expectations

- **Unit tests must not need a database.** `tests/conftest.py` provides in-memory fakes implementing the same ports as the PostgreSQL adapter. If a change is hard to unit test, that is usually a signal the logic belongs behind a port rather than inside an adapter.
- **Integration tests are marked** `@pytest.mark.integration` and skip themselves when no database is reachable.
- **Test behaviour, not implementation.** Assert on what a caller can observe.
- **Watch for vacuous assertions.** `assert "created" in output` passes when the summary table merely *prints the word*. Assert on the count.

## Code style

- Type hints on every public function.
- Docstrings where the *why* is not obvious from the name. Skip them where it is.
- Prefer readable code to clever abstractions.
- Keep `recall/core` free of SQLAlchemy, FastAPI, Celery and Typer. This is enforced by review, and it is the single most important rule in the codebase.
- No hardcoded models, credentials, or database assumptions.

## Adding a component

Connectors, chunkers, embedders, retrievers, rerankers, generators and evaluators are plugins. See [docs/contributing/plugins.md](docs/contributing/plugins.md). If adding one requires editing a file in `recall/core`, the abstraction is wrong — please open an issue rather than working around it.

## Adding a dependency

Explain in the pull request why it is necessary and what it replaces. Heavy or optional dependencies (anything pulling in PyTorch, a vendor SDK, or a native binary) belong behind an extra, imported lazily inside the function that needs them, and raising a clear "install it with…" error when absent.

## Benchmarks and claims

Do not add performance or quality numbers to the README, docs, or commit messages unless the harness that produced them is committed and re-runnable. Synthetic datasets must be labelled synthetic. This applies to prose as much as to charts: "faster" and "better" are claims.

If something is intentionally deferred, mark it `TODO / FUTURE` with a note on what it is waiting for.

## Commits and pull requests

- Small, focused commits with a message that explains *why*.
- Reference the issue if there is one.
- Update the docs in the same pull request as the behaviour change.
- New behaviour comes with tests.

## Reporting bugs

Include the Recall version, Python version, the relevant part of `recall.yaml` (with credentials removed), and the output of `recall status`. A failing test case is the fastest possible bug report.

## Security

Do not open a public issue for a security problem. See [SECURITY.md](SECURITY.md).

## Code of conduct

By participating you agree to uphold the [Code of Conduct](CODE_OF_CONDUCT.md).

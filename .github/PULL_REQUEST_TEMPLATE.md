## What and why

<!-- What does this change, and what problem does it solve? -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] New plugin (connector / chunker / embedder / retriever / reranker / evaluator)
- [ ] Documentation
- [ ] Refactor or infrastructure

## Checklist

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `mypy` passes
- [ ] `pytest` passes
- [ ] `pytest -m integration` passes (or the change cannot affect storage)
- [ ] New behaviour has tests
- [ ] Documentation updated in this PR
- [ ] `recall/core` still imports nothing from SQLAlchemy, FastAPI, Celery or Typer
- [ ] No new dependency, or the PR explains why it is necessary
- [ ] No performance or quality claims without a committed, re-runnable harness

## Notes for reviewers

<!-- Trade-offs, alternatives considered, anything deliberately left out. -->

"""File templates written by ``recall init``."""

from __future__ import annotations

# Placeholders are ``__TOKEN__`` rather than ``{token}``: the template is full
# of ``${VAR}`` interpolation syntax that str.format would try to expand.
RECALL_YAML = """\
# Recall configuration.
#
# ${VAR} is expanded from the environment; ${VAR:-default} supplies a fallback.
# Never put credentials in this file — reference environment variables instead.

project_name: __PROJECT_NAME__

database:
  url: ${DATABASE_URL:-postgresql+asyncpg://recall:recall@localhost:5432/recall}

redis:
  url: ${REDIS_URL:-redis://localhost:6379/0}

embedding:
  # Providers: sentence_transformers | openai | hash
  # `hash` needs no extra dependencies and is deterministic, which makes it a
  # good choice for tests and a quick first run. It is not a semantic model.
  provider: __EMBEDDING_PROVIDER__
  model: __EMBEDDING_MODEL__
  # Must match the model's output size. It is baked into the pgvector column by
  # the initial migration, so changing it requires a migration + re-index.
  dimensions: __EMBEDDING_DIMENSIONS__
  batch_size: 32

chunking:
  strategy: fixed
  chunk_size: 512
  overlap: 64

retrieval:
  default: dense
  top_k: 10

# Used from Milestone 2 onwards.
hybrid:
  dense_weight: 0.65
  lexical_weight: 0.35
  fusion: rrf

reranking:
  enabled: false
  strategy: none

logging:
  level: INFO
  format: console

experiments_dir: experiments
"""

ENV_EXAMPLE = """\
# Copy to .env and fill in. .env is git-ignored; .env.example is not.

DATABASE_URL=postgresql+asyncpg://recall:recall@localhost:5432/recall
REDIS_URL=redis://localhost:6379/0

# Only needed when embedding.provider or a generator is set to openai.
# OPENAI_API_KEY=

# Only needed for the GitHub / Notion connectors (Milestone 4).
# GITHUB_TOKEN=
# NOTION_API_KEY=
"""

GITIGNORE = """\
.env
__pycache__/
*.py[cod]
.venv/
venv/
.mypy_cache/
.pytest_cache/
.ruff_cache/
experiments/results/
"""

EXAMPLE_DOC = """\
# Authentication

Recall's example corpus uses this document to demonstrate ingestion and search.

## How authentication works

The API authenticates requests with a bearer token supplied in the
`Authorization` header. Tokens are issued by the auth service and are valid for
one hour, after which the client must exchange its refresh token for a new
access token.

## Token verification

Every request is verified in three steps: the signature is checked against the
service's public key, the expiry claim is compared against the current time,
and the token's scope is matched against the scope the endpoint requires.

## Rotating credentials

Signing keys rotate every 30 days. Both the current and previous key are
accepted during the overlap window so that in-flight tokens remain valid.
"""

# Getting started

## Requirements

- Python 3.12+
- Docker (for PostgreSQL with pgvector)

## Install

```bash
git clone https://github.com/TheWiseGhost/recall.git
cd recall
```

```bash
uv venv --python 3.12 && source .venv/bin/activate
```

```bash
uv pip install -e ".[dev,pdf]"
```

Extras:

| Extra | Adds | Needed for |
|---|---|---|
| `pdf` | PyMuPDF | the PDF connector |
| `local` | sentence-transformers (and PyTorch) | local embedding models |
| `openai` | openai | OpenAI embeddings |
| `dev` | pytest, ruff, mypy | development |

## Start the database

```bash
docker compose up -d postgres
```

The image is `pgvector/pgvector:pg17`, and the volume is created with the `vector` extension already installed.

## Initialize a project

```bash
recall init --embedding-provider hash
```

This writes `recall.yaml`, `.env.example`, `.gitignore` and an example corpus under `examples/documents/`.

`hash` is a deterministic, dependency-free embedder. It carries real lexical signal, which is enough to see the pipeline work end to end in seconds, but it is **not a semantic model** and must not be used for quality claims. Switch to `sentence_transformers` once you want meaningful retrieval.

## Create the schema

```bash
recall migrate
```

## Ingest and search

```bash
recall ingest ./examples/documents
```

```bash
recall search "How does authentication work?"
```

You should get ranked results with a score, the source document, the matching chunk, its metadata, and per-stage latency.

## Check the system

```bash
recall status
```

Reports the resolved configuration, database connectivity, the pgvector version, and current document/chunk/vector counts.

## Switching to a real embedding model

```bash
pip install -e ".[local]"
```

Edit `recall.yaml`:

```yaml
embedding:
  provider: sentence_transformers
  model: BAAI/bge-base-en-v1.5
  dimensions: 768
```

`dimensions` is baked into the pgvector column by the initial migration, so changing it means recreating the schema:

```bash
docker compose down -v && docker compose up -d postgres
recall migrate
recall ingest ./examples/documents --force
```

The first search will be slow while the model downloads.

## Running the tests

```bash
pytest                     # unit tests, no database needed
```

```bash
docker exec recall-postgres psql -U recall -d postgres -c "CREATE DATABASE recall_test OWNER recall;"
pytest -m integration      # against a real PostgreSQL
```

Integration tests skip themselves (rather than failing) when no database is reachable. Point them elsewhere with `RECALL_TEST_DATABASE_URL`.

## Troubleshooting

**`recall migrate` fails to connect.** Check `docker compose ps` and that `database.url` in `recall.yaml` matches the port Compose published. `DATABASE_URL` in the environment overrides the file's default.

**Search returns nothing after ingesting.** Run `recall status`. If `vectors` is 0, ingestion did not index anything — re-run `recall ingest` and read the summary table for failures.

**Search returns nothing after changing the embedding model.** Vectors are scoped to `provider:model`. Re-index with `recall ingest --force`.

**`dimension mismatch` errors.** The configured `embedding.dimensions` no longer matches the column. Recreate the schema and re-index as shown above.

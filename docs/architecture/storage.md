# Storage

PostgreSQL with the `pgvector` extension. One database holds documents, chunks, vectors and (from Milestone 2) the full-text index — so a filtered hybrid query is one query, not a fan-out and a join in application code.

## Why not a dedicated vector database

For the corpus sizes Recall targets, a separate vector store buys recall-per-millisecond that Recall cannot yet measure, and costs the ability to filter and join transactionally. `VectorIndex` in `recall/core/ports.py` is the seam: implementing it against Qdrant or Weaviate requires no change to any retriever. That is deliberately deferred until there are benchmarks that justify it.

## Schema

```
documents
  id            uuid        primary key   (uuid5 of source_type + source_id)
  source_id     text        unique with source_type
  source_type   varchar(64)
  title         text
  content       text
  uri           text
  metadata      jsonb       GIN indexed
  checksum      varchar(64)
  created_at    timestamptz
  updated_at    timestamptz

chunks
  id            uuid        primary key
  document_id   uuid        -> documents.id  ON DELETE CASCADE
  parent_id     uuid        -> chunks.id     ON DELETE SET NULL   (hierarchical chunking)
  content       text
  metadata      jsonb       GIN indexed
  position      int
  token_count   int
  checksum      varchar(64)
  start_char    int         offsets back into documents.content
  end_char      int
  content_tsv   tsvector    GENERATED, GIN indexed                (BM25 candidate selection)
  content_length int        GENERATED                             (BM25 |D|)

chunk_embeddings
  chunk_id      uuid        -> chunks.id     ON DELETE CASCADE  } composite
  model_key     varchar     "provider:model"                    } primary key
  document_id   uuid        -> documents.id  ON DELETE CASCADE   (denormalised)
  embedding     vector(N)   HNSW, vector_cosine_ops
  provider      varchar
  model         varchar
  dimensions    int
  created_at    timestamptz
```

## Decisions worth knowing about

### The vector dimension is a schema decision

pgvector needs a fixed dimension to build an ANN index, so the initial migration creates `vector(N)` using `embedding.dimensions` from your configuration. Switching to a model with a different output size requires a new migration and a full re-index:

```bash
docker compose down -v && docker compose up -d postgres
recall migrate
recall ingest ./docs --force
```

The dimension is passed to Alembic through the config it is invoked with, not re-discovered inside the migration, so the migration can never disagree with the settings the caller resolved.

### Vectors are keyed by `(chunk_id, model_key)`

One chunk can carry vectors from several embedding models at once, which is exactly what a "which embedding model is better?" experiment needs. Dense retrieval always scopes its query to the model it is configured with, so a half-migrated index returns correct results for whichever model you ask about rather than mixing incomparable score spaces.

### `document_id` is denormalised onto `chunk_embeddings`

It duplicates what a join to `chunks` would give, and it earns that duplication: per-document deletes and document-scoped filters avoid the join entirely.

### HNSW, not IVFFlat

IVFFlat needs a training pass over representative data and degrades badly when the index is built while nearly empty — the common case for a fresh project. HNSW builds incrementally and needs no training.

### Filtered search may fall back to an exact scan

pgvector applies its ANN index before `WHERE` predicates. A highly selective filter can therefore leave too few candidates and cause PostgreSQL to choose an exact scan. Results stay correct; latency rises on large corpora. This is a known characteristic, not a bug, and improving it (partial indexes per source, or pre-filtering into a CTE) is a Milestone 2+ concern.

### Filtering happens in SQL, never in Python

`recall/storage/postgres/filters.py` translates `SearchFilters` into predicates. Over-fetching and discarding rows in application code would silently change what `top_k` means, which would corrupt Recall's entire reason for existing: retrieval metrics.

Tag filtering uses a disjunction of `@>` containment checks rather than the more direct `?|` operator, because containment can use the GIN index on `metadata` and `?|` on a nested path cannot.

### The lexical columns are generated, not application state

`content_tsv` and `content_length` are both `GENERATED ALWAYS ... STORED`. PostgreSQL maintains them for every writer — the ORM, a raw `UPDATE`, a future second backend — so they cannot drift from `content`, and no backfill can be interrupted halfway.

`content_length` is BM25's `|D|`: the number of *positions* in the tsvector, i.e. tokens surviving stopword removal and stemming. It needs an aggregate over `unnest()`, which a generated expression may not contain, so migration 0002 adds an `IMMUTABLE` SQL function `recall_tsvector_length(tsvector)` and the column calls it. The cost is one extra `to_tsvector` per insert, since a generated column may not reference another generated column.

The text search configuration (`english`) is compiled into both columns, which makes it a schema decision exactly like the vector dimension. See [retrieval.md](retrieval.md).

### `metadata` is a hazard in ORM code

Both `documents` and `chunks` have a column named `metadata`, and `metadata` is also the name of SQLAlchemy's `Base.metadata`. The mapped attribute is therefore called `meta`, inserts use the attribute name, `on_conflict_do_update` uses `Column` objects rather than strings, and every `SELECT` labels its columns explicitly. Getting this wrong fails at runtime with a confusing `AttributeError` on `MetaData`.

## Migrations

Alembic runs through the async engine, so asyncpg is the only PostgreSQL driver Recall depends on.

```bash
recall migrate                  # upgrade to head
recall migrate --revision 0001  # to a specific revision
```

Because `env.py` calls `asyncio.run` internally, the migration helpers in `recall/storage/postgres/migrate.py` are synchronous and must not be called from inside a running event loop.

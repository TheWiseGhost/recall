# Architecture overview

Recall is organised around one rule: **the domain does not know about its infrastructure.**

```
recall/
  core/           domain models + protocols        (no SQLAlchemy, FastAPI, Celery, Typer)
    models.py       Document, Chunk, SearchResult, SearchFilters, SyncResult
    ports.py        DocumentRepository, ChunkRepository, VectorIndex, IngestStore
    registry.py     name -> factory, the plugin seam
    chunking/       Chunker protocol + strategies
    embeddings/     Embedder protocol + providers
    retrieval/      Retriever protocol + strategies

  connectors/     adapters: external source -> Document
  storage/        adapters: ports -> PostgreSQL + pgvector
  pipeline/       use cases: ingest, search, and the composition root
  config/         typed settings, YAML + env
  observability/  structured logging
  cli/            Typer entrypoints
```

Dependencies point **inward**. `storage/postgres` imports from `core`; `core` never imports from `storage`. The practical payoff is that the entire ingestion and retrieval logic can be unit-tested with in-memory fakes — see `tests/conftest.py` — and that a second storage backend is an additive change.

## The pipeline

```
Connector.discover()  ->  [SourceItem]
        |
   checksum comparison (see ingestion.md)
        |
Connector.fetch(item) ->  Document
        |
Chunker.chunk(document) -> [Chunk]
        |
Embedder.embed_documents([...]) -> [Vector]
        |
Storage.index_document(document, chunks, vectors, model)   <- one transaction
        |
        v
Retriever.search(query, top_k, filters) -> [SearchResult]
```

## Key decisions

### Domain models are Pydantic, database rows are SQLAlchemy, and they never mix

`recall/storage/postgres/mapping.py` is the only place that translates between them. Nothing outside `storage/` can accidentally depend on a column name, and `SearchResult` is a stable contract for the CLI, the API, and experiment result files.

### IDs are deterministic

`document_id = uuid5(namespace, "document:{source_type}:{source_id}")`, and `chunk_id` additionally folds in position and content checksum. Ingesting the same source twice produces the same primary keys, which is what makes ingestion idempotent without a separate lookup table. Changing a chunk's *content* changes its ID, so a rewritten chunk can never silently inherit an old chunk's meaning.

### Components are resolved by name

Every swappable piece lives in a `Registry` keyed by the string that appears in configuration. Configuration is validated against the registries at load time, so `chunking.strategy: telepathic` fails at startup with the list of what *is* available — not at the first ingest.

### `sync` is not on the `Connector` protocol

The spec sketches `Connector.sync()`. Recall deliberately keeps `discover`/`fetch` on the connector and puts reconciliation in `IngestionPipeline.sync(connector)`. Checksum comparison, re-chunking, re-embedding and pruning are identical for every source and are the most correctness-sensitive code in the system; duplicating them per connector would guarantee they drift. A connector author writes two methods and gets incremental sync for free.

### Timing is a first-class result, not a log line

`SearchResponse.timing` breaks a query into `embedding_ms`, `retrieval_ms`, `reranking_ms`, `generation_ms` and `total_ms`. Retrievers record their own stages against an *ambient* timer held in a `ContextVar`, so the `Retriever` protocol stays a single method and concurrent searches never mix up each other's numbers. Latency is one of the things experiments compare, so it cannot be an afterthought.

## What is not built yet

Milestone 1 covers ingestion, storage and dense retrieval. BM25, hybrid retrieval, reranking, context selection, generation, evaluation metrics, the experiment runner, the API, workers and the dashboard are all planned — see the roadmap in the [README](../../README.md). Where a seam for them already exists (the `reranking_ms` timing field, the `content_tsv` column, the `hybrid` config section), it is marked as such in the code.

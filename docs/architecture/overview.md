# Architecture overview

Recall is organised around one rule: **the domain does not know about its infrastructure.**

```
recall/
  core/           domain models + protocols        (no SQLAlchemy, FastAPI, Celery, Typer)
    models.py       Document, Chunk, SearchResult, SearchFilters, SyncResult
    ports.py        DocumentRepository, ChunkRepository, VectorIndex, LexicalIndex, IngestStore
    registry.py     name -> factory, the plugin seam
    chunking/       Chunker protocol + strategies (fixed, sentence, semantic, hierarchical)
    embeddings/     Embedder protocol + providers
    retrieval/      Retriever protocol + strategies (dense, bm25, hybrid) + fusion
    reranking/      Reranker protocol + strategies (none, cross_encoder)
    evaluation/     metrics, cost model, result shapes

  connectors/     adapters: external source -> Document
  storage/        adapters: ports -> PostgreSQL + pgvector
  evaluation/     adapters: datasets on disk, experiment runner, reports
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
await Chunker.chunk(document) -> [Chunk]
        |
Embedder.embed_documents([...]) -> [Vector]
        |
Storage.index_document(document, chunks, vectors, model)   <- one transaction
        |
        v
Retriever.search(query, top_k, filters) -> [SearchResult]
        |
Reranker.rerank(query, candidates, top_k) -> [SearchResult]      <- optional
```

`Chunker.chunk` is asynchronous because semantic chunking embeds candidate
sentences to find its boundaries. Strategies that need no I/O never await.

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

`SearchResponse.timing` breaks a query into `embedding_ms`, `retrieval_ms`, `fusion_ms`, `reranking_ms`, `generation_ms` and `total_ms`. Retrievers record their own stages against an *ambient* timer held in a `ContextVar`, so the `Retriever` protocol stays a single method and concurrent searches never mix up each other's numbers. Latency is one of the things experiments compare, so it cannot be an afterthought.

### Scoring that needs an index lives behind a port

Dense retrieval needs an ANN index; BM25 needs an inverted index and corpus-wide term statistics. Neither can be computed in `core` without dragging SQLAlchemy in. So each has a port — `VectorIndex` and `LexicalIndex` — implemented in `storage/postgres` and consumed by a thin retriever in `core/retrieval`. The retriever owns rank stamping and timing; the adapter owns the scoring SQL. See [retrieval.md](retrieval.md).

### The metric arithmetic has no dependencies at all

`core/evaluation/` computes metrics from a ranked list of label keys and a judgement mapping. It does not know what a retriever is, does not read a file and does not open a connection. Everything that translates chunks into label keys happens in `evaluation/` before it.

That separation is the point: retrieval metrics are where a subtle error produces confident, reproducible, *wrong* science, so the arithmetic has to be readable and checkable on its own. See [experiments](../experiments/index.md).

## What is not built yet

Milestones 1 and 2 are complete: ingestion, storage, dense/BM25/hybrid retrieval, reranking, four chunking strategies, and the evaluation harness. Context selection, generation, the API, workers and the dashboard are later — see the roadmap in the [README](../../README.md). Where a seam already exists (`chunks.parent_id` for parent-child selection), it is marked as such in the code.

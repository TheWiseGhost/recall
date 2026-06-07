# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Milestone 2 — retrieval research: BM25, hybrid retrieval, reciprocal rank
fusion, cross-encoder reranking, sentence/semantic/hierarchical chunking,
evaluation metrics, benchmark datasets, and the experiment runner.

## [0.1.0] — 2026-07-31

First release. Milestone 1 (MVP): ingest local files and PDFs, index them into
PostgreSQL with pgvector, and search them from the CLI.

### Added

**Core**
- Domain models: `Document`, `Chunk`, `SearchResult`, `SearchFilters`, `SourceItem`, `SyncResult`.
- Deterministic IDs (UUIDv5) and content checksums, making ingestion idempotent.
- `Registry`, the name-to-factory plugin seam used by every swappable component.
- Storage ports (`DocumentRepository`, `ChunkRepository`, `VectorIndex`, `IngestStore`) defined in the domain layer.
- Explicit error taxonomy separating transient failures from permanent ones.

**Connectors**
- Filesystem connector for `.txt`, `.md`, `.json` and `.html`, with directory exclusion, size limits, and dependency-free HTML/JSON text extraction.
- PDF connector on PyMuPDF, extracting per-page text, document metadata, and page offsets that let a chunk be traced back to a page.

**Chunking**
- `Chunker` protocol and `ChunkerBase`.
- Fixed-size chunker with configurable size and overlap, cutting on token boundaries and recording character offsets back into the source.
- Pluggable token counting with a dependency-free approximate counter.

**Embeddings**
- `Embedder` protocol with batching and dimension validation.
- Providers: `sentence_transformers`, `openai`, and `hash` — a deterministic, dependency-free embedder for tests and quick local runs.
- Model identity (`provider:model:dimensions`) stored alongside every vector.

**Storage**
- PostgreSQL + pgvector backend with HNSW cosine index.
- Metadata filtering pushed entirely into SQL: source type, source ID, document ID, file type, author, repository, tags, date ranges, and arbitrary metadata containment.
- Vectors keyed by `(chunk_id, model_key)`, so one corpus can hold several embedding models at once.
- Alembic migrations running through the async engine.
- Atomic `index_document`: document, chunks and vectors are written in one transaction.

**Retrieval**
- `Retriever` protocol and dense retrieval over pgvector.
- Per-stage timing (`embedding_ms`, `retrieval_ms`, `total_ms`) collected through a task-local timer, so concurrent searches keep separate numbers.

**Pipeline**
- Incremental synchronisation: two-stage checksum comparison, pruning of documents deleted at the source, per-item failure isolation, and retry with exponential backoff.

**Configuration and observability**
- Typed settings from YAML with `${VAR}` interpolation and environment overrides, validated against the component registries at load time.
- Structured logging with request IDs and automatic redaction of credential-shaped keys.

**CLI**
- `recall init`, `migrate`, `status`, `connectors`, `ingest`, `search`, `documents list`, `documents show`.

**Project**
- Docker Compose for PostgreSQL with pgvector.
- 250+ tests across unit, integration and end-to-end layers.
- CI running lint, strict type checking, unit tests and integration tests.
- Architecture, getting-started, plugin and experiment documentation.

### Known limitations

- Only dense retrieval is implemented. `retrieval.default` accepts `dense`; BM25 and hybrid arrive in Milestone 2.
- No evaluation metrics, experiment runner, or benchmark results. **No performance or quality claims are made in this release.**
- No API server, background workers, or web dashboard.
- The `hash` embedder is a hashing-trick baseline, not a semantic model. It must not be used for quality claims.
- Changing the embedding dimension requires a new migration and a full re-index.
- Filtered vector search may fall back to an exact scan when the filter is highly selective; see [docs/architecture/storage.md](docs/architecture/storage.md).
- Scanned PDFs are not OCR'd.

[Unreleased]: https://github.com/TheWiseGhost/recall/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/TheWiseGhost/recall/releases/tag/v0.1.0

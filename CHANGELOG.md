# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Milestone 2 — retrieval research. Still to come: evaluation metrics, benchmark
datasets, the experiment runner and the report generator.

### Added

**Chunking**
- `sentence` chunker: packs whole sentences into token-budgeted windows, with overlap counted in sentences. A sentence longer than `chunk_size` becomes its own oversized chunk rather than being split.
- `semantic` chunker: embeds each sentence and breaks where consecutive sentences are far apart. The threshold is a percentile of each document's own distances, since absolute cosine distances are not comparable across models.
- `hierarchical` chunker: emits large parent chunks and the small child chunks inside them, linked by `chunks.parent_id`. Positions are unique across both levels because chunk IDs fold in the position.
- Dependency-free sentence segmentation in `core/chunking/sentences.py`.
- `chunking.overlap_sentences`, `breakpoint_percentile`, `buffer_size`, `max_chunk_size`, `min_sentences` and `parent_chunk_size` configuration.

**Reranking**
- `Reranker` protocol and `reranker_registry`, with `none` (identity) and `cross_encoder` (sentence-transformers `CrossEncoder`, behind the `local` extra, imported lazily).
- `SearchService` composes retrieval and reranking: with a reranker active the candidate pool widens to `reranking.top_n`, and `SearchResponse.candidates` records it. `reranking_ms` is recorded per query.
- `SearchResult.retrieval_score` preserves the pre-rerank score, so a report can say how much the reranker changed rather than only that it ran.
- `reranking.device`, `batch_size` and `max_length` configuration; `reranking.strategy` is validated against the registry at load time.
- `recall search --rerank <strategy>|off`.

**Retrieval**
- `hybrid` retriever: fans out to named component retrievers concurrently and fuses their rankings. Generic over its components — "dense + BM25" is the shipped configuration, not an assumption in the code.
- Rank fusion in a `fusion_registry`: `rrf` (reciprocal rank fusion, the default) and `weighted` (min-max normalised weighted score fusion). Weights are relative and rescaled to sum to 1.
- `SearchResult.component_scores` and `component_ranks` record what each component scored and ranked a chunk, so a fused result can still be attributed.
- `RetrievalTiming.fusion_ms`. Concurrent component timings are merged with `max`, not `sum`, so the breakdown cannot exceed the directly-measured `total_ms`.
- `hybrid.components` and `hybrid.candidate_multiplier` configuration; `hybrid.fusion` and `hybrid.components` are validated against the registries at load time.
- `bm25` retriever: Okapi BM25 over PostgreSQL full-text search — IDF times saturated term frequency with length normalisation, *not* `ts_rank_cd` renamed. Verified against an independent reference implementation.
- `LexicalIndex` port in `core/ports.py`, implemented by `PostgresBM25Index`.
- Collection statistics (`N`, `avgdl`, `n(t)`) are computed over the *filtered* corpus in the same statement, so a filtered search is scored against the corpus it actually searched.
- `lexical.k1` and `lexical.b` configuration.

**Storage**
- Migration `0002`: `chunks.content_length`, a stored generated column holding BM25's `|D|`, plus the `recall_tsvector_length(tsvector)` function it calls. Maintained by PostgreSQL for every writer.
- `content_tsv` and `content_length` are now mapped on `ChunkRow` as computed columns.

**CLI**
- `recall search --strategy/-s` selects the retrieval strategy per query.

### Changed
- **`Chunker.chunk` is now async.** Semantic chunking has to embed candidate sentences to find its boundaries, which a synchronous protocol cannot express. `ChunkerBase` subclasses that implement the synchronous `split()` — the documented plugin pattern — are unaffected; override `split_async()` only when splitting needs I/O.
- `build_chunker(settings, embedder=...)` supplies the pipeline's embedder to the semantic chunker.
- `build_context` no longer hardcodes dense retrieval; `build_retriever` resolves the strategy through `retriever_registry` and supplies its dependencies.

### Known limitations
- Sentence segmentation is a regex heuristic with an explicit abbreviation list, not a trained model. An abbreviation outside the list ends a sentence early.
- `hierarchical` embeds both levels, so ingestion cost roughly doubles. Embedding children only becomes possible once parent-child context selection exists.
- Cross-encoder scores are raw model outputs — not probabilities, and not comparable to cosine similarity or BM25. Only the ordering is meaningful.
- `weighted` fusion min-max normalises per query, so the bottom of each component's list scores 0 and fused scores are not comparable across queries. `rrf` is the default for this reason.
- BM25 recomputes collection statistics per query, which means a scan for `avgdl`. Fine at research corpus sizes; a materialized stats table is a future change.
- The text search configuration is `english`, compiled into a generated column. Changing it is a migration.
- PostgreSQL stores at most 256 positions per lexeme, so `f(t,D)` saturates at 256 occurrences.

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

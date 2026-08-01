# Writing a plugin

Every extensible component in Recall follows the same two-step pattern:

1. Implement the protocol.
2. Register the class under the name configuration will use.

No file in `recall/core` needs to change. If your plugin lives in another package, import it once before configuration is loaded — an entry point or a line in your application's startup is enough.

---

## A connector

A connector answers two questions: what exists, and what does one item contain. Reconciliation, chunking, embedding, indexing and pruning are handled by `IngestionPipeline`, so you get incremental sync for free.

```python
from recall.connectors.base import connector_registry
from recall.core.models import Document, SourceItem, SourceType


@connector_registry.decorator("rss")
class RSSConnector:
    source_type = SourceType.FILESYSTEM  # or add a value to SourceType

    def __init__(self, *, feed_url: str) -> None:
        self.feed_url = feed_url

    async def discover(self) -> list[SourceItem]:
        return [
            SourceItem(
                source_id=entry["id"],
                source_type=self.source_type,
                uri=entry["link"],
                title=entry["title"],
                # Supply a checksum when you can get one cheaply: it lets the
                # pipeline skip the fetch entirely for unchanged items.
                checksum=entry.get("etag"),
                metadata={"published": entry["published"]},
            )
            for entry in await self._list_entries()
        ]

    async def fetch(self, item: SourceItem) -> Document:
        body = await self._download(item.uri)
        return Document.create(
            source_id=item.source_id,
            source_type=self.source_type,
            title=item.title or item.uri,
            content=body,
            uri=item.uri,
            metadata=item.metadata,
        )
```

Rules that matter:

- **Use `Document.create`.** It derives the deterministic ID and the content checksum. Constructing a `Document` by hand and inventing an ID breaks idempotency and incremental sync.
- **`source_id` must be stable.** It identifies the document forever. A path relative to a configured root is good; an absolute path that changes when the corpus moves is not.
- **Normalize in `fetch`.** The checksum is computed from the normalized content, so normalization is what makes "unchanged" mean unchanged.
- **Raise `DocumentParseError` for a bad item, `TransientError` for a retryable one.** The first is recorded as a failure; the second is retried with backoff.
- **Never read credentials from the database.** Take them as constructor arguments sourced from environment variables.

---

## A chunker

Subclass `ChunkerBase` and implement `split`, returning `(text, start_char, end_char)` triples in document order. The base class derives chunk IDs, propagates source metadata, counts tokens and records strategy provenance.

The `Chunker` protocol's `chunk()` is async — semantic chunking has to embed candidate sentences to find its boundaries — but you do not have to be. `ChunkerBase.chunk()` awaits `split_async()`, which defaults to calling your synchronous `split()`. Override `split_async` only if your splitting needs I/O.

```python
from recall.core.chunking.base import ChunkerBase, chunker_registry
from recall.core.models import Document


@chunker_registry.decorator("paragraph")
class ParagraphChunker(ChunkerBase):
    name = "paragraph"

    def __init__(self, *, min_chars: int = 100, token_counter=None) -> None:
        super().__init__(token_counter=token_counter)
        self.min_chars = min_chars

    def params(self) -> dict[str, object]:
        # Recorded on every chunk, so an experiment can be reproduced.
        return {"min_chars": self.min_chars}

    def split(self, document: Document) -> list[tuple[str, int, int]]:
        spans, cursor = [], 0
        for block in document.content.split("\n\n"):
            start = document.content.index(block, cursor)
            cursor = start + len(block)
            if len(block.strip()) >= self.min_chars:
                spans.append((block, start, cursor))
        return spans
```

Offsets must satisfy `document.content[start:end] == text`. Parent-child expansion and PDF page citation both depend on it.

---

## An embedder

Subclass `EmbedderBase` and implement `_embed_batch`. The base class handles batching, empty input, and dimension validation.

```python
from recall.core.embeddings.base import EmbedderBase, embedder_registry


@embedder_registry.decorator("cohere")
class CohereEmbedder(EmbedderBase):
    provider = "cohere"

    def __init__(
        self, *, model: str, dimensions: int, batch_size: int = 96, api_key: str | None = None
    ) -> None:
        super().__init__(model=model, dimensions=dimensions, batch_size=batch_size)
        self.api_key = api_key

    async def _embed_batch(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        input_type = "search_query" if is_query else "search_document"
        ...
        return vectors
```

`is_query` exists because several models (BGE, E5, Cohere) need asymmetric handling. Ignore it if yours does not.

Import optional dependencies **inside** the method, and raise `EmbeddingProviderUnavailableError` with the install command when they are missing — importing at module scope would make the whole registry fail to load for users who do not want your provider.

---

## A retriever

`dense`, `bm25` and `hybrid` are already registered, so pick a free name — registering over a taken one raises unless you pass `override=True`.

```python
from recall.core.models import SearchFilters, SearchResult
from recall.core.retrieval.base import rerank_positions, retriever_registry, stage


@retriever_registry.decorator("recency_boosted")
class RecencyBoostedRetriever:
    name = "recency_boosted"

    def __init__(self, *, inner, half_life_days: float = 30.0) -> None:
        self.inner = inner
        self.half_life_days = half_life_days

    async def search(
        self, query, top_k=10, filters: SearchFilters | None = None
    ) -> list[SearchResult]:
        with stage("retrieval"):
            results = await self.inner.search(query, top_k=top_k * 3, filters=filters)
        rescored = sorted(results, key=self._score, reverse=True)
        return rerank_positions(
            [r.model_copy(update={"retriever": self.name}) for r in rescored[:top_k]]
        )
```

- Wrap meaningful work in `stage(...)`. It records against the ambient timer when one is active and is a no-op otherwise, which is how `SearchResponse.timing` gets its per-stage breakdown without the protocol growing a parameter.
- Return 1-based, sequential ranks. `rerank_positions` does it for you.
- Push filters down to storage. Filtering in Python changes what `top_k` means and invalidates every metric.
- Over-fetch before reordering. Reranking a list of `top_k` can only shuffle what it was already given.

---

## A reranker

```python
from collections.abc import Sequence

from recall.core.models import SearchResult
from recall.core.reranking.base import preserve_retrieval_score, reranker_registry
from recall.core.retrieval.base import rerank_positions


@reranker_registry.decorator("llm")
class LLMReranker:
    name = "llm"

    async def rerank(
        self, query: str, results: Sequence[SearchResult], *, top_k: int
    ) -> list[SearchResult]:
        scores = await self._judge(query, [r.content for r in results])
        scored = [
            preserve_retrieval_score(result, score)
            for result, score in zip(results, scores, strict=True)
        ]
        scored.sort(key=lambda r: (-r.score, str(r.chunk_id)))
        return rerank_positions(scored[:top_k])
```

Use `preserve_retrieval_score`: it moves the pre-rerank score into `retrieval_score` so a report can say how much your reranker changed the ordering, not merely that it ran. Run blocking model calls off the event loop with `asyncio.to_thread`.

---

## A fusion strategy

```python
from recall.core.retrieval.fusion import fusion_registry


@fusion_registry.decorator("borda")
class BordaFusion:
    name = "borda"

    def fuse(self, lists, *, top_k: int) -> list[SearchResult]: ...
```

`fuse` receives `{retriever_name: ranked_results}` and returns one ranking. `hybrid.fusion: borda` then selects it.

---

## A metric

```python
from recall.core.evaluation.metrics import METRIC_LABELS, metric_registry


def r_precision(ranked, judgement, k):
    relevant = {key for key, grade in judgement.items() if grade > 0}
    cutoff = len(relevant)
    return sum(1 for key in ranked[:cutoff] if key in relevant) / cutoff if cutoff else 0.0


metric_registry.register("r_precision", r_precision)
METRIC_LABELS["r_precision"] = "r_precision@{k}"
```

A metric takes a ranked list of label keys, a `key -> grade` judgement and `k`, and returns a float. `metrics: [r_precision]` in an experiment config then selects it. Decide deliberately whether a repeated key should score twice — see [the metric definitions](../experiments/index.md#metrics) for how the built-ins answer that.

---

## Wiring it up

Concrete classes are instantiated in `recall/pipeline/factory.py` from configuration. A plugin that only needs the parameters already in `recall.yaml` works immediately:

```yaml
chunking:
  strategy: paragraph
```

Configuration is validated against the registries at load time, so a name that is not registered fails at startup with a list of what is.

## Testing a plugin

Recall's own fakes are reusable. `tests/conftest.py` provides `FakeStorage` (an in-memory `IngestStore` plus an exact-cosine vector index), `FakeLexicalIndex` and `ListConnector` — enough to test a chunker, embedder, retriever, reranker or connector against the real pipeline without a database.

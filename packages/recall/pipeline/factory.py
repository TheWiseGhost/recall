"""Composition root.

The single place where configuration turns into concrete objects. Everything
else receives its collaborators through constructor arguments, which is what
makes components swappable in experiments and replaceable in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Self

from recall.config.settings import Settings
from recall.core.chunking import create_chunker
from recall.core.chunking.base import Chunker
from recall.core.embeddings import create_embedder
from recall.core.embeddings.base import Embedder
from recall.core.errors import ConfigurationError
from recall.core.reranking import create_reranker
from recall.core.reranking.base import Reranker
from recall.core.retrieval import create_retriever
from recall.core.retrieval.base import Retriever
from recall.core.retrieval.fusion import Fusion, create_fusion
from recall.pipeline.ingest import IngestionPipeline
from recall.pipeline.search import SearchService
from recall.storage.postgres.storage import Storage, create_storage


@dataclass(slots=True)
class RecallContext:
    """A fully wired Recall instance."""

    settings: Settings
    storage: Storage
    chunker: Chunker
    embedder: Embedder
    retriever: Retriever
    reranker: Reranker | None = None

    @property
    def ingestion(self) -> IngestionPipeline:
        return IngestionPipeline(storage=self.storage, chunker=self.chunker, embedder=self.embedder)

    @property
    def search(self) -> SearchService:
        return SearchService(
            retriever=self.retriever,
            reranker=self.reranker,
            rerank_candidates=self.settings.reranking.top_n,
        )

    async def close(self) -> None:
        await self.storage.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()


def build_chunker(settings: Settings, *, embedder: Embedder | None = None) -> Chunker:
    """Build the configured chunker.

    Semantic chunking finds its boundaries by embedding candidate sentences, so
    it needs the same embedder the rest of the pipeline uses — using a
    different one would make the boundaries depend on a model nothing else
    knows about.
    """
    kwargs = settings.chunking.factory_kwargs()
    if settings.chunking.strategy == "semantic":
        if embedder is None:
            raise ConfigurationError(
                "chunking.strategy=semantic needs an embedder; build it with "
                "build_chunker(settings, embedder=...)"
            )
        kwargs["embedder"] = embedder
    return create_chunker(settings.chunking.strategy, **kwargs)


def build_embedder(settings: Settings) -> Embedder:
    return create_embedder(settings.embedding.provider, **settings.embedding.factory_kwargs())


def build_reranker(settings: Settings) -> Reranker | None:
    """Build the configured reranker, or ``None`` when reranking is off.

    ``None`` rather than the identity reranker, so a disabled reranker costs
    nothing at all — not even a pass over the results — and never widens the
    candidate pool. Selecting ``strategy: none`` with ``enabled: true`` is the
    separate, deliberate case that *does* widen the pool; see
    :class:`~recall.config.settings.RerankingSettings`.
    """
    if not settings.reranking.enabled:
        return None
    return create_reranker(settings.reranking.strategy, **settings.reranking.factory_kwargs())


def build_fusion(settings: Settings) -> Fusion:
    """Build the fusion strategy named by ``hybrid.fusion``."""
    kwargs: dict[str, object] = {"weights": settings.hybrid.weights()}
    if settings.hybrid.fusion == "rrf":
        kwargs["k"] = settings.hybrid.rrf_k
    return create_fusion(settings.hybrid.fusion, **kwargs)


def build_retriever(
    strategy: str, *, storage: Storage, embedder: Embedder, settings: Settings
) -> Retriever:
    """Instantiate ``strategy`` with the collaborators it needs.

    Retrievers are resolved through the registry — so a plugin registering its
    own ``dense`` wins — but each takes different dependencies, and deciding
    which to hand it is exactly the composition root's job.
    """
    if strategy == "hybrid":
        # Recursive, because a hybrid's components are themselves retrievers.
        # `hybrid` is excluded to stop a config naming itself from recursing
        # forever.
        components = {
            name: build_retriever(name, storage=storage, embedder=embedder, settings=settings)
            for name in settings.hybrid.components
            if name != "hybrid"
        }
        if not components:
            raise ConfigurationError(
                "hybrid.components must name retrievers other than 'hybrid' itself"
            )
        return create_retriever(
            "hybrid",
            components=components,
            fusion=build_fusion(settings),
            candidate_multiplier=settings.hybrid.candidate_multiplier,
        )

    dependencies: dict[str, dict[str, object]] = {
        "dense": {"embedder": embedder, "index": storage.vectors},
        "bm25": {"index": storage.lexical},
    }
    if strategy not in dependencies:
        available = ", ".join([*sorted(dependencies), "hybrid"])
        raise ConfigurationError(
            f"retrieval strategy {strategy!r} cannot be wired up. Available: {available}."
        )
    return create_retriever(strategy, **dependencies[strategy])


def build_context(settings: Settings, *, retrieval_strategy: str | None = None) -> RecallContext:
    """Wire storage, chunker, embedder and retriever from ``settings``."""
    storage = create_storage(settings.database, lexical=settings.lexical)
    embedder = build_embedder(settings)
    chunker = build_chunker(settings, embedder=embedder)

    strategy = retrieval_strategy or settings.retrieval.default
    retriever = build_retriever(strategy, storage=storage, embedder=embedder, settings=settings)

    return RecallContext(
        settings=settings,
        storage=storage,
        chunker=chunker,
        embedder=embedder,
        retriever=retriever,
        reranker=build_reranker(settings),
    )

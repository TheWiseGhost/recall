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
from recall.core.retrieval import create_retriever
from recall.core.retrieval.base import Retriever
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

    @property
    def ingestion(self) -> IngestionPipeline:
        return IngestionPipeline(storage=self.storage, chunker=self.chunker, embedder=self.embedder)

    @property
    def search(self) -> SearchService:
        return SearchService(retriever=self.retriever)

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


def build_chunker(settings: Settings) -> Chunker:
    return create_chunker(settings.chunking.strategy, **settings.chunking.factory_kwargs())


def build_embedder(settings: Settings) -> Embedder:
    return create_embedder(settings.embedding.provider, **settings.embedding.factory_kwargs())


def build_retriever(strategy: str, *, storage: Storage, embedder: Embedder) -> Retriever:
    """Instantiate ``strategy`` with the collaborators it needs.

    Retrievers are resolved through the registry — so a plugin registering its
    own ``dense`` wins — but each takes different dependencies, and deciding
    which to hand it is exactly the composition root's job. Adding a strategy
    that reuses an existing dependency set needs no change here.
    """
    dependencies: dict[str, dict[str, object]] = {
        "dense": {"embedder": embedder, "index": storage.vectors},
        "bm25": {"index": storage.lexical},
    }
    if strategy not in dependencies:
        available = ", ".join(sorted(dependencies))
        raise ConfigurationError(
            f"retrieval strategy {strategy!r} cannot be wired up. Available: {available}. "
            "Hybrid retrieval and reranking arrive later in Milestone 2."
        )
    return create_retriever(strategy, **dependencies[strategy])


def build_context(settings: Settings, *, retrieval_strategy: str | None = None) -> RecallContext:
    """Wire storage, chunker, embedder and retriever from ``settings``."""
    storage = create_storage(settings.database, lexical=settings.lexical)
    embedder = build_embedder(settings)
    chunker = build_chunker(settings)

    strategy = retrieval_strategy or settings.retrieval.default
    retriever = build_retriever(strategy, storage=storage, embedder=embedder)

    return RecallContext(
        settings=settings,
        storage=storage,
        chunker=chunker,
        embedder=embedder,
        retriever=retriever,
    )

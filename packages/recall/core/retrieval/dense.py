"""Dense retrieval over a vector index."""

from __future__ import annotations

from recall.core.embeddings.base import Embedder
from recall.core.models import SearchFilters, SearchResult
from recall.core.ports import VectorIndex
from recall.core.retrieval.base import rerank_positions, retriever_registry, stage


@retriever_registry.decorator("dense")
class DenseRetriever:
    """Embeds the query and asks the vector index for nearest neighbours.

    Filtering is delegated to the index so it happens in the database, not by
    over-fetching and discarding in Python.
    """

    name = "dense"

    def __init__(self, *, embedder: Embedder, index: VectorIndex) -> None:
        self.embedder = embedder
        self.index = index

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        with stage("embedding"):
            vector = await self.embedder.embed_query(query)
        with stage("retrieval"):
            results = await self.index.query(
                vector, top_k=top_k, filters=filters, model=self.embedder.info
            )
        return rerank_positions(
            [result.model_copy(update={"retriever": self.name}) for result in results]
        )

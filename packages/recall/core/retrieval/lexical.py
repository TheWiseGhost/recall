"""Lexical (sparse) retrieval.

The counterpart to :mod:`recall.core.retrieval.dense`. Where dense retrieval
asks "what is semantically near this query?", lexical retrieval asks "what
contains these exact terms, weighted by how rare and how saturated they are?".
Exact identifiers — function names, error codes, flags — are precisely the
signal embeddings blur, which is why the comparison is worth running.

Scoring lives behind the :class:`~recall.core.ports.LexicalIndex` port, because
it needs an inverted index and corpus-wide term statistics that only the
storage engine has. This module stays free of SQL.
"""

from __future__ import annotations

from recall.core.models import SearchFilters, SearchResult
from recall.core.ports import LexicalIndex
from recall.core.retrieval.base import rerank_positions, retriever_registry, stage


@retriever_registry.decorator("bm25")
class BM25Retriever:
    """Ranks chunks by BM25 over the lexical index.

    Filtering is delegated to the index, which applies it in SQL *before*
    computing collection statistics — so a filtered search is scored against
    the corpus it actually searched.
    """

    name = "bm25"

    def __init__(self, *, index: LexicalIndex) -> None:
        self.index = index

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        # No embedding stage: the whole point of BM25 is that there is no model
        # in the path. An experiment comparing it against dense retrieval will
        # see embedding_ms of 0 here, which is the honest number.
        with stage("retrieval"):
            results = await self.index.search(query, top_k=top_k, filters=filters)
        return rerank_positions(
            [result.model_copy(update={"retriever": self.name}) for result in results]
        )

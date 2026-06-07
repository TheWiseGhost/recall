"""Retrieval strategies.

Milestone 1 ships dense retrieval only. BM25, hybrid scoring and reciprocal
rank fusion arrive in Milestone 2 and plug into the same registry.
"""

from recall.core.retrieval.base import RetrievalTiming, Retriever, retriever_registry
from recall.core.retrieval.dense import DenseRetriever

__all__ = [
    "DenseRetriever",
    "RetrievalTiming",
    "Retriever",
    "create_retriever",
    "retriever_registry",
]


def create_retriever(strategy: str, **kwargs: object) -> Retriever:
    """Instantiate the retriever registered under ``strategy``."""
    return retriever_registry.create(strategy, **kwargs)

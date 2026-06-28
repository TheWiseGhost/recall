"""Retrieval strategies.

Importing the concrete modules is what populates the registry. Hybrid scoring
and reciprocal rank fusion plug into the same registry.
"""

from recall.core.retrieval.base import RetrievalTiming, Retriever, retriever_registry
from recall.core.retrieval.dense import DenseRetriever
from recall.core.retrieval.lexical import BM25Retriever

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "RetrievalTiming",
    "Retriever",
    "create_retriever",
    "retriever_registry",
]


def create_retriever(strategy: str, **kwargs: object) -> Retriever:
    """Instantiate the retriever registered under ``strategy``."""
    return retriever_registry.create(strategy, **kwargs)

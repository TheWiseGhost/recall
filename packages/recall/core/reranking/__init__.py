"""Reranking strategies.

Importing the concrete modules is what populates the registry. Whether
reranking pays for its latency is one of the questions Recall exists to answer,
so the identity reranker is a real component rather than a special case.
"""

from recall.core.reranking.base import NoOpReranker, Reranker, reranker_registry
from recall.core.reranking.cross_encoder import (
    CrossEncoderReranker,
    RerankerUnavailableError,
)

__all__ = [
    "CrossEncoderReranker",
    "NoOpReranker",
    "Reranker",
    "RerankerUnavailableError",
    "create_reranker",
    "reranker_registry",
]


def create_reranker(strategy: str, **kwargs: object) -> Reranker:
    """Instantiate the reranker registered under ``strategy``."""
    return reranker_registry.create(strategy, **kwargs)

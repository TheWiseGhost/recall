"""Embedding providers.

The embedding model is never hardcoded. It is selected by name from
configuration and its identity is stored alongside every vector, so a database
can always answer "which model produced this index?".
"""

from recall.core.embeddings.base import (
    Embedder,
    EmbedderBase,
    EmbeddingModelInfo,
    embedder_registry,
)
from recall.core.embeddings.hashing import HashingEmbedder
from recall.core.embeddings.openai import OpenAIEmbedder
from recall.core.embeddings.sentence_transformers import SentenceTransformersEmbedder

__all__ = [
    "Embedder",
    "EmbedderBase",
    "EmbeddingModelInfo",
    "HashingEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformersEmbedder",
    "create_embedder",
    "embedder_registry",
]


def create_embedder(provider: str, **kwargs: object) -> Embedder:
    """Instantiate the embedder registered under ``provider``."""
    return embedder_registry.create(provider, **kwargs)

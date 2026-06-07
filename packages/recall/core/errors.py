"""Explicit error taxonomy.

Callers should be able to distinguish *what kind* of thing went wrong without
string matching. Transient errors are retryable by background workers;
permanent ones must go to an error state instead of being retried forever.
"""

from __future__ import annotations


class RecallError(Exception):
    """Base class for every error Recall raises deliberately."""


class ConfigurationError(RecallError):
    """Configuration is missing, malformed, or internally inconsistent."""


class PluginNotFoundError(ConfigurationError):
    """A component was requested by name but nothing is registered under it."""


class UnsupportedFileTypeError(RecallError):
    """A connector was handed a file extension it cannot parse."""

    def __init__(self, path: str, suffix: str) -> None:
        super().__init__(f"Unsupported file type {suffix!r} for {path}")
        self.path = path
        self.suffix = suffix


class DocumentParseError(RecallError):
    """A source item was located but could not be turned into a Document."""


class ConnectorAuthError(RecallError):
    """A connector could not authenticate against its remote source."""


class TransientError(RecallError):
    """A failure that is expected to succeed on retry (network blips, 429s)."""


class EmbeddingError(RecallError):
    """The embedding provider failed permanently."""


class EmbeddingProviderUnavailableError(EmbeddingError):
    """The provider's optional dependency is not installed."""


class StorageError(RecallError):
    """The storage backend failed."""


class DimensionMismatchError(StorageError):
    """A vector was written whose dimension does not match the index."""

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(
            f"Vector index expects {expected}-dimensional vectors, got {actual}. "
            "Re-run migrations with the new embedding dimension, or re-index."
        )
        self.expected = expected
        self.actual = actual

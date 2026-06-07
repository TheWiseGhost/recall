"""Structured logging.

Every log line is a dict. A request/job ID is bound into a context variable so
it propagates through the whole pipeline without being threaded manually
through every function signature.
"""

from __future__ import annotations

import logging
import sys
import uuid
from typing import Any, Literal, TextIO, cast

import structlog

_configured = False

# Keys whose values must never reach a log sink.
_REDACTED_KEYS = frozenset(
    {"api_key", "apikey", "token", "password", "secret", "authorization", "access_token"}
)
_REDACTED = "***redacted***"


class _StderrProxy:
    """Resolves ``sys.stderr`` at write time rather than at configure time.

    structlog caches bound loggers, and a logger factory given a concrete
    stream holds that exact object forever. Any process that swaps its streams
    afterwards — a test runner capturing output, a worker that redirects logs —
    would then write to a stale or closed file.
    """

    def write(self, message: str) -> int:
        return sys.stderr.write(message)

    def flush(self) -> None:
        sys.stderr.flush()

    def isatty(self) -> bool:
        return sys.stderr.isatty()


def _redact(
    _logger: Any, _name: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    for key in list(event_dict):
        if key.lower() in _REDACTED_KEYS:
            event_dict[key] = _REDACTED
    return event_dict


def configure_logging(level: str = "INFO", fmt: Literal["console", "json"] = "console") -> None:
    """Configure structlog and the stdlib root logger. Idempotent."""
    global _configured

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=numeric_level, force=True)

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _redact,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        # The proxy is duck-typed as a text stream; structlog only writes and
        # flushes it.
        logger_factory=structlog.PrintLoggerFactory(file=cast(TextIO, _StderrProxy())),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, configuring logging on first use."""
    if not _configured:
        configure_logging()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def bind_request_id(request_id: str | None = None) -> str:
    """Bind a request/job ID into the logging context and return it."""
    resolved = request_id or uuid.uuid4().hex[:16]
    structlog.contextvars.bind_contextvars(request_id=resolved)
    return resolved


def clear_context() -> None:
    """Drop all bound context variables (call between jobs in a worker)."""
    structlog.contextvars.clear_contextvars()

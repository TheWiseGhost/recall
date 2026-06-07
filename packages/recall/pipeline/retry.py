"""Retry with exponential backoff for transient failures.

Only :class:`~recall.core.errors.TransientError` (and anything explicitly
listed by the caller) is retried. Permanent errors propagate immediately so
they can be recorded as a failure instead of being retried forever.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

from recall.core.errors import TransientError
from recall.observability.logging import get_logger

_log = get_logger(__name__)


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    jitter: float = 0.1,
    retry_on: tuple[type[BaseException], ...] = (TransientError,),
    description: str = "operation",
) -> T:
    """Run ``operation``, retrying transient failures with exponential backoff.

    Raises:
        The last exception, if every attempt fails.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except retry_on as exc:
            last = exc
            if attempt == attempts:
                break
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            delay += random.uniform(0, jitter * delay)
            _log.warning(
                "retrying_after_transient_error",
                operation=description,
                attempt=attempt,
                attempts=attempts,
                delay_seconds=round(delay, 3),
                error=str(exc),
            )
            await asyncio.sleep(delay)

    assert last is not None
    raise last

"""Retry with exponential backoff."""

from __future__ import annotations

import pytest

from recall.core.errors import EmbeddingError, TransientError
from recall.pipeline.retry import retry_async


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff delays instead of waiting for them."""
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("recall.pipeline.retry.asyncio.sleep", fake_sleep)
    return delays


class TestRetry:
    async def test_returns_immediately_on_success(self) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        assert await retry_async(operation) == "ok"
        assert calls == 1

    async def test_retries_transient_failures_then_succeeds(self) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise TransientError("rate limited")
            return "ok"

        assert await retry_async(operation, attempts=3) == "ok"
        assert calls == 3

    async def test_gives_up_after_the_attempt_budget(self) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            raise TransientError("still down")

        with pytest.raises(TransientError, match="still down"):
            await retry_async(operation, attempts=3)
        assert calls == 3

    async def test_permanent_errors_are_not_retried(self) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            raise EmbeddingError("bad request")

        with pytest.raises(EmbeddingError):
            await retry_async(operation, attempts=5)
        assert calls == 1

    async def test_backoff_grows(self, no_sleeping: list[float]) -> None:
        async def operation() -> str:
            raise TransientError("down")

        with pytest.raises(TransientError):
            await retry_async(operation, attempts=4, base_delay=1.0, jitter=0.0)
        assert no_sleeping == [1.0, 2.0, 4.0]

    async def test_backoff_is_capped(self, no_sleeping: list[float]) -> None:
        async def operation() -> str:
            raise TransientError("down")

        with pytest.raises(TransientError):
            await retry_async(operation, attempts=6, base_delay=1.0, max_delay=3.0, jitter=0.0)
        assert max(no_sleeping) == 3.0

    async def test_no_sleep_after_the_final_attempt(self, no_sleeping: list[float]) -> None:
        async def operation() -> str:
            raise TransientError("down")

        with pytest.raises(TransientError):
            await retry_async(operation, attempts=2, jitter=0.0)
        assert len(no_sleeping) == 1

    async def test_retry_on_is_configurable(self) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            raise ValueError("flaky")

        with pytest.raises(ValueError):
            await retry_async(operation, attempts=3, retry_on=(ValueError,))
        assert calls == 3

    async def test_zero_attempts_is_rejected(self) -> None:
        async def operation() -> str:
            return "ok"

        with pytest.raises(ValueError, match="at least 1"):
            await retry_async(operation, attempts=0)

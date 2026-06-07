"""A tiny name -> factory registry.

Every swappable component (chunker, embedder, retriever, reranker, connector,
...) is looked up by the string that appears in configuration. A plugin
therefore needs to do exactly two things: implement the protocol, and call
``register`` on the matching registry at import time. No core file needs to
change.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from recall.core.errors import PluginNotFoundError

type Factory[T] = Callable[..., T]


class Registry[T]:
    """Case-insensitive registry of named factories."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Factory[T]] = {}

    @staticmethod
    def _key(name: str) -> str:
        return name.strip().lower().replace("-", "_")

    def register(self, name: str, factory: Factory[T], *, override: bool = False) -> None:
        """Register ``factory`` under ``name``.

        Raises:
            ValueError: if the name is taken and ``override`` is not set.
        """
        key = self._key(name)
        if key in self._factories and not override:
            raise ValueError(f"{self.kind} {name!r} is already registered")
        self._factories[key] = factory

    def decorator(self, name: str, *, override: bool = False) -> Callable[[Factory[T]], Factory[T]]:
        """Class/function decorator form of :meth:`register`."""

        def wrap(factory: Factory[T]) -> Factory[T]:
            self.register(name, factory, override=override)
            return factory

        return wrap

    def create(self, name: str, /, **kwargs: Any) -> T:
        """Instantiate the component registered under ``name``."""
        key = self._key(name)
        try:
            factory = self._factories[key]
        except KeyError:
            raise PluginNotFoundError(
                f"Unknown {self.kind} {name!r}. Available: {', '.join(self.names()) or '(none)'}"
            ) from None
        return factory(**kwargs)

    def get(self, name: str) -> Factory[T]:
        """Return the raw factory without instantiating it."""
        key = self._key(name)
        try:
            return self._factories[key]
        except KeyError:
            raise PluginNotFoundError(
                f"Unknown {self.kind} {name!r}. Available: {', '.join(self.names()) or '(none)'}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._key(name) in self._factories

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._factories))

    def names(self) -> list[str]:
        return sorted(self._factories)

    def as_mapping(self) -> Mapping[str, Factory[T]]:
        return dict(self._factories)

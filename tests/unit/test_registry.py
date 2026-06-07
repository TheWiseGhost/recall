"""The plugin registry."""

from __future__ import annotations

import pytest

from recall.core.errors import PluginNotFoundError
from recall.core.registry import Registry


class Widget:
    def __init__(self, size: int = 1) -> None:
        self.size = size


@pytest.fixture
def registry() -> Registry[Widget]:
    return Registry("widget")


class TestRegistry:
    def test_register_and_create(self, registry: Registry[Widget]) -> None:
        registry.register("basic", Widget)
        widget = registry.create("basic", size=5)
        assert isinstance(widget, Widget)
        assert widget.size == 5

    def test_lookup_is_case_and_separator_insensitive(self, registry: Registry[Widget]) -> None:
        registry.register("cross-encoder", Widget)
        assert "cross_encoder" in registry
        assert "CROSS-ENCODER" in registry
        assert registry.create("Cross_Encoder") is not None

    def test_duplicate_registration_is_rejected(self, registry: Registry[Widget]) -> None:
        registry.register("basic", Widget)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("basic", Widget)

    def test_override_is_explicit(self, registry: Registry[Widget]) -> None:
        registry.register("basic", Widget)
        registry.register("basic", lambda **_: Widget(size=99), override=True)
        assert registry.create("basic").size == 99

    def test_unknown_name_lists_alternatives(self, registry: Registry[Widget]) -> None:
        registry.register("alpha", Widget)
        registry.register("beta", Widget)
        with pytest.raises(PluginNotFoundError, match="alpha, beta"):
            registry.create("gamma")

    def test_decorator_form(self, registry: Registry[Widget]) -> None:
        @registry.decorator("decorated")
        class Decorated(Widget):
            pass

        assert isinstance(registry.create("decorated"), Decorated)

    def test_names_are_sorted(self, registry: Registry[Widget]) -> None:
        for name in ("zeta", "alpha", "mu"):
            registry.register(name, Widget)
        assert registry.names() == ["alpha", "mu", "zeta"]
        assert list(registry) == ["alpha", "mu", "zeta"]

    def test_get_returns_factory_without_instantiating(self, registry: Registry[Widget]) -> None:
        registry.register("basic", Widget)
        assert registry.get("basic") is Widget


class TestBuiltInRegistries:
    def test_core_components_are_registered(self) -> None:
        from recall.connectors import connector_registry
        from recall.core.chunking import chunker_registry
        from recall.core.embeddings import embedder_registry
        from recall.core.retrieval import retriever_registry

        assert "fixed" in chunker_registry
        assert {"hash", "openai", "sentence_transformers"} <= set(embedder_registry.names())
        assert "dense" in retriever_registry
        assert {"filesystem", "pdf"} <= set(connector_registry.names())

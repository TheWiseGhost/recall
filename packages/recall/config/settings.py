"""Typed, validated configuration.

Precedence, lowest to highest:

1. defaults defined here
2. ``recall.yaml`` (discovered upwards from the cwd, or ``RECALL_CONFIG``)
3. environment variables (``RECALL_<SECTION>__<FIELD>``)

Everything is validated on load, so a malformed config fails at startup with a
pointed message instead of at the first search.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import Field, SecretStr, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from recall.config.interpolation import interpolate
from recall.core.errors import ConfigurationError
from recall.core.models import RecallModel
from recall.core.registry import Registry

CONFIG_FILENAMES = ("recall.yaml", "recall.yml")
DEFAULT_DATABASE_URL = "postgresql+asyncpg://recall:recall@localhost:5432/recall"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"


class DatabaseSettings(RecallModel):
    url: str = DEFAULT_DATABASE_URL
    pool_size: int = Field(default=5, ge=1, le=100)
    max_overflow: int = Field(default=10, ge=0, le=100)
    echo: bool = False
    statement_timeout_ms: int = Field(default=30_000, ge=0)

    @model_validator(mode="after")
    def _require_async_driver(self) -> Self:
        if not self.url.startswith("postgresql"):
            raise ValueError(
                f"database.url must be a PostgreSQL URL, got {self.url.split('://')[0]!r}"
            )
        if "+asyncpg" not in self.url:
            raise ValueError(
                "database.url must use the asyncpg driver, e.g. "
                "postgresql+asyncpg://user:pass@host:5432/recall"
            )
        return self


class RedisSettings(RecallModel):
    url: str = DEFAULT_REDIS_URL


class EmbeddingSettings(RecallModel):
    provider: str = "sentence_transformers"
    model: str = "BAAI/bge-base-en-v1.5"
    # Required, because the pgvector column is created with a fixed dimension.
    dimensions: int = Field(default=768, ge=2, le=16_000)
    batch_size: int = Field(default=32, ge=1, le=2048)
    device: str | None = None
    api_key: SecretStr | None = None
    base_url: str | None = None

    def factory_kwargs(self) -> dict[str, Any]:
        """Kwargs for :func:`recall.core.embeddings.create_embedder`."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "dimensions": self.dimensions,
            "batch_size": self.batch_size,
        }
        if self.provider == "sentence_transformers":
            kwargs["device"] = self.device
        elif self.provider == "openai":
            kwargs["api_key"] = self.api_key.get_secret_value() if self.api_key else None
            kwargs["base_url"] = self.base_url
        return kwargs


class ChunkingSettings(RecallModel):
    strategy: str = "fixed"
    chunk_size: int = Field(default=512, ge=16, le=16_000)
    overlap: int = Field(default=64, ge=0)

    @model_validator(mode="after")
    def _overlap_fits(self) -> Self:
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"chunking.overlap ({self.overlap}) must be smaller than "
                f"chunking.chunk_size ({self.chunk_size})"
            )
        return self

    def factory_kwargs(self) -> dict[str, Any]:
        if self.strategy == "fixed":
            return {"chunk_size": self.chunk_size, "overlap": self.overlap}
        return {}


class LexicalSettings(RecallModel):
    """BM25 parameters for lexical retrieval.

    The defaults are Robertson & Zaragoza's, and what Lucene ships. ``k1``
    controls how fast term-frequency gains saturate; ``b`` how strongly scores
    are normalised by chunk length, with ``b=0`` disabling it entirely. Both are
    worth sweeping — chunk length distribution differs wildly between corpora.

    The PostgreSQL text search configuration is deliberately *not* here: it is
    compiled into a generated column, so changing it is a migration, exactly
    like ``embedding.dimensions``.
    """

    k1: float = Field(default=1.2, ge=0.0, le=10.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)


class HybridSettings(RecallModel):
    """Hybrid retrieval: which signals to fuse, and how.

    Weights are relative — they are rescaled to sum to 1 before use, so
    ``0.65/0.35`` and ``65/35`` behave identically and doubling both changes
    nothing.
    """

    dense_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    lexical_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    fusion: Literal["weighted", "rrf"] = "rrf"
    rrf_k: int = Field(default=60, ge=1)
    # Each component is asked for `candidate_multiplier * top_k` results before
    # fusion. Fusion can only see what a component returned, so a chunk outside
    # one component's truncated list is scored as if that component rejected it.
    # Over-fetching pushes that boundary out.
    candidate_multiplier: int = Field(default=3, ge=1, le=20)
    # Which retrievers to fuse. Every name must be a registered retriever that
    # the composition root knows how to build.
    components: list[str] = Field(default_factory=lambda: ["dense", "bm25"])

    @model_validator(mode="after")
    def _weights_are_usable(self) -> Self:
        if self.dense_weight + self.lexical_weight <= 0:
            raise ValueError("hybrid.dense_weight and hybrid.lexical_weight must not both be zero")
        if not self.components:
            raise ValueError("hybrid.components must name at least one retriever")
        if len(set(self.components)) != len(self.components):
            raise ValueError(f"hybrid.components contains duplicates: {self.components}")
        return self

    def weights(self) -> dict[str, float]:
        """Per-component weights, keyed by retriever name.

        Anything other than ``dense`` and ``bm25`` gets weight 1: a component
        Recall has no dedicated setting for should not silently be weighted 0.
        """
        named = {"dense": self.dense_weight, "bm25": self.lexical_weight}
        return {name: named.get(name, 1.0) for name in self.components}


class RetrievalSettings(RecallModel):
    default: str = "dense"
    top_k: int = Field(default=10, ge=1, le=1000)


class RerankingSettings(RecallModel):
    """Reranking configuration.

    ``top_n`` is the candidate pool retrieval is asked for before reranking —
    a reranker can only reorder what it is given, so this is the knob that
    decides how much it can improve. It is also the knob that decides how much
    latency it costs, which is the whole trade-off.

    ``enabled: true`` with ``strategy: none`` is a deliberate configuration: it
    widens the pool and truncates without reordering, isolating the effect of
    the wider pool from the effect of the reranker itself.
    """

    enabled: bool = False
    strategy: str = "none"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = Field(default=50, ge=1)
    device: str | None = None
    batch_size: int = Field(default=32, ge=1, le=512)
    max_length: int = Field(default=512, ge=16, le=8192)

    def factory_kwargs(self) -> dict[str, Any]:
        """Kwargs for :func:`recall.core.reranking.create_reranker`."""
        if self.strategy == "cross_encoder":
            return {
                "model": self.model,
                "device": self.device,
                "batch_size": self.batch_size,
                "max_length": self.max_length,
            }
        return {}


class LoggingSettings(RecallModel):
    level: str = "INFO"
    format: Literal["console", "json"] = "console"


# The parsed YAML document, handed to the settings source below. A context
# variable rather than a class attribute so concurrent loads cannot interleave.
# `None` rather than `{}` as the default: a mutable default on a ContextVar is
# shared across every context that never sets it.
_yaml_data: ContextVar[dict[str, Any] | None] = ContextVar("recall_yaml_data", default=None)


class _YamlSource(PydanticBaseSettingsSource):
    """Feeds the parsed ``recall.yaml`` in as a settings source.

    It has to be a *source* rather than constructor kwargs: pydantic-settings
    ranks init kwargs above environment variables, which would make
    ``RECALL_CHUNKING__CHUNK_SIZE`` unable to override the file.
    """

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return (_yaml_data.get() or {}).get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(_yaml_data.get() or {})


class Settings(BaseSettings):
    """Root configuration object."""

    model_config = SettingsConfigDict(
        env_prefix="RECALL_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        nested_model_default_partial_update=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Highest priority first.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSource(settings_cls),
            file_secret_settings,
        )

    project_name: str = "recall"
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    lexical: LexicalSettings = Field(default_factory=LexicalSettings)
    hybrid: HybridSettings = Field(default_factory=HybridSettings)
    reranking: RerankingSettings = Field(default_factory=RerankingSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Where experiment artefacts are written.
    experiments_dir: Path = Path("experiments")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Settings:
        """Build settings from a config mapping, layering env vars on top."""
        token = _yaml_data.set(data)
        try:
            return cls()
        except Exception as exc:
            raise ConfigurationError(f"Invalid configuration: {exc}") from exc
        finally:
            _yaml_data.reset(token)


def find_config_file(start: Path | None = None) -> Path | None:
    """Locate ``recall.yaml``.

    ``RECALL_CONFIG`` wins if set; otherwise walk up from ``start`` (default:
    cwd) to the filesystem root.
    """
    explicit = os.environ.get("RECALL_CONFIG")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigurationError(f"RECALL_CONFIG points at a missing file: {path}")
        return path

    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        for name in CONFIG_FILENAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def load_settings(path: Path | None = None, *, start: Path | None = None) -> Settings:
    """Load and validate settings from ``path`` or a discovered config file."""
    config_path = path or find_config_file(start)
    data: dict[str, Any] = {}

    if config_path is not None:
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"{config_path} is not valid YAML: {exc}") from exc
        except OSError as exc:
            raise ConfigurationError(f"Could not read {config_path}: {exc}") from exc

        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigurationError(f"{config_path} must contain a YAML mapping at the top level")
        data = interpolate(raw)

    _apply_conventional_env_aliases(data)
    settings = Settings.from_mapping(data)
    _validate_component_names(settings)
    return settings


def _apply_conventional_env_aliases(data: dict[str, Any]) -> None:
    """Honour the conventional ``DATABASE_URL`` / ``REDIS_URL`` names.

    Docker Compose, Heroku and friends all set these; requiring
    ``RECALL_DATABASE__URL`` instead would be gratuitous. Explicit
    ``RECALL_*`` variables still win, since pydantic-settings applies them
    after this.
    """
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        data.setdefault("database", {})
        if isinstance(data["database"], dict):
            data["database"].setdefault("url", database_url)

    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        data.setdefault("redis", {})
        if isinstance(data["redis"], dict):
            data["redis"].setdefault("url", redis_url)


def _validate_component_names(settings: Settings) -> None:
    """Fail fast when configuration names a component that is not registered."""
    from recall.core.chunking import chunker_registry
    from recall.core.embeddings import embedder_registry
    from recall.core.reranking import reranker_registry
    from recall.core.retrieval import fusion_registry, retriever_registry

    checks: list[tuple[str, str, Registry[Any]]] = [
        ("chunking.strategy", settings.chunking.strategy, chunker_registry),
        ("embedding.provider", settings.embedding.provider, embedder_registry),
        ("retrieval.default", settings.retrieval.default, retriever_registry),
        ("hybrid.fusion", settings.hybrid.fusion, fusion_registry),
        ("reranking.strategy", settings.reranking.strategy, reranker_registry),
    ]
    checks += [
        (f"hybrid.components[{position}]", name, retriever_registry)
        for position, name in enumerate(settings.hybrid.components)
    ]
    for key, value, registry in checks:
        if value not in registry:
            raise ConfigurationError(
                f"{key}={value!r} is not registered. Available: {', '.join(registry.names())}"
            )

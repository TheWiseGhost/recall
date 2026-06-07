"""Configuration parsing, interpolation and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from recall.config.interpolation import interpolate
from recall.config.settings import Settings, find_config_file, load_settings
from recall.core.errors import ConfigurationError

VALID_URL = "postgresql+asyncpg://recall:recall@localhost:5432/recall"


class TestInterpolation:
    def test_expands_a_set_variable(self) -> None:
        assert interpolate("${HOST}", {"HOST": "db"}) == "db"

    def test_expands_inside_a_larger_string(self) -> None:
        assert interpolate("redis://${HOST}:6379/0", {"HOST": "cache"}) == "redis://cache:6379/0"

    def test_default_applies_when_unset(self) -> None:
        assert interpolate("${HOST:-localhost}", {}) == "localhost"

    def test_colon_dash_default_applies_when_empty(self) -> None:
        assert interpolate("${HOST:-localhost}", {"HOST": ""}) == "localhost"

    def test_dash_default_keeps_an_empty_value(self) -> None:
        assert interpolate("${HOST-localhost}", {"HOST": ""}) == ""

    def test_missing_required_variable_is_an_error(self) -> None:
        with pytest.raises(ConfigurationError, match="HOST"):
            interpolate("${HOST}", {})

    def test_recurses_into_containers(self) -> None:
        data = {"db": {"url": "${URL}"}, "hosts": ["${A}", "static"]}
        assert interpolate(data, {"URL": "u", "A": "a"}) == {
            "db": {"url": "u"},
            "hosts": ["a", "static"],
        }

    def test_leaves_non_strings_alone(self) -> None:
        assert interpolate({"n": 5, "b": True, "none": None}, {}) == {
            "n": 5,
            "b": True,
            "none": None,
        }


class TestDatabaseSettings:
    def test_rejects_a_non_postgres_url(self) -> None:
        with pytest.raises(ConfigurationError, match="PostgreSQL"):
            Settings.from_mapping({"database": {"url": "mysql+aiomysql://x/y"}})

    def test_rejects_a_synchronous_driver(self) -> None:
        with pytest.raises(ConfigurationError, match="asyncpg"):
            Settings.from_mapping({"database": {"url": "postgresql://u:p@h:5432/db"}})

    def test_accepts_the_async_driver(self) -> None:
        assert Settings.from_mapping({"database": {"url": VALID_URL}}).database.url == VALID_URL


class TestChunkingSettings:
    def test_rejects_overlap_at_or_above_chunk_size(self) -> None:
        with pytest.raises(ConfigurationError, match="overlap"):
            Settings.from_mapping({"chunking": {"chunk_size": 128, "overlap": 128}})

    def test_factory_kwargs_for_fixed(self) -> None:
        settings = Settings.from_mapping({"chunking": {"chunk_size": 256, "overlap": 32}})
        assert settings.chunking.factory_kwargs() == {"chunk_size": 256, "overlap": 32}


class TestValidationOnLoad:
    def test_unknown_top_level_key_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            Settings.from_mapping({"nonsense": True})

    def test_unregistered_component_name_fails_at_load(self, tmp_path: Path) -> None:
        config = tmp_path / "recall.yaml"
        config.write_text("chunking:\n  strategy: telepathic\n")
        with pytest.raises(ConfigurationError, match="not registered"):
            load_settings(config)

    def test_malformed_yaml_is_reported(self, tmp_path: Path) -> None:
        config = tmp_path / "recall.yaml"
        config.write_text("embedding: [unclosed\n")
        with pytest.raises(ConfigurationError, match="not valid YAML"):
            load_settings(config)

    def test_non_mapping_yaml_is_reported(self, tmp_path: Path) -> None:
        config = tmp_path / "recall.yaml"
        config.write_text("- just\n- a\n- list\n")
        with pytest.raises(ConfigurationError, match="mapping"):
            load_settings(config)

    def test_empty_file_falls_back_to_defaults(self, tmp_path: Path) -> None:
        config = tmp_path / "recall.yaml"
        config.write_text("")
        assert load_settings(config).chunking.strategy == "fixed"


class TestLoading:
    def test_reads_values_from_yaml(self, tmp_path: Path) -> None:
        config = tmp_path / "recall.yaml"
        config.write_text(
            "embedding:\n"
            "  provider: hash\n"
            "  model: hash-v1\n"
            "  dimensions: 128\n"
            "chunking:\n"
            "  chunk_size: 256\n"
            "  overlap: 16\n"
        )
        settings = load_settings(config)
        assert settings.embedding.provider == "hash"
        assert settings.embedding.dimensions == 128
        assert settings.chunking.chunk_size == 256

    def test_partial_section_keeps_other_defaults(self, tmp_path: Path) -> None:
        config = tmp_path / "recall.yaml"
        config.write_text("embedding:\n  provider: hash\n  dimensions: 64\n")
        settings = load_settings(config)
        assert settings.embedding.batch_size == 32

    def test_environment_variables_override_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tmp_path / "recall.yaml"
        config.write_text("chunking:\n  chunk_size: 256\n")
        monkeypatch.setenv("RECALL_CHUNKING__CHUNK_SIZE", "1024")
        assert load_settings(config).chunking.chunk_size == 1024

    def test_conventional_database_url_is_honoured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tmp_path / "recall.yaml"
        config.write_text("chunking:\n  chunk_size: 256\n")
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@elsewhere:5432/db")
        assert load_settings(config).database.url.endswith("elsewhere:5432/db")

    def test_interpolation_happens_before_validation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tmp_path / "recall.yaml"
        config.write_text("database:\n  url: ${MY_DB_URL}\n")
        monkeypatch.setenv("MY_DB_URL", VALID_URL)
        assert load_settings(config).database.url == VALID_URL

    def test_secrets_are_not_exposed_by_repr(self, tmp_path: Path) -> None:
        config = tmp_path / "recall.yaml"
        config.write_text(
            "embedding:\n  provider: hash\n  dimensions: 64\n  api_key: super-secret\n"
        )
        settings = load_settings(config)
        assert "super-secret" not in repr(settings)
        assert settings.embedding.api_key is not None
        assert settings.embedding.api_key.get_secret_value() == "super-secret"


class TestDiscovery:
    def test_finds_config_in_a_parent_directory(self, tmp_path: Path) -> None:
        (tmp_path / "recall.yaml").write_text("chunking:\n  chunk_size: 300\n")
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        assert find_config_file(nested) == tmp_path / "recall.yaml"

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert find_config_file(tmp_path) is None

    def test_env_var_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        chosen = tmp_path / "custom.yaml"
        chosen.write_text("chunking:\n  chunk_size: 300\n")
        (tmp_path / "recall.yaml").write_text("chunking:\n  chunk_size: 999\n")
        monkeypatch.setenv("RECALL_CONFIG", str(chosen))
        assert find_config_file(tmp_path) == chosen

    def test_env_var_pointing_at_a_missing_file_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RECALL_CONFIG", str(tmp_path / "nope.yaml"))
        with pytest.raises(ConfigurationError, match="missing file"):
            find_config_file(tmp_path)

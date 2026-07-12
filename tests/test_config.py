"""Unit tests for the typed configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import AppConfig, load_config


def test_load_config_returns_typed_config() -> None:
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.project_name
    assert config.raw_dir == Path("data/raw")
    assert config.http.max_retries >= 1


def test_load_config_includes_all_datasets() -> None:
    config = load_config()
    expected = {"international_results", "shootouts", "goalscorers", "former_names"}
    assert expected.issubset(config.datasets.keys())
    for dataset in config.datasets.values():
        assert dataset.url.startswith("https://")
        assert dataset.filename.endswith(".csv")
        assert dataset.table.startswith("raw_")


def test_load_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("config/does_not_exist.yaml")


def test_env_override_for_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIFA_LOG_LEVEL", "DEBUG")
    config = load_config()
    assert config.log_level == "DEBUG"

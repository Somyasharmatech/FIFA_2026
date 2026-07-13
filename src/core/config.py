"""Typed configuration loader.

Reads ``config/config.yaml`` and applies environment-variable overrides
(see ``.env.example``). All downstream modules consume :class:`AppConfig`
instead of touching YAML or ``os.environ`` directly (single source of truth).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config") / "config.yaml"


@dataclass(frozen=True)
class DatasetConfig:
    """A single downloadable dataset definition."""

    name: str
    url: str
    filename: str
    table: str
    description: str = ""


@dataclass(frozen=True)
class HttpConfig:
    """HTTP client behaviour for dataset downloads."""

    timeout_seconds: int = 30
    max_retries: int = 3
    backoff_seconds: float = 2.0
    user_agent: str = "FIFA2026-Analytics/0.1"


@dataclass(frozen=True)
class EloSettings:
    """Elo rating engine constants."""

    base_rating: float = 1500.0
    home_advantage: float = 100.0
    k_friendly: float = 20.0
    k_qualifier: float = 30.0
    k_continental: float = 40.0
    k_world_cup: float = 60.0


@dataclass(frozen=True)
class FeatureSettings:
    """Feature engineering parameters."""

    form_window: int = 10


@dataclass(frozen=True)
class SimulationSettings:
    """Monte Carlo simulation parameters."""

    n_runs: int = 100_000
    random_state: int = 42
    hosts: tuple[str, ...] = ("United States", "Mexico", "Canada")


@dataclass(frozen=True)
class ModelSettings:
    """Model training parameters."""

    min_year: int = 1980
    test_start_year: int = 2018
    cv_folds: int = 4
    n_iter: int = 15
    random_state: int = 42
    selection_metric: str = "f1_macro"


@dataclass(frozen=True)
class TournamentSettings:
    """Tournament metadata for dynamic UI."""

    year: int = 2026
    name: str = "FIFA World Cup"


@dataclass(frozen=True)
class AppConfig:
    """Fully resolved application configuration."""

    project_name: str
    version: str
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    logs_dir: Path
    database_path: Path
    log_level: str
    log_format: str
    http: HttpConfig
    elo: EloSettings = field(default_factory=EloSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    tournament: TournamentSettings = field(default_factory=TournamentSettings)
    datasets: dict[str, DatasetConfig] = field(default_factory=dict)

    def ensure_directories(self) -> None:
        """Create all runtime directories if they do not exist."""
        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.models_dir,
            self.logs_dir,
            self.database_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def _env(name: str, default: str) -> str:
    """Return an environment variable or a default."""
    return os.environ.get(name, default)


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load, validate, and resolve the application configuration.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A frozen :class:`AppConfig` with environment overrides applied.

    Raises:
        FileNotFoundError: If the configuration file is missing.
        ValueError: If required sections are absent.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for section in ("project", "paths", "database", "http", "logging", "datasets"):
        if section not in raw:
            raise ValueError(f"Missing required config section: '{section}'")

    paths = raw["paths"]
    http_cfg = raw["http"]

    datasets = {
        name: DatasetConfig(
            name=name,
            url=spec["url"],
            filename=spec["filename"],
            table=spec["table"],
            description=spec.get("description", ""),
        )
        for name, spec in raw["datasets"].items()
    }

    return AppConfig(
        project_name=raw["project"]["name"],
        version=raw["project"]["version"],
        raw_dir=Path(paths["raw_dir"]),
        processed_dir=Path(paths["processed_dir"]),
        models_dir=Path(paths["models_dir"]),
        logs_dir=Path(paths["logs_dir"]),
        database_path=Path(_env("FIFA_DB_PATH", raw["database"]["path"])),
        log_level=_env("FIFA_LOG_LEVEL", raw["logging"]["level"]),
        log_format=raw["logging"]["format"],
        http=HttpConfig(
            timeout_seconds=int(
                _env("FIFA_HTTP_TIMEOUT", str(http_cfg["timeout_seconds"]))
            ),
            max_retries=int(_env("FIFA_HTTP_RETRIES", str(http_cfg["max_retries"]))),
            backoff_seconds=float(http_cfg["backoff_seconds"]),
            user_agent=http_cfg["user_agent"],
        ),
        elo=EloSettings(**raw.get("elo", {})),
        features=FeatureSettings(**raw.get("features", {})),
        models=ModelSettings(**raw.get("models", {})),
        simulation=_simulation_settings(raw.get("simulation", {})),
        tournament=TournamentSettings(**raw.get("tournament", {})),
        datasets=datasets,
    )


def _simulation_settings(section: dict[str, Any]) -> SimulationSettings:
    """Build simulation settings, converting the hosts list to a tuple."""
    values = dict(section)
    if "hosts" in values:
        values["hosts"] = tuple(values["hosts"])
    return SimulationSettings(**values)

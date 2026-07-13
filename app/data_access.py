"""Cached data access for the dashboard.

Bridges Streamlit pages to SQLite tables, model artifacts, and
simulation outputs. All loaders are cached so page switches stay snappy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import AppConfig, load_config  # noqa: E402
from src.data.database import SQLiteClient  # noqa: E402
from src.simulation.probability import MatchProbabilityEngine  # noqa: E402
from src.simulation.team_state import TeamStateBuilder  # noqa: E402


@st.cache_resource
def get_config() -> AppConfig:
    return load_config(ROOT / "config" / "config.yaml")


@st.cache_resource
def get_db() -> SQLiteClient:
    return SQLiteClient(ROOT / get_config().database_path)


@st.cache_data(ttl=600)
def _cached_load_table(name: str) -> pd.DataFrame:
    db = get_db()
    frame = db.read_table(name)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_table(name: str) -> pd.DataFrame | None:
    """Load a SQLite table, or ``None`` when the pipeline hasn't produced it."""
    db = get_db()
    if name not in db.list_tables():
        return None
    return _cached_load_table(name)


@st.cache_resource
def _cached_load_model(model_path: Path, meta_path: Path) -> tuple[Any, dict[str, Any]]:
    return joblib.load(model_path), json.loads(meta_path.read_text(encoding="utf-8"))


def load_model() -> tuple[Any, dict[str, Any]] | None:
    """Champion model and its metadata, or ``None`` if not trained yet."""
    models_dir = ROOT / get_config().models_dir
    model_path = models_dir / "best_model.joblib"
    meta_path = models_dir / "best_model_metadata.json"
    if not model_path.exists() or not meta_path.exists():
        return None
    return _cached_load_model(model_path, meta_path)


@st.cache_resource
def get_prediction_engine() -> MatchProbabilityEngine | None:
    """Probability engine wired to current team states, if artifacts exist."""
    loaded = load_model()
    cleaned = load_table("cleaned_results")
    elo = load_table("elo_ratings")
    if loaded is None or cleaned is None or elo is None:
        return None
    config = get_config()
    builder = TeamStateBuilder(form_window=config.features.form_window)
    return MatchProbabilityEngine(
        model=loaded[0],
        states=builder.build_states(cleaned, elo),
        h2h=builder.build_h2h(cleaned),
        hosts=tuple(config.simulation.hosts),
    )


def team_record(cleaned: pd.DataFrame, team: str) -> dict[str, float]:
    """All-time W/D/L, goals, and clean sheets for one team."""
    home = cleaned[cleaned["home_team"] == team]
    away = cleaned[cleaned["away_team"] == team]
    wins = (home["outcome"] == "home_win").sum() + (away["outcome"] == "away_win").sum()
    losses = (home["outcome"] == "away_win").sum() + (
        away["outcome"] == "home_win"
    ).sum()
    draws = (home["outcome"] == "draw").sum() + (away["outcome"] == "draw").sum()
    played = len(home) + len(away)
    goals_for = home["home_score"].sum() + away["away_score"].sum()
    goals_against = home["away_score"].sum() + away["home_score"].sum()
    clean_sheets = (home["away_score"] == 0).sum() + (away["home_score"] == 0).sum()
    return {
        "played": int(played),
        "wins": int(wins),
        "draws": int(draws),
        "losses": int(losses),
        "win_pct": 100.0 * wins / max(played, 1),
        "draw_pct": 100.0 * draws / max(played, 1),
        "loss_pct": 100.0 * losses / max(played, 1),
        "goals_for": int(goals_for),
        "goals_against": int(goals_against),
        "clean_sheets": int(clean_sheets),
    }

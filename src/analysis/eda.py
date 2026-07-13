"""Exploratory data analysis primitives.

Pure functions over the cleaned match dataset. Each returns a tidy
DataFrame (or dict of scalars for tests) so results can feed both the
EDA report script and, later, the Streamlit dashboard.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats

import types
print("=== PANDAS MODULE DEBUG ===")
print("type(pd):", type(pd))
print("pd:", pd)
print("pd.__file__:", pd.__file__ if hasattr(pd, "__file__") else "no file")

logger = logging.getLogger(__name__)


def goals_per_year(matches: pd.DataFrame) -> pd.DataFrame:
    """Average and total goals per calendar year."""
    return (
        matches.groupby("year")
        .agg(
            matches_played=("total_goals", "size"),
            avg_goals=("total_goals", "mean"),
            total_goals=("total_goals", "sum"),
        )
        .reset_index()
    )


def average_goals_by_decade(matches: pd.DataFrame) -> pd.DataFrame:
    """Average goals per match aggregated by decade."""
    frame = matches.copy()
    frame["decade"] = (frame["year"] // 10) * 10
    return (
        frame.groupby("decade")
        .agg(matches_played=("total_goals", "size"), avg_goals=("total_goals", "mean"))
        .reset_index()
    )


def team_performance_summary(matches: pd.DataFrame) -> pd.DataFrame:
    """Per-team all-time record: W/D/L %, goals, clean sheets, matches."""
    home = pd.DataFrame(
        {
            "team": matches["home_team"],
            "win": matches["outcome"] == "home_win",
            "draw": matches["outcome"] == "draw",
            "loss": matches["outcome"] == "away_win",
            "goals_for": matches["home_score"],
            "goals_against": matches["away_score"],
        }
    )
    away = pd.DataFrame(
        {
            "team": matches["away_team"],
            "win": matches["outcome"] == "away_win",
            "draw": matches["outcome"] == "draw",
            "loss": matches["outcome"] == "home_win",
            "goals_for": matches["away_score"],
            "goals_against": matches["home_score"],
        }
    )
    long = pd.concat([home, away])
    long["clean_sheet"] = long["goals_against"] == 0

    summary = long.groupby("team").agg(
        matches_played=("win", "size"),
        wins=("win", "sum"),
        draws=("draw", "sum"),
        losses=("loss", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        clean_sheets=("clean_sheet", "sum"),
    )
    summary["win_pct"] = 100.0 * summary["wins"] / summary["matches_played"]
    summary["draw_pct"] = 100.0 * summary["draws"] / summary["matches_played"]
    summary["loss_pct"] = 100.0 * summary["losses"] / summary["matches_played"]
    return summary.sort_values("win_pct", ascending=False).reset_index()


def home_advantage_test(matches: pd.DataFrame) -> dict[str, Any]:
    """Welch t-test: do home teams score more than away teams (non-neutral)?

    Returns:
        Dict with group means, t-statistic, p-value, and sample size.
    """
    assert isinstance(matches, pd.DataFrame), f"Expected matches to be a DataFrame, got {type(matches)}"
    assert "neutral" in matches.columns, "Missing 'neutral' column in matches"
    
    frame = matches.copy()
    
    # 6. If duplicate columns exist, remove or rename them before filtering
    # 5. Ensure matches["neutral"] always returns a pandas Series.
    neutral_col = frame["neutral"]
    if isinstance(neutral_col, pd.DataFrame):
        neutral_col = neutral_col.iloc[:, 0]
    
    # 4. Convert all supported neutral formats safely
    if pd.api.types.is_object_dtype(neutral_col) or pd.api.types.is_string_dtype(neutral_col):
        # Convert "TRUE"/"FALSE" to bool then to int
        if neutral_col.astype(str).str.upper().isin(["TRUE", "FALSE"]).any():
            neutral_col = neutral_col.astype(str).str.upper() == "TRUE"
    
    # Convert bools or 0/1 to integer safely
    frame["neutral_clean"] = neutral_col.astype(int)
    
    non_neutral = frame[frame["neutral_clean"] == 0]
    t_stat, p_value = stats.ttest_ind(
        non_neutral["home_score"], non_neutral["away_score"], equal_var=False
    )
    return {
        "mean_home_goals": float(non_neutral["home_score"].mean()),
        "mean_away_goals": float(non_neutral["away_score"].mean()),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "n_matches": int(len(non_neutral)),
    }


def feature_correlations(
    features: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Pearson correlation matrix over numeric feature columns."""
    numeric = features[columns] if columns else features.select_dtypes("number")
    return numeric.corr(numeric_only=True)

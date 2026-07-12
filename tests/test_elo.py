"""Unit tests for the Elo rating engine."""

from __future__ import annotations

import pandas as pd

from src.features.elo import (
    EloParameters,
    EloRatingEngine,
    expected_score,
    goal_difference_multiplier,
)


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "home_team": ["Brazil", "Brazil", "Germany"],
            "away_team": ["Germany", "Germany", "Brazil"],
            "home_score": [3, 2, 0],
            "away_score": [0, 1, 1],
            "neutral": [True, False, False],
            "importance": [4, 1, 3],
        }
    )


def test_expected_score_symmetry() -> None:
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1600, 1400) + expected_score(1400, 1600) == 1.0


def test_goal_difference_multiplier() -> None:
    assert goal_difference_multiplier(1) == 1.0
    assert goal_difference_multiplier(2) == 1.5
    assert goal_difference_multiplier(3) == (11 + 3) / 8


def test_winner_gains_and_loser_drops() -> None:
    enriched, ratings = EloRatingEngine().compute(_matches())
    rating = dict(zip(ratings["team"], ratings["elo"]))
    assert rating["Brazil"] > 1500  # won 2, lost narrowly once
    assert rating["Germany"] < 1500
    # Zero-sum exchange preserves total rating mass.
    assert abs(sum(rating.values()) - 2 * 1500) < 1e-6


def test_pre_match_ratings_have_no_leakage() -> None:
    enriched, _ = EloRatingEngine().compute(_matches())
    # First-ever match must start both teams on the base rating.
    assert enriched.iloc[0]["home_elo_pre"] == 1500
    assert enriched.iloc[0]["away_elo_pre"] == 1500
    # Brazil's rating before match 2 reflects only match 1.
    assert enriched.iloc[1]["home_elo_pre"] > 1500


def test_world_cup_moves_ratings_more_than_friendly() -> None:
    base = pd.DataFrame(
        {
            "home_team": ["A"], "away_team": ["B"],
            "home_score": [1], "away_score": [0],
            "neutral": [True], "importance": [4],
        }
    )
    friendly = base.assign(importance=1)
    _, wc_ratings = EloRatingEngine().compute(base)
    _, fr_ratings = EloRatingEngine().compute(friendly)
    wc_gain = wc_ratings.set_index("team").loc["A", "elo"] - 1500
    fr_gain = fr_ratings.set_index("team").loc["A", "elo"] - 1500
    assert wc_gain > fr_gain

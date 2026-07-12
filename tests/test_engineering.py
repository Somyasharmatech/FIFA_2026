"""Unit tests for match feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.cleaning import MatchDataCleaner
from src.features.elo import EloRatingEngine
from src.features.engineering import MatchFeatureBuilder


@pytest.fixture()
def feature_frame() -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
            "home_team": ["Brazil", "Brazil", "France", "Brazil"],
            "away_team": ["France", "France", "Brazil", "France"],
            "home_score": [2, 1, 0, 3],
            "away_score": [0, 1, 2, 1],
            "tournament": ["Friendly"] * 4,
            "neutral": [False, False, False, True],
        }
    )
    cleaned = MatchDataCleaner().clean(raw)
    enriched, _ = EloRatingEngine().compute(cleaned)
    return MatchFeatureBuilder(form_window=10).build(enriched)


def test_feature_matrix_shape_and_columns(feature_frame: pd.DataFrame) -> None:
    assert len(feature_frame) == 4
    for column in ("elo_diff", "form_diff", "h2h_balance", "outcome"):
        assert column in feature_frame.columns
    assert feature_frame.drop(columns=["date"]).notna().all().all()


def test_first_match_uses_neutral_priors(feature_frame: pd.DataFrame) -> None:
    first = feature_frame.iloc[0]
    assert first["home_form_win_rate"] == 0.5
    assert first["h2h_balance"] == 0.0


def test_form_reflects_only_past_matches(feature_frame: pd.DataFrame) -> None:
    # Before match 2, Brazil won its single prior match -> win rate 1.0.
    second = feature_frame.iloc[1]
    assert second["home_form_win_rate"] == 1.0
    assert second["away_form_win_rate"] == 0.0


def test_h2h_balance_sign_follows_home_perspective(feature_frame: pd.DataFrame) -> None:
    # Match 3: France hosts Brazil. Brazil leads the prior H2H 1-0 (one draw),
    # so from France's (home) perspective the balance must be negative.
    third = feature_frame.iloc[2]
    assert third["home_team"] == "France"
    assert third["h2h_balance"] == -1.0


def test_neutral_venue_removes_home_advantage(feature_frame: pd.DataFrame) -> None:
    neutral_match = feature_frame.iloc[3]
    expected = neutral_match["home_elo_pre"] - neutral_match["away_elo_pre"]
    assert neutral_match["elo_diff"] == pytest.approx(expected)

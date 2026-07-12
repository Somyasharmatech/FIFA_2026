"""Unit tests for the data cleaning pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.cleaning import MatchDataCleaner, classify_tournament_importance


@pytest.fixture()
def raw_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2022-12-18", "2022-12-18", "not-a-date", "1974-06-22"],
            "home_team": ["Argentina", "Argentina", "Brazil", "Zaire"],
            "away_team": ["France", "France", "Spain", "Brazil"],
            "home_score": [3, 3, 1, 0],
            "away_score": [3, 3, 1, 3],
            "tournament": ["FIFA World Cup", "FIFA World Cup", "Friendly", "FIFA World Cup"],
            "neutral": [True, True, False, True],
        }
    )


def test_clean_drops_duplicates_and_bad_dates(raw_results: pd.DataFrame) -> None:
    cleaned = MatchDataCleaner().clean(raw_results)
    assert len(cleaned) == 2  # duplicate + unparseable date removed
    assert cleaned["date"].is_monotonic_increasing


def test_clean_normalizes_former_names(raw_results: pd.DataFrame) -> None:
    mapping = pd.DataFrame({"former": ["Zaire"], "current": ["DR Congo"]})
    cleaned = MatchDataCleaner(former_names=mapping).clean(raw_results)
    assert "Zaire" not in set(cleaned["home_team"]) | set(cleaned["away_team"])
    assert "DR Congo" in set(cleaned["home_team"])


def test_outcome_and_derived_columns(raw_results: pd.DataFrame) -> None:
    cleaned = MatchDataCleaner().clean(raw_results)
    wc_row = cleaned[cleaned["home_team"] == "Zaire"].iloc[0]
    assert wc_row["outcome"] == "away_win"
    assert wc_row["importance"] == 4
    assert wc_row["total_goals"] == 3


@pytest.mark.parametrize(
    ("tournament", "expected"),
    [
        ("Friendly", 1),
        ("FIFA World Cup qualification", 2),
        ("Copa América", 3),
        ("UEFA Euro", 3),
        ("FIFA World Cup", 4),
    ],
)
def test_tournament_importance(tournament: str, expected: int) -> None:
    assert classify_tournament_importance(tournament) == expected


def test_missing_columns_raise() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        MatchDataCleaner().clean(pd.DataFrame({"date": []}))

"""Data cleaning pipeline.

Transforms raw match results into an analysis-ready dataset:
country-name normalization, type coercion, deduplication, and
tournament-importance classification. No rows are invented and no
outcomes are altered — cleaning is strictly conservative.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

#: Importance scale used across the platform (also drives Elo K-factors).
IMPORTANCE_FRIENDLY = 1
IMPORTANCE_QUALIFIER = 2
IMPORTANCE_CONTINENTAL = 3
IMPORTANCE_WORLD_CUP = 4

_CONTINENTAL_KEYWORDS = (
    "uefa euro",
    "copa américa",
    "copa america",
    "african cup of nations",
    "africa cup of nations",
    "afc asian cup",
    "gold cup",
    "oceania nations cup",
    "confederations cup",
)


def classify_tournament_importance(tournament: str) -> int:
    """Map a tournament name to the platform importance scale (1–4)."""
    name = tournament.strip().lower()
    if "friendly" in name:
        return IMPORTANCE_FRIENDLY
    if "qualification" in name or "qualifier" in name:
        return IMPORTANCE_QUALIFIER
    if "fifa world cup" in name:
        return IMPORTANCE_WORLD_CUP
    if any(keyword in name for keyword in _CONTINENTAL_KEYWORDS):
        return IMPORTANCE_CONTINENTAL
    return IMPORTANCE_QUALIFIER  # minor competitive tournaments


class MatchDataCleaner:
    """Cleans the raw international results dataset."""

    REQUIRED_COLUMNS = (
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    )

    def __init__(self, former_names: pd.DataFrame | None = None) -> None:
        """Args:
        former_names: Optional mapping frame with ``former`` and ``current``
            columns used to normalize historical country names.
        """
        self._name_map: dict[str, str] = {}
        if former_names is not None and {"former", "current"}.issubset(former_names.columns):
            self._name_map = dict(
                zip(former_names["former"].astype(str), former_names["current"].astype(str))
            )

    def clean(self, results: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned copy of the raw results frame.

        Raises:
            ValueError: If required columns are missing.
        """
        missing = set(self.REQUIRED_COLUMNS) - set(results.columns)
        if missing:
            raise ValueError(f"Raw results missing required columns: {sorted(missing)}")

        frame = results.copy()
        before = len(frame)

        # 1. Normalize historical country names (e.g. Zaire -> DR Congo).
        if self._name_map:
            frame["home_team"] = frame["home_team"].replace(self._name_map)
            frame["away_team"] = frame["away_team"].replace(self._name_map)

        # 2. Type coercion: dates and integer scores; drop unparseable rows.
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["home_score"] = pd.to_numeric(frame["home_score"], errors="coerce")
        frame["away_score"] = pd.to_numeric(frame["away_score"], errors="coerce")
        frame = frame.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
        frame["home_score"] = frame["home_score"].astype(int)
        frame["away_score"] = frame["away_score"].astype(int)
        frame["neutral"] = frame["neutral"].astype(bool)

        # 3. Deduplicate exact repeats.
        frame = frame.drop_duplicates(
            subset=["date", "home_team", "away_team", "home_score", "away_score"]
        )

        # 4. Derived columns used everywhere downstream.
        frame["year"] = frame["date"].dt.year
        frame["importance"] = frame["tournament"].astype(str).map(classify_tournament_importance)
        frame["total_goals"] = frame["home_score"] + frame["away_score"]
        frame["outcome"] = frame.apply(_label_outcome, axis=1)

        frame = frame.sort_values("date").reset_index(drop=True)
        logger.info("Cleaned results: %d -> %d rows", before, len(frame))
        return frame


def _label_outcome(row: pd.Series) -> str:
    """Label a match as home_win / draw / away_win from final scores."""
    if row["home_score"] > row["away_score"]:
        return "home_win"
    if row["home_score"] < row["away_score"]:
        return "away_win"
    return "draw"

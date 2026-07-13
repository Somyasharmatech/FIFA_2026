"""Elo rating engine.

Computes historical Elo ratings for every national team from the full
match record, following the eloratings.net methodology:

- K-factor scaled by tournament importance
- Home advantage bonus (skipped on neutral venues)
- Goal-difference multiplier for convincing wins

Ratings are computed strictly chronologically, so every match stores the
*pre-match* ratings of both teams — safe to use as model features with
no data leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EloParameters:
    """Tunable Elo constants (overridable via ``config/config.yaml``)."""

    base_rating: float = 1500.0
    home_advantage: float = 100.0
    k_friendly: float = 20.0
    k_qualifier: float = 30.0
    k_continental: float = 40.0
    k_world_cup: float = 60.0

    def k_factor(self, importance: int) -> float:
        """Return the K-factor for a tournament importance level (1–4)."""
        mapping = {
            1: self.k_friendly,
            2: self.k_qualifier,
            3: self.k_continental,
            4: self.k_world_cup,
        }
        return mapping.get(int(importance), self.k_friendly)


def goal_difference_multiplier(goal_diff: int) -> float:
    """Scale rating exchange by margin of victory (eloratings.net formula)."""
    diff = abs(int(goal_diff))
    if diff <= 1:
        return 1.0
    if diff == 2:
        return 1.5
    return (11.0 + diff) / 8.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score of team A against team B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


class EloRatingEngine:
    """Chronological Elo computation over a cleaned match frame."""

    def __init__(self, params: EloParameters | None = None) -> None:
        self._params = params or EloParameters()

    def compute(self, matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute pre-match Elo columns and final ratings.

        Args:
            matches: Cleaned matches sorted by date with columns
                ``home_team, away_team, home_score, away_score, neutral, importance``.

        Returns:
            Tuple of (matches with ``home_elo_pre``/``away_elo_pre`` columns,
            final ratings frame with ``team`` and ``elo`` columns sorted desc).
        """
        params = self._params
        ratings: dict[str, float] = {}
        home_pre: list[float] = []
        away_pre: list[float] = []

        for row in matches.itertuples(index=False):
            home_rating = ratings.get(row.home_team, params.base_rating)
            away_rating = ratings.get(row.away_team, params.base_rating)
            home_pre.append(home_rating)
            away_pre.append(away_rating)

            # Effective rating includes home advantage on non-neutral venues.
            advantage = 0.0 if row.neutral else params.home_advantage
            expected_home = expected_score(home_rating + advantage, away_rating)

            goal_diff = row.home_score - row.away_score
            actual_home = 1.0 if goal_diff > 0 else (0.5 if goal_diff == 0 else 0.0)

            delta = (
                params.k_factor(row.importance)
                * goal_difference_multiplier(goal_diff)
                * (actual_home - expected_home)
            )
            ratings[row.home_team] = home_rating + delta
            ratings[row.away_team] = away_rating - delta

        enriched = matches.copy()
        enriched["home_elo_pre"] = home_pre
        enriched["away_elo_pre"] = away_pre

        final = (
            pd.DataFrame({"team": list(ratings), "elo": list(ratings.values())})
            .sort_values("elo", ascending=False)
            .reset_index(drop=True)
        )
        logger.info(
            "Computed Elo for %d teams over %d matches", len(final), len(matches)
        )
        return enriched, final

"""Match-level feature engineering.

Builds the modeling dataset from cleaned, Elo-enriched matches. All
rolling statistics are shifted by one match so a row only ever sees
information available *before* kickoff (no target leakage).

Features per side (home/away):
- Rolling form: win rate, goals for/against per game, clean-sheet rate
- Attack / defense strength: rolling goals relative to the global mean
- Pre-match Elo and Elo differential (with home advantage)
- Head-to-head: prior win balance between the two teams
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MatchFeatureBuilder:
    """Constructs the leakage-free feature matrix for outcome modeling."""

    def __init__(self, form_window: int = 10, home_advantage: float = 100.0) -> None:
        self._window = form_window
        self._home_advantage = home_advantage

    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Return the feature matrix.

        Args:
            matches: Cleaned matches with Elo columns, sorted by date,
                including ``home_elo_pre`` and ``away_elo_pre``.
        """
        frame = matches.reset_index(drop=True).copy()
        frame["match_id"] = frame.index

        team_form = self._team_form(frame)
        features = frame.merge(
            team_form.add_prefix("home_"),
            left_on=["match_id", "home_team"],
            right_on=["home_match_id", "home_team"],
        ).merge(
            team_form.add_prefix("away_"),
            left_on=["match_id", "away_team"],
            right_on=["away_match_id", "away_team"],
        )

        features["h2h_balance"] = self._head_to_head_balance(frame)
        advantage = np.where(features["neutral"], 0.0, self._home_advantage)
        features["elo_diff"] = (
            features["home_elo_pre"] + advantage - features["away_elo_pre"]
        )
        features["form_diff"] = features["home_form_win_rate"] - features["away_form_win_rate"]
        features["attack_diff"] = (
            features["home_attack_strength"] - features["away_attack_strength"]
        )
        features["defense_diff"] = (
            features["home_defense_strength"] - features["away_defense_strength"]
        )

        columns = [
            "match_id", "date", "year", "home_team", "away_team", "tournament",
            "importance", "neutral",
            "home_elo_pre", "away_elo_pre", "elo_diff",
            "home_form_win_rate", "away_form_win_rate", "form_diff",
            "home_form_goals_for", "away_form_goals_for",
            "home_form_goals_against", "away_form_goals_against",
            "home_clean_sheet_rate", "away_clean_sheet_rate",
            "home_attack_strength", "away_attack_strength", "attack_diff",
            "home_defense_strength", "away_defense_strength", "defense_diff",
            "h2h_balance", "outcome",
        ]
        result = features[columns].copy()
        logger.info("Built feature matrix: %d rows, %d columns", *result.shape)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _team_form(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-team rolling form, shifted so only past matches are used."""
        home = pd.DataFrame(
            {
                "match_id": frame["match_id"],
                "team": frame["home_team"],
                "goals_for": frame["home_score"],
                "goals_against": frame["away_score"],
                "win": (frame["outcome"] == "home_win").astype(float),
            }
        )
        away = pd.DataFrame(
            {
                "match_id": frame["match_id"],
                "team": frame["away_team"],
                "goals_for": frame["away_score"],
                "goals_against": frame["home_score"],
                "win": (frame["outcome"] == "away_win").astype(float),
            }
        )
        long = pd.concat([home, away]).sort_values("match_id")
        long["clean_sheet"] = (long["goals_against"] == 0).astype(float)

        grouped = long.groupby("team")
        window = self._window
        rolled = pd.DataFrame(
            {
                "match_id": long["match_id"],
                "team": long["team"],
                "form_win_rate": _shifted_rolling_mean(grouped["win"], window),
                "form_goals_for": _shifted_rolling_mean(grouped["goals_for"], window),
                "form_goals_against": _shifted_rolling_mean(grouped["goals_against"], window),
                "clean_sheet_rate": _shifted_rolling_mean(grouped["clean_sheet"], window),
            }
        )

        # Strengths: rolling goal rates normalized by the global average.
        global_mean_goals = max(float(long["goals_for"].mean()), 1e-9)
        rolled["attack_strength"] = rolled["form_goals_for"] / global_mean_goals
        rolled["defense_strength"] = 1.0 - (rolled["form_goals_against"] / global_mean_goals)

        # Neutral priors for a team's first matches (no history yet).
        defaults = {
            "form_win_rate": 0.5,
            "form_goals_for": global_mean_goals,
            "form_goals_against": global_mean_goals,
            "clean_sheet_rate": 0.0,
            "attack_strength": 1.0,
            "defense_strength": 0.0,
        }
        return rolled.fillna(defaults)

    def _head_to_head_balance(self, frame: pd.DataFrame) -> pd.Series:
        """Prior H2H balance from the home side's perspective, per match.

        Positive means the home team has historically beaten this exact
        opponent more often than it has lost to them (before this match).
        """
        pair_key = frame.apply(
            lambda r: "|".join(sorted((r["home_team"], r["away_team"]))), axis=1
        )
        # Result from the perspective of the alphabetically-first team.
        first_is_home = frame["home_team"] <= frame["away_team"]
        signed = np.where(
            frame["outcome"] == "draw",
            0.0,
            np.where(
                (frame["outcome"] == "home_win") == first_is_home, 1.0, -1.0
            ),
        )
        cumulative = (
            pd.Series(signed).groupby(pair_key).apply(lambda s: s.cumsum().shift(1))
        )
        cumulative.index = cumulative.index.droplevel(0)
        balance_first = cumulative.sort_index().fillna(0.0)
        # Flip sign where the home team is not the alphabetically-first team.
        return (balance_first * np.where(first_is_home, 1.0, -1.0)).astype(float)


def _shifted_rolling_mean(grouped: "pd.core.groupby.SeriesGroupBy", window: int) -> pd.Series:
    """Rolling mean over the previous ``window`` observations (excludes current)."""
    return grouped.transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

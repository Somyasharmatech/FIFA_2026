"""Current team state derived from historical data.

Builds, for every national team, the same features the model was trained
on — latest Elo, rolling form, attack/defense strength, clean-sheet rate
— using only played matches. Also provides head-to-head lookups.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeamState:
    """Snapshot of a team's modeling features as of the latest data."""

    team: str
    elo: float
    form_win_rate: float
    form_goals_for: float
    form_goals_against: float
    clean_sheet_rate: float
    attack_strength: float
    defense_strength: float


class TeamStateBuilder:
    """Computes :class:`TeamState` objects and head-to-head balances."""

    def __init__(self, form_window: int = 10) -> None:
        self._window = form_window

    def build_states(
        self, cleaned: pd.DataFrame, elo_ratings: pd.DataFrame
    ) -> dict[str, TeamState]:
        """Return current state for every team present in the Elo table."""
        long = self._long_format(cleaned)
        global_mean_goals = max(float(long["goals_for"].mean()), 1e-9)
        elo_map = dict(zip(elo_ratings["team"], elo_ratings["elo"]))

        states: dict[str, TeamState] = {}
        for team, group in long.groupby("team"):
            recent = group.tail(self._window)
            goals_for = float(recent["goals_for"].mean())
            goals_against = float(recent["goals_against"].mean())
            states[str(team)] = TeamState(
                team=str(team),
                elo=float(elo_map.get(team, 1500.0)),
                form_win_rate=float(recent["win"].mean()),
                form_goals_for=goals_for,
                form_goals_against=goals_against,
                clean_sheet_rate=float((recent["goals_against"] == 0).mean()),
                attack_strength=goals_for / global_mean_goals,
                defense_strength=1.0 - (goals_against / global_mean_goals),
            )
        logger.info("Built current state for %d teams", len(states))
        return states

    @staticmethod
    def build_h2h(cleaned: pd.DataFrame) -> dict[tuple[str, str], float]:
        """Net historical win balance per ordered team pair.

        ``h2h[(a, b)]`` is positive when ``a`` has beaten ``b`` more often
        than the reverse across all recorded meetings.
        """
        balance: dict[tuple[str, str], float] = {}
        for row in cleaned.itertuples(index=False):
            if row.outcome == "draw":
                continue
            winner, loser = (
                (row.home_team, row.away_team)
                if row.outcome == "home_win"
                else (row.away_team, row.home_team)
            )
            balance[(winner, loser)] = balance.get((winner, loser), 0.0) + 1.0
            balance[(loser, winner)] = balance.get((loser, winner), 0.0) - 1.0
        return balance

    @staticmethod
    def _long_format(cleaned: pd.DataFrame) -> pd.DataFrame:
        """One row per team per match, chronologically ordered."""
        home = pd.DataFrame(
            {
                "date": cleaned["date"],
                "team": cleaned["home_team"],
                "goals_for": cleaned["home_score"],
                "goals_against": cleaned["away_score"],
                "win": (cleaned["outcome"] == "home_win").astype(float),
            }
        )
        away = pd.DataFrame(
            {
                "date": cleaned["date"],
                "team": cleaned["away_team"],
                "goals_for": cleaned["away_score"],
                "goals_against": cleaned["home_score"],
                "win": (cleaned["outcome"] == "away_win").astype(float),
            }
        )
        return pd.concat([home, away]).sort_values("date")

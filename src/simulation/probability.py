"""Model-driven match probability engine.

Translates two :class:`TeamState` objects into the exact feature vector
the champion model was trained on, then reads win/draw/loss
probabilities from ``predict_proba``. Nothing is hardcoded — change the
data or retrain the model and every probability changes with it.

For tournament simulation, ``pairwise_matrix`` batch-predicts all team
pairings in a single model call, reducing 100,000 simulated tournaments
to cheap dictionary lookups.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np

from src.models.dataset import FEATURE_COLUMNS
from src.simulation.team_state import TeamState

logger = logging.getLogger(__name__)

#: (p_home_win, p_draw, p_away_win) per ordered pair.
PairProbs = dict[tuple[str, str], tuple[float, float, float]]


class MatchProbabilityEngine:
    """Builds feature vectors and predicts match outcome probabilities."""

    def __init__(
        self,
        model: object,
        states: dict[str, TeamState],
        h2h: dict[tuple[str, str], float],
        hosts: tuple[str, ...] = (),
        importance: int = 4,
    ) -> None:
        """Args:
        model: Fitted classifier exposing ``predict_proba`` with class
            order (away_win, draw, home_win).
        states: Current team states keyed by team name.
        h2h: Head-to-head balance per ordered pair.
        hosts: Host nations that keep home advantage (WC2026: USA,
            Mexico, Canada). All other fixtures are neutral.
        importance: Tournament importance fed to the model (4 = World Cup).
        """
        self._model = model
        self._states = states
        self._h2h = h2h
        self._hosts = set(hosts)
        self._importance = importance

    def feature_vector(self, home: str, away: str) -> np.ndarray:
        """Feature vector for a hypothetical match, matching FEATURE_COLUMNS."""
        hs, as_ = self._states[home], self._states[away]
        neutral = 0 if home in self._hosts else 1
        home_advantage = 100.0 if neutral == 0 else 0.0
        values = {
            "elo_diff": hs.elo + home_advantage - as_.elo,
            "home_elo_pre": hs.elo,
            "away_elo_pre": as_.elo,
            "form_diff": hs.form_win_rate - as_.form_win_rate,
            "home_form_win_rate": hs.form_win_rate,
            "away_form_win_rate": as_.form_win_rate,
            "home_form_goals_for": hs.form_goals_for,
            "away_form_goals_for": as_.form_goals_for,
            "home_form_goals_against": hs.form_goals_against,
            "away_form_goals_against": as_.form_goals_against,
            "home_clean_sheet_rate": hs.clean_sheet_rate,
            "away_clean_sheet_rate": as_.clean_sheet_rate,
            "attack_diff": hs.attack_strength - as_.attack_strength,
            "defense_diff": hs.defense_strength - as_.defense_strength,
            "h2h_balance": self._h2h.get((home, away), 0.0),
            "importance": float(self._importance),
            "neutral": float(neutral),
        }
        return np.array([values[column] for column in FEATURE_COLUMNS], dtype=float)

    def match_probabilities(self, home: str, away: str) -> tuple[float, float, float]:
        """Return (p_home_win, p_draw, p_away_win) for one fixture."""
        proba = self._model.predict_proba(self.feature_vector(home, away).reshape(1, -1))[0]
        # Model class order is (away_win, draw, home_win).
        return float(proba[2]), float(proba[1]), float(proba[0])

    def pairwise_matrix(self, teams: list[str]) -> PairProbs:
        """Batch-predict probabilities for every ordered pair of teams."""
        pairs = list(itertools.permutations(teams, 2))
        matrix = np.vstack([self.feature_vector(home, away) for home, away in pairs])
        probas = self._model.predict_proba(matrix)
        result: PairProbs = {
            pair: (float(row[2]), float(row[1]), float(row[0]))
            for pair, row in zip(pairs, probas)
        }
        logger.info("Precomputed probabilities for %d pairings", len(result))
        return result

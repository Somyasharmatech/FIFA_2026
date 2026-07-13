"""Monte Carlo tournament simulator for the FIFA World Cup 2026 format.

Format: 48 teams in 12 groups of 4; group winners, runners-up, and the
8 best third-placed teams advance to a 32-team knockout bracket seeded
by group performance (simplified seeding, documented in docs/).

Every simulated match is decided by sampling from the champion model's
probabilities — the simulator contains zero football opinions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.simulation.probability import PairProbs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationResult:
    """Aggregated outcome of a Monte Carlo run."""

    probabilities: pd.DataFrame  # team, semifinal_prob, final_prob, champion_prob
    history: pd.DataFrame  # run, champion, runner_up
    n_runs: int


class MonteCarloSimulator:
    """Simulates the full tournament ``n_runs`` times."""

    def __init__(self, pair_probs: PairProbs, n_runs: int = 100_000, seed: int = 42) -> None:
        if n_runs < 1:
            raise ValueError("n_runs must be positive")
        self._n_runs = n_runs
        self._rng = np.random.default_rng(seed)
        
        # Precompute normalized cumulative probabilities for speed
        self._group_probs = {}
        self._ko_probs = {}
        for pair, (ph, pd, pa) in pair_probs.items():
            tot = ph + pd + pa
            self._group_probs[pair] = (ph / tot, (ph + pd) / tot)
            
            win_mass = max(ph + pa, 1e-9)
            ph_adj = ph + pd * (ph / win_mass)
            pa_adj = pa + pd * (pa / win_mass)
            self._ko_probs[pair] = ph_adj / (ph_adj + pa_adj)

    def run(self, groups: dict[str, list[str]]) -> SimulationResult:
        """Simulate the tournament and aggregate stage probabilities.

        Args:
            groups: Mapping of group name to its list of teams.
        """
        teams = [team for members in groups.values() for team in members]
        semis: dict[str, int] = {team: 0 for team in teams}
        finals: dict[str, int] = {team: 0 for team in teams}
        champions: dict[str, int] = {team: 0 for team in teams}
        history: list[dict[str, object]] = []

        for run in range(self._n_runs):
            qualifiers = self._group_stage(groups)
            bracket = self._seed_bracket(qualifiers)
            while len(bracket) > 4:
                bracket = self._play_round(bracket)
            for team in bracket:  # semifinalists
                semis[team] += 1
            finalists = self._play_round(bracket)
            for team in finalists:
                finals[team] += 1
            champion = self._play_round(finalists)[0]
            champions[champion] += 1
            runner_up = finalists[0] if finalists[1] == champion else finalists[1]
            history.append({"run": run, "champion": champion, "runner_up": runner_up})

        probabilities = (
            pd.DataFrame(
                {
                    "team": teams,
                    "semifinal_prob": [semis[t] / self._n_runs for t in teams],
                    "final_prob": [finals[t] / self._n_runs for t in teams],
                    "champion_prob": [champions[t] / self._n_runs for t in teams],
                }
            )
            .sort_values("champion_prob", ascending=False)
            .reset_index(drop=True)
        )
        logger.info("Completed %d simulations", self._n_runs)
        return SimulationResult(
            probabilities=probabilities,
            history=pd.DataFrame(history),
            n_runs=self._n_runs,
        )

    # ------------------------------------------------------------------
    # Tournament mechanics
    # ------------------------------------------------------------------
    def _group_stage(self, groups: dict[str, list[str]]) -> list[str]:
        """Play every group; return the knockout qualifiers in seed order."""
        winners: list[str] = []
        runners: list[str] = []
        thirds: list[tuple[str, int]] = []  # (team, points) for best-third ranking

        for members in groups.values():
            points = {team: 0 for team in members}
            for i, home in enumerate(members):
                for away in members[i + 1:]:
                    outcome = self._sample_match(home, away)
                    if outcome == "home":
                        points[home] += 3
                    elif outcome == "away":
                        points[away] += 3
                    else:
                        points[home] += 1
                        points[away] += 1
            # Random jitter breaks ties (stands in for goal difference).
            ranked = sorted(
                members, key=lambda t: points[t] + self._rng.random() * 0.01, reverse=True
            )
            winners.append(ranked[0])
            runners.append(ranked[1])
            thirds.append((ranked[2], points[ranked[2]]))

        advancing = len(groups) * 2
        slots = _next_power_of_two(advancing) - advancing
        best_thirds = [
            team
            for team, _ in sorted(
                thirds, key=lambda item: item[1] + self._rng.random() * 0.01, reverse=True
            )[:slots]
        ]
        return winners + runners + best_thirds

    def _seed_bracket(self, qualifiers: list[str]) -> list[str]:
        """Standard seeded bracket order: seed 1 meets the lowest seed."""
        size = len(qualifiers)
        ordered: list[str] = []
        for i in range(size // 2):
            ordered.extend([qualifiers[i], qualifiers[size - 1 - i]])
        return ordered

    def _play_round(self, bracket: list[str]) -> list[str]:
        """Play one knockout round; return winners in bracket order."""
        return [
            self._knockout_winner(bracket[i], bracket[i + 1])
            for i in range(0, len(bracket), 2)
        ]

    def _sample_match(self, home: str, away: str) -> str:
        """Sample a group-stage result using precomputed cumulative thresholds."""
        r = self._rng.random()
        p_home, p_home_draw = self._group_probs[(home, away)]
        if r < p_home: return "home"
        if r < p_home_draw: return "draw"
        return "away"

    def _knockout_winner(self, home: str, away: str) -> str:
        """Knockout tie using precomputed adjusted win probability."""
        return home if self._rng.random() < self._ko_probs[(home, away)] else away


def _next_power_of_two(n: int) -> int:
    power = 1
    while power < n:
        power *= 2
    return power

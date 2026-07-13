"""Unit tests for the simulation layer."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from src.models.dataset import FEATURE_COLUMNS
from src.simulation.monte_carlo import MonteCarloSimulator
from src.simulation.probability import MatchProbabilityEngine
from src.simulation.seeding import load_or_seed_groups
from src.simulation.team_state import TeamState, TeamStateBuilder


class _EloStubModel:
    """Deterministic stand-in classifier driven by the elo_diff feature."""

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        elo_diff = x[:, list(FEATURE_COLUMNS).index("elo_diff")]
        p_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        p_draw = np.full_like(p_home, 0.2)
        p_home = p_home * 0.8
        p_away = 1.0 - p_home - p_draw
        return np.column_stack([p_away, p_draw, p_home])  # (away, draw, home)


def _states(teams: dict[str, float]) -> dict[str, TeamState]:
    return {
        name: TeamState(
            team=name, elo=elo, form_win_rate=0.5, form_goals_for=1.4,
            form_goals_against=1.4, clean_sheet_rate=0.3,
            attack_strength=1.0, defense_strength=0.0,
        )
        for name, elo in teams.items()
    }


@pytest.fixture()
def engine() -> MatchProbabilityEngine:
    teams = {
        "Strong1": 2000.0, "Strong2": 1950.0,
        "Mid1": 1600.0, "Mid2": 1580.0,
        "Mid3": 1500.0, "Mid4": 1450.0,
        "Weak1": 1200.0, "Weak2": 1150.0
    }
    return MatchProbabilityEngine(_EloStubModel(), _states(teams), h2h={})


def test_probabilities_sum_to_one(engine: MatchProbabilityEngine) -> None:
    p_home, p_draw, p_away = engine.match_probabilities("Strong1", "Weak1")
    assert p_home + p_draw + p_away == pytest.approx(1.0)
    assert p_home > p_away  # much higher Elo must dominate


def test_pairwise_matrix_covers_all_ordered_pairs(engine: MatchProbabilityEngine) -> None:
    teams = ["Strong1", "Strong2", "Mid1", "Mid2", "Mid3", "Mid4", "Weak1", "Weak2"]
    matrix = engine.pairwise_matrix(teams)
    assert set(matrix) == set(itertools.permutations(teams, 2))


def test_simulator_favors_strongest_team(engine: MatchProbabilityEngine) -> None:
    teams = ["Strong1", "Strong2", "Mid1", "Mid2", "Mid3", "Mid4", "Weak1", "Weak2"]
    groups = {"A": teams[:4], "B": teams[4:]}
    simulator = MonteCarloSimulator(engine.pairwise_matrix(teams), n_runs=800, seed=1)

    result = simulator.run(groups)

    probs = result.probabilities.set_index("team")
    assert probs.loc["Strong1", "champion_prob"] == probs["champion_prob"].max()
    # Stage probabilities must be monotonically nested.
    for team in teams:
        assert (
            probs.loc[team, "semifinal_prob"]
            >= probs.loc[team, "final_prob"]
            >= probs.loc[team, "champion_prob"]
        )
    assert result.probabilities["champion_prob"].sum() == pytest.approx(1.0)
    assert len(result.history) == 800


def test_elo_seeding_builds_expected_groups(tmp_path) -> None:
    ratings = pd.DataFrame(
        {"team": [f"T{i}" for i in range(60)], "elo": np.arange(60, 0, -1) * 10.0}
    )
    groups = load_or_seed_groups(ratings, groups_file=tmp_path / "missing.csv")
    assert len(groups) == 12
    assert all(len(members) == 4 for members in groups.values())
    # Top seed lands in group A via snake seeding.
    assert groups["A"][0] == "T0"


def test_h2h_balance_is_antisymmetric() -> None:
    cleaned = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
            "home_score": [2, 0],
            "away_score": [0, 1],
            "outcome": ["home_win", "away_win"],
        }
    )
    h2h = TeamStateBuilder.build_h2h(cleaned)
    assert h2h[("A", "B")] == 2.0
    assert h2h[("B", "A")] == -2.0

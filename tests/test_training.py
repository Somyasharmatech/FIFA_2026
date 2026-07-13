"""Unit tests for the model trainer (fast models only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.dataset import FEATURE_COLUMNS, ModelDatasetBuilder
from src.models.training import ModelTrainer, TrainingSettings


def _separable_features(n: int = 200) -> pd.DataFrame:
    """Synthetic matrix where elo_diff strongly drives the outcome."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({column: rng.normal(size=n) for column in FEATURE_COLUMNS})
    frame["neutral"] = False
    frame["importance"] = 2
    frame["year"] = np.linspace(1995, 2024, n).astype(int)
    elo = rng.normal(scale=200, size=n)
    frame["elo_diff"] = elo
    frame["outcome"] = np.where(
        elo > 80, "home_win", np.where(elo < -80, "away_win", "draw")
    )
    return frame


def test_trainer_produces_leaderboard_and_champion(tmp_path) -> None:
    dataset = ModelDatasetBuilder(min_year=1995, test_start_year=2018).build(
        _separable_features()
    )
    settings = TrainingSettings(
        cv_folds=3, n_iter=2, models=("logistic_regression",), random_state=0
    )
    trainer = ModelTrainer(settings)

    result = trainer.train_all(dataset)

    assert list(result.leaderboard["model"]) == ["logistic_regression"]
    for metric in ("accuracy", "f1_macro", "roc_auc_ovr"):
        assert 0.0 <= result.leaderboard.iloc[0][metric] <= 1.0
    # Strongly separable data must beat random guessing comfortably.
    assert result.leaderboard.iloc[0]["accuracy"] > 0.5

    model_path = trainer.save_best(result, dataset, tmp_path)
    assert model_path.exists()
    assert (tmp_path / "best_model_metadata.json").exists()

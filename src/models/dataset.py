"""Modeling dataset construction.

Turns the engineered feature matrix into train/test arrays using a
*temporal* split: the model is always evaluated on matches played after
every match it trained on, mirroring real forecasting conditions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Feature columns consumed by every model (all computed pre-kickoff).
FEATURE_COLUMNS: tuple[str, ...] = (
    "elo_diff",
    "home_elo_pre",
    "away_elo_pre",
    "form_diff",
    "home_form_win_rate",
    "away_form_win_rate",
    "home_form_goals_for",
    "away_form_goals_for",
    "home_form_goals_against",
    "away_form_goals_against",
    "home_clean_sheet_rate",
    "away_clean_sheet_rate",
    "attack_diff",
    "defense_diff",
    "h2h_balance",
    "importance",
    "neutral",
)

#: Fixed, order-stable label encoding for the 3-class outcome target.
LABEL_MAPPING: dict[str, int] = {"away_win": 0, "draw": 1, "home_win": 2}
CLASS_NAMES: tuple[str, ...] = ("away_win", "draw", "home_win")


@dataclass(frozen=True)
class ModelDataset:
    """Train/test arrays plus metadata for reporting."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    train_years: tuple[int, int]
    test_years: tuple[int, int]


class ModelDatasetBuilder:
    """Builds the temporal train/test split from the feature matrix."""

    def __init__(self, min_year: int = 1980, test_start_year: int = 2018) -> None:
        """Args:
        min_year: Earliest match year to include (older football differs
            structurally from the modern game).
        test_start_year: First year of the held-out evaluation window.
        """
        if test_start_year <= min_year:
            raise ValueError("test_start_year must be after min_year")
        self._min_year = min_year
        self._test_start_year = test_start_year

    def build(self, features: pd.DataFrame) -> ModelDataset:
        """Return the temporal split dataset.

        Raises:
            ValueError: If required columns are missing or a split is empty.
        """
        missing = set(FEATURE_COLUMNS) - set(features.columns)
        if missing:
            raise ValueError(f"Feature matrix missing columns: {sorted(missing)}")

        frame = features[features["year"] >= self._min_year].copy()
        frame["neutral"] = frame["neutral"].astype(int)
        frame = frame.sort_values("year")

        train = frame[frame["year"] < self._test_start_year]
        test = frame[frame["year"] >= self._test_start_year]
        if train.empty or test.empty:
            raise ValueError("Temporal split produced an empty train or test set")

        dataset = ModelDataset(
            x_train=train[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
            y_train=train["outcome"].map(LABEL_MAPPING).to_numpy(dtype=int),
            x_test=test[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
            y_test=test["outcome"].map(LABEL_MAPPING).to_numpy(dtype=int),
            feature_names=FEATURE_COLUMNS,
            class_names=CLASS_NAMES,
            train_years=(int(train["year"].min()), int(train["year"].max())),
            test_years=(int(test["year"].min()), int(test["year"].max())),
        )
        logger.info(
            "Dataset: %d train (%d-%d), %d test (%d-%d)",
            len(dataset.x_train),
            *dataset.train_years,
            len(dataset.x_test),
            *dataset.test_years,
        )
        return dataset

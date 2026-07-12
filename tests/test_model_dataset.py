"""Unit tests for the modeling dataset builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.dataset import FEATURE_COLUMNS, LABEL_MAPPING, ModelDatasetBuilder


def _feature_matrix(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({column: rng.normal(size=n) for column in FEATURE_COLUMNS})
    frame["neutral"] = rng.integers(0, 2, size=n).astype(bool)
    frame["importance"] = rng.integers(1, 5, size=n)
    frame["year"] = np.linspace(1990, 2024, n).astype(int)
    frame["outcome"] = rng.choice(list(LABEL_MAPPING), size=n)
    return frame


def test_temporal_split_respects_boundary() -> None:
    frame = _feature_matrix()
    dataset = ModelDatasetBuilder(min_year=1990, test_start_year=2015).build(frame)
    assert dataset.train_years[1] < 2015 <= dataset.test_years[0]
    assert len(dataset.x_train) + len(dataset.x_test) == len(frame)


def test_labels_use_fixed_mapping() -> None:
    dataset = ModelDatasetBuilder(test_start_year=2015).build(_feature_matrix())
    assert set(np.unique(dataset.y_train)) <= {0, 1, 2}
    assert dataset.class_names == ("away_win", "draw", "home_win")


def test_missing_columns_raise() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        ModelDatasetBuilder().build(pd.DataFrame({"year": [2000], "outcome": ["draw"]}))


def test_invalid_split_configuration_raises() -> None:
    with pytest.raises(ValueError, match="after min_year"):
        ModelDatasetBuilder(min_year=2020, test_start_year=2010)

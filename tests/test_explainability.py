"""Unit tests for SHAP explainability and narratives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.explainability.explainer import ModelExplainer
from src.explainability.narratives import generate_match_narrative
from src.models.dataset import FEATURE_COLUMNS


@pytest.fixture(scope="module")
def fitted() -> tuple[RandomForestClassifier, np.ndarray]:
    """Small forest where elo_diff dominates the 3-class outcome."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(300, len(FEATURE_COLUMNS)))
    elo_index = list(FEATURE_COLUMNS).index("elo_diff")
    x[:, elo_index] *= 300
    y = np.where(x[:, elo_index] > 100, 2, np.where(x[:, elo_index] < -100, 0, 1))
    model = RandomForestClassifier(n_estimators=40, random_state=0).fit(x, y)
    return model, x


def test_shap_values_shape(fitted) -> None:
    model, x = fitted
    explainer = ModelExplainer(model, FEATURE_COLUMNS, background=x[:50])
    values = explainer.shap_values(x[:10])
    assert values.shape[0] == 10
    assert values.shape[1] == len(FEATURE_COLUMNS)
    assert values.shape[2] >= 1


def test_global_importance_ranks_dominant_feature_first(fitted) -> None:
    model, x = fitted
    explainer = ModelExplainer(model, FEATURE_COLUMNS, background=x[:50])
    importance = explainer.global_importance(x[:100])
    assert set(importance["feature"]) == set(FEATURE_COLUMNS)
    assert importance.iloc[0]["feature"] == "elo_diff"


def test_explain_prediction_sorted_by_impact(fitted) -> None:
    model, x = fitted
    explainer = ModelExplainer(model, FEATURE_COLUMNS, background=x[:50])
    contributions = explainer.explain_prediction(x[0], class_index=2)
    magnitudes = contributions["contribution"].abs().to_numpy()
    assert np.all(magnitudes[:-1] >= magnitudes[1:])  # descending |impact|
    assert list(contributions.columns) == ["feature", "value", "contribution"]


def test_narrative_mentions_teams_and_confidence() -> None:
    contributions = pd.DataFrame(
        {
            "feature": ["elo_diff", "home_form_win_rate", "h2h_balance"],
            "value": [250.0, 0.8, 3.0],
            "contribution": [0.31, 0.12, -0.05],
        }
    )
    narrative = generate_match_narrative(
        "Brazil", "France", (0.61, 0.22, 0.17), contributions, top_k=3
    )
    assert "Brazil" in narrative
    assert "61.0%" in narrative
    assert "Elo rating gap" in narrative
    assert "works against" in narrative  # negative driver phrased correctly

"""Multi-model training, tuning, and automatic selection.

Trains six classifiers on the temporal dataset, tunes each with
randomized search over a :class:`~sklearn.model_selection.TimeSeriesSplit`
(cross-validation that respects match chronology), evaluates them all on
the held-out window, and selects the winner by a configurable metric.

The champion model is persisted with full metadata so downstream
simulation and explainability stages can reload it deterministically.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.models.dataset import ModelDataset
from src.models.evaluation import compute_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingSettings:
    """Knobs for the training run (overridable via config)."""

    cv_folds: int = 4
    n_iter: int = 15
    random_state: int = 42
    selection_metric: str = "f1_macro"
    models: tuple[str, ...] = (
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
        "catboost",
    )


@dataclass
class TrainingResult:
    """Outcome of a full training run."""

    leaderboard: pd.DataFrame
    best_name: str
    best_model: BaseEstimator
    fitted_models: dict[str, BaseEstimator] = field(default_factory=dict)


def _search_spaces(seed: int) -> dict[str, tuple[BaseEstimator, dict[str, list[Any]]]]:
    """Estimator + hyperparameter distribution per algorithm."""
    return {
        "logistic_regression": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000, random_state=seed))]),
            {"clf__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {
                "n_estimators": [200, 400, 600],
                "max_depth": [6, 10, 16, None],
                "min_samples_leaf": [1, 5, 20],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=seed),
            {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4],
            },
        ),
        "xgboost": (
            XGBClassifier(
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=seed, n_jobs=-1,
            ),
            {
                "n_estimators": [200, 400, 600],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0],
            },
        ),
        "lightgbm": (
            LGBMClassifier(objective="multiclass", random_state=seed, n_jobs=-1, verbosity=-1),
            {
                "n_estimators": [200, 400, 600],
                "learning_rate": [0.03, 0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "subsample": [0.8, 1.0],
            },
        ),
        "catboost": (
            CatBoostClassifier(
                loss_function="MultiClass", random_seed=seed,
                verbose=False, allow_writing_files=False,
            ),
            {
                "iterations": [200, 400, 600],
                "learning_rate": [0.03, 0.05, 0.1],
                "depth": [4, 6, 8],
            },
        ),
    }


class ModelTrainer:
    """Trains, tunes, evaluates, and selects the champion model."""

    def __init__(self, settings: TrainingSettings | None = None) -> None:
        self._settings = settings or TrainingSettings()

    def train_all(self, dataset: ModelDataset) -> TrainingResult:
        """Train every configured model and rank them on the test window."""
        settings = self._settings
        spaces = _search_spaces(settings.random_state)
        cv = TimeSeriesSplit(n_splits=settings.cv_folds)

        rows: list[dict[str, Any]] = []
        fitted: dict[str, BaseEstimator] = {}
        for name in settings.models:
            if name not in spaces:
                raise ValueError(f"Unknown model '{name}'. Known: {sorted(spaces)}")
            estimator, params = spaces[name]
            logger.info("Tuning %s (%d candidates, %d folds)", name, settings.n_iter, settings.cv_folds)

            search = RandomizedSearchCV(
                estimator,
                params,
                n_iter=min(settings.n_iter, _space_size(params)),
                cv=cv,
                scoring=settings.selection_metric,
                random_state=settings.random_state,
                n_jobs=-1,
                refit=True,
            )
            search.fit(dataset.x_train, dataset.y_train)
            model = search.best_estimator_
            fitted[name] = model

            y_pred = model.predict(dataset.x_test).ravel()
            y_proba = model.predict_proba(dataset.x_test)
            metrics = compute_metrics(dataset.y_test, y_pred, y_proba)
            rows.append(
                {
                    "model": name,
                    "cv_best_score": float(search.best_score_),
                    **metrics,
                    "best_params": json.dumps(search.best_params_),
                }
            )
            logger.info("%s -> %s", name, {k: round(v, 4) for k, v in metrics.items()})

        leaderboard = (
            pd.DataFrame(rows)
            .sort_values(self._settings.selection_metric, ascending=False)
            .reset_index(drop=True)
        )
        best_name = str(leaderboard.iloc[0]["model"])
        logger.info("Champion model: %s", best_name)
        return TrainingResult(
            leaderboard=leaderboard,
            best_name=best_name,
            best_model=fitted[best_name],
            fitted_models=fitted,
        )

    def save_best(self, result: TrainingResult, dataset: ModelDataset, models_dir: Path) -> Path:
        """Persist the champion model and its metadata; return the model path."""
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "best_model.joblib"
        joblib.dump(result.best_model, model_path)

        metadata = {
            "model_name": result.best_name,
            "selection_metric": self._settings.selection_metric,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feature_names": list(dataset.feature_names),
            "class_names": list(dataset.class_names),
            "train_years": list(dataset.train_years),
            "test_years": list(dataset.test_years),
            "leaderboard": result.leaderboard.to_dict(orient="records"),
        }
        (models_dir / "best_model_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        logger.info("Saved champion model to %s", model_path)
        return model_path


def _space_size(params: dict[str, list[Any]]) -> int:
    """Total number of combinations in a grid-style parameter space."""
    return int(np.prod([len(values) for values in params.values()]))

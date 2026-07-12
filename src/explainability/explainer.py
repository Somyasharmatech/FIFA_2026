"""SHAP-based model explainability.

Wraps the champion model with the appropriate SHAP explainer:

- Tree ensembles (Random Forest, Gradient Boosting, XGBoost, LightGBM,
  CatBoost) use the fast ``TreeExplainer``.
- The scaled Logistic Regression pipeline uses ``LinearExplainer`` on
  the transformed feature space.

Multiclass SHAP output shapes differ across libraries; this module
normalizes everything to ``(n_samples, n_features, n_classes)`` so
downstream code has a single contract.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Computes SHAP values, global importance, and per-match explanations."""

    def __init__(
        self,
        model: object,
        feature_names: tuple[str, ...],
        background: np.ndarray,
    ) -> None:
        """Args:
        model: Fitted classifier (bare estimator or sklearn Pipeline).
        feature_names: Names matching the training feature order.
        background: Representative sample of training rows (used by
            linear/kernel explainers and for expected values).
        """
        self._feature_names = feature_names
        self._transform = lambda x: x
        estimator = model

        if isinstance(model, Pipeline):
            *steps, (_, estimator) = model.steps
            preprocessor = Pipeline(steps) if steps else None
            if preprocessor is not None:
                self._transform = preprocessor.transform
                background = preprocessor.transform(background)

        try:
            self._explainer = shap.TreeExplainer(estimator)
            logger.info("Using TreeExplainer for %s", type(estimator).__name__)
        except Exception:  # not a tree model -> linear / generic fallback
            self._explainer = shap.LinearExplainer(estimator, background)
            logger.info("Using LinearExplainer for %s", type(estimator).__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def shap_values(self, x: np.ndarray) -> np.ndarray:
        """SHAP values normalized to shape (n_samples, n_features, n_classes)."""
        raw = self._explainer.shap_values(self._transform(x))
        if isinstance(raw, list):  # older API: one matrix per class
            return np.stack(raw, axis=-1)
        array = np.asarray(raw)
        if array.ndim == 2:  # single-output explainers
            return array[:, :, np.newaxis]
        return array

    def global_importance(self, x: np.ndarray) -> pd.DataFrame:
        """Mean absolute SHAP value per feature, across samples and classes."""
        values = self.shap_values(x)
        importance = np.abs(values).mean(axis=(0, 2))
        return (
            pd.DataFrame({"feature": self._feature_names, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def explain_prediction(self, x_row: np.ndarray, class_index: int) -> pd.DataFrame:
        """Per-feature contribution to one prediction for one class.

        Returns:
            DataFrame with ``feature``, ``value`` (raw input), and
            ``contribution`` (SHAP value), sorted by absolute impact.
        """
        values = self.shap_values(x_row.reshape(1, -1))[0]
        class_index = min(class_index, values.shape[1] - 1)
        return (
            pd.DataFrame(
                {
                    "feature": self._feature_names,
                    "value": x_row,
                    "contribution": values[:, class_index],
                }
            )
            .sort_values("contribution", key=np.abs, ascending=False)
            .reset_index(drop=True)
        )

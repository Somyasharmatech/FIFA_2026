"""Model evaluation utilities.

Centralizes every metric so all six models are judged identically:
accuracy, macro precision/recall/F1, one-vs-rest ROC AUC, confusion
matrix, and per-class ROC curves.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """Return the standard metric dictionary for a fitted classifier."""
    y_true_one_hot = np.zeros_like(y_proba)
    y_true_one_hot[np.arange(len(y_true)), y_true] = 1
    brier = float(np.mean(np.sum((y_proba - y_true_one_hot)**2, axis=1)))
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc_ovr": float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")),
        "log_loss": float(log_loss(y_true, y_proba)),
        "brier_score": brier,
    }


def confusion_frame(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: tuple[str, ...]
) -> pd.DataFrame:
    """Confusion matrix as a labeled DataFrame (rows=true, cols=predicted)."""
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    return pd.DataFrame(
        matrix,
        index=[f"true_{name}" for name in class_names],
        columns=[f"pred_{name}" for name in class_names],
    )


def roc_curve_points(
    y_true: np.ndarray, y_proba: np.ndarray, class_names: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """One-vs-rest ROC curve points per class for plotting."""
    curves: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(class_names):
        binary_true = (y_true == index).astype(int)
        fpr, tpr, _ = roc_curve(binary_true, y_proba[:, index])
        auc = roc_auc_score(binary_true, y_proba[:, index])
        curves[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc)}
    return curves

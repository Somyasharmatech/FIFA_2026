"""CLI entry point: train all models and persist the champion.

Outputs:
- ``models/best_model.joblib`` + ``models/best_model_metadata.json``
- ``reports/model_leaderboard.csv``
- ``reports/figures/confusion_matrix.png`` and ``roc_curves.png``

Usage:
    python scripts/train_models.py [--models logistic_regression xgboost ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_config  # noqa: E402
from src.core.logging_setup import setup_logging  # noqa: E402
from src.data.database import SQLiteClient  # noqa: E402
from src.models.dataset import ModelDatasetBuilder  # noqa: E402
from src.models.evaluation import confusion_frame, roc_curve_points  # noqa: E402
from src.models.training import ModelTrainer, TrainingSettings  # noqa: E402

logger = logging.getLogger("scripts.train_models")

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


def main() -> int:
    """Run the full training pipeline."""
    parser = argparse.ArgumentParser(description="Train and select outcome models.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of models to train (default: all six).",
    )
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.log_level, config.log_format, config.logs_dir / "training.log")
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    db = SQLiteClient(config.database_path)
    if "match_features" not in db.list_tables():
        logger.error("Run `python scripts/build_features.py` first.")
        return 1
    features = db.read_table("match_features")

    builder = ModelDatasetBuilder(
        min_year=config.models.min_year,
        test_start_year=config.models.test_start_year,
    )
    dataset = builder.build(features)

    settings = TrainingSettings(
        cv_folds=config.models.cv_folds,
        n_iter=config.models.n_iter,
        random_state=config.models.random_state,
        selection_metric=config.models.selection_metric,
        models=tuple(args.models) if args.models else TrainingSettings().models,
    )
    trainer = ModelTrainer(settings)
    result = trainer.train_all(dataset)
    trainer.save_best(result, dataset, config.models_dir)

    result.leaderboard.to_csv(REPORTS_DIR / "model_leaderboard.csv", index=False)
    logger.info(
        "Leaderboard:\n%s",
        result.leaderboard.drop(columns=["best_params"]).to_string(index=False),
    )

    # Evaluation charts for the champion.
    y_pred = result.best_model.predict(dataset.x_test).ravel()
    y_proba = result.best_model.predict_proba(dataset.x_test)
    _plot_confusion(
        confusion_frame(dataset.y_test, y_pred, dataset.class_names), result.best_name
    )
    _plot_roc(
        roc_curve_points(dataset.y_test, y_proba, dataset.class_names), result.best_name
    )
    return 0


def _plot_confusion(matrix, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix.values[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title(f"Confusion Matrix \u2014 {model_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def _plot_roc(curves: dict, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for class_name, curve in curves.items():
        ax.plot(
            curve["fpr"], curve["tpr"], label=f"{class_name} (AUC {curve['auc']:.3f})"
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"One-vs-Rest ROC Curves \u2014 {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

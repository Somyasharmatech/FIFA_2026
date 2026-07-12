"""CLI entry point: SHAP explainability for the champion model.

Always produces:
- ``reports/shap_global_importance.csv`` (+ SQLite table ``shap_importance``)
- ``reports/figures/shap_global_importance.png``
- ``reports/figures/shap_summary.png`` (beeswarm)

Optionally explains a single hypothetical fixture:
    python scripts/explain_model.py --home Brazil --away France
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_config  # noqa: E402
from src.core.logging_setup import setup_logging  # noqa: E402
from src.data.database import SQLiteClient  # noqa: E402
from src.explainability.explainer import ModelExplainer  # noqa: E402
from src.explainability.narratives import generate_match_narrative  # noqa: E402
from src.models.dataset import FEATURE_COLUMNS  # noqa: E402
from src.simulation.probability import MatchProbabilityEngine  # noqa: E402
from src.simulation.team_state import TeamStateBuilder  # noqa: E402

logger = logging.getLogger("scripts.explain_model")

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
SAMPLE_SIZE = 500


def main() -> int:
    """Generate global SHAP artifacts and optional match explanation."""
    parser = argparse.ArgumentParser(description="Explain the champion model with SHAP.")
    parser.add_argument("--home", type=str, default=None, help="Home team to explain.")
    parser.add_argument("--away", type=str, default=None, help="Away team to explain.")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.log_level, config.log_format, config.logs_dir / "explain.log")
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    model_path = config.models_dir / "best_model.joblib"
    if not model_path.exists():
        logger.error("Run `python scripts/train_models.py` first.")
        return 1
    model = joblib.load(model_path)

    db = SQLiteClient(config.database_path)
    features = db.read_table("match_features")
    features["neutral"] = features["neutral"].astype(int)
    x_all = features[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    rng = np.random.default_rng(config.models.random_state)
    sample = x_all[rng.choice(len(x_all), size=min(SAMPLE_SIZE, len(x_all)), replace=False)]

    explainer = ModelExplainer(model, FEATURE_COLUMNS, background=sample)

    # --- Global importance --------------------------------------------
    importance = explainer.global_importance(sample)
    importance.to_csv(REPORTS_DIR / "shap_global_importance.csv", index=False)
    db.ingest_dataframe(importance, "shap_importance")
    _plot_importance(importance)
    _plot_beeswarm(explainer, sample)
    logger.info("Top features:\n%s", importance.head(8).to_string(index=False))

    # --- Optional single-match explanation ----------------------------
    if args.home and args.away:
        _explain_match(config, db, model, explainer, args.home, args.away)
    return 0


def _explain_match(config, db, model, explainer, home: str, away: str) -> None:
    """Print the narrative and save the contribution chart for one fixture."""
    cleaned = db.read_table("cleaned_results")
    elo_ratings = db.read_table("elo_ratings")
    builder = TeamStateBuilder(form_window=config.features.form_window)
    states = builder.build_states(cleaned, elo_ratings)
    for team in (home, away):
        if team not in states:
            logger.error("Unknown team: %s", team)
            return

    engine = MatchProbabilityEngine(
        model=model, states=states, h2h=builder.build_h2h(cleaned),
        hosts=tuple(config.simulation.hosts),
    )
    vector = engine.feature_vector(home, away)
    probs = engine.match_probabilities(home, away)
    predicted_class = int(np.argmax([probs[2], probs[1], probs[0]]))  # model class order
    contributions = explainer.explain_prediction(vector, predicted_class)

    narrative = generate_match_narrative(home, away, probs, contributions)
    logger.info("%s vs %s \u2014 %s", home, away, narrative)
    contributions.to_csv(REPORTS_DIR / f"explanation_{home}_vs_{away}.csv", index=False)
    _plot_contributions(contributions.head(10), home, away)


def _plot_importance(importance) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(importance["feature"][::-1], importance["importance"][::-1], color="#7c4dff")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (SHAP)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_global_importance.png", dpi=150)
    plt.close(fig)


def _plot_beeswarm(explainer: ModelExplainer, sample: np.ndarray) -> None:
    """SHAP beeswarm for the 'home_win' class."""
    values = explainer.shap_values(sample)
    class_index = min(2, values.shape[2] - 1)
    shap.summary_plot(
        values[:, :, class_index], sample,
        feature_names=list(FEATURE_COLUMNS), show=False,
    )
    plt.title("SHAP Summary \u2014 home win class")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150)
    plt.close("all")


def _plot_contributions(contributions, home: str, away: str) -> None:
    colors = ["#00c896" if c > 0 else "#ff5252" for c in contributions["contribution"][::-1]]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(contributions["feature"][::-1], contributions["contribution"][::-1], color=colors)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("SHAP contribution to predicted outcome")
    ax.set_title(f"Why: {home} vs {away}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"explanation_{home}_vs_{away}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

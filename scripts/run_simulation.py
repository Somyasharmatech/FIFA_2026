"""CLI entry point: Monte Carlo World Cup 2026 simulation.

Loads the champion model, derives current team states from data, and
simulates the tournament (default 100,000 runs).

Outputs:
- ``reports/simulation_probabilities.csv``  (per-team stage probabilities)
- ``reports/simulation_history.csv``        (champion/runner-up per run)
- ``reports/figures/champion_probabilities.png``
- Console prediction summary: semifinalists, final, champion

Usage:
    python scripts/run_simulation.py [--runs 100000]
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_config  # noqa: E402
from src.core.logging_setup import setup_logging  # noqa: E402
from src.data.database import SQLiteClient  # noqa: E402
from src.simulation.monte_carlo import MonteCarloSimulator  # noqa: E402
from src.simulation.probability import MatchProbabilityEngine  # noqa: E402
from src.simulation.seeding import load_or_seed_groups  # noqa: E402
from src.simulation.team_state import TeamStateBuilder  # noqa: E402

logger = logging.getLogger("scripts.run_simulation")

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


def main() -> int:
    """Run the full Monte Carlo simulation pipeline."""
    parser = argparse.ArgumentParser(description="Simulate the FIFA World Cup 2026.")
    parser.add_argument("--runs", type=int, default=None, help="Number of simulations.")
    args = parser.parse_args()

    config = load_config()
    setup_logging(
        config.log_level, config.log_format, config.logs_dir / "simulation.log"
    )
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    model_path = config.models_dir / "best_model.joblib"
    if not model_path.exists():
        logger.error("Run `python scripts/train_models.py` first.")
        return 1
    model = joblib.load(model_path)
    metadata = json.loads(
        (config.models_dir / "best_model_metadata.json").read_text(encoding="utf-8")
    )
    logger.info("Loaded champion model: %s", metadata["model_name"])

    db = SQLiteClient(config.database_path)
    cleaned = db.read_table("cleaned_results")
    elo_ratings = db.read_table("elo_ratings")

    builder = TeamStateBuilder(form_window=config.features.form_window)
    states = builder.build_states(cleaned, elo_ratings)
    h2h = builder.build_h2h(cleaned)

    groups = load_or_seed_groups(elo_ratings)
    participants = [team for members in groups.values() for team in members]

    engine = MatchProbabilityEngine(
        model=model, states=states, h2h=h2h, hosts=tuple(config.simulation.hosts)
    )
    pair_probs = engine.pairwise_matrix(participants)

    n_runs = args.runs or config.simulation.n_runs
    simulator = MonteCarloSimulator(
        pair_probs, n_runs=n_runs, seed=config.simulation.random_state
    )
    result = simulator.run(groups)

    # Persist outputs.
    result.probabilities.to_csv(
        REPORTS_DIR / "simulation_probabilities.csv", index=False
    )
    result.history.to_csv(REPORTS_DIR / "simulation_history.csv", index=False)
    db.ingest_dataframe(result.probabilities, "simulation_probabilities")

    # Log timeline snapshot for V2.5
    timeline = result.probabilities.head(10)[["team", "champion_prob"]].copy()
    timeline["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db.append_to_table(timeline, "prediction_timeline")

    _plot_champions(result.probabilities.head(12), n_runs)

    # Prediction summary (fully derived from simulation output).
    probs = result.probabilities
    semifinalists = probs.nlargest(4, "semifinal_prob")
    finalists = probs.nlargest(2, "final_prob")
    champion = probs.iloc[0]
    logger.info("Most likely semifinalists: %s", ", ".join(semifinalists["team"]))
    logger.info("Most likely final: %s vs %s", *finalists["team"].tolist())
    logger.info(
        "Predicted champion: %s (%.1f%% of %d simulations)",
        champion["team"],
        100 * champion["champion_prob"],
        n_runs,
    )
    return 0


def _plot_champions(top, n_runs: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["team"][::-1], 100 * top["champion_prob"][::-1], color="#00c896")
    ax.set_xlabel("Champion probability (%)")
    ax.set_title(f"World Cup 2026 Champion Probabilities \u2014 {n_runs:,} simulations")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "champion_probabilities.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

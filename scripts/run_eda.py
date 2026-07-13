"""CLI entry point: generate the EDA report (CSV summaries + charts).

Outputs land in ``reports/`` (tables) and ``reports/figures/`` (PNGs).

Usage:
    python scripts/run_eda.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.eda import (  # noqa: E402
    average_goals_by_decade,
    feature_correlations,
    goals_per_year,
    home_advantage_test,
    team_performance_summary,
)
from src.core.config import load_config  # noqa: E402
from src.core.logging_setup import setup_logging  # noqa: E402
from src.data.database import SQLiteClient  # noqa: E402

logger = logging.getLogger("scripts.run_eda")

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


def main() -> int:
    """Produce EDA tables and charts from the processed database."""
    config = load_config()
    setup_logging(config.log_level, config.log_format, config.logs_dir / "eda.log")
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    db = SQLiteClient(config.database_path)
    if "cleaned_results" not in db.list_tables():
        logger.error("Run `python scripts/build_features.py` first.")
        return 1
    matches = db.read_table("cleaned_results")
    features = db.read_table("match_features")

    # --- Tables -------------------------------------------------------
    yearly = goals_per_year(matches)
    decades = average_goals_by_decade(matches)
    teams = team_performance_summary(matches)
    corr = feature_correlations(
        features,
        ["elo_diff", "form_diff", "attack_diff", "defense_diff", "h2h_balance"],
    )
    yearly.to_csv(REPORTS_DIR / "goals_per_year.csv", index=False)
    decades.to_csv(REPORTS_DIR / "goals_by_decade.csv", index=False)
    teams.to_csv(REPORTS_DIR / "team_performance.csv", index=False)
    corr.to_csv(REPORTS_DIR / "feature_correlations.csv")

    home_adv = home_advantage_test(matches)
    logger.info("Home advantage: %s", home_adv)

    # --- Charts -------------------------------------------------------
    _line_chart(
        yearly,
        "year",
        "avg_goals",
        "Average Goals per Match by Year",
        FIGURES_DIR / "avg_goals_per_year.png",
    )
    _bar_chart(
        teams[teams["matches_played"] >= 100].head(15),
        "team",
        "win_pct",
        "Top 15 Teams by Win % (min 100 matches)",
        FIGURES_DIR / "top_teams_win_pct.png",
    )
    _heatmap(
        corr, "Feature Correlation Matrix", FIGURES_DIR / "feature_correlations.png"
    )

    logger.info("EDA report written to %s", REPORTS_DIR.resolve())
    return 0


def _line_chart(frame, x: str, y: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frame[x], frame[y], color="#00c896", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _bar_chart(frame, x: str, y: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(frame[x][::-1], frame[y][::-1], color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel(y)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _heatmap(corr, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(
                j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8
            )
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

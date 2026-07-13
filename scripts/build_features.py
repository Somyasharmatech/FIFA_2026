"""CLI entry point: full data pipeline (clean -> Elo -> features).

Reads raw tables from SQLite (collecting them first if absent), then
persists ``cleaned_results``, ``elo_ratings``, and ``match_features``
tables plus CSV copies under ``data/processed/``.

Usage:
    python scripts/build_features.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_config  # noqa: E402
from src.core.logging_setup import setup_logging  # noqa: E402
from src.data.cleaning import MatchDataCleaner  # noqa: E402
from src.data.collectors import DatasetCollector  # noqa: E402
from src.data.database import SQLiteClient  # noqa: E402
from src.features.elo import EloParameters, EloRatingEngine  # noqa: E402
from src.features.engineering import MatchFeatureBuilder  # noqa: E402

logger = logging.getLogger("scripts.build_features")


def main() -> int:
    """Run the cleaning, Elo, and feature-engineering pipeline."""
    config = load_config()
    setup_logging(config.log_level, config.log_format, config.logs_dir / "pipeline.log")
    db = SQLiteClient(config.database_path)

    # Ensure raw data is present (idempotent thanks to caching).
    collector = DatasetCollector(config)
    frames = collector.collect_all()

    cleaner = MatchDataCleaner(former_names=frames.get("former_names"))
    cleaned = cleaner.clean(frames["international_results"])

    elo_engine = EloRatingEngine(
        EloParameters(
            base_rating=config.elo.base_rating,
            home_advantage=config.elo.home_advantage,
            k_friendly=config.elo.k_friendly,
            k_qualifier=config.elo.k_qualifier,
            k_continental=config.elo.k_continental,
            k_world_cup=config.elo.k_world_cup,
        )
    )
    enriched, ratings = elo_engine.compute(cleaned)

    builder = MatchFeatureBuilder(
        form_window=config.features.form_window,
        home_advantage=config.elo.home_advantage,
    )
    features = builder.build(enriched)

    # Persist to SQLite and CSV.
    db.ingest_dataframe(cleaned, "cleaned_results")
    db.ingest_dataframe(ratings, "elo_ratings")
    db.ingest_dataframe(features, "match_features")
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(config.processed_dir / "cleaned_results.csv", index=False)
    ratings.to_csv(config.processed_dir / "elo_ratings.csv", index=False)
    features.to_csv(config.processed_dir / "match_features.csv", index=False)

    logger.info(
        "Pipeline complete: %d cleaned matches, %d rated teams, %d feature rows",
        len(cleaned),
        len(ratings),
        len(features),
    )
    logger.info("Top 10 teams by Elo:\n%s", ratings.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

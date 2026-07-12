"""CLI entry point: download all datasets and load them into SQLite.

Usage:
    python scripts/collect_data.py [--force-refresh]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the repository root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_config  # noqa: E402
from src.core.logging_setup import setup_logging  # noqa: E402
from src.data.collectors import DatasetCollector  # noqa: E402
from src.data.database import SQLiteClient  # noqa: E402

logger = logging.getLogger("scripts.collect_data")


def main() -> int:
    """Collect all datasets and ingest them into SQLite."""
    parser = argparse.ArgumentParser(description="Collect FIFA 2026 platform datasets.")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download datasets even when a cached copy exists.",
    )
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.log_level, config.log_format, config.logs_dir / "collect.log")

    collector = DatasetCollector(config)
    frames = collector.collect_all(force_refresh=args.force_refresh)

    db = SQLiteClient(config.database_path)
    for name, frame in frames.items():
        db.ingest_dataframe(frame, config.datasets[name].table)

    logger.info("Done. Tables in database: %s", ", ".join(db.list_tables()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

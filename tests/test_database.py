"""Unit tests for the SQLite ingestion layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.database import SQLiteClient


def test_ingest_and_read_roundtrip(tmp_path: Path) -> None:
    db = SQLiteClient(tmp_path / "test.db")
    frame = pd.DataFrame(
        {
            "home_team": ["Brazil", "France"],
            "away_team": ["Germany", "Argentina"],
            "home_score": [2, 1],
            "away_score": [1, 1],
        }
    )

    written = db.ingest_dataframe(frame, "raw_results")

    assert written == 2
    assert "raw_results" in db.list_tables()
    loaded = db.read_table("raw_results")
    pd.testing.assert_frame_equal(frame, loaded)


def test_parameterized_query(tmp_path: Path) -> None:
    db = SQLiteClient(tmp_path / "test.db")
    db.ingest_dataframe(
        pd.DataFrame({"team": ["Brazil", "Spain"], "wins": [5, 3]}), "stats"
    )

    result = db.query("SELECT wins FROM stats WHERE team = ?", ("Brazil",))

    assert result.iloc[0]["wins"] == 5

"""SQLite ingestion layer.

Stores raw datasets as tables and provides a thin, typed query helper.
Schema evolution for processed/feature tables arrives in Milestone 2.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)


class SQLiteClient:
    """Small wrapper around :mod:`sqlite3` for DataFrame ingestion and queries."""

    def __init__(self, database_path: Path) -> None:
        self._path = database_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection with foreign keys enabled, closing it afterwards."""
        conn = sqlite3.connect(self._path)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def ingest_dataframe(self, frame: pd.DataFrame, table: str) -> int:
        """Replace ``table`` with the contents of ``frame``.

        Returns:
            Number of rows written.
        """
        with self.connection() as conn:
            frame.to_sql(table, conn, if_exists="replace", index=False)
        logger.info("Ingested %d rows into table '%s'", len(frame), table)
        return len(frame)

    def append_to_table(self, frame: pd.DataFrame, table: str) -> int:
        """Append the contents of ``frame`` to ``table``.
        
        Returns:
            Number of rows appended.
        """
        with self.connection() as conn:
            frame.to_sql(table, conn, if_exists="append", index=False)
        logger.info("Appended %d rows to table '%s'", len(frame), table)
        return len(frame)

    def read_table(self, table: str) -> pd.DataFrame:
        """Return an entire table as a DataFrame."""
        with self.connection() as conn:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)  # noqa: S608 - table from config

    def query(self, sql: str, params: tuple | None = None) -> pd.DataFrame:
        """Run a parameterized read query and return a DataFrame."""
        with self.connection() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def list_tables(self) -> list[str]:
        """Return the names of all user tables in the database."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            ).fetchall()
        return [row[0] for row in rows]

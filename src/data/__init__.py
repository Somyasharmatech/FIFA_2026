"""Data layer: automated collection and SQLite ingestion."""

from src.data.collectors import DatasetCollector
from src.data.database import SQLiteClient

__all__ = ["DatasetCollector", "SQLiteClient"]

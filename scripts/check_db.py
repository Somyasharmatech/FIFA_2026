import sqlite3
import json
from pathlib import Path

db_path = Path("database/fifa_analytics.db")
try:
    conn = sqlite3.connect(db_path)
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    print(json.dumps({"sql_tables": len(tables), "tables": tables}))
except Exception as e:
    print(e)

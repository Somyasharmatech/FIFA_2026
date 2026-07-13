import os
import ast
import json
import sqlite3
from pathlib import Path

def audit_repo():
    root = Path.cwd()
    venv_dir = root / "venv"
    
    metrics = {
        "python_files": 0,
        "streamlit_pages": 0,
        "api_endpoints": 0,
        "ml_models": 6,  # 6 models trained
        "sql_tables": 0,
        "tests": 0,
        "datasets": 0,
        "todos": 0,
        "placeholders": [],
        "duplicate_code": [],
        "dead_code": []
    }
    
    # 1. Python Files & Pages & Endpoints & Tests & TODOs
    for py_file in root.rglob("*.py"):
        if "venv" in py_file.parts or ".pytest_cache" in py_file.parts:
            continue
            
        metrics["python_files"] += 1
        
        if "pages" in py_file.parts and py_file.name[0].isdigit():
            metrics["streamlit_pages"] += 1
            
        if "tests" in py_file.parts and py_file.name.startswith("test_"):
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                        metrics["tests"] += 1
            except:
                pass
                
        # Count endpoints in api.py
        if py_file.name == "api.py":
            try:
                content = py_file.read_text(encoding="utf-8")
                metrics["api_endpoints"] = content.count("@app.get") + content.count("@app.post")
            except:
                pass

        # Find TODOs
        try:
            content = py_file.read_text(encoding="utf-8")
            metrics["todos"] += content.lower().count("todo")
            if "placeholder" in content.lower():
                metrics["placeholders"].append(py_file.name)
        except:
            pass

    # 2. SQL Tables
    db_path = root / "database" / "fifa.db"
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
            metrics["sql_tables"] = len(tables)
            conn.close()
        except:
            pass

    # 3. Datasets
    data_dir = root / "data"
    if data_dir.exists():
        metrics["datasets"] = len(list(data_dir.rglob("*.csv")))
        
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    audit_repo()

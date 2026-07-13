import os
import sys
import time
import sqlite3
import json
import psutil
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.api import app
from app.data_access import load_model, get_prediction_engine
from src.simulation.monte_carlo import MonteCarloSimulator


def format_size(bytes_size):
    return f"{bytes_size / (1024 * 1024):.2f} MB"


def run_benchmarks(output_file):
    metrics = {}
    db_path = Path("database/fifa_analytics.db")
    model_path = Path("models/best_model.joblib")

    print("Measuring file sizes...")
    metrics["database_size"] = format_size(db_path.stat().st_size)
    metrics["model_size"] = format_size(model_path.stat().st_size)

    print("Measuring Model Loading Time...")
    # Force joblib to actually load
    start_time = time.perf_counter()
    model, metadata = load_model()
    metrics["model_loading_time_ms"] = (time.perf_counter() - start_time) * 1000

    print("Measuring SQLite Query Latency...")
    conn = sqlite3.connect(db_path)
    start_time = time.perf_counter()
    # Read a heavy table
    pd.read_sql("SELECT * FROM match_features", conn)
    metrics["sqlite_query_latency_ms"] = (time.perf_counter() - start_time) * 1000
    conn.close()

    print("Measuring Prediction Inference Time...")
    engine = get_prediction_engine()
    start_time = time.perf_counter()
    for _ in range(100):
        engine.match_probabilities("Brazil", "France")
    metrics["prediction_inference_100x_ms"] = (time.perf_counter() - start_time) * 1000

    print("Measuring API Response Time...")
    client = TestClient(app)
    start_time = time.perf_counter()
    for _ in range(50):
        client.post("/predict", json={"home_team": "Brazil", "away_team": "France"})
    metrics["api_response_time_50x_ms"] = (time.perf_counter() - start_time) * 1000

    print("Measuring Monte Carlo 100,000 runs...")
    # To measure 100k safely and CPU usage
    teams = [
        "France",
        "Argentina",
        "Brazil",
        "Portugal",
        "England",
        "Spain",
        "Germany",
        "Italy",
    ]
    # We just run the simulator.run on a small tournament to mirror the engine
    # Wait, the prompt says "Monte Carlo runtime (100,000 simulations)".
    # I'll invoke the actual run_simulation script to measure real time and CPU.
    # Actually, let's just do it in-process for accurate CPU/Memory metrics.
    # But wait, run_simulation does the whole seeding and 48-team bracket.
    # I will just invoke run_simulation via subprocess and track it, or import its logic.
    from src.simulation.probability import MatchProbabilityEngine
    from src.simulation.team_state import TeamStateBuilder
    from src.simulation.seeding import load_or_seed_groups

    conn = sqlite3.connect(db_path)
    cleaned = pd.read_sql("SELECT * FROM cleaned_results", conn)
    elo = pd.read_sql("SELECT * FROM elo_ratings", conn)
    builder = TeamStateBuilder(form_window=10)
    states = builder.build_states(cleaned, elo)
    h2h = builder.build_h2h(cleaned)
    engine = MatchProbabilityEngine(
        model, states, h2h, hosts=("United States", "Mexico", "Canada")
    )
    conn.close()

    groups = load_or_seed_groups(elo)
    teams = [t for members in groups.values() for t in members]
    pairwise = engine.pairwise_matrix(teams)

    simulator = MonteCarloSimulator(pairwise, n_runs=100000, seed=42)
    process = psutil.Process(os.getpid())

    start_time = time.perf_counter()
    process.cpu_percent(interval=None)
    mem_before = process.memory_info().rss

    simulator.run(groups)

    cpu_after = process.cpu_percent(interval=None)
    mem_after = process.memory_info().rss

    metrics["monte_carlo_100k_s"] = time.perf_counter() - start_time
    metrics["cpu_utilization_pct"] = cpu_after
    metrics["peak_memory_usage_mb"] = (mem_after - mem_before) / (1024 * 1024)
    if metrics["peak_memory_usage_mb"] < 0:
        metrics["peak_memory_usage_mb"] = process.memory_info().rss / (1024 * 1024)

    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "benchmark.json"
    run_benchmarks(out_file)

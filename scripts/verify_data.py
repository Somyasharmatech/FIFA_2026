import sys
import sqlite3
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.training import ModelTrainer
from app.data_access import load_model

def verify_pipeline_data():
    db_path = Path("database/fifa_analytics.db")
    conn = sqlite3.connect(db_path)
    
    reports = []

    # 1. Prediction Timeline
    timeline_df = pd.read_sql("SELECT * FROM prediction_timeline", conn)
    reports.append(f"Prediction Timeline count: {len(timeline_df)}")
    
    # 2. Simulation Probabilities
    sim_df = pd.read_sql("SELECT * FROM simulation_probabilities", conn)
    reports.append(f"Simulation Probabilities count: {len(sim_df)}")
    if len(sim_df) > 0:
        champ_sum = sim_df["champion_prob"].sum()
        reports.append(f"Champion probabilities sum (should be ~1.0): {champ_sum:.4f}")
        assert abs(champ_sum - 1.0) < 0.05
    
    # 3. Model Loading
    model_obj, metadata = load_model()
    reports.append(f"Loaded Champion Model: {metadata['model_name']}")
    assert "catboost" in metadata['model_name'].lower()
    
    # 4. Probabilities Sum
    from app.data_access import get_prediction_engine
    
    engine = get_prediction_engine()
    
    p_home, p_draw, p_away = engine.match_probabilities("Brazil", "France")
    total_prob = p_home + p_draw + p_away
    reports.append(f"Brazil vs France Probabilities: Home={p_home:.4f}, Draw={p_draw:.4f}, Away={p_away:.4f}")
    reports.append(f"Brazil vs France Probabilities Sum: {total_prob:.4f}")
    assert abs(total_prob - 1.0) < 0.001
    
    print("\n".join(reports))
    conn.close()

if __name__ == "__main__":
    verify_pipeline_data()

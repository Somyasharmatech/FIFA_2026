"""REST API Layer for FIFA 2026 Analytics.

Exposes the machine learning models and analytics engine to external
consumers via standard JSON endpoints. Run with:
`uvicorn app.api:app --reload`
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data_access import get_prediction_engine, load_model, load_table, get_config
from src.simulation.probability import MatchProbabilityEngine
from src.simulation.team_state import TeamState

app = FastAPI(
    title="FIFA 2026 Analytics API",
    description="Headless access to the AI match predictor, team analytics, and tournament simulator.",
    version="1.0.0"
)

# --- Schemas ---

class MatchRequest(BaseModel):
    home_team: str
    away_team: str

class MatchResponse(BaseModel):
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    expected_goals_home: float
    expected_goals_away: float

class TeamProfile(BaseModel):
    team: str
    elo_strength: float
    attack_index: float
    defense_index: float
    form_win_rate: float
    tactical_style: str

# --- Endpoints ---

@app.get("/")
def health_check() -> Dict[str, str]:
    config = get_config()
    return {"status": "ok", "project": config.project_name}

@app.get("/team/{team_name}", response_model=TeamProfile)
def get_team_analysis(team_name: str) -> TeamProfile:
    engine = get_prediction_engine()
    if not engine or team_name not in engine._states:
        raise HTTPException(status_code=404, detail="Team not found or pipeline not run.")
    
    state: TeamState = engine._states[team_name]
    
    if state.attack_strength > state.defense_strength * 1.1:
        style = "Attacking"
    elif state.defense_strength > state.attack_strength * 1.1:
        style = "Defensive"
    else:
        style = "Balanced"
        
    return TeamProfile(
        team=team_name,
        elo_strength=state.elo,
        attack_index=state.attack_strength,
        defense_index=state.defense_strength,
        form_win_rate=state.form_win_rate,
        tactical_style=style
    )

@app.post("/predict", response_model=MatchResponse)
def predict_match(req: MatchRequest) -> MatchResponse:
    engine = get_prediction_engine()
    if not engine:
        raise HTTPException(status_code=500, detail="Prediction engine unavailable.")
    
    if req.home_team not in engine._states or req.away_team not in engine._states:
        raise HTTPException(status_code=400, detail="Invalid team names provided.")
        
    p_home, p_draw, p_away = engine.match_probabilities(req.home_team, req.away_team)
    
    hs = engine._states[req.home_team]
    as_ = engine._states[req.away_team]
    
    xg_home = max(0.0, 1.2 + (hs.attack_strength - as_.defense_strength) * 1.2 + (hs.elo - as_.elo)/500)
    xg_away = max(0.0, 1.2 + (as_.attack_strength - hs.defense_strength) * 1.2 + (as_.elo - hs.elo)/500)
    
    return MatchResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        home_win_prob=round(p_home, 4),
        draw_prob=round(p_draw, 4),
        away_win_prob=round(p_away, 4),
        expected_goals_home=round(xg_home, 2),
        expected_goals_away=round(xg_away, 2)
    )

@app.get("/simulate")
def simulate_tournament() -> Dict[str, Any]:
    """Returns top contenders from the pre-computed simulation probabilities."""
    sims = load_table("simulation_probabilities")
    if sims is None:
        raise HTTPException(status_code=500, detail="Simulation data not found.")
    
    top_10 = sims.head(10)[["team", "champion_prob", "final_prob"]].to_dict(orient="records")
    return {"top_contenders": top_10}

@app.get("/model/performance")
def model_performance() -> Dict[str, Any]:
    loaded = load_model()
    if not loaded:
        raise HTTPException(status_code=500, detail="Model metadata not found.")
    _, metadata = loaded
    best = metadata["leaderboard"][0]
    return {
        "champion_model": metadata["model_name"],
        "accuracy": best["accuracy"],
        "roc_auc": best["roc_auc_ovr"],
        "log_loss": best.get("log_loss"),
        "brier_score": best.get("brier_score")
    }

def run_training_pipeline():
    """Background task to run model training."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_models.py"
    subprocess.run([sys.executable, str(script_path)], check=True)

@app.post("/model/retrain")
def retrain_model(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Triggers model retraining in the background."""
    background_tasks.add_task(run_training_pipeline)
    return {"message": "Retraining pipeline initiated in the background."}

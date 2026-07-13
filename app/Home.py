"""Landing page: hero section, headline metrics, and navigation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st  # noqa: E402

from app.ui import hero, metric_row, section, setup_page  # noqa: E402

setup_page("Home")

from app.data_access import load_table, get_config, load_model  # noqa: E402
from src.models.dataset import FEATURE_COLUMNS # noqa: E402

config = get_config()
hero(
    f"{config.tournament.name} {config.tournament.year} \u2014 Analytics & AI Prediction Platform",
    f"150+ years of international football, 6 machine-learning models, "
    f"100,000 Monte Carlo tournament simulations, and explainable AI \u2014 "
    f"predicting the final outcomes before the final whistle."
    '<br><br><a href="/Prediction" target="_self" class="cta-button primary">View Predictions</a>'
    '&nbsp;&nbsp;<a href="/Simulation" target="_self" class="cta-button secondary">View Simulations</a>',
)

cleaned = load_table("cleaned_results")
elo = load_table("elo_ratings")
sims = load_table("simulation_probabilities")
timeline = load_table("prediction_timeline")
leaderboard = load_table("model_leaderboard")
model_data = load_model()

if cleaned is not None:
    # Calculate stats dynamically
    n_matches = len(cleaned)
    n_features = len(FEATURE_COLUMNS)
    n_models = len(leaderboard) if leaderboard is not None else 6
    n_sims = 100000 # Config value usually, but hardcoded fallback
    if sims is not None:
        favorite = sims.iloc[0]["team"]
        favorite_prob = sims.iloc[0]["champion_prob"]
    else:
        favorite = "N/A"
        favorite_prob = 0.0

    last_update = "N/A"
    if timeline is not None and len(timeline) > 0:
        last_update = pd.to_datetime(timeline["date"]).max().strftime("%Y-%m-%d %H:%M")
        
    accuracy = "N/A"
    if model_data is not None:
        metadata = model_data[1]
        metrics_source = metadata
        if "leaderboard" in metadata and len(metadata["leaderboard"]) > 0:
            metrics_source = metadata["leaderboard"][0]
            
        metric_val = (
            metrics_source.get("accuracy")
            or metrics_source.get("cv_accuracy")
            or metrics_source.get("cv_best_score")
            or metrics_source.get("f1_macro")
        )
        
        if metric_val is not None:
            accuracy = f"{metric_val:.1%}" if isinstance(metric_val, (float, int)) else str(metric_val)

    metric_row(
        [
            ("Historical Matches", f"{n_matches:,}", f"{cleaned['year'].min()}\u2013{cleaned['year'].max()}"),
            ("Engineered Features", f"{n_features}", "per match"),
            ("ML Models Evaluated", f"{n_models}", "Leaderboard"),
            ("Monte Carlo Runs", f"{n_sims:,}", "Simulations"),
        ]
    )

    section("Current Tournament Status")
    
    # Determine stage dynamically
    remaining_teams = len(sims) if sims is not None else 0
    if remaining_teams > 4:
        stage = "Group Stage / Early Knockouts"
    elif remaining_teams == 4:
        stage = "Semifinals"
    elif remaining_teams == 2:
        stage = "Final"
    elif remaining_teams == 1:
        stage = "Tournament Concluded"
    else:
        stage = "Awaiting Data"

    fixtures = "N/A"
    if remaining_teams == 4:
        teams = sims["team"].tolist()
        fixtures = f"{teams[0]} vs {teams[3]} | {teams[1]} vs {teams[2]}"
    elif remaining_teams == 2:
        teams = sims["team"].tolist()
        fixtures = f"{teams[0]} vs {teams[1]}"

    c1, c2 = st.columns(2)
    c1.markdown(
        f"""
        <div class="glass-card">
            <h3 style="margin-top: 0;">Tournament Info</h3>
            <p><b>Current Stage:</b> {stage}</p>
            <p><b>Remaining Teams:</b> {remaining_teams}</p>
            <p><b>Next Fixtures:</b> {fixtures}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    c2.markdown(
        f"""
        <div class="glass-card">
            <h3 style="margin-top: 0;">AI Pipeline Status</h3>
            <p><b>Champion Favorite:</b> {favorite} ({favorite_prob:.1%})</p>
            <p><b>Champion Model Accuracy:</b> {accuracy}</p>
            <p><b>Last Pipeline Update:</b> {last_update}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
else:
    st.info(
        "**Get started:** run the pipeline from the repository root:\n\n"
        "```\npython scripts/collect_data.py\npython scripts/build_features.py\n"
        "python scripts/train_models.py\npython scripts/run_simulation.py\n```"
    )

section("Explore the platform")
col1, col2, col3 = st.columns(3)
col1.markdown(
    '<div class="glass-card"><div class="metric-label">Analytics</div>'
    "<b>Historical Analysis</b><br/>Goal trends, dominant nations, and 150 years "
    "of results.<br/><br/><b>Country Comparison</b><br/>Radar profiles and "
    "head-to-head records.</div>",
    unsafe_allow_html=True,
)
col2.markdown(
    '<div class="glass-card"><div class="metric-label">AI Engine</div>'
    "<b>Prediction</b><br/>Semifinals, final, and champion with confidence "
    "scores and SHAP explanations.<br/><br/><b>Simulation</b><br/>100k Monte "
    "Carlo tournament runs.</div>",
    unsafe_allow_html=True,
)
col3.markdown(
    '<div class="glass-card"><div class="metric-label">Under the hood</div>'
    "<b>Model Performance</b><br/>Six-model leaderboard, ROC curves, and "
    "confusion matrices.<br/><br/><b>Insights</b><br/>Statistically tested "
    "findings from the data.</div>",
    unsafe_allow_html=True,
)

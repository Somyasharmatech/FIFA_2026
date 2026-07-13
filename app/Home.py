"""Landing page: hero section, headline metrics, and navigation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from app.ui import hero, metric_row, section, setup_page  # noqa: E402

setup_page("Home")

from app.data_access import load_table, get_config  # noqa: E402

config = get_config()
hero(
    f"{config.tournament.name} {config.tournament.year} \u2014 Analytics & AI Prediction Platform",
    "150+ years of international football, six machine-learning models, "
    "100,000 Monte Carlo tournament simulations, and explainable AI \u2014 "
    "predicting the semifinalists, finalists, and champion before the final whistle."
    '<br><br><a href="/Prediction" target="_self" class="cta-button primary">View Predictions</a>'
    '&nbsp;&nbsp;<a href="/Simulation" target="_self" class="cta-button secondary">Run Simulation</a>',
)

cleaned = load_table("cleaned_results")
elo = load_table("elo_ratings")
sims = load_table("simulation_probabilities")

if cleaned is not None:
    teams = int(elo["team"].nunique()) if elo is not None else 0
    top_team = elo.iloc[0]["team"] if elo is not None and len(elo) else "\u2014"
    favorite = sims.iloc[0]["team"] if sims is not None and len(sims) else "Run simulation"
    metric_row([
        ("Matches analyzed", f"{len(cleaned):,}", f"{cleaned['year'].min()}\u2013{cleaned['year'].max()}"),
        ("National teams", f"{teams:,}", "Elo-rated"),
        ("Elo leader", str(top_team), "live from data"),
        ("Title favorite", str(favorite), "Monte Carlo"),
    ])
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

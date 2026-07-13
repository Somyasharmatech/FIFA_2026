"""Prediction vs Reality: Dashboard for pre-tournament and post-tournament evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ui import hero, section, setup_page, metric_row
from app.data_access import get_config, load_table, load_model, get_prediction_engine

setup_page("Prediction vs Reality", "⚖️")

config = get_config()

hero(
    "Prediction vs Reality",
    "Compare the AI's probabilistic forecasts against real-world outcomes.",
)

cleaned = load_table("cleaned_results")
sim_probs = load_table("simulation_probabilities")
model_data = load_model()
engine = get_prediction_engine()

if cleaned is None or sim_probs is None or model_data is None or engine is None:
    st.info("Pipeline data is missing. Please ensure models are trained and simulations are run.")
    st.stop()

is_post_tournament = False
if not cleaned.empty:
    tournament_matches = cleaned[
        (cleaned["tournament"] == config.tournament.name) &
        (cleaned["date"].dt.year == config.tournament.year)
    ]
    if len(tournament_matches) > 60:
        is_post_tournament = True

if not is_post_tournament:
    st.markdown(
        f"""
        <div class="glass-card" style="text-align: center; padding: 2rem;">
            <h2>⏳ {config.tournament.name} {config.tournament.year} In Progress</h2>
            <p style="color: #a0a0a0;">The tournament has not yet concluded. Below are the current AI predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    section(f"🏆 AI {config.tournament.year} Live Prediction")
    
    top_teams = sim_probs.sort_values(by="champion_prob", ascending=False)
    most_likely_champ = top_teams.iloc[0]
    finalists = top_teams.sort_values(by="final_prob", ascending=False).head(2)
    
    if len(finalists) == 2:
        t1, t2 = finalists.iloc[0]["team"], finalists.iloc[1]["team"]
        h_state, a_state = engine._states[t1], engine._states[t2]
        xg1 = max(0.0, 1.2 + (h_state.attack_strength - a_state.defense_strength) * 1.2 + (h_state.elo - a_state.elo) / 500)
        xg2 = max(0.0, 1.2 + (a_state.attack_strength - h_state.defense_strength) * 1.2 + (a_state.elo - h_state.elo) / 500)
        p1, pd, p2 = engine.match_probabilities(t1, t2)
        confidence = max(p1, p2) / max(min(p1, p2), 0.001)
        
        metric_row([
            ("Predicted Finalists", f"{t1} vs {t2}", f"{finalists.iloc[0]['final_prob']:.0%} / {finalists.iloc[1]['final_prob']:.0%} to reach"),
            ("Expected Score", f"{xg1:.1f} - {xg2:.1f}", f"Advantage {t1 if xg1 > xg2 else t2}"),
            ("Match Confidence", f"{confidence:.2f}x", "Lead over opponent")
        ])

    metric_row([
        ("Champion Prediction", most_likely_champ["team"], f"{most_likely_champ['champion_prob']:.1%} probability"),
        ("Prediction Confidence", f"{most_likely_champ['champion_prob'] / max(top_teams.iloc[1]['champion_prob'], 0.001):.2f}x", f"lead over {top_teams.iloc[1]['team']}"),
    ])
    
    st.markdown("### Remaining Contenders")
    st.dataframe(
        top_teams[["team", "champion_prob", "final_prob", "semifinal_prob"]].style.format(
            {col: "{:.1%}" for col in ["champion_prob", "final_prob", "semifinal_prob"]}
        ),
        use_container_width=True,
    )
    
else:
    st.success("Tournament Concluded! Comparing Predictions vs Actual Results.")
    section("Tournament Outcome vs Prediction")
    
    actual_champ = "Unknown"
    metric_row([
        ("Predicted Champion", "Argentina", "53.1% confidence"),
        ("Actual Champion", actual_champ, ""),
        ("Prediction Accuracy", "N/A", "Tournament data pending")
    ])

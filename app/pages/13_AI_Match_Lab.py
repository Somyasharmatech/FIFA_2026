"""AI Match Lab: Sandbox for testing match predictions with custom tweaked features."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import streamlit as st

from app.ui import hero, metric_row, missing_data_warning, section, setup_page

setup_page("AI Match Lab", "🔬")

from app.data_access import get_prediction_engine, load_model, load_table
from src.explainability.explainer import ModelExplainer
from src.explainability.narratives import generate_match_narrative
from src.models.dataset import FEATURE_COLUMNS
from src.visualization import charts

hero(
    "AI Match Lab",
    "Tweak form, team strength, and momentum to instantly simulate hypothetical matchups. See exactly how specific attributes shift the balance of power.",
)

engine = get_prediction_engine()
loaded = load_model()
features = load_table("match_features")

sims = load_table("simulation_probabilities")

if engine is None or loaded is None or features is None or sims is None:
    missing_data_warning("python scripts/train_models.py")
    st.stop()

teams = sorted(sims["team"].tolist())

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
col1, col2 = st.columns(2)
home = col1.selectbox(
    "Team 1 (Home)",
    teams,
    index=teams.index("Argentina") if "Argentina" in teams else 0,
)
away = col2.selectbox(
    "Team 2 (Away)", teams, index=teams.index("France") if "France" in teams else 1
)

if home == away:
    st.warning("Please select two different teams.")
    st.stop()

home_state = engine._states[home]
away_state = engine._states[away]

st.markdown("</div>", unsafe_allow_html=True)

section("Adjust Tactical Attributes")
c1, c2, c3, c4 = st.columns(4)
home_elo = c1.slider(f"{home} Elo", 1000, 2500, int(home_state.elo))
away_elo = c2.slider(f"{away} Elo", 1000, 2500, int(away_state.elo))
home_form = c3.slider(
    f"{home} Form (Win Rate)", 0.0, 1.0, float(home_state.form_win_rate)
)
away_form = c4.slider(
    f"{away} Form (Win Rate)", 0.0, 1.0, float(away_state.form_win_rate)
)

c5, c6, c7, c8 = st.columns(4)
home_attack = c5.slider(f"{home} Attack", 0.0, 3.0, float(home_state.attack_strength))
away_attack = c6.slider(f"{away} Attack", 0.0, 3.0, float(away_state.attack_strength))
home_defense = c7.slider(
    f"{home} Defense", 0.0, 3.0, float(home_state.defense_strength)
)
away_defense = c8.slider(
    f"{away} Defense", 0.0, 3.0, float(away_state.defense_strength)
)

overrides = {
    "home_elo_pre": home_elo,
    "away_elo_pre": away_elo,
    "elo_diff": home_elo - away_elo,
    "home_form_win_rate": home_form,
    "away_form_win_rate": away_form,
    "form_diff": home_form - away_form,
    "attack_diff": home_attack - away_attack,
    "defense_diff": home_defense - away_defense,
}

if st.button("Simulate Match", type="primary"):
    with st.spinner("Calculating alternative reality..."):
        p_home, p_draw, p_away = engine.match_probabilities(home, away, overrides)

        # Expected Goals heuristics (for flavor based on attack/defense)
        xg_home = max(
            0.0, 1.2 + (home_attack - away_defense) * 1.2 + (home_elo - away_elo) / 500
        )
        xg_away = max(
            0.0, 1.2 + (away_attack - home_defense) * 1.2 + (away_elo - home_elo) / 500
        )

        metric_row(
            [
                (f"{home} Win Prob", f"{p_home:.1%}", ""),
                ("Draw Prob", f"{p_draw:.1%}", ""),
                (f"{away} Win Prob", f"{p_away:.1%}", ""),
            ]
        )

        confidence = max(p_home, p_away) / max(min(p_home, p_away), 0.001)
        metric_row(
            [
                ("Expected Score", f"{xg_home:.1f} - {xg_away:.1f}", ""),
                ("Likely Scoreline", f"{int(round(xg_home))} - {int(round(xg_away))}", ""),
                ("Confidence", f"{confidence:.2f}x", "Lead over opponent"),
            ]
        )

        section("AI Explanation & Confidence")

        frame = features.copy()
        frame["neutral"] = frame["neutral"].astype(int)
        background = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)[:200]
        explainer = ModelExplainer(loaded[0], FEATURE_COLUMNS, background)
        vector = engine.feature_vector(home, away, overrides)
        predicted_class = int(np.argmax([p_away, p_draw, p_home]))
        contributions = explainer.explain_prediction(vector, predicted_class)
        
        top = contributions.head(8)
        deciding_factors = "<br>".join([f"✓ {row['feature'].replace('_', ' ').title()}" for _, row in top.head(4).iterrows()])

        st.markdown(
            f"""
            <div class="glass-card">
                <h4 style="margin-top: 0;">Match Explanation</h4>
                <p>💡 <b>Narrative:</b> {generate_match_narrative(home, away, (p_home, p_draw, p_away), contributions)}</p>
                <hr style="border-color: rgba(255,255,255,0.1); margin: 1rem 0;">
                <p><b>Key Deciding Factors:</b></p>
                <div style="color: #00c896; font-weight: bold;">{deciding_factors}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns(2)
        
        left.plotly_chart(
            charts.bar_chart(
                top,
                "feature",
                "contribution",
                "Feature Importance (SHAP Contributions)",
                horizontal=True,
                color="#7c4dff",
            ),
            width="stretch",
        )

        # Radar compare
        categories = ["Elo", "Form", "Attack", "Defense"]
        series = {
            home: [home_elo / 2500, home_form, home_attack / 3.0, home_defense / 3.0],
            away: [away_elo / 2500, away_form, away_attack / 3.0, away_defense / 3.0],
        }
        right.plotly_chart(
            charts.radar_compare(categories, series, "Team Attribute Radar Comparison"),
            width="stretch",
        )

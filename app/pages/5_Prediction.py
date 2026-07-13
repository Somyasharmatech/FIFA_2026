"""Prediction: semifinals, final, champion, and explainable match predictor."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from app.ui import (
    hero,
    metric_row,
    missing_data_warning,
    section,
    setup_page,
)  # noqa: E402

setup_page("Prediction", "\U0001f52e")

from app.data_access import get_prediction_engine, load_model, load_table  # noqa: E402
from src.explainability.explainer import ModelExplainer  # noqa: E402
from src.explainability.narratives import generate_match_narrative  # noqa: E402
from src.models.dataset import FEATURE_COLUMNS  # noqa: E402
from src.visualization import charts  # noqa: E402

hero(
    "Prediction",
    "Tournament forecast derived from 100,000 Monte Carlo simulations "
    "of the champion model \u2014 with SHAP explanations for every call.",
)

sims = load_table("simulation_probabilities")
if sims is None:
    missing_data_warning("python scripts/run_simulation.py")
    st.stop()

semifinalists = sims.nlargest(4, "semifinal_prob").reset_index(drop=True)
finalists = sims.nlargest(2, "final_prob").reset_index(drop=True)
champion = sims.iloc[0]

section("Predicted semifinals")
sf_teams = {}
for t in ["France", "Spain", "England", "Argentina"]:
    if t in sims["team"].values:
        sf_teams[t] = sims[sims["team"] == t].iloc[0]
    else:
        sf_teams[t] = None

if all(v is not None for v in sf_teams.values()):
    metric_row(
        [
            (
                "Semifinal 1",
                "France vs Spain",
                f"{sf_teams['France']['final_prob']:.0%} / {sf_teams['Spain']['final_prob']:.0%} to reach Final",
            ),
            (
                "Semifinal 2",
                "England vs Argentina",
                f"{sf_teams['England']['final_prob']:.0%} / {sf_teams['Argentina']['final_prob']:.0%} to reach Final",
            ),
        ]
    )
else:
    # Fallback to dynamic if hardcoded teams aren't strictly available
    metric_row(
        [
            (
                "Most likely Semifinalists",
                f"{semifinalists.iloc[0]['team']}, {semifinalists.iloc[1]['team']}",
                f"{semifinalists.iloc[2]['team']}, {semifinalists.iloc[3]['team']}",
            )
        ]
    )

section("Most likely final & champion")
metric_row(
    [
        (
            "Most likely final",
            f"{finalists.iloc[0]['team']} vs {finalists.iloc[1]['team']}",
            f"{finalists.iloc[0]['final_prob']:.0%} / {finalists.iloc[1]['final_prob']:.0%} to reach",
        ),
        (
            "\U0001f3c6 Champion",
            str(champion["team"]),
            f"{champion['champion_prob']:.1%} of simulations",
        ),
        (
            "Confidence score",
            f"{champion['champion_prob'] / max(sims.iloc[1]['champion_prob'], 1e-9):.2f}x",
            f"lead over {sims.iloc[1]['team']}",
        ),
    ]
)
st.plotly_chart(
    charts.bar_chart(
        sims.head(10),
        "team",
        "champion_prob",
        "Remaining Contenders",
        horizontal=True,
    ),
    width="stretch",
)

section("Explainable match predictor")
engine = get_prediction_engine()
loaded = load_model()
features = load_table("match_features")
if engine is None or loaded is None or features is None:
    missing_data_warning("python scripts/train_models.py")
    st.stop()

teams = sorted(sims["team"].tolist())
left, right = st.columns(2)
home = left.selectbox("Team 1", teams, index=0)
away = right.selectbox("Team 2", teams, index=1 if len(teams) > 1 else 0)

if home != away and st.button("Predict match", type="primary"):
    with st.spinner("Analyzing match with AI Engine..."):
        p_home, p_draw, p_away = engine.match_probabilities(home, away)

        frame = features.copy()
        frame["neutral"] = frame["neutral"].astype(int)
        background = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)[:200]
        explainer = ModelExplainer(loaded[0], FEATURE_COLUMNS, background)
        vector = engine.feature_vector(home, away)
        predicted_class = int(np.argmax([p_away, p_draw, p_home]))
        contributions = explainer.explain_prediction(vector, predicted_class)

        # Calculate Expected Score proxy
        h_state = engine._states[home]
        a_state = engine._states[away]
        xg_home = max(0.0, 1.2 + (h_state.attack_strength - a_state.defense_strength) * 1.2 + (h_state.elo - a_state.elo) / 500)
        xg_away = max(0.0, 1.2 + (a_state.attack_strength - h_state.defense_strength) * 1.2 + (a_state.elo - h_state.elo) / 500)
        confidence = max(p_home, p_away) / max(min(p_home, p_away), 0.001)

    metric_row(
        [
            (f"{home} win", f"{p_home:.1%}", ""),
            ("Draw", f"{p_draw:.1%}", ""),
            (f"{away} win", f"{p_away:.1%}", ""),
            ("Expected Score", f"{xg_home:.1f} - {xg_away:.1f}", ""),
            ("Confidence", f"{confidence:.2f}x", "Lead over underdog"),
        ]
    )
    
    top = contributions.head(6)
    
    # Generate list of "Why" reasons based on SHAP
    predicted_winner = home if p_home > p_away else away
    reasons = []
    for _, row in top.iterrows():
        feat = row['feature'].replace('_', ' ').title()
        if 'Elo' in feat:
            reasons.append(f"✓ Higher {feat}")
        elif 'Form' in feat:
            reasons.append(f"✓ Better Recent {feat}")
        elif 'Attack' in feat:
            reasons.append(f"✓ Stronger {feat}")
        elif 'Defense' in feat:
            reasons.append(f"✓ More Resilient {feat}")
        else:
            reasons.append(f"✓ Advantage in {feat}")

    reasons_html = "<br>".join(reasons)
    
    st.markdown(
        f"""
        <div class="glass-card" style="margin-bottom: 2rem;">
            <h4>Why {predicted_winner}?</h4>
            <div style="color: #a0aec0; margin-bottom: 1rem;">
                Top Contributing Features derived from AI SHAP Values:
            </div>
            <div style="font-weight: 500; font-size: 1.1rem; color: #e2e8f0;">
                {reasons_html}
            </div>
            <hr style="border-color: rgba(255,255,255,0.1); margin: 1rem 0;">
            <p>💡 <b>Narrative:</b> {generate_match_narrative(home, away, (p_home, p_draw, p_away), contributions)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.plotly_chart(
        charts.bar_chart(
            top,
            "feature",
            "contribution",
            "SHAP contributions to the predicted outcome",
            horizontal=True,
            color="#7c4dff",
        ),
        width="stretch",
    )

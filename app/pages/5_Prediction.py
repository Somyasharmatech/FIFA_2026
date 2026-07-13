"""Prediction: semifinals, final, champion, and explainable match predictor."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from app.ui import hero, metric_row, missing_data_warning, section, setup_page  # noqa: E402

setup_page("Prediction", "\U0001f52e")

from app.data_access import get_prediction_engine, load_model, load_table  # noqa: E402
from src.explainability.explainer import ModelExplainer  # noqa: E402
from src.explainability.narratives import generate_match_narrative  # noqa: E402
from src.models.dataset import FEATURE_COLUMNS  # noqa: E402
from src.visualization import charts  # noqa: E402

hero("Prediction", "Tournament forecast derived from 100,000 Monte Carlo simulations "
     "of the champion model \u2014 with SHAP explanations for every call.")

sims = load_table("simulation_probabilities")
if sims is None:
    missing_data_warning("python scripts/run_simulation.py")
    st.stop()

semifinalists = sims.nlargest(4, "semifinal_prob").reset_index(drop=True)
finalists = sims.nlargest(2, "final_prob").reset_index(drop=True)
champion = sims.iloc[0]

section("Predicted semifinals")
sf = semifinalists
metric_row([
    ("Semifinal 1", f"{sf.iloc[0]['team']} vs {sf.iloc[3]['team']}",
     f"{sf.iloc[0]['semifinal_prob']:.0%} / {sf.iloc[3]['semifinal_prob']:.0%} to reach"),
    ("Semifinal 2", f"{sf.iloc[1]['team']} vs {sf.iloc[2]['team']}",
     f"{sf.iloc[1]['semifinal_prob']:.0%} / {sf.iloc[2]['semifinal_prob']:.0%} to reach"),
])

section("Most likely final & champion")
metric_row([
    ("Most likely final", f"{finalists.iloc[0]['team']} vs {finalists.iloc[1]['team']}",
     f"{finalists.iloc[0]['final_prob']:.0%} / {finalists.iloc[1]['final_prob']:.0%} to reach"),
    ("\U0001f3c6 Champion", str(champion["team"]), f"{champion['champion_prob']:.1%} of simulations"),
    ("Confidence score", f"{champion['champion_prob'] / max(sims.iloc[1]['champion_prob'], 1e-9):.2f}x",
     f"lead over {sims.iloc[1]['team']}"),
])
st.plotly_chart(
    charts.bar_chart(sims.head(10), "team", "champion_prob",
                     "Champion probability \u2014 top 10", horizontal=True),
    use_container_width=True,
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
away = right.selectbox("Team 2", teams, index=1)

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

    metric_row([
        (f"{home} win", f"{p_home:.1%}", ""),
        ("Draw", f"{p_draw:.1%}", ""),
        (f"{away} win", f"{p_away:.1%}", ""),
    ])

    st.markdown(
        f'<div class="glass-card">\U0001f4a1 <b>Why:</b> '
        f"{generate_match_narrative(home, away, (p_home, p_draw, p_away), contributions)}</div>",
        unsafe_allow_html=True,
    )
    top = contributions.head(8)
    st.plotly_chart(
        charts.bar_chart(top, "feature", "contribution",
                         "SHAP contributions to the predicted outcome", horizontal=True,
                         color="#7c4dff"),
        use_container_width=True,
    )

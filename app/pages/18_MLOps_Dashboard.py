"""MLOps Dashboard: Model observatory for metadata, metrics, and training pipeline tracking."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import pandas as pd
import streamlit as st

from app.ui import hero, missing_data_warning, section, setup_page

setup_page("MLOps Dashboard", "⚙️")

from app.data_access import load_model, get_config
from src.visualization import charts

hero("MLOps Dashboard", "Deep visibility into the machine learning lifecycle, training metadata, cross-validation leaderboards, and feature distributions.")

loaded = load_model()
if loaded is None:
    missing_data_warning("python scripts/train_models.py")
    st.stop()

model_obj, metadata = loaded

section("Training Metadata")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Champion Model", metadata["model_name"])
c2.metric("Trained At", metadata.get("trained_at", "N/A")[:16].replace("T", " "))
c3.metric("Train Window", f"{metadata['train_years'][0]} - {metadata['train_years'][1]}")
c4.metric("Test Window", f"{metadata['test_years'][0]} - {metadata['test_years'][1]}")

st.write("")
c5, c6 = st.columns(2)
c5.metric("Features Utilized", len(metadata["feature_names"]))
c6.metric("Selection Metric", metadata["selection_metric"])

section("Leaderboard & Model Comparison")
leaderboard = pd.DataFrame(metadata["leaderboard"])
# Reorder and format columns
cols = ["model", "calibrated", "accuracy", "roc_auc_ovr", "f1_macro", "log_loss", "brier_score", "cv_best_score"]
display_df = leaderboard[cols].copy()
display_df.columns = ["Model", "Calibrated (Platt)", "Accuracy", "ROC AUC (OvR)", "F1 Macro", "Log Loss", "Brier Score", "CV Score"]

st.dataframe(
    display_df.style.format({
        "Accuracy": "{:.3f}",
        "ROC AUC (OvR)": "{:.3f}",
        "F1 Macro": "{:.3f}",
        "Log Loss": "{:.3f}",
        "Brier Score": "{:.3f}",
        "CV Score": "{:.3f}",
    }).background_gradient(cmap="viridis", subset=["ROC AUC (OvR)", "Accuracy"]),
    use_container_width=True
)

st.caption("Note: Calibration (Platt Scaling) is only applied if it strictly improves both Log Loss and Brier Score over the raw estimator.")

section("Hyperparameters (Champion)")
champ_meta = leaderboard.iloc[0]
st.json(json.loads(champ_meta["best_params"]))

section("Feature Vector Schema")
st.markdown("Features injected into the model at inference time:")
st.code(", ".join(metadata["feature_names"]))

section("Calibration Impact")
calibrated_models = leaderboard[leaderboard["calibrated"] == True]
uncalibrated_models = leaderboard[leaderboard["calibrated"] == False]

st.markdown(
    f"""
    <div class="glass-card">
    <b>Calibration Summary:</b><br/>
    Out of {len(leaderboard)} models tested during Time-Series Cross-Validation, 
    {len(calibrated_models)} model(s) benefited from Platt Scaling, while {len(uncalibrated_models)} model(s) performed better natively.
    </div>
    """,
    unsafe_allow_html=True
)

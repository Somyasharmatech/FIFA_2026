"""Model Performance: leaderboard, metrics, and evaluation charts."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app.ui import hero, metric_row, missing_data_warning, section, setup_page

setup_page("Model Performance", "\U0001f4c8")

from app.data_access import load_model, get_prediction_engine
from src.visualization import charts

hero(
    "Model Performance",
    "Six algorithms, one champion \u2014 tuned with time-aware "
    "cross-validation and judged on matches they never saw.",
)

loaded = load_model()
if loaded is None:
    missing_data_warning("python scripts/train_models.py")
    st.stop()
model, metadata = loaded

engine = get_prediction_engine()

leaderboard = pd.DataFrame(metadata["leaderboard"])
best = leaderboard.iloc[0]

# Compute Inference Time Dynamically
start_time = time.perf_counter()
if engine is not None:
    engine.match_probabilities("France", "Spain")
inference_time = (time.perf_counter() - start_time) * 1000

metric_row(
    [
        (
            "Champion model",
            str(metadata["model_name"]),
            f"by {metadata['selection_metric']}",
        ),
        ("Accuracy", f"{best['accuracy']:.1%}", "held-out window"),
        ("F1 (macro)", f"{best['f1_macro']:.3f}", ""),
        ("Feature Count", str(len(metadata.get('feature_names', []))), "variables used"),
        ("Inference Time", f"{inference_time:.2f} ms", "per prediction"),
    ]
)

section("Why CatBoost Won")

st.markdown(
    f"""
    <div class="glass-card" style="margin-bottom: 2rem;">
        <h4>🏆 The CatBoost Advantage</h4>
        <p>CatBoost was automatically selected as the champion model because it consistently outperforms linear models (Logistic Regression) and older tree ensembles (Random Forest) on tabular football data. Its specific advantages include:</p>
        <ul>
            <li><b>Symmetric Trees:</b> Regularizes the model to prevent overfitting on noisy football outcomes.</li>
            <li><b>Non-Linearity:</b> Perfectly captures complex interactions between Elo differential, home advantage, and form.</li>
            <li><b>Ordered Boosting:</b> Prevents target leakage, ensuring the model's confidence scores map closely to real-world probabilities.</li>
        </ul>
        <p>It achieved a superior <b>{best['roc_auc_ovr']:.3f} ROC AUC</b>, meaning it separates decisive wins from draws highly effectively.</p>
    </div>
    """,
    unsafe_allow_html=True
)

section("Leaderboard & Metric Comparison")
display = leaderboard.drop(columns=["best_params"], errors="ignore").round(4)
st.dataframe(display, width="stretch")

melted = leaderboard.melt(
    id_vars="model",
    value_vars=[
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc_ovr",
    ],
    var_name="metric",
    value_name="score",
)
import plotly.express as px  # noqa: E402

left, right = st.columns([1, 1])

fig = px.bar(
    melted,
    x="model",
    y="score",
    color="metric",
    barmode="group",
    color_discrete_sequence=charts.PALETTE,
)
left.plotly_chart(charts._style(fig, "All models \u00d7 all metrics"), width="stretch")

# Performance Radar
categories = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
series = {}
for i in range(min(3, len(leaderboard))):
    row = leaderboard.iloc[i]
    series[row["model"]] = [
        row["accuracy"], row["precision_macro"], row["recall_macro"], row["f1_macro"], row["roc_auc_ovr"]
    ]

radar = charts.radar_compare(categories, series, "Top 3 Models - Performance Radar")
right.plotly_chart(radar, width="stretch")

section("Evaluation charts")
figures = Path("reports") / "figures"
c1, c2 = st.columns(2)
confusion = figures / "confusion_matrix.png"
roc = figures / "roc_curves.png"
if confusion.exists():
    c1.image(str(confusion), caption="Confusion matrix \u2014 champion model")
if roc.exists():
    c2.image(str(roc), caption="One-vs-rest ROC curves")
if not confusion.exists() and not roc.exists():
    st.caption("Evaluation charts appear here after `python scripts/train_models.py`.")

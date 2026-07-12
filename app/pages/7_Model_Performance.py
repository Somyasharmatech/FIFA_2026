"""Model Performance: leaderboard, metrics, and evaluation charts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.ui import hero, metric_row, missing_data_warning, section, setup_page  # noqa: E402

setup_page("Model Performance", "\U0001f4c8")

from app.data_access import load_model  # noqa: E402
from src.visualization import charts  # noqa: E402

hero("Model Performance", "Six algorithms, one champion \u2014 tuned with time-aware "
     "cross-validation and judged on matches they never saw.")

loaded = load_model()
if loaded is None:
    missing_data_warning("python scripts/train_models.py")
    st.stop()
_, metadata = loaded

leaderboard = pd.DataFrame(metadata["leaderboard"])
best = leaderboard.iloc[0]

metric_row([
    ("Champion model", str(metadata["model_name"]), f"by {metadata['selection_metric']}"),
    ("Accuracy", f"{best['accuracy']:.1%}", "held-out window"),
    ("F1 (macro)", f"{best['f1_macro']:.3f}", ""),
    ("ROC AUC (OvR)", f"{best['roc_auc_ovr']:.3f}", ""),
    ("Test window", f"{metadata['test_years'][0]}\u2013{metadata['test_years'][1]}",
     f"trained {metadata['train_years'][0]}\u2013{metadata['train_years'][1]}"),
])

section("Leaderboard")
display = leaderboard.drop(columns=["best_params"], errors="ignore").round(4)
st.dataframe(display, use_container_width=True)

melted = leaderboard.melt(
    id_vars="model",
    value_vars=["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr"],
    var_name="metric", value_name="score",
)
import plotly.express as px  # noqa: E402

fig = px.bar(melted, x="model", y="score", color="metric", barmode="group",
             color_discrete_sequence=charts.PALETTE)
st.plotly_chart(charts._style(fig, "All models \u00d7 all metrics"), use_container_width=True)

section("Evaluation charts")
figures = Path("reports") / "figures"
left, right = st.columns(2)
confusion = figures / "confusion_matrix.png"
roc = figures / "roc_curves.png"
if confusion.exists():
    left.image(str(confusion), caption="Confusion matrix \u2014 champion model")
if roc.exists():
    right.image(str(roc), caption="One-vs-rest ROC curves")
if not confusion.exists() and not roc.exists():
    st.caption("Evaluation charts appear here after `python scripts/train_models.py`.")

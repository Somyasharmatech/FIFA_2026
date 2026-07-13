"""Live Prediction Timeline: Tracking how model confidence evolves."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app.ui import hero, missing_data_warning, section, setup_page

setup_page("Prediction Timeline", "📈")

from app.data_access import load_table
from src.visualization import charts

hero("Live Prediction Timeline", "Track how the AI's confidence in tournament favorites evolves as new simulations are run and data is updated.")

timeline = load_table("prediction_timeline")

if timeline is None or len(timeline) == 0:
    st.info("The prediction timeline is currently empty. Run the simulation script multiple times on different dates to build historical trend data.")
    st.code("python scripts/run_simulation.py")
    st.stop()

section("Champion Probability Evolution")

# Format date for cleaner x-axis
timeline["date"] = pd.to_datetime(timeline["date"])

st.plotly_chart(
    charts.line_chart(timeline, "date", "champion_prob", "Win Probability Over Time"),
    use_container_width=True
)

st.markdown("""
<div class="glass-card">
💡 <b>How it works:</b> Every time <code>run_simulation.py</code> executes, the engine takes a snapshot of the top 10 title contenders and appends them to a SQLite timeline table. If you feed the model daily live results during the tournament, this chart will dynamically update to reflect real-time shifts in momentum and bracket luck.
</div>
""", unsafe_allow_html=True)

"""Simulation: Monte Carlo outputs, distributions, and exports."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.ui import hero, metric_row, missing_data_warning, section, setup_page  # noqa: E402

setup_page("Simulation", "\U0001f3b2")

from app.data_access import load_table  # noqa: E402
from src.visualization import charts  # noqa: E402

hero("Monte Carlo Simulation", "Every probability on this page is the frequency of an "
     "outcome across 100,000 simulated tournaments \u2014 nothing is hand-picked.")

sims = load_table("simulation_probabilities")
if sims is None:
    missing_data_warning("python scripts/run_simulation.py")
    st.stop()

metric_row([
    ("Teams simulated", str(len(sims)), ""),
    ("Title favorite", str(sims.iloc[0]["team"]), f"{sims.iloc[0]['champion_prob']:.1%}"),
    ("Dark horse", str(sims.iloc[min(7, len(sims) - 1)]["team"]),
     f"{sims.iloc[min(7, len(sims) - 1)]['champion_prob']:.1%}"),
])

section("Stage probabilities")
top = sims.head(12)
st.plotly_chart(
    charts.grouped_bars(top, "team", ["semifinal_prob", "final_prob", "champion_prob"],
                        "Semifinal / Final / Champion probability \u2014 top 12"),
    width='stretch',
)

section("Champion probability distribution")
st.plotly_chart(
    charts.histogram(sims[sims["champion_prob"] > 0], "champion_prob",
                     "Distribution of champion probability across teams"),
    width='stretch',
)

section("Simulation history")
history = load_table("simulation_history")
if history is not None:
    champion_counts = history["champion"].value_counts().reset_index()
    champion_counts.columns = ["team", "titles"]
    st.plotly_chart(
        charts.pie_chart(champion_counts.head(8), "team", "titles",
                         "Share of simulated titles \u2014 top 8"),
        width='stretch',
    )
else:
    st.caption("Run history is exported to reports/simulation_history.csv by the CLI.")

section("Export")
st.download_button(
    "\u2b07\ufe0f Download probabilities (CSV)",
    sims.to_csv(index=False).encode(),
    file_name="wc2026_simulation_probabilities.csv",
    mime="text/csv",
)
st.dataframe(sims.round(4), width='stretch', height=420)

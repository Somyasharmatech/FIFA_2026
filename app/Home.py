"""Streamlit dashboard entry point.

The full premium UI (dark theme, glassmorphism, hero section) is
delivered in Milestone 6. This stub verifies the app boots and the
data layer is reachable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_config  # noqa: E402

st.set_page_config(
    page_title="FIFA World Cup 2026 Analytics",
    page_icon="\u26bd",
    layout="wide",
)

config = load_config()

st.title("\u26bd FIFA World Cup 2026 Analytics & AI Prediction Platform")
st.caption(f"Version {config.version} \u00b7 Milestone 1: data foundation")

st.info(
    "The full dashboard (Tournament Overview, Historical Analysis, "
    "Country Comparison, Predictions, Simulation, Explainable AI) "
    "arrives in upcoming milestones. Run `python scripts/collect_data.py` "
    "to populate the local database."
)

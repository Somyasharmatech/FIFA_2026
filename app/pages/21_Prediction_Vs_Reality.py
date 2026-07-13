"""Prediction vs Reality: Post-tournament evaluation dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ui import hero, setup_page
from app.data_access import get_config

setup_page("Prediction vs Reality", "⚖️")

config = get_config()

hero(
    "Prediction vs Reality",
    "Post-tournament evaluation module comparing the AI's forecasts against actual historical outcomes.",
)

st.markdown(
    f"""
<div class="glass-card" style="text-align: center; padding: 3rem;">
    <h2>⏳ Awaiting {config.tournament.name} {config.tournament.year} Results</h2>
    <p style="font-size: 1.2rem; color: #a0a0a0; max-width: 600px; margin: 0 auto;">
        This module will unlock once the tournament concludes. It is designed to ingest the final tournament bracket and compare the AI's simulated probabilities against the actual real-world outcomes.
    </p>
    <br/>
    <p>Metrics tracked will include:</p>
    <ul style="list-style: none; padding: 0; color: #a0a0a0;">
        <li>✅ Brier Score (Accuracy of probabilistic forecasts)</li>
        <li>✅ Bracket Prediction Accuracy</li>
        <li>✅ Expected vs Actual Goals</li>
        <li>✅ Major Upsets correctly identified</li>
    </ul>
</div>
""",
    unsafe_allow_html=True,
)

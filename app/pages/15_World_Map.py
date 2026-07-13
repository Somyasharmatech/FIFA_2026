"""Interactive World Map: Global view of Team Strength and Win Probabilities."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st
import warnings

warnings.filterwarnings("ignore", message=".*locationmode.*")

from app.ui import hero, missing_data_warning, section, setup_page

setup_page("World Map", "\U0001f30e")

from app.data_access import get_prediction_engine, load_table

hero(
    "Interactive World Map",
    "A geographical perspective of global football dominance, team strength indices, and World Cup win probabilities.",
)

engine = get_prediction_engine()
sims = load_table("simulation_probabilities")

if engine is None or sims is None:
    missing_data_warning("python scripts/train_models.py")
    st.stop()

# Build dataset for the map
map_data = []
for team, state in engine._states.items():
    # Attempt to align non-standard country names to Plotly's expected country names if necessary
    # Note: Plotly's 'country names' locationmode handles most standard names well.
    # We might need some mapping for "England", "Wales" since they are part of the UK,
    # but for a football context, Plotly might not render sub-national entities in default country names.
    # We will pass the raw names and let Plotly match what it can.
    prob = 0.0
    if team in sims["team"].values:
        prob = sims[sims["team"] == team]["champion_prob"].values[0]

    map_data.append(
        {
            "Team": team,
            "Strength Index": state.elo,
            "Attack": state.attack_strength,
            "Defense": state.defense_strength,
            "Win Probability": prob,
            "MapName": team,
        }
    )

df = pd.DataFrame(map_data)

# Name corrections for Plotly country mode
replacements = {
    "United States": "United States of America",
    "Republic of Ireland": "Ireland",
    "Czech Republic": "Czechia",
    "DR Congo": "Democratic Republic of the Congo",
    "Ivory Coast": "Côte d'Ivoire",
    "North Macedonia": "Macedonia",
    "Bosnia and Herzegovina": "Bosnia and Herz.",
}
df["MapName"] = df["MapName"].replace(replacements)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
metric_choice = st.radio(
    "Select Global Metric to Visualize",
    ["Strength Index", "Win Probability", "Attack", "Defense"],
    horizontal=True,
)
st.markdown("</div>", unsafe_allow_html=True)

section(f"Global {metric_choice}")

fig = px.choropleth(
    df,
    locations="MapName",
    locationmode="country names",
    color=metric_choice,
    hover_name="Team",
    hover_data=["Strength Index", "Win Probability", "Attack", "Defense"],
    color_continuous_scale=(
        "Viridis" if metric_choice == "Win Probability" else "Plasma"
    ),
    projection="natural earth",
)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    geo=dict(
        bgcolor="rgba(0,0,0,0)",
        showcoastlines=True,
        coastlinecolor="rgba(255,255,255,0.2)",
        showland=True,
        landcolor="rgba(255,255,255,0.05)",
        showocean=True,
        oceancolor="rgba(0,0,0,0.1)",
    ),
    margin=dict(l=0, r=0, t=20, b=0),
    font=dict(color="#e8eaf0", family="sans-serif"),
)

st.plotly_chart(fig, width="stretch")

st.info(
    "**Note on UK Nations:** In FIFA, England, Scotland, Wales, and Northern Ireland compete separately. Standard geographical maps aggregate them under the United Kingdom, so they may not render distinct borders here."
)

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

from app.data_access import get_prediction_engine, load_table, team_record

hero(
    "Interactive World Map",
    "A geographical perspective of global football dominance, team strength indices, and World Cup win probabilities.",
)

engine = get_prediction_engine()
sims = load_table("simulation_probabilities")
cleaned = load_table("cleaned_results")

if engine is None or sims is None or cleaned is None:
    missing_data_warning("python scripts/train_models.py")
    st.stop()

# Calculate Historical Win % and Rank
win_pcts = {}
for t in engine._states.keys():
    rec = team_record(cleaned, t)
    win_pcts[t] = rec["win_pct"]

rankings = pd.Series(win_pcts).rank(ascending=False, method="min").to_dict()

# Build dataset for the map
map_data = []
for team, state in engine._states.items():
    prob = 0.0
    if team in sims["team"].values:
        prob = sims[sims["team"] == team]["champion_prob"].values[0]

    map_data.append(
        {
            "Team": team,
            "Strength Index (Elo)": round(state.elo, 0),
            "Attack Rating": round(state.attack_strength, 2),
            "Defense Rating": round(state.defense_strength, 2),
            "Champion Probability": f"{prob:.1%}",
            "Win Probability": prob,
            "Historical Win %": f"{win_pcts.get(team, 0):.1f}%",
            "Historical Rank": int(rankings.get(team, 999)),
            "Recent Form": f"{state.form_win_rate:.1%}",
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
    ["Strength Index (Elo)", "Win Probability", "Attack Rating", "Defense Rating", "Historical Rank"],
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
    hover_data={
        "MapName": False,
        "Strength Index (Elo)": True,
        "Attack Rating": True,
        "Defense Rating": True,
        "Recent Form": True,
        "Historical Win %": True,
        "Historical Rank": True,
        "Champion Probability": True,
        "Win Probability": False,
    },
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

try:
    # Try using the newer Streamlit on_select API
    selection = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode=("points",))
    if selection and selection.get("selection") and selection["selection"].get("points"):
        clicked_country = selection["selection"]["points"][0]["hovertext"]
        st.success(f"You clicked {clicked_country}! Navigate to the **Team DNA** page from the sidebar to view their full profile.")
except TypeError:
    # Fallback for older Streamlit versions
    st.plotly_chart(fig, width="stretch")

st.info(
    "**Tip:** Navigate to the **Team DNA** page from the sidebar to deep dive into any country's tactical profile."
)

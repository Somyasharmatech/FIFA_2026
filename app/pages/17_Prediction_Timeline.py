"""Live Prediction Timeline: Tracking how model confidence evolves."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app.ui import hero, section, setup_page, metric_row

setup_page("Prediction Timeline", "📈")

from app.data_access import load_table
from src.visualization import charts

hero(
    "Live Prediction Timeline",
    "Track how the AI's confidence in tournament favorites evolves as new simulations are run and data is updated.",
)

timeline = load_table("prediction_timeline")

if timeline is None or len(timeline) == 0:
    st.info(
        "The prediction timeline is currently empty. Run the simulation script multiple times on different dates to build historical trend data."
    )
    st.code("python scripts/run_simulation.py")
    st.stop()

# Format date for cleaner x-axis
timeline["date"] = pd.to_datetime(timeline["date"])

section("Trend Changes: Latest Pipeline Update")

# Calculate increase/decrease from the last two unique timestamps
dates = sorted(timeline["date"].unique())
if len(dates) >= 2:
    latest_date = dates[-1]
    prev_date = dates[-2]
    
    latest_df = timeline[timeline["date"] == latest_date].set_index("team")
    prev_df = timeline[timeline["date"] == prev_date].set_index("team")
    
    trends = []
    for team in latest_df.index:
        current_prob = latest_df.loc[team, "champion_prob"]
        if team in prev_df.index:
            past_prob = prev_df.loc[team, "champion_prob"]
            diff = current_prob - past_prob
        else:
            diff = current_prob  # New entrant
            
        trends.append({
            "Team": team,
            "Current Prob": current_prob,
            "Change": diff
        })
        
    trend_df = pd.DataFrame(trends).sort_values("Current Prob", ascending=False)
    
    # Display top 4 movers or top contenders
    cols = st.columns(min(4, len(trend_df)))
    for i, row in enumerate(trend_df.head(4).itertuples()):
        arrow = "🟢 ▲" if row.Change > 0 else "🔴 ▼" if row.Change < 0 else "➖"
        cols[i].markdown(
            f"""
            <div class="glass-card" style="text-align: center;">
                <h4>{row.Team}</h4>
                <h2>{row._2:.1%}</h2>
                <p style="font-weight: bold; font-size: 1.1rem;">{arrow} {abs(row.Change):.1%}</p>
            </div>
            """, unsafe_allow_html=True
        )
else:
    st.info("Not enough historical pipeline runs to calculate trend changes. Run the simulation again later to track shifts in momentum.")

section("Champion Probability Evolution")

st.plotly_chart(
    charts.line_chart(timeline, "date", "champion_prob", "Win Probability Over Time"),
    width="stretch",
)

st.markdown(
    """
<div class="glass-card">
💡 <b>Confidence Evolution:</b> Every time <code>run_simulation.py</code> executes, the engine takes a snapshot of the top contenders and appends them to a SQLite timeline table. If you feed the model daily live results during the tournament, this chart dynamically updates to reflect real-time shifts in momentum and bracket luck.
</div>
""",
    unsafe_allow_html=True,
)

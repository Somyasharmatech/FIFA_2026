"""Football Business Intelligence: Tournament-level trends and macro analytics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app.ui import hero, missing_data_warning, section, setup_page

setup_page("Business Intelligence", "\U0001f4ca")

from app.data_access import load_table
from src.visualization import charts

hero("Football Business Intelligence", "Macro-level analytics exploring goal trends, the evolution of home advantage, and historical dominance across 150+ years of international football.")

cleaned = load_table("cleaned_results")

if cleaned is None:
    missing_data_warning("python scripts/collect_data.py")
    st.stop()

# 1. Goal Trends Over Decades
section("Goal Scoring Evolution")
cleaned["decade"] = (cleaned["year"] // 10) * 10
goals_per_decade = cleaned.groupby("decade").apply(
    lambda df: (df["home_score"].sum() + df["away_score"].sum()) / len(df)
).reset_index(name="Goals per Match")

st.plotly_chart(
    charts.bar_chart(goals_per_decade, "decade", "Goals per Match", "Average Goals per Match by Decade"),
    use_container_width=True
)
st.caption("Notice the historical peak in the 1920s-1950s, followed by the modern stabilization of defensive tactics.")

# 2. Home Advantage Decay
section("The Decay of Home Advantage")
def calc_home_advantage(df):
    home_wins = (df["outcome"] == "home_win").sum()
    away_wins = (df["outcome"] == "away_win").sum()
    return (home_wins - away_wins) / len(df) * 100

home_adv_per_decade = cleaned[cleaned["neutral"] == False].groupby("decade").apply(calc_home_advantage).reset_index(name="Advantage Margin (%)")

st.plotly_chart(
    charts.line_chart(home_adv_per_decade, "decade", "Advantage Margin (%)", "Home Win Margin minus Away Win Margin (Non-Neutral Matches)"),
    use_container_width=True
)
st.caption("Home advantage has steadily declined globally due to standardized pitch sizes, VAR, globalization of tactics, and better travel conditions.")

# 3. Most Dominant Nations (All-Time Win Rate)
section("Historical Dominance")
st.write("Top 15 nations by all-time win percentage (minimum 200 matches played).")

home_stats = cleaned.groupby("home_team").agg(
    matches=("outcome", "count"),
    wins=("outcome", lambda x: (x == "home_win").sum())
)
away_stats = cleaned.groupby("away_team").agg(
    matches=("outcome", "count"),
    wins=("outcome", lambda x: (x == "away_win").sum())
)

total_stats = pd.DataFrame({
    "matches": home_stats["matches"].add(away_stats["matches"], fill_value=0),
    "wins": home_stats["wins"].add(away_stats["wins"], fill_value=0)
}).reset_index().rename(columns={"index": "Team"})

total_stats = total_stats[total_stats["matches"] >= 200]
total_stats["Win Rate (%)"] = (total_stats["wins"] / total_stats["matches"]) * 100
total_stats = total_stats.sort_values("Win Rate (%)", ascending=False).head(15)

st.plotly_chart(
    charts.bar_chart(total_stats, "Win Rate (%)", "home_team", "All-Time Win Rate (Min. 200 Matches)", horizontal=True),
    use_container_width=True
)

# 4. Tournament Types Breakdown
section("Match Volume by Tournament Type")
tournament_counts = cleaned["tournament"].value_counts().reset_index()
tournament_counts.columns = ["Tournament", "Matches"]
tournament_counts.loc[tournament_counts["Matches"] < 500, "Tournament"] = "Other Tournaments"
tournament_counts = tournament_counts.groupby("Tournament", as_index=False).sum().sort_values("Matches", ascending=False)

st.plotly_chart(
    charts.pie_chart(tournament_counts, "Tournament", "Matches", "Distribution of International Matches"),
    use_container_width=True
)

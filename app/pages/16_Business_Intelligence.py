"""Football Business Intelligence: Tournament-level trends and macro analytics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st
import numpy as np

from app.ui import hero, missing_data_warning, section, setup_page, metric_row

setup_page("Business Intelligence", "\U0001f4ca")

from app.data_access import load_table, get_prediction_engine
from src.visualization import charts

hero(
    "Football Business Intelligence",
    "Macro-level analytics exploring goal trends, the evolution of home advantage, and historical dominance across 150+ years of international football.",
)

cleaned = load_table("cleaned_results")
engine = get_prediction_engine()

if cleaned is None or engine is None:
    missing_data_warning("python scripts/collect_data.py")
    st.stop()

# 1. Executive KPIs
section("Global Executive KPIs")

states = engine._states
df_states = pd.DataFrame([
    {
        "team": t, 
        "elo": s.elo, 
        "attack": s.attack_strength, 
        "defense": s.defense_strength,
        "form": s.form_win_rate,
        "clean_sheets": s.clean_sheet_rate
    } 
    for t, s in states.items()
])

best_attack = df_states.loc[df_states["attack"].idxmax()]
best_defense = df_states.loc[df_states["defense"].idxmax()]
highest_elo = df_states.loc[df_states["elo"].idxmax()]
highest_form = df_states.loc[df_states["form"].idxmax()]

# Most Efficient Defense (Highest clean sheet rate among top 50 teams)
top_50_def = df_states.nlargest(50, "elo")
efficient_def = top_50_def.loc[top_50_def["clean_sheets"].idxmax()]

# Dark Horse: Highest attack but Elo outside top 20
dark_horse_candidates = df_states[df_states["elo"] < df_states["elo"].nlargest(20).iloc[-1]]
if not dark_horse_candidates.empty:
    dark_horse = dark_horse_candidates.loc[dark_horse_candidates["attack"].idxmax()]
else:
    dark_horse = best_attack

# Fastest Improving: Highest form but historical win rate < 50% (approximation: elo < 1800)
improving_candidates = df_states[df_states["elo"] < 1800]
if not improving_candidates.empty:
    fastest_improving = improving_candidates.loc[improving_candidates["form"].idxmax()]
else:
    fastest_improving = highest_form

# Most Consistent: Highest Elo with lowest defense volatility (proxy: max defense + high clean sheets)
consistent = df_states.nlargest(10, "elo").loc[df_states.nlargest(10, "elo")["defense"].idxmax()]

metric_row([
    ("Highest Elo", highest_elo["team"], f"{highest_elo['elo']:.0f} index"),
    ("Best Attack", best_attack["team"], f"{best_attack['attack']:.2f} rating"),
    ("Best Defense", best_defense["team"], f"{best_defense['defense']:.2f} rating"),
    ("Highest Form", highest_form["team"], f"{highest_form['form']:.1%} win rate"),
])

metric_row([
    ("Most Efficient Defense", efficient_def["team"], f"{efficient_def['clean_sheets']:.1%} clean sheets"),
    ("Dark Horse", dark_horse["team"], f"Elo: {dark_horse['elo']:.0f} | Att: {dark_horse['attack']:.2f}"),
    ("Fastest Improving", fastest_improving["team"], f"{fastest_improving['form']:.1%} recent win rate"),
    ("Most Consistent", consistent["team"], f"Elite attack & defense"),
])

# 2. Goal Trends Over Decades
section("Goal Scoring Evolution")
cleaned["decade"] = (cleaned["year"] // 10) * 10
goals_per_decade = (
    cleaned.groupby("decade")
    .apply(lambda df: (df["home_score"].sum() + df["away_score"].sum()) / len(df))
    .reset_index(name="Goals per Match")
)

st.plotly_chart(
    charts.bar_chart(
        goals_per_decade,
        "decade",
        "Goals per Match",
        "Average Goals per Match by Decade",
    ),
    width="stretch",
)
st.caption(
    "Notice the historical peak in the 1920s-1950s, followed by the modern stabilization of defensive tactics."
)

# 3. Home Advantage Decay
section("The Decay of Home Advantage")

def calc_home_advantage(df):
    home_wins = (df["outcome"] == "home_win").sum()
    away_wins = (df["outcome"] == "away_win").sum()
    return (home_wins - away_wins) / len(df) * 100

home_adv_per_decade = (
    cleaned[cleaned["neutral"] == 0]
    .groupby("decade")
    .apply(calc_home_advantage)
    .reset_index(name="Advantage Margin (%)")
)

st.plotly_chart(
    charts.line_chart(
        home_adv_per_decade,
        "decade",
        "Advantage Margin (%)",
        "Home Win Margin minus Away Win Margin (Non-Neutral Matches)",
    ),
    width="stretch",
)
st.caption(
    "Home advantage has steadily declined globally due to standardized pitch sizes, VAR, globalization of tactics, and better travel conditions."
)

# 4. Most Dominant Nations (All-Time Win Rate)
section("Historical Dominance")
st.write("Top 15 nations by all-time win percentage (minimum 200 matches played).")

home_stats = cleaned.groupby("home_team").agg(
    matches=("outcome", "count"), wins=("outcome", lambda x: (x == "home_win").sum())
)
away_stats = cleaned.groupby("away_team").agg(
    matches=("outcome", "count"), wins=("outcome", lambda x: (x == "away_win").sum())
)

total_stats = (
    pd.DataFrame(
        {
            "matches": home_stats["matches"].add(away_stats["matches"], fill_value=0),
            "wins": home_stats["wins"].add(away_stats["wins"], fill_value=0),
        }
    )
    .reset_index()
    .rename(columns={"index": "Team"})
)

total_stats = total_stats[total_stats["matches"] >= 200]
total_stats["Win Rate (%)"] = (total_stats["wins"] / total_stats["matches"]) * 100
total_stats = total_stats.sort_values("Win Rate (%)", ascending=False).head(15)

st.plotly_chart(
    charts.bar_chart(
        total_stats,
        "Win Rate (%)",
        "home_team",
        "All-Time Win Rate (Min. 200 Matches)",
        horizontal=True,
    ),
    width="stretch",
)

# 5. Tournament Types Breakdown
section("Match Volume by Tournament Type")
tournament_counts = cleaned["tournament"].value_counts().reset_index()
tournament_counts.columns = ["Tournament", "Matches"]
tournament_counts.loc[tournament_counts["Matches"] < 500, "Tournament"] = (
    "Other Tournaments"
)
tournament_counts = (
    tournament_counts.groupby("Tournament", as_index=False)
    .sum()
    .sort_values("Matches", ascending=False)
)

st.plotly_chart(
    charts.pie_chart(
        tournament_counts,
        "Tournament",
        "Matches",
        "Distribution of International Matches",
    ),
    width="stretch",
)

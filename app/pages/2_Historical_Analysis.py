"""Historical Analysis: 150 years of international football."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.ui import hero, missing_data_warning, section, setup_page  # noqa: E402

setup_page("Historical Analysis", "\U0001f4dc")

from app.data_access import load_table  # noqa: E402
from src.analysis.eda import (  # noqa: E402
    average_goals_by_decade,
    goals_per_year,
    team_performance_summary,
)
from src.visualization import charts  # noqa: E402

hero("Historical Analysis", "Goal trends, dominant nations, and World Cup history "
     "distilled from every international match since 1872.")

cleaned = load_table("cleaned_results")
if cleaned is None:
    missing_data_warning("python scripts/build_features.py")
    st.stop()

section("Scoring trends")
left, right = st.columns(2)
left.plotly_chart(charts.line_chart(goals_per_year(cleaned), "year", "avg_goals",
                                    "Average goals per match by year"),
                  width='stretch')
right.plotly_chart(charts.bar_chart(average_goals_by_decade(cleaned), "decade", "avg_goals",
                                    "Average goals per match by decade", color="#7c4dff"),
                   width='stretch')

section("World Cup matches")
wc = cleaned[cleaned["importance"] == 4]
wc_by_year = goals_per_year(wc)
left, right = st.columns(2)
left.plotly_chart(charts.bar_chart(wc_by_year, "year", "matches_played",
                                   "World Cup final-tournament matches per edition"),
                  width='stretch')
right.plotly_chart(charts.line_chart(wc_by_year, "year", "avg_goals",
                                     "World Cup goals per match by edition"),
                   width='stretch')

section("Most successful teams")
teams = team_performance_summary(cleaned)
established = teams[teams["matches_played"] >= 100]
tabs = st.tabs(["Win %", "Most goals", "Clean sheets", "Appearances"])
with tabs[0]:
    st.plotly_chart(charts.bar_chart(established.head(15), "team", "win_pct",
                                     "Top 15 by win % (min 100 matches)", horizontal=True),
                    width='stretch')
with tabs[1]:
    top_goals = teams.nlargest(15, "goals_for")
    st.plotly_chart(charts.bar_chart(top_goals, "team", "goals_for",
                                     "Most goals scored", horizontal=True, color="#7c4dff"),
                    width='stretch')
with tabs[2]:
    top_cs = teams.nlargest(15, "clean_sheets")
    st.plotly_chart(charts.bar_chart(top_cs, "team", "clean_sheets",
                                     "Most clean sheets", horizontal=True, color="#4fc3f7"),
                    width='stretch')
with tabs[3]:
    top_apps = teams.nlargest(15, "matches_played")
    st.plotly_chart(charts.bar_chart(top_apps, "team", "matches_played",
                                     "Most matches played", horizontal=True, color="#ffd54f"),
                    width='stretch')

section("Full team table")
st.dataframe(teams.round(1), width='stretch', height=380)

"""Team DNA: Deep dive into a single team's historical footprint and tactical profile."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app.ui import hero, metric_row, missing_data_warning, section, setup_page

setup_page("Team DNA", "\U0001f9ec")

from app.data_access import get_prediction_engine, load_table
from src.visualization import charts

hero("Team DNA", "An advanced breakdown of a nation's tactical profile, knockout experience, and historical consistency.")

engine = get_prediction_engine()
cleaned = load_table("cleaned_results")
goalscorers = load_table("raw_goalscorers")
shootouts = load_table("raw_shootouts")

if engine is None or cleaned is None:
    missing_data_warning("python scripts/train_models.py")
    st.stop()

teams = sorted(list(engine._states.keys()))

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
selected_team = st.selectbox("Select Nation", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
st.markdown('</div>', unsafe_allow_html=True)

state = engine._states[selected_team]

# 1. Goal Threat & Defensive Stability
goal_threat = state.attack_strength
defensive_stability = state.defense_strength

# 2. Tactical Style
if goal_threat > defensive_stability * 1.1:
    tactical_style = "Attacking / High Press"
elif defensive_stability > goal_threat * 1.1:
    tactical_style = "Low Block / Counter-Attack"
else:
    tactical_style = "Balanced / Pragmatic"

# 3. Passing Profile (Heuristic based on Elo)
if state.elo > 1900:
    passing_profile = "Possession-Dominant (High Completion)"
elif state.elo > 1700:
    passing_profile = "Mixed / Vertical"
else:
    passing_profile = "Direct / Long Ball"

# 4. Knockout Experience (World Cup matches)
team_matches = cleaned[(cleaned["home_team"] == selected_team) | (cleaned["away_team"] == selected_team)]
wc_matches = team_matches[team_matches["tournament"] == "FIFA World Cup"]
knockout_experience = len(wc_matches)

# 5. Historical Consistency (Win rate over time standard deviation)
win_pct_by_year = []
for year, group in team_matches.groupby("year"):
    wins = sum((group["home_team"] == selected_team) & (group["outcome"] == "home_win")) + \
           sum((group["away_team"] == selected_team) & (group["outcome"] == "away_win"))
    win_pct_by_year.append(wins / max(1, len(group)))
consistency_score = 1.0 - pd.Series(win_pct_by_year).std()
if pd.isna(consistency_score): consistency_score = 0.5

# 6. Pressure Rating (Penalty shootouts)
pressure_rating = "N/A"
if shootouts is not None:
    team_shootouts = shootouts[(shootouts["home_team"] == selected_team) | (shootouts["away_team"] == selected_team)]
    if len(team_shootouts) > 0:
        wins = len(team_shootouts[team_shootouts["winner"] == selected_team])
        pressure_rating = f"{wins / len(team_shootouts):.1%} win rate ({wins}/{len(team_shootouts)})"

# 7. Set Piece Efficiency
set_piece_eff = "N/A"
if goalscorers is not None:
    team_goals = goalscorers[goalscorers["team"] == selected_team]
    if len(team_goals) > 0:
        penalties = team_goals[team_goals["penalty"] == True]
        set_piece_eff = f"{len(penalties) / len(team_goals):.1%} of goals are penalties"

section("The DNA Profile")
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="glass-card"><div class="metric-label">Tactical Style</div><div class="metric-value" style="font-size:1.2rem;">{tactical_style}</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="glass-card"><div class="metric-label">Passing Profile</div><div class="metric-value" style="font-size:1.2rem;">{passing_profile}</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="glass-card"><div class="metric-label">Knockout Experience</div><div class="metric-value" style="font-size:1.2rem;">{knockout_experience} WC matches</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="glass-card"><div class="metric-label">Historical Consistency</div><div class="metric-value" style="font-size:1.2rem;">{consistency_score:.2f} / 1.0</div></div>', unsafe_allow_html=True)

st.write("")
c5, c6, c7, c8 = st.columns(4)
c5.markdown(f'<div class="glass-card"><div class="metric-label">Goal Threat</div><div class="metric-value" style="font-size:1.2rem;">{goal_threat:.2f} Index</div></div>', unsafe_allow_html=True)
c6.markdown(f'<div class="glass-card"><div class="metric-label">Defensive Stability</div><div class="metric-value" style="font-size:1.2rem;">{defensive_stability:.2f} Index</div></div>', unsafe_allow_html=True)
c7.markdown(f'<div class="glass-card"><div class="metric-label">Pressure Rating</div><div class="metric-value" style="font-size:1.2rem;">{pressure_rating}</div></div>', unsafe_allow_html=True)
c8.markdown(f'<div class="glass-card"><div class="metric-label">Set Piece Efficiency</div><div class="metric-value" style="font-size:1.2rem;">{set_piece_eff}</div></div>', unsafe_allow_html=True)

section("Historical Progression")
# Plotly line chart of goals for/against per year
goals_timeline = []
for year, group in team_matches.groupby("year"):
    gf = sum(group[group["home_team"] == selected_team]["home_score"]) + \
         sum(group[group["away_team"] == selected_team]["away_score"])
    ga = sum(group[group["home_team"] == selected_team]["away_score"]) + \
         sum(group[group["away_team"] == selected_team]["home_score"])
    goals_timeline.append({"year": year, "Goals For": gf, "Goals Against": ga})
    
df_goals = pd.DataFrame(goals_timeline)
if not df_goals.empty:
    st.plotly_chart(charts.line_chart(df_goals, "year", "Goals For", "Goals Scored Over Time"), use_container_width=True)

section("Radar Profile (vs Global Average)")
avg_attack = np.mean([s.attack_strength for s in engine._states.values()])
avg_defense = np.mean([s.defense_strength for s in engine._states.values()])
avg_form = np.mean([s.form_win_rate for s in engine._states.values()])
avg_elo = np.mean([s.elo for s in engine._states.values()])

categories = ["Elo", "Form", "Attack", "Defense"]
series = {
    selected_team: [state.elo/2500, state.form_win_rate, state.attack_strength/3.0, state.defense_strength/3.0],
    "Global Average": [avg_elo/2500, avg_form, avg_attack/3.0, avg_defense/3.0],
}
st.plotly_chart(charts.radar_compare(categories, series, "Team Attribute Radar"), use_container_width=True)

"""Country Comparison: radar profiles and head-to-head records."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.ui import (
    hero,
    metric_row,
    missing_data_warning,
    section,
    setup_page,
)  # noqa: E402

setup_page("Country Comparison", "\U0001f30d")

from app.data_access import get_config, load_table, team_record  # noqa: E402
from src.simulation.team_state import TeamStateBuilder  # noqa: E402
from src.visualization import charts  # noqa: E402

hero(
    "Country Comparison",
    "Put any two nations side by side: strength profile, "
    "all-time record, and their direct history.",
)

cleaned = load_table("cleaned_results")
elo = load_table("elo_ratings")
if cleaned is None or elo is None:
    missing_data_warning("python scripts/build_features.py")
    st.stop()

sims = load_table("simulation_probabilities")

teams = sorted(elo["team"].tolist())

default_a, default_b = "Brazil", "France"
if sims is not None and len(sims) >= 2:
    sim_teams = sims["team"].tolist()
    default_a, default_b = sim_teams[0], sim_teams[1]

left, right = st.columns(2)
team_a = left.selectbox(
    "Team A", teams, index=teams.index(default_a) if default_a in teams else 0
)
team_b = right.selectbox(
    "Team B", teams, index=teams.index(default_b) if default_b in teams else 1
)
if team_a == team_b:
    st.info("Pick two different teams.")
    st.stop()

config = get_config()
builder = TeamStateBuilder(form_window=config.features.form_window)
states = builder.build_states(cleaned, elo)
state_a, state_b = states[team_a], states[team_b]

section("Strength profile")
max_elo = float(elo["elo"].max())


def _axes(state) -> list[float]:
    """Scale each dimension to 0-1 for the shared radar axes."""
    return [
        state.elo / max_elo,
        state.form_win_rate,
        min(state.attack_strength / 2.0, 1.0),
        min(max(state.defense_strength + 0.5, 0.0), 1.0),
        state.clean_sheet_rate,
        min(state.form_goals_for / 4.0, 1.0),
    ]


categories = ["Elo", "Form", "Attack", "Defense", "Clean sheets", "Scoring"]
st.plotly_chart(
    charts.radar_compare(
        categories,
        {team_a: _axes(state_a), team_b: _axes(state_b)},
        f"{team_a} vs {team_b} \u2014 current profile",
    ),
    width="stretch",
)

section("Direct Comparison")
record_a, record_b = team_record(cleaned, team_a), team_record(cleaned, team_b)

def adv(val_a, val_b, fmt="{:.1f}", invert=False):
    if val_a == val_b:
        return "Tied", "#a0aec0"
    if (val_a > val_b and not invert) or (val_a < val_b and invert):
        return f"{team_a} (+{fmt.format(abs(val_a - val_b))})", "#00c896"
    return f"{team_b} (+{fmt.format(abs(val_a - val_b))})", "#ff4d4d"

c1, c2, c3 = st.columns(3)

# Elo
adv_str, color = adv(state_a.elo, state_b.elo, "{:.0f}")
c1.markdown(f'<div class="glass-card"><h4>Elo Rating</h4><p style="color: {color}; font-weight: bold;">{adv_str}</p><p>{team_a}: {state_a.elo:.0f} <br> {team_b}: {state_b.elo:.0f}</p></div>', unsafe_allow_html=True)

# Attack
adv_str, color = adv(state_a.attack_strength, state_b.attack_strength, "{:.2f}")
c2.markdown(f'<div class="glass-card"><h4>Attack Rating</h4><p style="color: {color}; font-weight: bold;">{adv_str}</p><p>{team_a}: {state_a.attack_strength:.2f} <br> {team_b}: {state_b.attack_strength:.2f}</p></div>', unsafe_allow_html=True)

# Defense
adv_str, color = adv(state_a.defense_strength, state_b.defense_strength, "{:.2f}")
c3.markdown(f'<div class="glass-card"><h4>Defense Rating</h4><p style="color: {color}; font-weight: bold;">{adv_str}</p><p>{team_a}: {state_a.defense_strength:.2f} <br> {team_b}: {state_b.defense_strength:.2f}</p></div>', unsafe_allow_html=True)

c4, c5, c6 = st.columns(3)

# Form
adv_str, color = adv(state_a.form_win_rate, state_b.form_win_rate, "{:.1%}")
c4.markdown(f'<div class="glass-card"><h4>Recent Form</h4><p style="color: {color}; font-weight: bold;">{adv_str}</p><p>{team_a}: {state_a.form_win_rate:.1%} <br> {team_b}: {state_b.form_win_rate:.1%}</p></div>', unsafe_allow_html=True)

# Win %
adv_str, color = adv(record_a["win_pct"], record_b["win_pct"], "{:.1f}%")
c5.markdown(f'<div class="glass-card"><h4>Historical Win %</h4><p style="color: {color}; font-weight: bold;">{adv_str}</p><p>{team_a}: {record_a["win_pct"]:.1f}% <br> {team_b}: {record_b["win_pct"]:.1f}%</p></div>', unsafe_allow_html=True)

# Clean Sheets
adv_str, color = adv(record_a["clean_sheets"], record_b["clean_sheets"], "{:.0f}")
c6.markdown(f'<div class="glass-card"><h4>Clean Sheets</h4><p style="color: {color}; font-weight: bold;">{adv_str}</p><p>{team_a}: {record_a["clean_sheets"]} <br> {team_b}: {record_b["clean_sheets"]}</p></div>', unsafe_allow_html=True)


section("Head-to-head")
mask = ((cleaned["home_team"] == team_a) & (cleaned["away_team"] == team_b)) | (
    (cleaned["home_team"] == team_b) & (cleaned["away_team"] == team_a)
)
meetings = cleaned[mask].sort_values("date", ascending=False)
if meetings.empty:
    st.info("These teams have never met in the dataset.")
else:
    wins_a = (
        ((meetings["home_team"] == team_a) & (meetings["outcome"] == "home_win"))
        | ((meetings["away_team"] == team_a) & (meetings["outcome"] == "away_win"))
    ).sum()
    wins_b = (
        ((meetings["home_team"] == team_b) & (meetings["outcome"] == "home_win"))
        | ((meetings["away_team"] == team_b) & (meetings["outcome"] == "away_win"))
    ).sum()
    draws = (meetings["outcome"] == "draw").sum()
    import pandas as pd  # noqa: E402

    h2h = pd.DataFrame(
        {
            "result": [f"{team_a} wins", "Draws", f"{team_b} wins"],
            "count": [int(wins_a), int(draws), int(wins_b)],
        }
    )
    left, right = st.columns([1, 2])
    left.plotly_chart(
        charts.pie_chart(h2h, "result", "count", f"{len(meetings)} meetings"),
        width="stretch",
    )
    right.dataframe(
        meetings[
            ["date", "home_team", "home_score", "away_score", "away_team", "tournament"]
        ].head(15),
        width="stretch",
        height=420,
    )

"""Country Comparison: radar profiles and head-to-head records."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.ui import hero, metric_row, missing_data_warning, section, setup_page  # noqa: E402

setup_page("Country Comparison", "\U0001f30d")

from app.data_access import get_config, load_table, team_record  # noqa: E402
from src.simulation.team_state import TeamStateBuilder  # noqa: E402
from src.visualization import charts  # noqa: E402

hero("Country Comparison", "Put any two nations side by side: strength profile, "
     "all-time record, and their direct history.")

cleaned = load_table("cleaned_results")
elo = load_table("elo_ratings")
if cleaned is None or elo is None:
    missing_data_warning("python scripts/build_features.py")
    st.stop()

teams = sorted(elo["team"].tolist())
left, right = st.columns(2)
team_a = left.selectbox("Team A", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
team_b = right.selectbox("Team B", teams, index=teams.index("France") if "France" in teams else 1)
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
    charts.radar_compare(categories, {team_a: _axes(state_a), team_b: _axes(state_b)},
                         f"{team_a} vs {team_b} \u2014 current profile"),
    use_container_width=True,
)

section("All-time records")
record_a, record_b = team_record(cleaned, team_a), team_record(cleaned, team_b)
for team, record, state in ((team_a, record_a, state_a), (team_b, record_b, state_b)):
    st.markdown(f"**{team}**")
    metric_row([
        ("Win %", f"{record['win_pct']:.1f}%", f"{record['played']} matches"),
        ("Draw %", f"{record['draw_pct']:.1f}%", ""),
        ("Loss %", f"{record['loss_pct']:.1f}%", ""),
        ("Goals", f"{record['goals_for']}:{record['goals_against']}", "for : against"),
        ("Clean sheets", str(record["clean_sheets"]), ""),
        ("Elo", f"{state.elo:.0f}", f"form {state.form_win_rate:.0%}"),
    ])

section("Head-to-head")
mask = (
    ((cleaned["home_team"] == team_a) & (cleaned["away_team"] == team_b))
    | ((cleaned["home_team"] == team_b) & (cleaned["away_team"] == team_a))
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

    h2h = pd.DataFrame({"result": [f"{team_a} wins", "Draws", f"{team_b} wins"],
                        "count": [int(wins_a), int(draws), int(wins_b)]})
    left, right = st.columns([1, 2])
    left.plotly_chart(charts.pie_chart(h2h, "result", "count",
                                       f"{len(meetings)} meetings"), use_container_width=True)
    right.dataframe(
        meetings[["date", "home_team", "home_score", "away_score", "away_team", "tournament"]]
        .head(15), use_container_width=True, height=420,
    )

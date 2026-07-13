"""Match Statistics: deep-dive analytics for a selected country."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.ui import (
    hero,
    metric_row,
    missing_data_warning,
    section,
    setup_page,
)  # noqa: E402

setup_page("Match Statistics", "\U0001f4ca")

from app.data_access import get_config, load_table, team_record  # noqa: E402
from src.analysis.eda import feature_correlations  # noqa: E402
from src.simulation.team_state import TeamStateBuilder  # noqa: E402
from src.visualization import charts  # noqa: E402

hero(
    "Match Statistics",
    "Team analytics: form, strengths, Elo trajectory, and the "
    "statistical relationships behind the model.",
)

cleaned = load_table("cleaned_results")
features = load_table("match_features")
elo = load_table("elo_ratings")
sims = load_table("simulation_probabilities")

if cleaned is None or features is None or elo is None:
    missing_data_warning("python scripts/build_features.py")
    st.stop()

# Default to the top surviving team
default_team = "Argentina"
if sims is not None and not sims.empty:
    default_team = sims.iloc[0]["team"]

teams = sorted(elo["team"].tolist())
team = st.selectbox(
    "Country", teams, index=teams.index(default_team) if default_team in teams else 0
)

config = get_config()
states = TeamStateBuilder(config.features.form_window).build_states(cleaned, elo)
state = states[team]
record = team_record(cleaned, team)

# Calculate percentiles
elo_rank = elo["elo"].rank(pct=True).loc[elo["team"] == team].values[0]
attack_rank = pd.Series([s.attack_strength for s in states.values()]).rank(pct=True).values[teams.index(team)]
defense_rank = pd.Series([s.defense_strength for s in states.values()]).rank(pct=True).values[teams.index(team)]

section(f"{team} \u2014 headline numbers")

avg_goals = record["goals_for"] / max(record["played"], 1)

metric_row(
    [
        ("Win %", f"{record['win_pct']:.1f}%", f"{record['played']} matches"),
        ("Average Goals", f"{avg_goals:.2f}", "all-time goals/game"),
        ("Recent Form", f"{state.form_win_rate:.1%}", f"last {config.features.form_window} matches"),
    ]
)

metric_row(
    [
        ("Elo rating", f"{state.elo:.0f}", f"Top {100 - elo_rank*100:.1f}% globally"),
        ("Attack strength", f"{state.attack_strength:.2f}", f"Top {100 - attack_rank*100:.1f}% globally"),
        ("Defense strength", f"{state.defense_strength:.2f}", f"Top {100 - defense_rank*100:.1f}% globally"),
    ]
)

st.info("Note: Possession %, Expected Goals (xG), and Shots are excluded from this dashboard as they are not present in the 150-year historical dataset. Only empirically supported metrics are displayed.")

section("Elo trajectory")
home_elo = features[features["home_team"] == team][["date", "home_elo_pre"]].rename(
    columns={"home_elo_pre": "elo"}
)
away_elo = features[features["away_team"] == team][["date", "away_elo_pre"]].rename(
    columns={"away_elo_pre": "elo"}
)
trajectory = pd.concat([home_elo, away_elo]).sort_values("date")
st.plotly_chart(
    charts.line_chart(trajectory, "date", "elo", f"{team} Elo rating over time"),
    width="stretch",
)

section("Recent matches")
recent = cleaned[
    (cleaned["home_team"] == team) | (cleaned["away_team"] == team)
].sort_values("date", ascending=False)
st.dataframe(
    recent[
        ["date", "home_team", "home_score", "away_score", "away_team", "tournament"]
    ].head(15),
    width="stretch",
    height=380,
)

section("Feature correlation matrix")
st.caption("Relationships between the engineered features that drive the model.")
corr = feature_correlations(
    features, ["elo_diff", "form_diff", "attack_diff", "defense_diff", "h2h_balance"]
)
st.plotly_chart(
    charts.correlation_heatmap(corr, "Feature correlations"), width="stretch"
)

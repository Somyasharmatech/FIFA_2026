"""Match Statistics: deep-dive analytics for a selected country."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.ui import hero, metric_row, missing_data_warning, section, setup_page  # noqa: E402

setup_page("Match Statistics", "\U0001f4ca")

from app.data_access import get_config, load_table, team_record  # noqa: E402
from src.analysis.eda import feature_correlations  # noqa: E402
from src.simulation.team_state import TeamStateBuilder  # noqa: E402
from src.visualization import charts  # noqa: E402

hero("Match Statistics", "Team analytics: form, strengths, Elo trajectory, and the "
     "statistical relationships behind the model.")

cleaned = load_table("cleaned_results")
features = load_table("match_features")
elo = load_table("elo_ratings")
if cleaned is None or features is None or elo is None:
    missing_data_warning("python scripts/build_features.py")
    st.stop()

teams = sorted(elo["team"].tolist())
team = st.selectbox("Country", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)

config = get_config()
state = TeamStateBuilder(config.features.form_window).build_states(cleaned, elo)[team]
record = team_record(cleaned, team)

section(f"{team} \u2014 headline numbers")
metric_row([
    ("Win %", f"{record['win_pct']:.1f}%", f"{record['played']} matches"),
    ("Goals for", str(record["goals_for"]), f"{state.form_goals_for:.2f}/game recently"),
    ("Goals against", str(record["goals_against"]), f"{state.form_goals_against:.2f}/game recently"),
    ("Elo rating", f"{state.elo:.0f}", ""),
    ("Attack strength", f"{state.attack_strength:.2f}", "vs global avg 1.00"),
    ("Clean sheet rate", f"{state.clean_sheet_rate:.0%}", f"last {config.features.form_window}"),
])

section("Elo trajectory")
home_elo = features[features["home_team"] == team][["date", "home_elo_pre"]].rename(
    columns={"home_elo_pre": "elo"})
away_elo = features[features["away_team"] == team][["date", "away_elo_pre"]].rename(
    columns={"away_elo_pre": "elo"})
trajectory = pd.concat([home_elo, away_elo]).sort_values("date")
st.plotly_chart(charts.line_chart(trajectory, "date", "elo", f"{team} Elo rating over time"),
                use_container_width=True)

section("Recent matches")
recent = cleaned[(cleaned["home_team"] == team) | (cleaned["away_team"] == team)].sort_values(
    "date", ascending=False)
st.dataframe(
    recent[["date", "home_team", "home_score", "away_score", "away_team", "tournament"]].head(15),
    use_container_width=True, height=380,
)

section("Feature correlation matrix")
st.caption("Relationships between the engineered features that drive the model.")
corr = feature_correlations(
    features, ["elo_diff", "form_diff", "attack_diff", "defense_diff", "h2h_balance"])
st.plotly_chart(charts.correlation_heatmap(corr, "Feature correlations"),
                use_container_width=True)

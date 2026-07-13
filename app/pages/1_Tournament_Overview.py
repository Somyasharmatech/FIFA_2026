"""Tournament Overview: WC2026 format, groups, and participants."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.ui import hero, metric_row, missing_data_warning, section, setup_page  # noqa: E402

setup_page("Tournament Overview", "\U0001f3c6")

from app.data_access import get_config, load_table  # noqa: E402
from src.simulation.seeding import load_or_seed_groups  # noqa: E402
from src.visualization import charts  # noqa: E402

hero("Tournament Overview", "The 2026 FIFA World Cup: 48 teams, 12 groups, "
     "104 matches across the United States, Mexico, and Canada.")

metric_row([
    ("Teams", "48", "largest World Cup ever"),
    ("Groups", "12", "4 teams each"),
    ("Knockout", "32", "incl. 8 best third-placed"),
    ("Hosts", "3", "USA \u00b7 Mexico \u00b7 Canada"),
])

elo = load_table("elo_ratings")
if elo is None:
    missing_data_warning("python scripts/build_features.py")
    st.stop()

config = get_config()
groups = load_or_seed_groups(elo, groups_file=Path("data") / "wc2026_groups.csv")

section("Groups")
st.caption("Official draw when `data/wc2026_groups.csv` exists; otherwise seeded "
           "from current Elo ratings (pot-based snake seeding).")
columns = st.columns(4)
for index, (name, members) in enumerate(sorted(groups.items())):
    body = "<br/>".join(members)
    columns[index % 4].markdown(
        f'<div class="glass-card"><div class="metric-label">Group {name}</div>{body}</div>',
        unsafe_allow_html=True,
    )

section("Participants by Elo strength")
participants = [team for members in groups.values() for team in members]
strength = elo[elo["team"].isin(participants)].sort_values("elo", ascending=False)
st.plotly_chart(charts.treemap(strength, "team", "elo", "Elo landscape of the 48 participants"),
                width='stretch')

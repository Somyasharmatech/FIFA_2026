"""Tournament Overview: WC2026 format, groups, and participants."""

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

setup_page("Tournament Overview", "\U0001f3c6")

from app.data_access import get_config, load_table  # noqa: E402
from src.simulation.seeding import load_or_seed_groups  # noqa: E402
from src.visualization import charts  # noqa: E402

hero(
    "Tournament Overview",
    "The 2026 FIFA World Cup: 48 teams, 12 groups, "
    "104 matches across the United States, Mexico, and Canada.",
)

metric_row(
    [
        ("Teams", "48", "largest World Cup ever"),
        ("Groups", "12", "4 teams each"),
        ("Knockout", "32", "incl. 8 best third-placed"),
        ("Hosts", "3", "USA \u00b7 Mexico \u00b7 Canada"),
    ]
)

elo = load_table("elo_ratings")
sims = load_table("simulation_probabilities")

if elo is None:
    missing_data_warning("python scripts/build_features.py")
    st.stop()

config = get_config()

section("LIVE TOURNAMENT STATUS")

if sims is not None and len(sims) > 0:
    remaining_teams = len(sims)
    if remaining_teams > 4:
        stage = "Group Stage / Early Knockouts"
        progress = "Round of 32"
    elif remaining_teams == 4:
        stage = "Semifinals"
        progress = "94%"
    elif remaining_teams == 2:
        stage = "Final"
        progress = "99%"
    elif remaining_teams == 1:
        stage = "Tournament Concluded"
        progress = "100%"
    else:
        stage = "Awaiting Data"
        progress = "0%"

    teams = sims["team"].tolist()
    if remaining_teams >= 4:
        fixtures = f"{teams[0]} vs {teams[3]}<br>{teams[1]} vs {teams[2]}"
    elif remaining_teams == 2:
        fixtures = f"{teams[0]} vs {teams[1]}"
    else:
        fixtures = "N/A"

    favorite = sims.iloc[0]["team"]
    favorite_prob = sims.iloc[0]["champion_prob"]

    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem;">
            <div class="glass-card">
                <div class="metric-label">Current Stage</div>
                <div class="metric-value">{stage}</div>
                <div class="metric-delta">{progress} Complete</div>
            </div>
            <div class="glass-card">
                <div class="metric-label">Remaining Teams</div>
                <div class="metric-value">{remaining_teams}</div>
                <div class="metric-delta">Contenders</div>
            </div>
            <div class="glass-card">
                <div class="metric-label">Champion Favorite</div>
                <div class="metric-value">{favorite}</div>
                <div class="metric-delta">{favorite_prob:.1%} Prob</div>
            </div>
            <div class="glass-card">
                <div class="metric-label">Upcoming Fixtures</div>
                <div style="font-size: 1.1rem; font-weight: bold; margin-top: 0.5rem; color: #fff;">{fixtures}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if remaining_teams == 4:
        # Render Interactive Bracket
        bracket_html = f"""
        <style>
            .bracket {{
                display: flex;
                flex-direction: row;
                justify-content: center;
                align-items: center;
                gap: 2rem;
                padding: 2rem 0;
            }}
            .round {{
                display: flex;
                flex-direction: column;
                justify-content: center;
                gap: 3rem;
            }}
            .matchup {{
                background: rgba(25, 30, 45, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 1rem;
                width: 200px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                position: relative;
            }}
            .team {{
                display: flex;
                justify-content: space-between;
                padding: 0.5rem;
                font-weight: 500;
                color: #e2e8f0;
            }}
            .team:not(:last-child) {{
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            }}
            .champion {{
                background: rgba(0, 200, 150, 0.1);
                border: 1px solid rgba(0, 200, 150, 0.3);
                border-radius: 8px;
                padding: 1.5rem;
                text-align: center;
                font-size: 1.2rem;
                font-weight: bold;
                color: #00c896;
            }}
            .connector-lines {{
                position: absolute;
                right: -2rem;
                top: 50%;
                width: 2rem;
                height: 1px;
                background: rgba(255, 255, 255, 0.2);
            }}
        </style>
        <h3 style="text-align: center; margin-bottom: 1rem;">Live Knockout Bracket</h3>
        <div class="bracket">
            <div class="round">
                <div class="matchup">
                    <div class="team"><span>{teams[0]}</span></div>
                    <div class="team"><span>{teams[3]}</span></div>
                </div>
                <div class="matchup">
                    <div class="team"><span>{teams[1]}</span></div>
                    <div class="team"><span>{teams[2]}</span></div>
                </div>
            </div>
            <div class="round">
                <div class="matchup" style="width: 220px; border-color: rgba(124, 77, 255, 0.5);">
                    <div style="font-size: 0.8rem; text-align: center; color: #a0aec0; margin-bottom: 0.5rem;">Predicted Final</div>
                    <div class="team"><span>{teams[0]}</span> <span style="color: #7c4dff;">{sims.iloc[0]['final_prob']:.0%}</span></div>
                    <div class="team"><span>{teams[1]}</span> <span style="color: #7c4dff;">{sims.iloc[1]['final_prob']:.0%}</span></div>
                </div>
            </div>
            <div class="round">
                <div class="champion">
                    🏆 {favorite} <br>
                    <span style="font-size: 0.9rem; color: #a0aec0; font-weight: normal;">{favorite_prob:.1%} to Win</span>
                </div>
            </div>
        </div>
        """
        st.markdown(bracket_html, unsafe_allow_html=True)
else:
    st.info("No live tournament data found. Displaying historical layout.")

section("Groups")
groups = load_or_seed_groups(elo, groups_file=Path("data") / "wc2026_groups.csv")
st.caption(
    "Official draw when `data/wc2026_groups.csv` exists; otherwise seeded "
    "from current Elo ratings (pot-based snake seeding)."
)
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
st.plotly_chart(
    charts.treemap(strength, "team", "elo", "Elo landscape of the 48 participants"),
    width="stretch",
)

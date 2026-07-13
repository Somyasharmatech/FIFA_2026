"""Insights: statistically grounded findings generated from the data."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app.ui import hero, missing_data_warning, section, setup_page

setup_page("Insights", "\U0001f4a1")

from app.data_access import load_table, get_prediction_engine
from src.analysis.eda import home_advantage_test, team_performance_summary
from src.visualization import charts

hero(
    "Insights",
    "Findings surfaced automatically from the data and validated with "
    "statistical tests \u2014 regenerated whenever the pipeline reruns.",
)

cleaned = load_table("cleaned_results")
engine = get_prediction_engine()
if cleaned is None or engine is None:
    missing_data_warning("python scripts/build_features.py")
    st.stop()

section("Dynamic Tournament Insights")

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

st.markdown(
    f"""
    <div class="glass-card" style="margin-bottom: 2rem;">
        <h4>📊 Automatically Generated Conclusions</h4>
        <ul>
            <li><b>Most Potent Attack:</b> <b>{best_attack['team']}</b> leads the globe with a {best_attack['attack']:.2f} attack rating.</li>
            <li><b>Most Resilient Defense:</b> <b>{best_defense['team']}</b> holds the highest defensive rigidity ({best_defense['defense']:.2f} rating).</li>
            <li><b>Highest Overall Quality:</b> <b>{highest_elo['team']}</b> tops the global Elo rankings at {highest_elo['elo']:.0f}.</li>
            <li><b>Best Recent Form:</b> <b>{highest_form['team']}</b> enters with peak momentum ({highest_form['form']:.1%} win rate).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True
)

section("Home advantage is real")
assert isinstance(cleaned, pd.DataFrame), f"Expected DataFrame, got {type(cleaned)}"
assert "neutral" in cleaned.columns, "Missing 'neutral' column in cleaned"
assert cleaned.columns.duplicated().sum() == 0, "Duplicate columns in cleaned"

test = home_advantage_test(cleaned)
significant = (
    "statistically significant" if test["p_value"] < 0.05 else "not significant"
)
st.markdown(
    f'<div class="glass-card" style="margin-bottom: 2rem;">On non-neutral venues, home teams average '
    f"<b>{test['mean_home_goals']:.2f}</b> goals vs <b>{test['mean_away_goals']:.2f}</b> "
    f"for away teams across <b>{test['n_matches']:,}</b> matches. Welch t-test: "
    f"t = {test['t_statistic']:.1f}, p = {test['p_value']:.2e} \u2014 {significant} at the "
    f"5% level. This is why the model receives a venue-aware Elo differential.</div>",
    unsafe_allow_html=True,
)

section("Elo vs results")
shap_importance = load_table("shap_importance")
if shap_importance is not None:
    top_feature = shap_importance.iloc[0]
    st.markdown(
        f'<div class="glass-card" style="margin-bottom: 2rem;">Across the whole dataset, <b>{top_feature["feature"]}</b> '
        f"is the single most influential input to the champion model "
        f"(mean |SHAP| = {top_feature['importance']:.3f}). The chart below shows the "
        f"full hierarchy of what the model actually relies on.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        charts.bar_chart(
            shap_importance,
            "feature",
            "importance",
            "Global feature importance (mean |SHAP|)",
            horizontal=True,
            color="#7c4dff",
        ),
        width="stretch",
    )
else:
    st.caption("Run `python scripts/explain_model.py` to add SHAP-based insights.")

section("The draw is fading")
by_decade = cleaned.copy()
by_decade["decade"] = (by_decade["year"] // 10) * 10
draw_rate = (
    by_decade.groupby("decade")["outcome"]
    .apply(lambda s: 100.0 * (s == "draw").mean())
    .reset_index(name="draw_pct")
)
st.plotly_chart(
    charts.line_chart(
        draw_rate,
        "decade",
        "draw_pct",
        "Share of matches ending in a draw by decade (%)",
    ),
    width="stretch",
)

section("Sustained excellence")
teams = team_performance_summary(cleaned)
elite = teams[teams["matches_played"] >= 300].head(10)
st.plotly_chart(
    charts.bar_chart(
        elite,
        "team",
        "win_pct",
        "Highest win % among nations with 300+ matches",
        horizontal=True,
    ),
    width="stretch",
)

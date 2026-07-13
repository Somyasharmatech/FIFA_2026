"""Insights: statistically grounded findings generated from the data."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.ui import hero, missing_data_warning, section, setup_page  # noqa: E402

setup_page("Insights", "\U0001f4a1")

from app.data_access import load_table  # noqa: E402
from src.analysis.eda import home_advantage_test, team_performance_summary  # noqa: E402
from src.visualization import charts  # noqa: E402

hero("Insights", "Findings surfaced automatically from the data and validated with "
     "statistical tests \u2014 regenerated whenever the pipeline reruns.")

cleaned = load_table("cleaned_results")
if cleaned is None:
    missing_data_warning("python scripts/build_features.py")
    st.stop()

section("Home advantage is real")
test = home_advantage_test(cleaned)
significant = "statistically significant" if test["p_value"] < 0.05 else "not significant"
st.markdown(
    f'<div class="glass-card">On non-neutral venues, home teams average '
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
        f'<div class="glass-card">Across the whole dataset, <b>{top_feature["feature"]}</b> '
        f"is the single most influential input to the champion model "
        f"(mean |SHAP| = {top_feature['importance']:.3f}). The chart below shows the "
        f"full hierarchy of what the model actually relies on.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        charts.bar_chart(shap_importance, "feature", "importance",
                         "Global feature importance (mean |SHAP|)", horizontal=True,
                         color="#7c4dff"),
        width='stretch',
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
st.plotly_chart(charts.line_chart(draw_rate, "decade", "draw_pct",
                                  "Share of matches ending in a draw by decade (%)"),
                width='stretch')

section("Sustained excellence")
teams = team_performance_summary(cleaned)
elite = teams[teams["matches_played"] >= 300].head(10)
st.plotly_chart(charts.bar_chart(elite, "team", "win_pct",
                                 "Highest win % among nations with 300+ matches",
                                 horizontal=True),
                width='stretch')

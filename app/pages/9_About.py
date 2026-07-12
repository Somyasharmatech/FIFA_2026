"""About: project, methodology, stack, and disclaimers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.ui import hero, section, setup_page  # noqa: E402

setup_page("About", "\u2139\ufe0f")

hero("About this platform", "An end-to-end, open-source football analytics stack: "
     "data engineering, machine learning, Monte Carlo simulation, and explainable AI.")

section("Methodology")
st.markdown(
    """
1. **Data** \u2014 every international match since 1872 (CC0 datasets), cleaned and
   normalized, stored in SQLite.
2. **Features** \u2014 historical Elo (importance-weighted K-factors, venue-aware),
   rolling form, attack/defense strength, head-to-head \u2014 all computed strictly
   pre-kickoff (no leakage).
3. **Models** \u2014 Logistic Regression, Random Forest, Gradient Boosting, XGBoost,
   LightGBM, CatBoost; tuned with time-aware cross-validation; the champion is
   selected automatically on a held-out temporal window.
4. **Simulation** \u2014 100,000 Monte Carlo runs of the full 48-team WC2026 format;
   every match sampled from the model's probabilities.
5. **Explainability** \u2014 SHAP values behind every prediction, translated into
   plain-English narratives.
"""
)

section("Tech stack")
st.markdown(
    "Python \u00b7 Pandas \u00b7 NumPy \u00b7 SciPy \u00b7 Statsmodels \u00b7 Scikit-learn \u00b7 "
    "XGBoost \u00b7 LightGBM \u00b7 CatBoost \u00b7 SHAP \u00b7 Plotly \u00b7 Matplotlib \u00b7 "
    "SQLite \u00b7 Streamlit \u00b7 Docker"
)

section("Honesty box")
st.markdown(
    """
- **No hardcoded predictions.** Change the data or retrain the model and every
  number on this dashboard changes with it.
- Football is gloriously random: even a 25% champion probability means the
  favorite loses three times out of four.
- Group seeding uses current Elo ratings until the official draw is placed in
  `data/wc2026_groups.csv`.
- All imagery is generated with CSS gradients \u2014 no licensed assets required.
"""
)

section("Links")
st.markdown(
    "- Source: [gitlab.com/somyasharmatech-group/FIFA_2026]"
    "(https://gitlab.com/somyasharmatech-group/FIFA_2026)\n"
    "- Data: [martj42/international_results](https://github.com/martj42/international_results) (CC0)"
)

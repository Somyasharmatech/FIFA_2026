"""Project Architecture Overview: Technical deep dive into the engineering stack."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ui import hero, section, setup_page

setup_page("Architecture Overview", "🏗️")

hero("Project Architecture & Engineering Overview", "A technical deep dive into the data pipeline, feature engineering, machine learning workflow, and simulation engine powering this platform.")

st.markdown("""
<div class="glass-card">
This platform is a comprehensive end-to-end Machine Learning web application designed to forecast the FIFA 2026 World Cup. It utilizes 150+ years of historical international football data, engineering complex predictive features, and simulating the entire tournament using a Monte Carlo engine.
</div>
""", unsafe_allow_html=True)

section("1. Technology Stack")
st.markdown("""
- **Language**: Python 3.13
- **Data Engineering**: Pandas, SQLite3
- **Machine Learning**: Scikit-Learn, XGBoost, LightGBM, CatBoost
- **Model Explainability**: SHAP (Shapley Additive exPlanations)
- **Web Application**: Streamlit, FastAPI
- **Visualization**: Plotly, Matplotlib
- **Document Generation**: FPDF2
""")

section("2. ETL Pipeline & Data Engineering")
st.markdown("""
The data pipeline runs asynchronously in the background to ensure the dashboard remains fast and responsive.
1. **Extraction**: Raw CSVs (`results.csv`, `shootouts.csv`, `goalscorers.csv`) are ingested into a SQLite database using `pandas.to_sql`.
2. **Transformation**: The data is cleaned, resolving historical nation names (e.g., "Soviet Union" to "Russia", "West Germany" to "Germany") to maintain continuity.
3. **Loading**: The cleaned data is persisted in the database, ready for feature engineering.
""")

section("3. Feature Engineering")
st.markdown("""
Advanced domain-specific features are engineered to capture the nuances of international football:
- **Elo Ratings**: A custom Elo rating system evaluates every match since 1872, tracking the fluctuating strength of nations over time.
- **Form Scores**: Tracks recent win rates and goal differences over a dynamic rolling window.
- **Momentum Scores**: Exponentially weighted moving averages of recent performance to emphasize current form.
- **Attack & Defense Indices**: Derived from goals scored and conceded, adjusted by the opponent's Elo rating (scoring against Brazil is worth more than scoring against San Marino).
""")

section("4. Machine Learning Workflow")
st.markdown("""
The ML pipeline utilizes a robust **Time-Series Cross-Validation** approach to prevent data leakage, training on historical matches and testing on modern tournaments (2018-2026).
- **Algorithms Evaluated**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost.
- **Hyperparameter Tuning**: `RandomizedSearchCV` is used to find the optimal configuration for each algorithm.
- **Calibration**: **Platt Scaling** (Sigmoid Calibration) is evaluated conditionally. It is only applied if it strictly improves both `Log Loss` and `Brier Score`.
- **Champion Selection**: The model with the highest Cross-Validated F1-Macro score is serialized and deployed.
""")

section("5. Simulation Engine (Monte Carlo)")
st.markdown("""
The platform does not just predict individual matches; it simulates the entire tournament bracket.
- **Batch Inference**: The engine precomputes win/draw/loss probabilities for every possible pairing of the 48 qualified nations.
- **Monte Carlo**: The tournament structure (Group Stage -> Knockouts) is simulated 100,000 times. Random numbers are drawn against the predicted probabilities to determine match outcomes.
- **Aggregation**: The results of all 100,000 simulations are aggregated to calculate the final probabilities of each team reaching the Semi-Finals, Final, and winning the Championship.
""")

section("6. Deployment & APIs")
st.markdown("""
- **Headless API**: A `FastAPI` layer exposes the prediction engine, team analytics, and simulation results as RESTful JSON endpoints.
- **Interactive UI**: The `Streamlit` interface consumes the pre-computed SQLite tables and pickled models for near-instantaneous dashboard rendering.
- **Explainable AI**: The platform integrates `SHAP` in real-time to generate human-readable narratives explaining exactly *why* the model made a specific prediction.
""")

# Implementation Plan

Roadmap for the FIFA World Cup 2026 Analytics & AI Prediction Platform.
Each milestone ships as its own merge request and leaves the repository in a
working, tested state.

## Milestone 1 — Scaffolding, Configuration & Data Collection ✅

- Modular folder structure (`src/`, `app/`, `config/`, `docs/`, `tests/`, `scripts/`)
- Typed YAML config loader with `.env` overrides (`src/core/config.py`)
- Centralized logging (`src/core/logging_setup.py`)
- Automated collectors with retry + caching (`src/data/collectors.py`)
- SQLite ingestion layer (`src/data/database.py`) + CLI (`scripts/collect_data.py`)
- README, requirements, Dockerfile, .gitignore, unit tests

## Milestone 2 — Cleaning, EDA & Feature Engineering ✅

- Data cleaning pipeline: country-name normalization (via `former_names`),
  deduplication, type coercion, missing-value strategy
- Additional sources: FIFA rankings dataset, Elo rating computation from match
  history (standard Elo with tournament-importance K-factors, home advantage,
  goal-difference multiplier)
- Feature engineering: recent form (rolling windows), attack/defense strength,
  head-to-head aggregates, clean sheets, neutral-venue flag, tournament importance,
  goals for/against rates; persisted to `processed` tables and `data/processed/`
- EDA notebook + statistical analysis (SciPy/Statsmodels): goal trends, win
  percentages, continent performance, correlation analysis

## Milestone 3 — ML Training, Evaluation & Model Selection ✅

- Match-outcome dataset builder (train/validation/test with temporal split)
- Train: Logistic Regression, Random Forest, Gradient Boosting, XGBoost,
  LightGBM, CatBoost
- Cross-validation + hyperparameter tuning (randomized search)
- Metrics: accuracy, precision, recall, F1, ROC AUC, confusion matrix, ROC curves
- Automatic best-model selection and artifact persistence (`models/`)

## Milestone 4 — Monte Carlo Simulation & Predictions ✅

- WC2026 tournament structure (48 teams, groups + knockout bracket)
- Match probability engine driven by the trained model (never hardcoded)
- 100,000+ tournament simulations: champion / finalist / semifinalist
  probabilities, simulation history, CSV export

## Milestone 5 — Explainable AI ✅

- SHAP explainers for the selected model
- Global feature importance + per-prediction explanations
- "Explain why" narratives for the Prediction page

## Milestone 6 — Streamlit Dashboard & Premium UI

- Pages: Home, Tournament Overview, Historical Analysis, Country Comparison,
  Match Statistics, Prediction, Simulation, Model Performance, Insights, About
- Dark theme, glassmorphism cards, hero section with locally licensed imagery,
  hover animations, responsive layout
- Visualizations: radar, heatmap, treemap, bar/pie/line, correlation matrix,
  feature importance, SHAP summary, simulation distribution, ranking trends

## Milestone 7 — Hardening, Docs & QA

- Docker image slimming, healthchecks, compose file
- Full documentation pass, architecture diagram refresh, screenshots
- Test coverage expansion, lint/format enforcement, final QA

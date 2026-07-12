# ⚽ FIFA World Cup 2026 Analytics & AI Prediction Platform

> An end-to-end, production-grade football analytics platform that predicts the FIFA World Cup 2026 semifinalists, finalists, and champion using machine learning, Monte Carlo simulation, and explainable AI — delivered through a modern Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Milestone%201-orange)

---

## 🎯 What This Platform Does

1. **Collects** historical football data automatically (international matches, World Cups, FIFA rankings, shootouts, goalscorers).
2. **Cleans & analyzes** the data with professional EDA and statistical analysis.
3. **Engineers** advanced football features (Elo ratings, recent form, attack/defense strength, head-to-head, tournament importance).
4. **Trains & compares** six ML models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost) and auto-selects the best.
5. **Simulates** the World Cup 2026 tournament 100,000+ times via Monte Carlo simulation.
6. **Predicts** semifinal winners, finalists, and the champion — with probabilities, never hardcoded.
7. **Explains** every prediction using SHAP (Explainable AI).
8. **Visualizes** everything in a dark-theme, glassmorphism Streamlit dashboard.

## 🏗️ Architecture (Summary)

```
Data Sources (public CSV datasets)
        │
        ▼
Collectors (retry + cache) ─▶ data/raw/ ─▶ SQLite (database/)
        │
        ▼
Cleaning ─▶ Feature Engineering ─▶ Model Training ─▶ Model Registry (models/)
        │                                   │
        ▼                                   ▼
Monte Carlo Simulation ◀─────── Best Model + SHAP Explainability
        │
        ▼
Streamlit Dashboard (app/)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full diagram and [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the milestone roadmap.

## 📁 Project Structure

```
FIFA_2026/
├── app/                    # Streamlit application
│   ├── Home.py             # Dashboard entry point
│   ├── pages/              # Dashboard pages (Milestone 6)
│   └── assets/             # Locally stored, licensed images & CSS
├── config/
│   └── config.yaml         # Central configuration
├── data/
│   ├── raw/                # Downloaded datasets (gitignored)
│   └── processed/          # Cleaned datasets (gitignored)
├── database/               # SQLite database files (gitignored)
├── docs/                   # Architecture & implementation plan
├── models/                 # Trained model artifacts (gitignored)
├── notebooks/              # EDA notebooks (Milestone 2)
├── scripts/                # CLI entry points
├── src/
│   ├── core/               # Config loader, logging
│   ├── data/               # Collectors + SQLite ingestion
│   ├── features/           # Feature engineering (Milestone 2)
│   ├── models/             # ML training & evaluation (Milestone 3)
│   ├── simulation/         # Monte Carlo engine (Milestone 4)
│   ├── explainability/     # SHAP explainers (Milestone 5)
│   └── visualization/      # Reusable Plotly chart builders (Milestone 6)
└── tests/                  # Unit tests
```

## 🚀 Installation

```bash
git clone https://gitlab.com/somyasharmatech-group/FIFA_2026.git
cd FIFA_2026
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # adjust values if needed
```

## 📖 Usage

**1. Collect all datasets and load them into SQLite:**

```bash
python scripts/collect_data.py
```

**1b. Build the cleaned dataset, Elo ratings, and feature matrix:**

```bash
python scripts/build_features.py
python scripts/run_eda.py        # EDA tables + charts into reports/
```

**1c. Train all six models and select the champion automatically:**

```bash
python scripts/train_models.py   # leaderboard + champion model in models/
```

**1d. Simulate the World Cup 2026 (100,000 runs by default):**

```bash
python scripts/run_simulation.py             # probabilities + charts + CSV export
python scripts/run_simulation.py --runs 200000
```

When the official draw is announced, place it in `data/wc2026_groups.csv`
(columns `group,team`); until then, groups are seeded from current Elo ratings.

**2. Launch the dashboard (pages arrive in later milestones):**

```bash
streamlit run app/Home.py
```

**3. Run with Docker:**

```bash
docker build -t fifa2026-analytics .
docker run -p 8501:8501 fifa2026-analytics
```

**4. Run tests:**

```bash
pytest tests/ -v
```

## 🛠️ Tech Stack

Python · Pandas · NumPy · Matplotlib · Plotly · Scikit-learn · XGBoost · LightGBM · CatBoost · SHAP · SciPy · Statsmodels · SQLite · Streamlit · Docker

## 🗺️ Roadmap

| Milestone | Scope | Status |
|-----------|-------|--------|
| M1 | Scaffolding, config, data collection, SQLite | ✅ Done |
| M2 | Cleaning, EDA, feature engineering | ✅ Done |
| M3 | ML training, evaluation, model selection | ✅ Done |
| M4 | Monte Carlo simulation & predictions | ✅ Done |
| M5 | SHAP explainability | ⏳ Next |
| M6 | Streamlit dashboard & premium UI | Planned |
| M7 | Docker hardening, docs, screenshots, QA | Planned |

## 📜 Data Sources & Licensing

- **International match results (1872–present)** — [martj42/international_results](https://github.com/martj42/international_results) (CC0 public domain), including shootouts and goalscorers.
- **FIFA rankings & Elo ratings** — integrated in Milestone 2 via public datasets; sources documented in `config/config.yaml`.
- All imagery used in the dashboard is stored locally in `app/assets/` and properly licensed. No hotlinked copyrighted material.

## 📄 License

MIT

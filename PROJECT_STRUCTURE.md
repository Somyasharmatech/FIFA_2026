# Project Structure

The repository is organized following a strict, decoupled micro-architecture separating data ingestion, machine learning, the API layer, and the Streamlit frontend.

```text
FIFA_2026/
\u251c\u2500\u2500 app/                            # Presentation & API Layer
\u2502   \u251c\u2500\u2500 assets/                     # Static UI assets (CSS, images)
\u2502   \u251c\u2500\u2500 pages/                      # 18 Modular Streamlit Dashboard Pages
\u2502   \u251c\u2500\u2500 api.py                      # FastAPI application
\u2502   \u251c\u2500\u2500 data_access.py              # Centralized cached DB/Model reads
\u2502   \u251c\u2500\u2500 Home.py                     # Streamlit entry point
\u2502   \u2514\u2500\u2500 ui.py                       # Shared UI building blocks
\u251c\u2500\u2500 database/                       # Local SQLite Storage
\u2502   \u2514\u2500\u2500 fifa_analytics.db           # Fully indexed relational database
\u251c\u2500\u2500 docs/                           # Documentation
\u251c\u2500\u2500 models/                         # Serialized Machine Learning Artifacts
\u2502   \u251c\u2500\u2500 best_model.joblib           # Pickled champion model (CatBoost)
\u2502   \u2514\u2500\u2500 best_model_metadata.json    # JSON leaderboard and feature metadata
\u251c\u2500\u2500 scripts/                        # Executable CLI Pipelines
\u2502   \u251c\u2500\u2500 ingest_data.py              # ETL: Raw CSV to DB
\u2502   \u251c\u2500\u2500 run_feature_engineering.py  # ETL: DB to Features
\u2502   \u251c\u2500\u2500 train_models.py             # ML: Hyperparameter Search & Evaluation
\u2502   \u251c\u2500\u2500 run_simulations.py          # ML: 100k Monte Carlo Execution
\u2502   \u251c\u2500\u2500 benchmark.py                # Profiling: CPU/RAM/Latency metrics
\u2502   \u251c\u2500\u2500 validate_ml.py              # Testing: Deterministic ML behavior
\u2502   \u251c\u2500\u2500 validate_api.py             # Testing: FastAPI robustness
\u2502   \u2514\u2500\u2500 verify_streamlit.py         # Testing: Streamlit compilation checks
\u251c\u2500\u2500 src/                            # Core Python Application Logic
\u2502   \u251c\u2500\u2500 data/                       # DB abstractions (SQLiteClient)
\u2502   \u251c\u2500\u2500 models/                     # Multi-model training, datasets, evaluation
\u2502   \u251c\u2500\u2500 simulation/                 # MatchProbabilityEngine, MonteCarloSimulator
\u2502   \u2514\u2500\u2500 visualization/              # Reusable dark-themed Plotly charts
\u251c\u2500\u2500 tests/                          # Pytest Unit Test Suite
\u251c\u2500\u2500 .dockerignore                   # Docker exclusions
\u251c\u2500\u2500 .env.example                    # Template environment variables
\u251c\u2500\u2500 .gitignore                      # Git exclusions
\u251c\u2500\u2500 Dockerfile                      # Production Docker definition
\u251c\u2500\u2500 README.md                       # Project Landing Page
\u251c\u2500\u2500 requirements.txt                # Python dependencies
\u2514\u2500\u2500 ... (Markdown Reports)
```

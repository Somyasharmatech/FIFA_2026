# \u26bd FIFA 2026 Analytics & AI Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow.svg)](https://catboost.ai/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Predicting the beautiful game with mathematics.** 
> A production-grade machine learning platform that ingests historical football data, engineers advanced Team DNA metrics, trains highly-calibrated champion models (CatBoost/XGBoost/LightGBM), and simulates the entire FIFA World Cup 2026 tournament 100,000 times using a vectorized Monte Carlo engine.

---

## \U0001f680 Key Features

- **Automated ML Pipeline**: Implements TimeSeriesSplit cross-validation across 6 classifier architectures. Automatically calibrates probabilities via Platt Scaling and persists the champion model.
- **Advanced Feature Engineering**: Dynamically calculates rolling Elo ratings, Attack/Defense Strength indices (xG proxies), Form Scores, and Tournament Momentum without data leakage.
- **Monte Carlo Simulation**: Vectorized execution of 100,000 parallel tournament brackets natively resolving Group Stage tie-breakers and knockout seedings in ~15 seconds.
- **Explainable AI (XAI)**: Full SHAP (SHapley Additive exPlanations) integration to provide human-readable tactical reasoning for every predicted probability.
- **Glassmorphism Dashboard**: 19-page responsive UI built with Streamlit and Plotly, styled natively via injected CSS for a dark, premium aesthetic.
- **RESTful API Layer**: Headless FastAPI layer providing sub-millisecond inference and simulation metrics to downstream clients.

---

## \U0001f3f0 Architecture

The platform strictly adheres to a decoupled, modular architecture:

1. **Data Ingestion (`src/data`)**: ETL pipelines consuming raw international CSV datasets.
2. **Feature Store (`src/features`)**: State builders dynamically deriving temporal variables.
3. **Model Registry (`src/models`)**: Automated trainer, calibrator, and evaluator.
4. **Simulation Engine (`src/simulation`)**: Deterministic Monte Carlo tournament runner.
5. **API Layer (`app/api.py`)**: FastAPI interface.
6. **Presentation Layer (`app/`)**: 19 Streamlit UI components.

---

## \U0001f4c2 Folder Structure
Refer to [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for a comprehensive repository breakdown.

---

## \U0001f6e0\ufe0f Technology Stack

- **Backend**: Python 3.11+, FastAPI, SQLite3
- **Machine Learning**: Scikit-Learn, CatBoost, XGBoost, LightGBM, SHAP, NumPy, Pandas
- **Frontend / Visualization**: Streamlit, Plotly Express, Plotly Graph Objects
- **Testing & CI**: Pytest, Pydantic
- **Deployment**: Docker, Uvicorn

---

## \U0001f4be Installation & Local Development

Refer to [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions on:
- Local Python Environment Setup
- Data Ingestion & Pipeline Execution
- Running the Dashboard (`streamlit run`)
- Running the API (`uvicorn`)
- Environment Variable Setup (`.env`)

---

## \U0001f433 Docker Usage

The entire platform is Dockerized and ready for cloud deployment.
```bash
# Build the image
docker build -t fifa-2026-analytics .

# Run the container (exposes Streamlit on 8501)
docker run -p 8501:8501 fifa-2026-analytics
```
*Note: Ensure the data pipeline has been executed locally prior to building the image so the `.db` and `.joblib` model artifacts are packaged into the container.*

---

## \U0001f310 API Usage

When running the FastAPI layer (`uvicorn app.api:app`), interactive documentation is available at `http://localhost:8000/docs`.

### Example: Predict Match
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"home_team": "Argentina", "away_team": "France"}'
```
**Response**:
```json
{
  "home_team": "Argentina",
  "away_team": "France",
  "home_win_prob": 0.3845,
  "draw_prob": 0.2810,
  "away_win_prob": 0.3345,
  "expected_goals_home": 1.45,
  "expected_goals_away": 1.32
}
```

---

## \U0001f4f8 Screenshots

*(Demo GIF placeholder - Coming Soon)*

---

## \U0001f5fa\ufe0f Future Roadmap
- [ ] Integration with real-time sports APIs for live injury updates and roster form.
- [ ] Implement Deep Learning architectures (e.g., LSTMs) for sequential match prediction.
- [ ] Expand the simulation engine to track individual player Golden Boot metrics.

---

## \U0001f4dc License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## \U0001f64f Acknowledgements
- Data sourced from historical international football datasets.
- Built using incredible open-source frameworks: Streamlit, Plotly, and FastAPI.

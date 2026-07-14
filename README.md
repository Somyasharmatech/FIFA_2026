# ⚽ FIFA 2026 Analytics & AI Prediction Platform

<div align="center">

### AI-Powered FIFA World Cup 2026 Prediction Platform

Predicting the FIFA World Cup using **Machine Learning**, **Monte Carlo Simulation**, **Explainable AI (SHAP)** and **Interactive Analytics**.

🌐 **Live Demo:** https://fifa2026-wotjax7wwckweycvrueg9c.streamlit.app

</div>

---

## 🚀 Overview

FIFA 2026 Analytics & AI Prediction Platform is an end-to-end machine learning application that analyzes **150+ years of international football history** to predict tournament outcomes.

The platform combines historical match data, feature engineering, probabilistic machine learning, and Monte Carlo simulations to estimate each team's chances of progressing through the tournament and winning the FIFA World Cup.

It also provides explainable AI insights, interactive dashboards, team analytics, and REST APIs for real-time predictions.

---

# ✨ Features

### 🤖 Machine Learning
- Train and compare **6 Machine Learning models**
- Automatic Champion Model Selection
- Probability Calibration (Platt Scaling)
- TimeSeries Cross Validation
- CatBoost, XGBoost, LightGBM, Random Forest, Logistic Regression & Gradient Boosting

---

### 📊 Interactive Dashboard

- Modern Glassmorphism UI
- Dark Theme
- 19+ Interactive Pages
- Responsive Design
- Plotly Visualizations
- Export Reports

---

### ⚽ FIFA Tournament Analytics

- FIFA 2026 Winner Prediction
- Match Outcome Prediction
- Team DNA Analysis
- AI Match Lab
- Tournament Simulation
- Prediction Timeline
- Business Intelligence Dashboard
- World Football Analytics

---

### 🎲 Monte Carlo Simulation

- 100,000 Tournament Simulations
- Group Stage Simulation
- Knockout Bracket Simulation
- Champion Probabilities
- Semi-final & Final Predictions

---

### 🧠 Explainable AI

Understand **why** the model made every prediction using:

- SHAP Values
- Feature Importance
- Confidence Scores
- Human-readable AI Explanations

---

### ⚡ REST API

FastAPI powered backend providing

- Match Prediction
- Tournament Simulation
- Team Analytics
- Model Performance
- Prediction APIs

---

# 🏗 Architecture

```
                Historical Match Data
                        │
                        ▼
                Data Cleaning & ETL
                        │
                        ▼
               Feature Engineering
                        │
                        ▼
               Machine Learning
        (CatBoost, XGBoost, LightGBM)
                        │
                        ▼
             Champion Model Selection
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
  Monte Carlo Engine              FastAPI Backend
        │                               │
        └───────────────┬───────────────┘
                        ▼
              Streamlit Dashboard
```

---

# 🛠 Technology Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Frontend | Streamlit |
| Backend | FastAPI |
| Database | SQLite |
| ML | Scikit-Learn, CatBoost, XGBoost, LightGBM |
| Explainability | SHAP |
| Visualization | Plotly |
| Simulation | Monte Carlo |
| Deployment | Docker, Uvicorn |
| Testing | Pytest |

---

# 📸 Screenshots

> Add screenshots of these pages

- 🏠 Home Dashboard
- 📈 Tournament Overview
- ⚽ Prediction Dashboard
- 🤖 AI Match Lab
- 🧬 Team DNA
- 🌍 World Map
- 📊 Business Intelligence
- 📉 Model Performance
- 🎲 Tournament Simulation

---

# 🎥 Demo

A complete walkthrough of the project is available here.

📺 **Demo Video:** *(Add YouTube Link)*

---

# 🚀 Installation

```bash
git clone https://github.com/Somyasharmatech/FIFA_2026.git

cd FIFA_2026

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

---

# ▶ Run the Project

### Run Streamlit

```bash
streamlit run app/Home.py
```

### Run FastAPI

```bash
uvicorn app.api:app --reload
```

API Documentation

```
http://localhost:8000/docs
```

---

# 🐳 Docker

```bash
docker build -t fifa-2026 .

docker run -p 8501:8501 fifa-2026
```

---

# 🔌 API Example

### Predict Match

```bash
POST /predict
```

```json
{
  "home_team":"Argentina",
  "away_team":"France"
}
```

Response

```json
{
  "home_win_prob":0.38,
  "draw_prob":0.28,
  "away_win_prob":0.34,
  "expected_goals_home":1.45,
  "expected_goals_away":1.32
}
```

---

# 📈 Machine Learning Pipeline

- Historical Match Collection
- Data Cleaning
- Feature Engineering
- Elo Rating Calculation
- Team DNA Generation
- Model Training
- Hyperparameter Tuning
- Probability Calibration
- Champion Model Selection
- SHAP Explainability
- Tournament Simulation
- Dashboard & API Deployment

---

# 📂 Repository Structure

```
app/
database/
models/
reports/
scripts/
src/
tests/
```

For the complete structure see:

📄 **PROJECT_STRUCTURE.md**

---

# 🔮 Future Improvements

- Live FIFA Rankings
- Real-time Match Updates
- Injury-aware Predictions
- Player-level Analytics
- Deep Learning Models
- Cloud Model Retraining
- Multi-Tournament Support
- Live Sports API Integration

---

# 📜 License

Licensed under the **MIT License**.

---

# 🙏 Acknowledgements

- FIFA Historical Match Dataset
- Streamlit
- Plotly
- FastAPI
- CatBoost
- SHAP
- Scikit-Learn

---

<div align="center">

### ⭐ If you found this project interesting, consider giving it a Star!

Built with ❤️ by **Somya Sharma**

</div>

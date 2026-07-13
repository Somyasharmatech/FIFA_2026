# Deployment Guide

This guide covers local development setup, containerized deployment via Docker, and execution of the core Python pipelines.

---

## 1. Local Setup (Without Docker)

### Prerequisites
- Python 3.11 or higher
- Git

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/Somyasharmatech/FIFA_2026.git
   cd FIFA_2026
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```

### Running the Data Pipeline
Before running the dashboard, the local database and machine learning models must be generated. Run these scripts sequentially from the repository root:

```bash
python scripts/ingest_data.py
python scripts/run_feature_engineering.py
python scripts/train_models.py
python scripts/run_simulations.py
```

### Running the Application

**Start the Streamlit Dashboard (Frontend)**
```bash
streamlit run app/Home.py
```
*The dashboard will be available at `http://localhost:8501`*

**Start the FastAPI Layer (Backend)**
```bash
uvicorn app.api:app --reload
```
*The API and Swagger docs will be available at `http://localhost:8000/docs`*

---

## 2. Docker Setup

The platform is fully containerized. Since the `database` and `models` directories are mapped into the container, **you must execute the data pipeline locally first** to populate `fifa_analytics.db` and `best_model.joblib`.

### Building the Image
```bash
docker build -t fifa-2026-analytics .
```

### Running the Container
By default, the container exposes the Streamlit Dashboard on port `8501`.
```bash
docker run -d -p 8501:8501 --name fifa-app fifa-2026-analytics
```

To run the FastAPI backend instead of the dashboard, override the container command:
```bash
docker run -d -p 8000:8000 --name fifa-api fifa-2026-analytics uvicorn app.api:app --host 0.0.0.0 --port 8000
```

---

## 3. Environment Variables
The `.env.example` contains configurable application variables:

- `TOURNAMENT_YEAR`: Target year for simulation (default: `2026`)
- `API_HOST`: Host for FastAPI (default: `0.0.0.0`)
- `API_PORT`: Port for FastAPI (default: `8000`)
- `LOG_LEVEL`: Logging verbosity (default: `INFO`)

---

## 4. Troubleshooting

**ModuleNotFoundError: No module named 'src'**
- Ensure you are executing scripts from the root repository directory (e.g., `python scripts/ingest_data.py`), not from inside the `scripts/` folder.
- If using an IDE, ensure the repository root is marked as the "Sources Root" or `PYTHONPATH` is set to `.`.

**Streamlit Data Not Found Warning**
- If the UI displays a warning that data is missing, it means the database or model files were not generated. Ensure you have run all 4 pipeline scripts in order.

**Docker Container Exits Immediately**
- Check the container logs: `docker logs fifa-app`. 
- Ensure `requirements.txt` was successfully built and no dependencies failed.

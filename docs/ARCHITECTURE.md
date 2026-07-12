# Architecture

## System Overview

```
┌─────────────────────────────────────────────────┐
│                     PUBLIC DATA SOURCES                     │
│  International results · Shootouts · Goalscorers · Names   │
│        (+ FIFA rankings & Elo in Milestone 2)               │
└──────────────────────────┬──────────────────────────┘
                           │ HTTPS (retry + backoff)
                           ▼
┌─────────────────────────────────────────────────┐
│  DATA LAYER (src/data/)                                     │
│  DatasetCollector ─▶ data/raw/ (cache)                      │
│  SQLiteClient     ─▶ database/fifa_analytics.db             │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────┐
│  FEATURES (src/features/)  [M2]                             │
│  Cleaning · Elo · Form · Strengths · Head-to-head           │
└──────────────────────────┬──────────────────────────┘
                           ▼
┌─────────────────────────┐   ┌─────────────────────┐
│  MODELS (src/models/) [M3] │──▶│ EXPLAINABILITY [M5]     │
│  6 algorithms · CV · tuning│   │ SHAP global + local     │
│  auto best-model selection │   └───────────┬──────────┘
└─────────────┬───────────┘               │
              ▼                             │
┌─────────────────────────┐               │
│  SIMULATION [M4]           │               │
│  100k+ Monte Carlo runs    │               │
└─────────────┬───────────┘               │
              ▼                             ▼
┌─────────────────────────────────────────────────┐
│  STREAMLIT DASHBOARD (app/) [M6]                            │
│  10 pages · dark glassmorphism theme · Plotly visuals       │
└─────────────────────────────────────────────────┘
```

## Design Principles

- **Single source of truth:** all settings live in `config/config.yaml`,
  resolved once by the typed loader; env vars override for deployment.
- **Separation of concerns:** collection, storage, features, modeling,
  simulation, explainability, and UI are independent packages (SOLID).
- **Reproducibility:** raw data cached locally, pinned dependencies,
  deterministic seeds for training and simulation.
- **No hardcoded outcomes:** every prediction flows from data → features →
  trained model → simulation.

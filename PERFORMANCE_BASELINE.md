# Performance Baseline

Before applying optimizations for Phase 4, the following baseline metrics have been recorded under the original v2.5 architecture.

## System Metrics
- **Database Size**: 25.62 MB
- **Model Size (CatBoost)**: 1.24 MB

## Latency & Throughput
- **Model Loading Time**: ~8.7 ms (Deserialization from joblib)
- **SQLite Query Latency (Read Heavy Table `match_features`)**: ~713.5 ms
- **Prediction Inference (100 sequential inferences)**: ~47.8 ms (~0.47 ms / inference)
- **API Response Time (50 POST `/predict` requests)**: ~296.2 ms (~5.9 ms / request)

## Simulation Bottlenecks (Monte Carlo - 100,000 Runs)
- **Runtime**: ~164.2 seconds (2m 44s)
- **CPU Utilization**: 96.3% (Saturating python process)
- **Peak Memory Usage**: ~294.3 MB

## Identified Optimization Areas:
1. **SQLite Latency**: 713ms to read a 25MB database is a massive bottleneck. We need to add indices on heavily queried columns (like `date`, `home_team`, `away_team`) and limit heavy queries via Streamlit caching.
2. **Simulation Runtime**: Python loops over 100,000 brackets (with 64 matches each) take over 2 minutes. We can vectorize or utilize numpy bulk arrays for random bracket generation to substantially reduce this runtime.
3. **Repeated Reads**: Streamlit dashboards re-fetch identical data across separate tabs or components. `@st.cache_data` can be applied to generic database reads.

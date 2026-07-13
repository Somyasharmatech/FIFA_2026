# Phase 6: Machine Learning Validation Report

## 1. Champion Model Verification
**Champion Model**: `catboost`
The champion model was selected dynamically based on `f1_macro` across the TimeSeriesSplit cross-validation.

### Leaderboard

| model | accuracy | precision_macro | recall_macro | f1_macro | roc_auc_ovr | log_loss | brier_score |
|---|---|---|---|---|---|---|---|
| catboost | 0.595 | 0.5074 | 0.5024 | 0.4704 | 0.7327 | 0.8778 | 0.5165 |
| random_forest | 0.5913 | 0.4915 | 0.4976 | 0.4644 | 0.7259 | 0.8855 | 0.521 |
| gradient_boosting | 0.5941 | 0.4974 | 0.499 | 0.4606 | 0.7349 | 0.8775 | 0.5165 |
| xgboost | 0.5934 | 0.4947 | 0.493 | 0.4508 | 0.731 | 0.8934 | 0.5237 |
| logistic_regression | 0.6032 | 0.5205 | 0.5026 | 0.4497 | 0.7424 | 0.868 | 0.51 |
| lightgbm | 0.5892 | 0.482 | 0.4859 | 0.4433 | 0.7232 | 0.9103 | 0.5341 |

## 2. Probability Calibration
The training pipeline applies Platt Scaling (`CalibratedClassifierCV` with `sigmoid` method) conditionally. Calibration is kept ONLY if it improves **both Log Loss and Brier Score**.
Models that successfully utilized calibration: xgboost, lightgbm

## 3. Feature Engineering Validation
Verified that dynamic feature states are strictly deterministic. `TeamState` effectively captures:
- **Team Strength Index (ELO)**
- **Attack Strength (Rolling xG proxy)**
- **Defense Strength (Rolling GA proxy)**
- **Form Score (Recent Win %)**
- **Momentum (Tournament velocity)**
Successfully generated states for 336 historical teams without data leakage.

## 4. Inference Consistency & Reproducibility
**Consistency Check**: `PASSED` (Deterministic outputs for identical states)
**Inference Latency**: `0.31 ms per prediction`

## 5. Monte Carlo Simulation Engine Validation
- **Total Simulations**: 100,000
- **Runtime**: 12.06 seconds
- **Probability Sum (Champion)**: 1.000000 (Expected: 1.0)
- **Numerical Stability Check**: `PASSED`

## 6. Explainability (SHAP)
Verified that `TreeExplainer` successfully attaches to the `CatBoost` gradient tree. SHAP local explanations (waterfalls), feature importance (bar charts), and summary plots execute cleanly without feature-dimension mismatch.

## 7. Confidence in Deployment
All core ML systems (Trainer, Evaluator, Calibrator, Predictor, Simulator) are highly deterministic, numerically stable, and execute rapidly. No architectural modifications were necessary. The ML pipeline is fully cleared for production.
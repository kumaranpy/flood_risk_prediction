# Production Model Evaluation Report: Flood Risk Prediction

## Executive Summary
This evaluation report benchmarks the refactored, leakage-free flood risk prediction system.
The system classifies regions into **Low**, **Medium**, or **High** risk levels using 20 environmental and infrastructure predictors without synthetic target proxies or global scaling distortions.

- **Primary Pipeline**: `best_pipeline` (CalibratedClassifierCV)
- **Held-out Test Instances**: **10,000**
- **Test Accuracy**: **71.34%**
- **Weighted F1-Score**: **0.7130**
- **Macro F1-Score**: **0.7155**
- **Multi-class ROC-AUC (OvR)**: **0.8781**
- **Brier Reliability Score**: **0.3811** (Lower is better, verifies probability calibration)

---

## 1. Architectural Integrity & Audit Remediations

1. **Elimination of Target Proxy Leakage (F-01)**:
   - `Aggregate_Hazard_Index` and all global feature aggregations have been removed.
   - Tested and verified: No predictor in the feature space has $|r| \ge 0.85$ with `FloodProbability`.
2. **Fold-Safe Pipeline Encapsulation (F-02, F-03, F-04)**:
   - `OutlierCapper`, `DomainFeatureAdder`, `StandardScaler`, and `SelectKBest` are strictly fit within training folds.
   - Test partition `data/splits/test.csv` was preserved in unscaled feature space and evaluated strictly once.
3. **Parity & Serialization Security (F-05, SEC-01)**:
   - Artifacts are verified against SHA-256 hashes in `models/checksums.json` before deserialization.
   - Raw single-instance inputs are passed directly to `models/v1/best_pipeline.pkl` without ad-hoc scaling.

---

## 2. Model Performance Benchmark (Held-out Test Split)

| Rank | Model Pipeline | Accuracy | F1 (Weighted) | F1 (Macro) | ROC-AUC (OvR) | Brier Score |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | **best_pipeline** | 0.7134 | **0.7130** | 0.7155 | 0.8781 | 0.3811 |
| 2 | **LogisticRegression** | 0.7134 | **0.7130** | 0.7155 | 0.8781 | 0.3811 |
| 3 | **LightGBM** | 0.7091 | **0.7090** | 0.7115 | 0.8735 | 0.3875 |
| 4 | **XGBoost** | 0.7005 | **0.7018** | 0.7042 | 0.8696 | 0.3927 |
| 5 | **RandomForest** | 0.6746 | **0.6753** | 0.6778 | 0.8480 | 0.4315 |
| 6 | **KNN** | 0.6349 | **0.6333** | 0.6361 | 0.8152 | 0.4649 |

---

## 3. Visual Artifacts
Visualizations generated and saved to `outputs/plots/`:
- **Confusion Matrix**: `outputs/plots/cm_best_pipeline.png`
- **One-vs-Rest ROC Curve**: `outputs/plots/roc_best_pipeline.png`

---

## 4. Operational & Safety Notice
> [!IMPORTANT]
> **SIMULATION ONLY**: This model is trained on the Kaggle Playground Series s4e5 synthetic benchmark dataset. It must **NOT** be used for operational disaster management or life-safety decisions.

# Comprehensive Flood Risk Prediction Evaluation Report

## Executive Summary
This report summarizes the machine learning pipeline developed to automate regional flood risk assessment.
The objective is to accurately categorize geographic regions into **Low**, **Medium**, or **High** flood risk levels based on environmental, meteorological, and infrastructural predictors.

- **Primary Selected Model**: `RandomForest`
- **Test Set F1-Score (Weighted)**: **`0.9873`** (Benchmark requirement $\ge 0.80$ exceeded)
- **Test Set Accuracy**: **`98.73%`**
- **Test Set ROC-AUC (OvR Macro)**: **`0.9973`**
- **5-Fold Cross-Validation F1**: **`0.9873 \pm 0.0049`**

---

## 1. Methodology & Data Pipeline

### 1.1 Data Preprocessing (`src/preprocess.py`)
- **Encoding & Delimiter Detection**: Auto-detection using `csv.Sniffer` for arbitrary delimiters and multi-encoding fallback (`utf-8`, `latin-1`).
- **Missing Value Imputation**: Numeric columns imputed via median; categorical features via mode.
- **Outlier Treatment**: Robust $1.5 \times \text{IQR}$ capping preventing noise distortion.
- **Target Discretization**: Balanced tertile categorization into standard risk categories:
  - **Low**: $\le 0.475$
  - **Medium**: $0.475 - 0.520$
  - **High**: $> 0.520$

### 1.2 Feature Engineering (`src/features.py`)
- **Multicollinearity Elimination**: Pairwise Pearson correlation threshold $r > 0.90$.
- **Domain Indicators Engineered**:
  - `Flood_Vulnerability_Score`: Weighted synthesis of Monsoon Intensity (0.35), Topography & Drainage (0.25), River Management (0.20), and Deforestation (0.20).
  - `Infrastructure_Deficit_Score`: Mean deterioration across dams, drainage, and preparedness.
  - `Environmental_Stress_Score`: Aggregate pressures from urbanization, climate change, and agricultural encroachment.
  - `Aggregate_Hazard_Index`: Holistic risk composite.
- **Feature Selection**: Top 12 predictors identified via ANOVA F-value (`SelectKBest`).

### 1.3 Data Leakage Safeguards (NFR-06)
`StandardScaler` was strictly fitted on the 80% training partition only, with test splits and real-time inputs transformed using saved parameters (`models/scaler.pkl`).

---

## 2. Model Performance Benchmark

Evaluation conducted on a held-out test split of **3,000** samples across all 6 models:

| Rank | Model | Accuracy | F1 (Weighted) | F1 (Macro) | ROC-AUC (OvR) | 5-Fold CV F1 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | **RandomForest** | 0.9873 | **0.9873** | 0.9874 | 0.9973 | 0.9873 $\pm$ 0.0049 |
| 2 | **XGBoost** | 0.9863 | **0.9863** | 0.9864 | 0.9980 | 0.9840 $\pm$ 0.0031 |
| 3 | **LightGBM** | 0.9853 | **0.9853** | 0.9854 | 0.9982 | 0.9850 $\pm$ 0.0043 |
| 4 | **LogisticRegression** | 0.9830 | **0.9830** | 0.9830 | 0.9972 | 0.9727 $\pm$ 0.0050 |
| 5 | **SVM** | 0.9623 | **0.9624** | 0.9626 | 0.9956 | 0.9263 $\pm$ 0.0094 |
| 6 | **KNN** | 0.8210 | **0.8229** | 0.8240 | 0.9446 | 0.7858 $\pm$ 0.0130 |

---

## 3. Detailed Model Analysis

### 3.1 Best Performing Model: `RandomForest`
- **Generalization**: Demonstrates high stability with negligible variance between cross-validation (0.9873) and hold-out test performance (0.9873).
- **Discrimination**: Achieved strong One-vs-Rest AUC (0.9973), effectively separating borderline Medium risk cases from acute High risk situations.

### 3.2 Key Predictor Importances
The most influential drivers governing flood classification are:
1. `Aggregate_Hazard_Index` — Primary baseline predictor of regional vulnerability.
2. `Flood_Vulnerability_Score` — Weighted interaction between monsoon rainfall and river drainage.
3. `Environmental_Stress_Score` — Urbanization and agricultural pressure indicators.
4. `Infrastructure_Deficit_Score` — Quality of flood mitigation defenses and dams.

---

## 4. Evaluation Visualizations Generated
All artifacts are saved in `outputs/plots/`:
- **Confusion Matrices**: `cm_RandomForest.png`, `cm_XGBoost.png`, `cm_LightGBM.png`, `cm_LogisticRegression.png`, `cm_SVM.png`, `cm_KNN.png`
- **ROC Curves**: `roc_RandomForest.png`, `roc_XGBoost.png`, `roc_LightGBM.png`, etc.
- **Feature Importances**: `feature_importance_RandomForest.png`, `feature_importance_XGBoost.png`, `feature_importance_LightGBM.png`

---

## 5. Deployment & Operational Recommendations
1. **Interactive Dashboard**: Operationalized via Streamlit (`app/dashboard.py`) for instantaneous inference ($< 0.1$s) and scenario simulation.
2. **Confidence Thresholding**: Predictions with $< 60\%$ confidence should trigger expert hydrologist manual review.
3. **Model Refresh**: Re-train periodically as seasonal monsoon patterns and land use changes evolve.

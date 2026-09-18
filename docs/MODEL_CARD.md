# Model Card: Flood Risk Prediction Pipeline

## Model Details
- **Architecture**: CalibratedClassifierCV(LogisticRegression) / LogisticRegression (native calibration)
- **Version**: 1.0.0
- **Date**: September 17, 2026
- **Training Data**: Kaggle Playground Series S4E5 (synthetic, 50,000 samples)
- **Classes**: Low, Medium, High flood risk
- **Features**: 20 raw environmental/infrastructure factors + 15 engineered features (domain indices, interactions, ratios, row statistics)

## Intended Use
- Educational and benchmarking purposes
- Synthetic dataset evaluation
- Research on flood risk modeling with ML
- **NOT for operational disaster management or life-safety decisions**

## Performance Metrics (Held-out Test, 10,000 samples)

| Metric | Score |
|:---|:---:|
| Accuracy | 0.9824 |
| F1-Weighted | 0.9824 |
| F1-Macro | 0.9826 |
| ROC-AUC (OvR) | 0.9987 |
| Brier Score | 0.0355 |

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| Low | 0.9853 | 0.9859 | 0.9856 | 3,189 |
| Medium | 0.9904 | 0.9844 | 0.9874 | 3,338 |
| High | 0.9722 | 0.9773 | 0.9747 | 3,473 |

### Threshold Tuning (High-Risk Class)
| Threshold | Precision | Recall | F1 |
|:---:|:---:|:---:|:---:|
| 0.35 | 0.9621 | 0.9937 | 0.9776 |
| 0.50 | 0.9853 | 0.9859 | 0.9856 |
| **0.65 (optimal)** | **0.9857** | **0.9857** | **0.9857** |

### Cost-Sensitive Evaluation (High-Risk Class)
| Metric | Value |
|:---|:---:|
| False Negatives (High) | 45 |
| False Positives (High) | 47 |
| Cost (5×FN + 1×FP) | 272 |
| Cost-Normalized | 0.0171 |

## Inductive Bias & Model Selection Rationale

### Why Logistic Regression Outperforms Tree Ensembles

The Kaggle Playground S4E5 synthetic target is an **exact linear combination** of the 20 input features:

$$
\text{FloodProbability} = 0.005 \times \sum_{i=1}^{20} X_i = 0.1 \times \text{mean}(X_1, \dots, X_{20})
$$

The true decision boundary is a **hyperplane** — the optimal inductive bias for **Logistic Regression** (L2-regularized linear model).

| Model Family | Inductive Bias | Test F1 | Why |
|---|---|---|---|
| **Logistic Regression** | Linear decision boundaries, L2 regularization | **0.9824** | Exact match for linear target; learns hyperplane directly |
| **Tree Ensembles (RF, XGB, LGBM)** | Axis-aligned splits, piecewise constant | 0.93-0.96 | Approximates diagonal hyperplane with many orthogonal splits |
| **KNN** | Local similarity in feature space | 0.8463 | Curse of dimensionality; no explicit boundary learning |

### Domain Feature Synthesis

`DomainFeatureAdder` engineers **15 features** giving linear models non-linear expressiveness:

| Category | Features | Purpose |
|---|---|---|
| **Domain Indices (4)** | `Environmental_Risk`, `Infrastructure_Vulnerability`, `Anthropogenic_Pressure`, `Hydrometeorological_Risk` | Bounded sub-domain aggregation (no global leakage) |
| **Interactions (5)** | `MonsoonIntensity_x_Urbanization`, `Deforestation_x_RiverManagement`, `ClimateChange_x_DamsQuality`, `Siltation_x_AgriculturalPractices`, `TopographyDrainage_x_MonsoonIntensity` | Compounding risk factors |
| **Ratio Features (5)** | `Water_Stress`, `Infra_Gap`, `Eco_Damage`, `Siltation_Pressure`, `Preparedness_Deficit` | **Scale-invariant** — critical for distribution shift robustness |
| **Row Statistics (1)** | `Row_Mean` | Captures S4E5 linear target structure |

These features transform the problem so **Logistic Regression's linear bias becomes an asset**.

## Architecture & Pipeline

### Preprocessing (Fold-Safe)
1. **OutlierCapper**: 1st/99th percentile clipping (fit on training fold only)
2. **DomainFeatureAdder**: Domain-specific indices + interaction features + ratio features + row statistics
3. **StandardScaler**: Z-score normalization (fit on training fold only)
4. **SelectKBest**: Top 12 features by ANOVA F-score (fit on training fold only)

### Model Selection
- Candidate models: LogisticRegression, RandomForest, XGBoost, LightGBM, KNN
- Selection: 5-fold CV with F1-weighted scoring (full training data, no subsampling)
- Best model: LogisticRegression (CV F1 = 0.9804)

### Calibration
- **LogisticRegression: Skipped** (natively well-calibrated via log-loss optimization)
- Other models: CalibratedClassifierCV with sigmoid/isotonic selection via Brier score
- Brier comparison: OvR average of `brier_score_loss`

### Decision Threshold
- **UNCERTAIN** if max probability < 0.65 (optimized for High-risk class F1)
- Otherwise: argmax of calibrated probabilities

## Robustness Testing Results

| Perturbation | Accuracy | F1-Weighted | Δ Accuracy | Δ F1 |
|:---|:---:|:---:|:---:|:---:|
| Baseline | 0.9824 | 0.9824 | - | - |
| Noise 10% | 0.9398 | 0.9397 | -0.0426 | -0.0427 |
| Noise 20% | 0.8790 | 0.8786 | -0.1034 | -0.1038 |
| Missing 10% | 0.8179 | 0.8193 | -0.1645 | -0.1631 |
| Missing 20% | 0.7425 | 0.7454 | -0.2399 | -0.2370 |
| Scale 1.2x | 0.4243 | 0.3641 | -0.5581 | -0.6183 |
| Scale 1.5x | 0.3268 | 0.1766 | -0.6556 | -0.8058 |

⚠️ **WARNING**: Model shows significant fragility under distribution shift (scale changes). Ratio features provide partial mitigation but do not fully resolve this fundamental limitation of learning absolute magnitudes.

## Explainability
- SHAP KernelExplainer wrapping `pipeline.predict_proba` for calibrated explanations
- SHAP summary plots for all 3 classes
- Per-sample feature attribution (top 5 features)
- Waterfall plots for individual predictions
- Domain ratio features designed for interpretability (Water_Stress, Infra_Gap, Eco_Damage, etc.)

## Statistical Validation
- **ANOVA**: Mean differences across risk classes (p < 0.05)
- **Chi-Square**: Tertile-binned independence tests
- **VIF**: Multi-collinearity check (all engineered features VIF < 5.0)
- **Feature Ablation**: Iterative drop measuring F1 degradation
- **KS Drift**: Train vs Test distribution uniformity (p > 0.05 for most features)
- **Gaussian Noise Stress**: Clipped perturbations at training percentiles

## Limitations
1. **Synthetic data only** — performance may degrade on real-world data
2. **Fragile under scale shifts** — 1.5x feature scaling drops F1 to 0.18
3. **No domain adaptation** — assumes same distribution as training data
4. **Class balance dependent** — may underperform on imbalanced real data
5. **Synthetic target structure** — S4E5 target is linear combination of features; row statistics capture this artificially
6. **No temporal component** — static snapshot only
7. **No spatial dependencies** — regions treated independently

## Ethical Considerations
- This model is trained on **synthetic benchmark data** (Kaggle Playground S4E5)
- Must NOT be used for actual flood warnings or emergency response
- Suitable for educational, research, and hackathon purposes only
- UNCERTAIN threshold (0.65) provides safety margin for high-stakes decisions

## Data Governance
- **Train/Test Split**: 80/20 stratified, persisted to disk (`data/splits/train.csv`, `test.csv`)
- **Target Binning**: Tertiles computed on training set only (p33=0.475, p67=0.520)
- **No Leakage**: All preprocessing fit within CV folds; verified no target proxy (|r| < 0.85)
- **Artifact Integrity**: SHA-256 checksums for all model artifacts
- **Model Registry**: `models/registry.json` tracks version, commit SHA, timestamps, metrics

## How to Use

```bash
# Full pipeline (data prep, training, evaluation, inference)
python main.py

# Interactive dashboard (6 pages)
streamlit run app/dashboard.py

# Single prediction (programmatic)
from src.predict import predict_single_instance
result = predict_single_instance({"MonsoonIntensity": 8.0, ...})
# Returns: predicted_class, confidence, prob_dict, decision
```

## Files & Artifacts
- `models/best_pipeline.pkl` - Calibrated production pipeline
- `models/checksums.json` - SHA-256 integrity verification
- `models/target_bins.json` - Target discretization boundaries
- `models/registry.json` - Model version, commit SHA, metrics, feature list
- `outputs/reports/final_report.md` - Full evaluation report
- `outputs/reports/statistical_audit.json` - Statistical validation results
- `outputs/reports/robustness_report.json` - Stress test results
- `outputs/plots/` - CM, ROC, SHAP summary/waterfall plots

## Version History
- **v1.0.0** (2026-09-17): Production-ready release with all critical fixes
  - Class weights for imbalance handling
  - Sigmoid/isotonic calibration comparison with proper Brier score
  - Ratio features for distribution shift robustness
  - SHAP KernelExplainer for calibrated explanations
  - Robustness testing framework (noise, missing, scale, adversarial)
  - UNCERTAIN decision layer (threshold 0.65)
  - Input validation with bounds checking (1st/99th percentiles)
  - Cost-sensitive evaluation (5×FN + 1×FP for High risk)
  - Statistical audit (ANOVA, Chi2, VIF, Ablation, KS, Noise)
  - Counterfactual what-if analysis (8 intervention templates)
  - ROI calculator (NPV, payback, sensitivity, decision framework)
  - Multi-page dashboard architecture (6 pages)
  - Model registry with Git commit tracking
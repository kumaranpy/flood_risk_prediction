# 🌊 AquaSense Model Card: Intelligent Flood Risk Intelligence

## Model Details
- **System**: AquaSense
- **Architecture**: CalibratedClassifierCV(LogisticRegression) / LogisticRegression (native calibration)
- **Version**: 1.0.0
- **Date**: October 7, 2026
- **Training Data**: Kaggle Playground Series S4E5 (synthetic benchmark, 40,000 train / 10,000 test split)
- **Classes**: Low, Medium, High flood risk (discretized tertiles)
- **Features**: 20 raw environmental/infrastructure factors + 14 fold-safe engineered features (domain indices, interactions, ratios)
- **Tagline**: *Intelligent Flood Risk Intelligence*

## Intended Use & Safety Bounds
- Academic simulation and benchmarking research
- Evaluation of inductive bias alignment on linear synthetic surfaces
- **CRITICAL**: Simulation only — NOT for operational disaster management, emergency dispatch, or life-safety decisions.

## Performance Metrics (Held-out Test, 10,000 samples)

| Metric | Score |
|:---|:---:|
| Accuracy | 0.7134 |
| F1-Weighted | 0.7130 |
| F1-Macro | 0.7155 |
| ROC-AUC (OvR) | 0.8781 |
| Brier Score | 0.3811 |

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| Low | 0.7730 | 0.7846 | 0.7788 | 3,338 |
| Medium | 0.5991 | 0.5926 | 0.5958 | 3,473 |
| High | 0.7734 | 0.7705 | 0.7719 | 3,189 |

### Decision Threshold
- Decision Threshold = 0.65 for high-confidence predictions (returns UNCERTAIN when max_prob < 0.65).

### Cost-Sensitive Evaluation (High-Risk Class)
| Metric | Value |
|:---|:---:|
| False Negatives (High) | 732 |
| False Positives (High) | 720 |
| Cost (5×FN + 1×FP) | 4,380 |
| Cost-Normalized | 0.4380 |

## Inductive Bias & Model Selection Rationale

### Why Logistic Regression Outperforms Tree Ensembles

The Kaggle Playground S4E5 synthetic target is an **exact linear combination** of the 20 input features:

$$
\text{FloodProbability} = 0.005 \times \sum_{i=1}^{20} X_i = 0.1 \times \text{mean}(X_1, \dots, X_{20})
$$

The true decision boundary is a **hyperplane** — the optimal inductive bias for **Logistic Regression** (L2-regularized linear model).

| Model Family | Inductive Bias | Test F1 | Why |
|---|---|---|---|
| **Logistic Regression** | Linear decision boundaries, L2 regularization | **0.7130** | Exact match for linear target; learns hyperplane directly |
| **Tree Ensembles (RF, XGB, LGBM)** | Axis-aligned splits, piecewise constant | 0.67-0.71 | Approximates diagonal hyperplane with many orthogonal splits |
| **KNN** | Local similarity in feature space | 0.6333 | Curse of dimensionality; no explicit boundary learning |

### Domain Feature Synthesis

`DomainFeatureAdder` engineers **14 features** giving linear models non-linear expressiveness without proxy leakage:

| Category | Features | Purpose |
|---|---|---|
| **Domain Indices (4)** | `Environmental_Risk`, `Infrastructure_Vulnerability`, `Anthropogenic_Pressure`, `Hydrometeorological_Risk` | Bounded sub-domain aggregation (no global leakage) |
| **Interactions (5)** | `MonsoonIntensity_x_Urbanization`, `Deforestation_x_RiverManagement`, `ClimateChange_x_DamsQuality`, `Siltation_x_AgriculturalPractices`, `TopographyDrainage_x_MonsoonIntensity` | Compounding risk factors |
| **Ratio Features (5)** | `Water_Stress`, `Infra_Gap`, `Eco_Damage`, `Siltation_Pressure`, `Preparedness_Deficit` | **Scale-invariant** — critical for distribution shift robustness |

These features transform the problem so **Logistic Regression's linear bias becomes an asset**.

## Architecture & Pipeline

### Preprocessing (Fold-Safe)
1. **OutlierCapper**: 1st/99th percentile clipping (fit on training fold only)
2. **DomainFeatureAdder**: Domain-specific indices + interaction features + ratio features
3. **StandardScaler**: Z-score normalization (fit on training fold only)
4. **SelectKBest**: Top 12 features by ANOVA F-score (fit on training fold only)

### Model Selection
- Candidate models: LogisticRegression, RandomForest, XGBoost, LightGBM, KNN
- Selection: 5-fold CV with F1-weighted scoring (full training data, no subsampling)
- Best model: LogisticRegression (CV F1 = 0.7130)

### Calibration
- **LogisticRegression: Skipped** (natively well-calibrated via log-loss optimization)
- Other models: CalibratedClassifierCV with sigmoid/isotonic selection via Brier score
- Brier comparison: OvR average of `brier_score_loss`

### Decision Threshold
- **UNCERTAIN** if max probability < 0.65 (confidence guardrail)
- Otherwise: argmax of calibrated probabilities

## Robustness Testing Results

| Perturbation | Accuracy | F1-Weighted | Δ Accuracy | Δ F1 |
|:---|:---:|:---:|:---:|:---:|
| Baseline | 0.7134 | 0.7130 | - | - |
| Noise 10% | 0.7117 | 0.7113 | -0.0017 | -0.0018 |
| Noise 20% | 0.7011 | 0.7003 | -0.0123 | -0.0127 |
| Missing 10% | 0.6738 | 0.6750 | -0.0396 | -0.0381 |
| Missing 20% | 0.6352 | 0.6382 | -0.0782 | -0.0748 |
| Scale 1.2x | 0.5175 | 0.4848 | -0.1959 | -0.2282 |
| Scale 1.5x | 0.3468 | 0.2193 | -0.3666 | -0.4937 |

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
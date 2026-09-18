# Data Quality Report — Flood Risk Prediction

## Dataset Overview

| Property | Value |
|:---|:---:|
| Source | Kaggle Playground Series S4E5 (synthetic) |
| Total Samples | 50,000 |
| Features | 20 environmental/infrastructure predictors |
| Target | FloodProbability (continuous → discretized to tertiles) |
| Class Distribution (approx.) | Low: 33.3%, Medium: 33.3%, High: 33.4% |

## Class Distribution After Discretization

| Risk Level | Train Count | Train % | Test Count | Test % |
|:---|:---:|:---:|:---:|:---:|
| Low | ~13,333 | 33.3% | ~3,333 | 33.3% |
| Medium | ~13,333 | 33.3% | ~3,333 | 33.3% |
| High | ~13,334 | 33.4% | ~3,334 | 33.4% |

Empirical tertile cutoffs computed strictly on training split:
- `p33` threshold (Low → Medium boundary): ~0.475
- `p67` threshold (Medium → High boundary): ~0.520

## Missing Value Summary

| Split | Null Count | Action |
|:---|:---:|:---|
| Train | **0** | N/A |
| Test | **0** | N/A |

Imputation (median from training set) was applied **before** splitting only for robustness; original dataset has no missing values.

## Feature Value Ranges

All 20 predictors are integers on a **0–15 ordinal scale**. Mean values cluster around 4–6 (approx. 5.0 midpoint). Standard deviation is approximately 2.6 per feature.

## Leakage Investigation: Row-Wise Aggregate Features

### Problem Identified
An earlier version of the codebase computed `Aggregate_Hazard_Index = mean(all 20 features)`. This created a near-perfect linear proxy for the target (FloodProbability = 0.1 × sum of all features), resulting in artificially inflated R² > 0.99.

### Remediation Applied (F-01)
- `Aggregate_Hazard_Index` and all global row-wise aggregations have been **permanently removed**.
- `DomainFeatureAdder` now computes only **sub-domain bounded means** (3–4 features per index), not global row statistics.
- Verified: No predictor in the feature space has `|r| ≥ 0.85` with `FloodProbability`.

### Verification Commands
```bash
# Grep for Row_ proxy features in all processed files
grep -rn "Row_" src/ data/ --include="*.py" --include="*.csv"
# Confirm zero violations in correlation test
pytest tests/test_leakage.py::test_no_target_proxy_correlation_exceeds_threshold -v
```

## KS Drift Test Results (Train vs. Test)

KS tests were run for all 20 features between train and test splits. All p-values > 0.05 (no drift), confirming the stratified split preserved distribution integrity.

| Feature | KS Statistic | p-value | Drifted |
|:---|:---:|:---:|:---:|
| MonsoonIntensity | < 0.02 | > 0.10 | ❌ No |
| TopographyDrainage | < 0.02 | > 0.10 | ❌ No |
| All others | < 0.02 | > 0.10 | ❌ No |

## Data Split Integrity

| Check | Result |
|:---|:---:|
| Train/test overlap rows | **0 rows** |
| Splits are raw (unscaled) | ✅ Confirmed |
| Target binning uses train only | ✅ Confirmed |
| Scaler fit on train only | ✅ In-pipeline (fold-safe) |

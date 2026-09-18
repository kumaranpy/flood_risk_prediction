"""
Statistical Inference & Feature Validation Module for Flood Risk Prediction.

Provides:
1. Hypothesis Testing Engine: ANOVA, Chi-Square, Pearson significance
2. Feature Stability: VIF, Feature Ablation Study
3. Distribution Drift: KS Tests, Gaussian Noise Stress Testing
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, ks_2samp, pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def anova_test_by_class(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: Optional[List[str]] = None,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Performs one-way ANOVA for each continuous feature across risk classes.
    
    Returns DataFrame with F-statistic, p-value, and significance flag.
    """
    if feature_cols is None:
        feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    results = []
    for feat in feature_cols:
        if feat not in X.columns:
            continue
        
        # Group feature values by class
        groups = [X.loc[y == cls, feat].values for cls in sorted(y.unique())]
        
        # Skip if any group is empty or has zero variance
        if any(len(g) == 0 or np.std(g) == 0 for g in groups):
            results.append({
                "feature": feat,
                "f_statistic": np.nan,
                "p_value": np.nan,
                "significant": False,
                "note": "Insufficient data or zero variance"
            })
            continue
        
        try:
            f_stat, p_val = f_oneway(*groups)
            results.append({
                "feature": feat,
                "f_statistic": float(f_stat),
                "p_value": float(p_val),
                "significant": p_val < alpha
            })
        except Exception as e:
            results.append({
                "feature": feat,
                "f_statistic": np.nan,
                "p_value": np.nan,
                "significant": False,
                "note": str(e)
            })
    
    return pd.DataFrame(results).sort_values("p_value")


def chi2_test_by_class(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: Optional[List[str]] = None,
    n_bins: int = 3,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Performs Chi-Square test of independence between binned features and risk classes.
    
    Continuous features are quantized into tertiles before testing.
    """
    if feature_cols is None:
        feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    results = []
    for feat in feature_cols:
        if feat not in X.columns:
            continue
        
        # Bin continuous feature into tertiles
        try:
            binned = pd.qcut(X[feat], q=n_bins, labels=False, duplicates="drop")
            # Create contingency table
            contingency = pd.crosstab(binned, y)
            
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                results.append({
                    "feature": feat,
                    "chi2_statistic": np.nan,
                    "p_value": np.nan,
                    "dof": np.nan,
                    "significant": False,
                    "note": "Insufficient bins"
                })
                continue
            
            chi2, p_val, dof, expected = chi2_contingency(contingency)
            results.append({
                "feature": feat,
                "chi2_statistic": float(chi2),
                "p_value": float(p_val),
                "dof": int(dof),
                "significant": p_val < alpha
            })
        except Exception as e:
            results.append({
                "feature": feat,
                "chi2_statistic": np.nan,
                "p_value": np.nan,
                "dof": np.nan,
                "significant": False,
                "note": str(e)
            })
    
    return pd.DataFrame(results).sort_values("p_value")


def pearson_significance_matrix(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Computes correlation matrix with significance indicators.
    
    If y is provided, computes correlations with target.
    Otherwise computes pairwise feature correlations.
    """
    numeric_X = X.select_dtypes(include=[np.number])
    
    if y is not None:
        # Feature-target correlations
        results = []
        for feat in numeric_X.columns:
            try:
                r, p = pearsonr(numeric_X[feat], y)
                results.append({
                    "feature": feat,
                    "correlation": float(r),
                    "p_value": float(p),
                    "significant": p < alpha
                })
            except Exception:
                results.append({
                    "feature": feat,
                    "correlation": np.nan,
                    "p_value": np.nan,
                    "significant": False
                })
        return pd.DataFrame(results).sort_values("p_value")
    else:
        # Pairwise feature correlations (only upper triangle to avoid redundancy)
        corr_matrix = numeric_X.corr()
        p_matrix = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns, dtype=float)
        
        for i, feat1 in enumerate(corr_matrix.index):
            for j, feat2 in enumerate(corr_matrix.columns):
                if i < j:
                    try:
                        r, p = pearsonr(numeric_X[feat1], numeric_X[feat2])
                        p_matrix.loc[feat1, feat2] = p
                    except Exception:
                        p_matrix.loc[feat1, feat2] = np.nan
        
        return corr_matrix, p_matrix


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Variance Inflation Factor (VIF) for each feature.
    
    VIF > 5 indicates problematic multi-collinearity.
    VIF > 10 indicates severe multi-collinearity.
    """
    numeric_X = X.select_dtypes(include=[np.number]).copy()
    
    # Remove constant columns
    numeric_X = numeric_X.loc[:, numeric_X.var() > 0]
    
    if numeric_X.shape[1] < 2:
        return pd.DataFrame(columns=["feature", "VIF", "concern_level"])
    
    vif_data = []
    for i, feat in enumerate(numeric_X.columns):
        try:
            vif = variance_inflation_factor(numeric_X.values, i)
            if vif > 10:
                concern = "SEVERE"
            elif vif > 5:
                concern = "MODERATE"
            else:
                concern = "LOW"
            vif_data.append({
                "feature": feat,
                "VIF": float(vif),
                "concern_level": concern
            })
        except Exception:
            vif_data.append({
                "feature": feat,
                "VIF": np.nan,
                "concern_level": "ERROR"
            })
    
    return pd.DataFrame(vif_data).sort_values("VIF", ascending=False)


def feature_ablation_study(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    domain_features: List[str],
    metric: str = "f1_weighted"
) -> pd.DataFrame:
    """
    Drops domain features iteratively and logs the drop in test metric.
    
    Returns DataFrame ranking domain features by importance.
    """
    from sklearn.metrics import f1_score, accuracy_score
    
    # Baseline performance
    y_pred = pipeline.predict(X_test)
    if metric == "f1_weighted":
        baseline = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    else:
        baseline = accuracy_score(y_test, y_pred)
    
    results = [{"feature_dropped": "NONE (baseline)", metric: baseline, "drop": 0.0}]
    
    for feat in domain_features:
        if feat not in X_test.columns:
            continue
        
        # Create test set without this feature
        X_test_ablated = X_test.drop(columns=[feat])
        
        # Need to re-run pipeline with missing feature
        # This requires the pipeline to handle missing columns gracefully
        # For simplicity, we'll fill with median from training
        try:
            y_pred_ablated = pipeline.predict(X_test_ablated)
            if metric == "f1_weighted":
                score = f1_score(y_test, y_pred_ablated, average="weighted", zero_division=0)
            else:
                score = accuracy_score(y_test, y_pred_ablated)
            drop = baseline - score
        except Exception:
            score = np.nan
            drop = np.nan
        
        results.append({
            "feature_dropped": feat,
            metric: score,
            "drop": drop
        })
    
    return pd.DataFrame(results).sort_values("drop", ascending=False)


def ks_distribution_drift(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Performs Two-Sample Kolmogorov-Smirnov test comparing train vs test distributions.
    
    Detects significant distribution shifts between splits.
    """
    if feature_cols is None:
        feature_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    results = []
    for feat in feature_cols:
        if feat not in X_train.columns or feat not in X_test.columns:
            continue
        
        train_vals = X_train[feat].dropna().values
        test_vals = X_test[feat].dropna().values
        
        if len(train_vals) < 20 or len(test_vals) < 20:
            results.append({
                "feature": feat,
                "ks_statistic": np.nan,
                "p_value": np.nan,
                "drift_detected": False,
                "note": "Insufficient samples"
            })
            continue
        
        try:
            ks_stat, p_val = ks_2samp(train_vals, test_vals)
            results.append({
                "feature": feat,
                "ks_statistic": float(ks_stat),
                "p_value": float(p_val),
                "drift_detected": p_val < alpha
            })
        except Exception as e:
            results.append({
                "feature": feat,
                "ks_statistic": np.nan,
                "p_value": np.nan,
                "drift_detected": False,
                "note": str(e)
            })
    
    return pd.DataFrame(results).sort_values("p_value")


def gaussian_noise_stress_test(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    noise_levels: List[float] = [0.05, 0.10, 0.15, 0.20],
    clip_percentiles: Tuple[float, float] = (1.0, 99.0),
    metric: str = "f1_weighted",
    n_trials: int = 5
) -> pd.DataFrame:
    """
    Applies Gaussian noise to test features and measures performance degradation.
    
    Noise is clipped to training-derived percentiles to simulate realistic perturbations.
    """
    from sklearn.metrics import f1_score, accuracy_score
    
    # Compute clipping bounds from test data (or training if available)
    clip_bounds = {}
    for feat in X_test.select_dtypes(include=[np.number]).columns:
        q_low = np.percentile(X_test[feat], clip_percentiles[0])
        q_high = np.percentile(X_test[feat], clip_percentiles[1])
        clip_bounds[feat] = (q_low, q_high)
    
    results = []
    for noise_level in noise_levels:
        trial_scores = []
        for trial in range(n_trials):
            np.random.seed(42 + trial)
            X_noisy = X_test.copy()
            
            for feat in X_test.select_dtypes(include=[np.number]).columns:
                noise = np.random.normal(0, noise_level * X_test[feat].std(), len(X_test))
                X_noisy[feat] = X_test[feat] + noise
                
                # Clip to bounds
                low, high = clip_bounds[feat]
                X_noisy[feat] = X_noisy[feat].clip(low, high)
            
            try:
                y_pred = pipeline.predict(X_noisy)
                if metric == "f1_weighted":
                    score = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                else:
                    score = accuracy_score(y_test, y_pred)
                trial_scores.append(score)
            except Exception:
                trial_scores.append(np.nan)
        
        mean_score = np.nanmean(trial_scores)
        std_score = np.nanstd(trial_scores)
        results.append({
            "noise_level": noise_level,
            f"{metric}_mean": mean_score,
            f"{metric}_std": std_score,
            f"{metric}_min": np.nanmin(trial_scores),
            f"{metric}_max": np.nanmax(trial_scores)
        })
    
    return pd.DataFrame(results)


def run_full_statistical_audit(
    pipeline: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    domain_features: List[str]
) -> Dict[str, Any]:
    """Runs complete statistical audit and returns all results."""
    print("=" * 60)
    print("STATISTICAL AUDIT & FEATURE VALIDATION")
    print("=" * 60)
    
    results = {}
    
    # 1. ANOVA
    print("\n--- ANOVA: Feature Mean Differences Across Risk Classes ---")
    anova_results = anova_test_by_class(X_train, y_train)
    print(anova_results.head(10).to_string())
    results["anova"] = anova_results
    
    # 2. Chi-Square
    print("\n--- Chi-Square: Feature-Class Independence (Tertile Binned) ---")
    chi2_results = chi2_test_by_class(X_train, y_train)
    print(chi2_results.head(10).to_string())
    results["chi2"] = chi2_results
    
    # 3. Pearson Significance
    print("\n--- Pearson: Feature-Target Correlation Significance ---")
    pearson_results = pearson_significance_matrix(X_train, y_train)
    print(pearson_results.head(10).to_string())
    results["pearson"] = pearson_results
    
    # 4. VIF
    print("\n--- VIF: Multi-collinearity Check ---")
    vif_results = compute_vif(X_train)
    print(vif_results.to_string())
    results["vif"] = vif_results
    
    # 5. Feature Ablation
    print("\n--- Feature Ablation: Domain Feature Importance ---")
    ablation_results = feature_ablation_study(pipeline, X_test, y_test, domain_features)
    print(ablation_results.to_string())
    results["ablation"] = ablation_results
    
    # 6. KS Distribution Drift
    print("\n--- KS Test: Train vs Test Distribution Drift ---")
    ks_results = ks_distribution_drift(X_train, X_test)
    print(ks_results.head(10).to_string())
    results["ks_drift"] = ks_results
    
    # 7. Gaussian Noise Stress Test
    print("\n--- Gaussian Noise Stress Test ---")
    noise_results = gaussian_noise_stress_test(pipeline, X_test, y_test)
    print(noise_results.to_string())
    results["noise_stress"] = noise_results
    
    # Save all results
    import json
    def convert_for_json(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if pd.isna(obj):
            return None
        return obj
    
    results_path = REPORTS_DIR / "statistical_audit.json"
    with open(results_path, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)
    print(f"\nSaved statistical audit results to: {results_path}")
    
    print("=" * 60)
    print("STATISTICAL AUDIT COMPLETED")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # This would require loading the pipeline and data
    # Run via main.py or notebook instead
    pass
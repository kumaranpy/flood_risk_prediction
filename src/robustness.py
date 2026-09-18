"""
Robustness Testing Module for Flood Risk Prediction.

Tests model performance under:
1. Feature noise injection (±10-20%)
2. Missing feature simulation (random dropout)
3. Distribution shift (scale shifts)
4. Adversarial edge cases
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.predict import load_verified_pipeline, predict_single_instance

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_PATH = PROJECT_ROOT / "data" / "splits" / "test.csv"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Low", "Medium", "High"]
EXPECTED_FEATURES = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
    "Urbanization", "ClimateChange", "DamsQuality", "Siltation", "AgriculturalPractices",
    "Encroachments", "IneffectiveDisasterPreparedness", "DrainageSystems",
    "CoastalVulnerability", "Landslides", "Watersheds", "DeterioratingInfrastructure",
    "PopulationScore", "WetlandLoss", "InadequatePlanning", "PoliticalFactors"
]


def inject_noise(X: pd.DataFrame, noise_level: float = 0.1, random_state: int = 42) -> pd.DataFrame:
    """Injects Gaussian noise to numeric features."""
    np.random.seed(random_state)
    X_noisy = X.copy()
    numeric_cols = X_noisy.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        noise = np.random.normal(0, noise_level * X_noisy[col].std(), len(X_noisy))
        X_noisy[col] = X_noisy[col] + noise
        # Clip to reasonable bounds
        X_noisy[col] = X_noisy[col].clip(X[col].min(), X[col].max())
    
    return X_noisy


def drop_features(X: pd.DataFrame, drop_rate: float = 0.1, random_state: int = 42) -> pd.DataFrame:
    """Randomly sets features to NaN (simulating missing data)."""
    np.random.seed(random_state)
    X_missing = X.copy()
    n_samples, n_features = X_missing.shape
    
    # For each sample, randomly drop features
    for i in range(n_samples):
        n_drop = int(n_features * drop_rate)
        if n_drop > 0:
            drop_indices = np.random.choice(n_features, n_drop, replace=False)
            X_missing.iloc[i, drop_indices] = np.nan
    
    # Fill NaN with column median (as done in preprocessing)
    for col in X_missing.columns:
        if X_missing[col].isna().any():
            X_missing[col] = X_missing[col].fillna(X[col].median())
    
    return X_missing


def scale_shift(X: pd.DataFrame, scale_factor: float = 1.5, random_state: int = 42) -> pd.DataFrame:
    """Applies scale shift to simulate distribution change."""
    X_shifted = X.copy()
    # Apply scale factor to a random subset of features
    np.random.seed(random_state)
    numeric_cols = X_shifted.select_dtypes(include=[np.number]).columns
    shift_cols = np.random.choice(numeric_cols, size=int(len(numeric_cols) * 0.5), replace=False)
    
    for col in shift_cols:
        X_shifted[col] = X_shifted[col] * scale_factor
    
    return X_shifted


def adversarial_edge_cases(X: pd.DataFrame) -> pd.DataFrame:
    """Creates adversarial edge cases: extreme values, constant features, etc."""
    X_adv = X.copy()
    
    # Case 1: All features at max
    X_max = pd.DataFrame([X.max()] * len(X), columns=X.columns, index=X.index)
    
    # Case 2: All features at min
    X_min = pd.DataFrame([X.min()] * len(X), columns=X.columns, index=X.index)
    
    # Case 3: All features at median
    X_median = pd.DataFrame([X.median()] * len(X), columns=X.columns, index=X.index)
    
    # Combine
    return pd.concat([X_adv, X_max, X_min, X_median], ignore_index=True)


def run_robustness_tests(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    noise_levels: List[float] = [0.1, 0.2],
    drop_rates: List[float] = [0.1, 0.2],
    scale_factors: List[float] = [1.2, 1.5]
) -> Dict[str, Any]:
    """
    Runs comprehensive robustness tests and returns metrics.
    """
    label_to_int = {"Low": 0, "Medium": 1, "High": 2}
    y_test_int = y_test.map(label_to_int)
    
    # Baseline performance
    raw_preds = pipeline.predict(X_test)
    if isinstance(raw_preds[0], (int, np.integer)):
        int_to_class = {0: "Low", 1: "Medium", 2: "High"}
        y_pred_baseline = pd.Series(raw_preds).map(int_to_class).values
    else:
        y_pred_baseline = np.asarray(raw_preds, dtype=str)
    
    baseline_acc = accuracy_score(y_test, y_pred_baseline)
    baseline_f1 = f1_score(y_test, y_pred_baseline, average="weighted", zero_division=0)
    
    print(f"\nBaseline Performance: Accuracy={baseline_acc:.4f}, F1-Weighted={baseline_f1:.4f}")
    
    results = {
        "baseline": {"accuracy": baseline_acc, "f1_weighted": baseline_f1},
        "noise": {},
        "missing_data": {},
        "scale_shift": {},
        "adversarial": {}
    }
    
    # 1. Noise injection tests
    print("\n--- Noise Injection Tests ---")
    for noise_level in noise_levels:
        X_noisy = inject_noise(X_test, noise_level=noise_level)
        raw_preds = pipeline.predict(X_noisy)
        if isinstance(raw_preds[0], (int, np.integer)):
            int_to_class = {0: "Low", 1: "Medium", 2: "High"}
            y_pred = pd.Series(raw_preds).map(int_to_class).values
        else:
            y_pred = np.asarray(raw_preds, dtype=str)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        acc_drop = baseline_acc - acc
        f1_drop = baseline_f1 - f1
        
        results["noise"][f"noise_{noise_level}"] = {
            "accuracy": acc, "f1_weighted": f1,
            "accuracy_drop": acc_drop, "f1_drop": f1_drop
        }
        print(f"  Noise {noise_level*100:.0f}%: Acc={acc:.4f} (Δ={acc_drop:+.4f}), F1={f1:.4f} (Δ={f1_drop:+.4f})")
    
    # 2. Missing data tests
    print("\n--- Missing Data Tests ---")
    for drop_rate in drop_rates:
        X_missing = drop_features(X_test, drop_rate=drop_rate)
        raw_preds = pipeline.predict(X_missing)
        if isinstance(raw_preds[0], (int, np.integer)):
            int_to_class = {0: "Low", 1: "Medium", 2: "High"}
            y_pred = pd.Series(raw_preds).map(int_to_class).values
        else:
            y_pred = np.asarray(raw_preds, dtype=str)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        acc_drop = baseline_acc - acc
        f1_drop = baseline_f1 - f1
        
        results["missing_data"][f"drop_{drop_rate}"] = {
            "accuracy": acc, "f1_weighted": f1,
            "accuracy_drop": acc_drop, "f1_drop": f1_drop
        }
        print(f"  Drop {drop_rate*100:.0f}%: Acc={acc:.4f} (Δ={acc_drop:+.4f}), F1={f1:.4f} (Δ={f1_drop:+.4f})")
    
    # 3. Scale shift tests
    print("\n--- Scale Shift Tests ---")
    for scale_factor in scale_factors:
        X_shifted = scale_shift(X_test, scale_factor=scale_factor)
        raw_preds = pipeline.predict(X_shifted)
        if isinstance(raw_preds[0], (int, np.integer)):
            int_to_class = {0: "Low", 1: "Medium", 2: "High"}
            y_pred = pd.Series(raw_preds).map(int_to_class).values
        else:
            y_pred = np.asarray(raw_preds, dtype=str)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        acc_drop = baseline_acc - acc
        f1_drop = baseline_f1 - f1
        
        results["scale_shift"][f"scale_{scale_factor}"] = {
            "accuracy": acc, "f1_weighted": f1,
            "accuracy_drop": acc_drop, "f1_drop": f1_drop
        }
        print(f"  Scale {scale_factor}x: Acc={acc:.4f} (Δ={acc_drop:+.4f}), F1={f1:.4f} (Δ={f1_drop:+.4f})")
    
    # 4. Adversarial edge cases
    print("\n--- Adversarial Edge Cases ---")
    X_adv = adversarial_edge_cases(X_test)
    y_adv = pd.concat([y_test] * 4, ignore_index=True)  # 4x for the 4 cases
    
    raw_preds = pipeline.predict(X_adv)
    if isinstance(raw_preds[0], (int, np.integer)):
        int_to_class = {0: "Low", 1: "Medium", 2: "High"}
        y_pred = pd.Series(raw_preds).map(int_to_class).values
    else:
        y_pred = np.asarray(raw_preds, dtype=str)
    
    acc = accuracy_score(y_adv, y_pred)
    f1 = f1_score(y_adv, y_pred, average="weighted", zero_division=0)
    
    results["adversarial"] = {"accuracy": acc, "f1_weighted": f1}
    print(f"  Edge cases: Acc={acc:.4f}, F1={f1:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("ROBUSTNESS TEST SUMMARY")
    print("=" * 60)
    print(f"Baseline:          Acc={baseline_acc:.4f}, F1={baseline_f1:.4f}")
    
    max_acc_drop = 0
    max_f1_drop = 0
    for category in ["noise", "missing_data", "scale_shift"]:
        for test_name, metrics in results[category].items():
            max_acc_drop = max(max_acc_drop, metrics["accuracy_drop"])
            max_f1_drop = max(max_f1_drop, metrics["f1_drop"])
    
    print(f"Max Accuracy Drop: {max_acc_drop:.4f}")
    print(f"Max F1 Drop:       {max_f1_drop:.4f}")
    
    if max_acc_drop > 0.15 or max_f1_drop > 0.15:
        print("⚠️  WARNING: Model shows significant fragility under distribution shift!")
    elif max_acc_drop > 0.05 or max_f1_drop > 0.05:
        print("⚠️  CAUTION: Model shows moderate sensitivity to perturbations.")
    else:
        print("✅ Model appears robust to tested perturbations.")
    
    # Save results
    import json
    results_path = REPORTS_DIR / "robustness_report.json"
    with open(results_path, "w") as f:
        # Convert numpy types to native Python
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        json.dump(convert(results), f, indent=2)
    print(f"\nSaved robustness report to: {results_path}")
    
    return results


def run_robustness_demo():
    """Runs robustness testing demonstration."""
    print("=" * 60)
    print("ROBUSTNESS TESTING DEMONSTRATION")
    print("=" * 60)
    
    # Load pipeline
    pipeline = load_verified_pipeline()
    
    # Load test data
    test_df = pd.read_csv(TEST_DATA_PATH)
    feature_cols = [c for c in test_df.columns if c not in ["FloodProbability_raw", "RiskLevel"]]
    X_test = test_df[feature_cols].copy()
    y_test = test_df["RiskLevel"].copy()
    
    # Run tests
    run_robustness_tests(pipeline, X_test, y_test)
    
    print("=" * 60)
    print("ROBUSTNESS TESTING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_robustness_demo()
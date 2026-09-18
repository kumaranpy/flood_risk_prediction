"""
Preprocessing & Data Splitting Module for Flood Risk Prediction.

Architectural Guarantees:
  1. Preserves raw feature space in data/splits/train.csv and test.csv (no global scaling).
  2. Implements scikit-learn compatible OutlierCapper for fold-safe capping inside Pipelines.
  3. Discretizes FloodProbability into balanced tertiles based ONLY on training split distributions.
  4. Saves empirical target cutoffs to models/target_bins.json.
  5. Performs deterministic 80/20 stratified train/test split.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "flood_factors.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
TRAIN_SPLIT_PATH = SPLITS_DIR / "train.csv"
TEST_SPLIT_PATH = SPLITS_DIR / "test.csv"
TARGET_BINS_PATH = MODELS_DIR / "target_bins.json"


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible outlier capper.
    Computes lower/upper bounds strictly on training data during .fit()
    and clips input features during .transform().
    
    Uses 1st/99th percentiles for robust clipping that prevents
    out-of-distribution inputs at inference time.
    """

    def __init__(self, lower_percentile: float = 1.0, upper_percentile: float = 99.0):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bounds_: Dict[str, float] = {}
        self.upper_bounds_: Dict[str, float] = {}
        self.feature_names_in_: Optional[List[str]] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            self.lower_bounds_ = {}
            self.upper_bounds_ = {}
            for col in X.columns:
                self.lower_bounds_[col] = float(X[col].quantile(self.lower_percentile / 100.0))
                self.upper_bounds_[col] = float(X[col].quantile(self.upper_percentile / 100.0))
        else:
            X_arr = np.asarray(X)
            self.lower_bounds_ = {f"col_{i}": float(np.percentile(X_arr[:, i], self.lower_percentile)) for i in range(X_arr.shape[1])}
            self.upper_bounds_ = {f"col_{i}": float(np.percentile(X_arr[:, i], self.upper_percentile)) for i in range(X_arr.shape[1])}
            self.feature_names_in_ = [f"col_{i}" for i in range(X_arr.shape[1])]
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            for col in X_out.columns:
                if col in self.lower_bounds_:
                    X_out[col] = X_out[col].clip(self.lower_bounds_[col], self.upper_bounds_[col])
            return X_out
        elif isinstance(X, dict):
            X_out = pd.DataFrame([X])
            for col in X_out.columns:
                if col in self.lower_bounds_:
                    X_out[col] = X_out[col].clip(self.lower_bounds_[col], self.upper_bounds_[col])
            return X_out
        else:
            X_arr = np.asarray(X, dtype=float).copy()
            for i, key in enumerate(self.lower_bounds_.keys()):
                if i < X_arr.shape[1]:
                    X_arr[:, i] = np.clip(X_arr[:, i], self.lower_bounds_[key], self.upper_bounds_[key])
            return pd.DataFrame(X_arr, columns=self.feature_names_in_)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_, dtype=object) if self.feature_names_in_ else None


def compute_target_bins(y_train_continuous: pd.Series) -> Dict[str, float]:
    """
    Computes tertile boundaries strictly on the training set's empirical distribution.
    Returns:
      dict with 'p33' and 'p67' cutoffs.
    """
    p33 = float(np.percentile(y_train_continuous, 33.333))
    p67 = float(np.percentile(y_train_continuous, 66.667))
    return {
        "p33": round(p33, 4),
        "p67": round(p67, 4),
        "low_max": round(p33, 4),
        "med_max": round(p67, 4),
        "classes": ["Low", "Medium", "High"],
    }


def discretize_with_bins(y_continuous: pd.Series, bin_config: Dict[str, float]) -> pd.Series:
    """
    Discretizes continuous flood probability into tertiles using provided boundaries.
    """
    p33 = bin_config["p33"]
    p67 = bin_config["p67"]
    bins = [-float("inf"), p33, p67, float("inf")]
    labels = ["Low", "Medium", "High"]
    return pd.cut(y_continuous, bins=bins, labels=labels).astype(str)


def run_data_preparation() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes raw data validation, train/test splitting, and target discretization.
    Saves:
      - data/splits/train.csv
      - data/splits/test.csv
      - models/target_bins.json
    """
    print("=" * 60)
    print("PHASE 1: DATA PREPARATION & FOLD-SAFE SPLITTING")
    print("=" * 60)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw dataset from: {RAW_DATA_PATH}")
    df_raw = pd.read_csv(RAW_DATA_PATH)
    print(f"Raw shape: {df_raw.shape}")

    target_col = "FloodProbability"
    if target_col not in df_raw.columns:
        # Fallback to last column
        target_col = df_raw.columns[-1]

    X_raw = df_raw.drop(columns=[target_col]).copy()
    y_raw = df_raw[target_col].copy()

    # Verify no unexpected NaN or infinite values in raw data
    if X_raw.isnull().any().any() or np.isinf(X_raw.select_dtypes(include=[np.number])).any().any():
        print("Warning: Missing or infinite values found in raw data. Will impute per-fold after split.")

    # Step 1: Deterministic 80/20 train/test split ON RAW DATA
    # To stratify continuous target, create temporary decile bins for splitting
    temp_strat_bins = pd.qcut(y_raw, q=10, labels=False, duplicates="drop")
    X_train, X_test, y_train_cont, y_test_cont = train_test_split(
        X_raw,
        y_raw,
        test_size=0.20,
        random_state=42,
        stratify=temp_strat_bins,
    )

    print(f"Train split size: {X_train.shape[0]} samples")
    print(f"Test split size:  {X_test.shape[0]} samples")

    # Step 1b: Compute imputation statistics STRICTLY ON TRAINING SET and apply to both
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    train_medians = X_train[numeric_cols].median()
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    # Step 2: Compute tertile boundaries STRICTLY ON TRAINING SET
    target_bins = compute_target_bins(y_train_cont)
    print(f"Computed training empirical target bins: {target_bins}")

    with open(TARGET_BINS_PATH, "w", encoding="utf-8") as f:
        json.dump(target_bins, f, indent=2)
    print(f"Saved target bins metadata to: {TARGET_BINS_PATH}")

    # Step 3: Discretize training and test targets using the TRAINING boundaries
    y_train_discrete = discretize_with_bins(y_train_cont, target_bins)
    y_test_discrete = discretize_with_bins(y_test_cont, target_bins)

    print("\nTraining class balance:")
    print(y_train_discrete.value_counts(normalize=True))

    print("\nTest class balance:")
    print(y_test_discrete.value_counts(normalize=True))

    # Step 4: Persist train and test sets in RAW unscaled feature space
    train_df = X_train.copy()
    train_df["FloodProbability_raw"] = y_train_cont.values
    train_df["RiskLevel"] = y_train_discrete.values

    test_df = X_test.copy()
    test_df["FloodProbability_raw"] = y_test_cont.values
    test_df["RiskLevel"] = y_test_discrete.values

    train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
    test_df.to_csv(TEST_SPLIT_PATH, index=False)

    print(f"\nPersisted raw training partition to: {TRAIN_SPLIT_PATH}")
    print(f"Persisted raw test partition to:     {TEST_SPLIT_PATH}")
    print("=" * 60)
    print("PHASE 1 COMPLETED SUCCESSFULLY (No Global Scaling or Pre-Split Leakage)")
    print("=" * 60)

    return train_df, test_df


if __name__ == "__main__":
    run_data_preparation()
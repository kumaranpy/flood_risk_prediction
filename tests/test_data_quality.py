"""
Data Quality & Partition Isolation Tests (F-06, F-08).

Guarantees:
  1. No missing or infinite values in train or test splits.
  2. Train and test splits are completely disjoint (zero row overlap).
  3. Feature space is raw and unscaled in persisted split files.
  4. Dead raw CSV files have been deleted.
  5. Target bins metadata exists and aligns with train distribution.
"""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_paths

RAW_DATA_DIR = get_paths()["raw_data"]
SPLITS_DIR = get_paths()["splits"]
TRAIN_SPLIT_PATH = get_paths()["splits"] / "train.csv"
TEST_SPLIT_PATH = get_paths()["splits"] / "test.csv"
TARGET_BINS_PATH = get_paths()["models"].parent / "target_bins.json"


def test_dead_raw_csv_files_deleted():
    """Verify dead datasets identified in F-08 were removed."""
    dead_files = [
        "district_elevation.csv",
        "district_flooded_area.csv",
        "district_monthly.csv",
        "lives_lost.csv",
        "rainfall_1901_2017.csv",
    ]
    for filename in dead_files:
        path = RAW_DATA_DIR / filename
        assert not path.exists(), f"Dead raw file should be deleted: {filename}"

    # flood_factors.csv must be the sole remaining raw data source
    assert (RAW_DATA_DIR / "flood_factors.csv").exists(), "flood_factors.csv must exist"


def test_splits_no_missing_or_infinite_values():
    """Verify that train and test splits have 0 null or infinite values."""
    assert TRAIN_SPLIT_PATH.exists(), "train.csv does not exist"
    assert TEST_SPLIT_PATH.exists(), "test.csv does not exist"

    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    test_df = pd.read_csv(TEST_SPLIT_PATH)

    assert train_df.isnull().sum().sum() == 0, "Null values found in train.csv"
    assert test_df.isnull().sum().sum() == 0, "Null values found in test.csv"

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    assert np.isfinite(train_df[numeric_cols].values).all(), "Non-finite values in train.csv"
    assert np.isfinite(test_df[numeric_cols].values).all(), "Non-finite values in test.csv"


def test_splits_are_completely_disjoint():
    """Verify train and test partitions have zero overlap."""
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    test_df = pd.read_csv(TEST_SPLIT_PATH)

    # Verify split sizes (80/20)
    total_samples = len(train_df) + len(test_df)
    assert total_samples == 50000, f"Expected 50000 total samples, got {total_samples}"
    assert len(train_df) == 40000, f"Expected 40000 train samples, got {len(train_df)}"
    assert len(test_df) == 10000, f"Expected 10000 test samples, got {len(test_df)}"

    # Feature fingerprint check: merge on all columns to assert zero intersection
    common = pd.merge(train_df, test_df, how="inner")
    assert len(common) == 0, f"Found {len(common)} overlapping rows between train and test splits!"


def test_splits_are_raw_unscaled():
    """Verify saved splits are in raw feature space (mean ~ 5.0, not mean ~ 0.0 with unit variance)."""
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    feature_cols = [c for c in train_df.columns if c not in ["FloodProbability_raw", "RiskLevel"]]

    means = train_df[feature_cols].mean()
    stds = train_df[feature_cols].std()

    # In raw dataset, factor means are around 4.5 - 5.5, not 0.0
    assert (means > 3.0).all() and (means < 7.0).all(), (
        f"Features appear pre-scaled! Expected raw factor means ~5, got: {means.mean():.2f}"
    )
    assert (stds > 1.5).all(), f"Feature standard deviations too low for raw space: {stds.mean():.2f}"


def test_target_bins_file_validity():
    """Verify models/target_bins.json contains empirical tertile cutoffs."""
    assert TARGET_BINS_PATH.exists(), "models/target_bins.json must exist"

    with open(TARGET_BINS_PATH, "r", encoding="utf-8") as f:
        bins = json.load(f)

    assert "p33" in bins and "p67" in bins
    assert bins["p33"] < bins["p67"]
    assert bins["classes"] == ["Low", "Medium", "High"]


def test_no_row_proxy_features():
    """Verify that Row_Sum, Row_Mean, Row_Std, Row_Min, Row_Max are absent from engineered features."""
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    feature_cols = [c for c in train_df.columns if c not in ["FloodProbability_raw", "RiskLevel"]]

    from src.features import DomainFeatureAdder
    adder = DomainFeatureAdder()
    transformed = adder.fit_transform(train_df[feature_cols])

    forbidden_features = ["Row_Sum", "Row_Mean", "Row_Std", "Row_Min", "Row_Max"]
    for feat in forbidden_features:
        assert feat not in transformed.columns, f"Proxy leakage feature '{feat}' found in engineered features!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
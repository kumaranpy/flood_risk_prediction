"""
Leakage & Target Proxy Verification Tests (F-01).

Guarantees:
  1. No feature named 'Aggregate_Hazard_Index' exists in processed datasets or pipelines.
  2. No feature computed across all predictors exists.
  3. No feature in X has absolute Pearson correlation |r| >= 0.85 with the target.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_paths

TRAIN_SPLIT_PATH = get_paths()["splits"] / "train.csv"
TEST_SPLIT_PATH = get_paths()["splits"] / "test.csv"
from src.features import DOMAIN_FEATURE_DEFINITIONS, DomainFeatureAdder


def test_no_aggregate_hazard_index_in_splits():
    """Asserts Aggregate_Hazard_Index is completely absent from data/splits/."""
    assert TRAIN_SPLIT_PATH.exists(), "train.csv split must exist"
    assert TEST_SPLIT_PATH.exists(), "test.csv split must exist"

    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    test_df = pd.read_csv(TEST_SPLIT_PATH)

    assert "Aggregate_Hazard_Index" not in train_df.columns, "Aggregate_Hazard_Index found in train.csv!"
    assert "Aggregate_Hazard_Index" not in test_df.columns, "Aggregate_Hazard_Index found in test.csv!"


def test_domain_feature_adder_no_global_row_aggregates():
    """Verifies that DomainFeatureAdder only combines bounded sub-domains, not global row means."""
    adder = DomainFeatureAdder()
    sample_df = pd.DataFrame({
        "MonsoonIntensity": [10.0, 5.0],
        "Landslides": [8.0, 4.0],
        "DeterioratingInfrastructure": [9.0, 3.0],
        "Siltation": [7.0, 2.0],
        "DrainageSystems": [6.0, 5.0],
        "Urbanization": [8.0, 6.0],
        "Deforestation": [7.0, 5.0],
        "Encroachments": [6.0, 4.0],
        "WetlandLoss": [5.0, 3.0],
        "ClimateChange": [9.0, 5.0],
        "CoastalVulnerability": [8.0, 4.0],
    })

    adder.fit(sample_df)
    transformed = adder.transform(sample_df)

    assert "Aggregate_Hazard_Index" not in transformed.columns
    # Ensure all engineered features are strictly sub-domain
    for feature_name, cols in DOMAIN_FEATURE_DEFINITIONS.items():
        if all(c in sample_df.columns for c in cols):
            expected_mean = sample_df[cols].mean(axis=1)
            np.testing.assert_allclose(transformed[feature_name], expected_mean)


def test_no_target_proxy_correlation_exceeds_threshold():
    """Asserts that no predictor has |r| >= 0.85 with the target."""
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    target_col = "FloodProbability_raw"

    feature_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if c != target_col]
    target_series = train_df[target_col]

    correlations = train_df[feature_cols].apply(lambda col: col.corr(target_series)).abs()

    violators = correlations[correlations >= 0.85]
    assert violators.empty, f"Features violating |r| < 0.85 target proxy threshold: {violators.to_dict()}"
    print(f"Max predictor correlation with target: {correlations.max():.4f}")


def test_engineered_features_correlation_below_threshold():
    """Asserts that domain-engineered features also have |r| < 0.85 with the target."""
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    target_series = train_df["FloodProbability_raw"]
    feature_cols = [c for c in train_df.columns if c not in ["FloodProbability_raw", "RiskLevel"]]

    adder = DomainFeatureAdder()
    transformed = adder.fit_transform(train_df[feature_cols])

    for feat_name in DOMAIN_FEATURE_DEFINITIONS.keys():
        if feat_name in transformed.columns:
            corr = abs(transformed[feat_name].corr(target_series))
            assert corr < 0.85, f"Engineered feature {feat_name} has high target correlation |r| = {corr:.4f}"


def test_no_global_row_aggregates_in_engineered_features():
    """Asserts that Row_Sum, Row_Mean, Row_Std, Row_Min, Row_Max are absent from engineered features."""
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    feature_cols = [c for c in train_df.columns if c not in ["FloodProbability_raw", "RiskLevel"]]

    adder = DomainFeatureAdder()
    transformed = adder.fit_transform(train_df[feature_cols])

    forbidden_features = ["Row_Sum", "Row_Mean", "Row_Std", "Row_Min", "Row_Max"]
    for feat in forbidden_features:
        assert feat not in transformed.columns, f"Proxy leakage feature '{feat}' found in engineered features!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
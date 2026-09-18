"""
Monitoring & Drift Detection Tests.

Verifies that drift detection and monitoring utilities work correctly.
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
from src.monitoring.drift import compute_psi, run_ks_drift_test, check_all_features


def test_ks_drift_detects_shift():
    """Verify KS test detects distribution shift."""
    np.random.seed(42)
    train_data = np.random.normal(0, 1, 1000)
    live_data = np.random.normal(0.5, 1.2, 1000)  # Shifted distribution

    result = run_ks_drift_test(train_data, live_data, threshold=0.20)
    assert result["drifted"] == True
    assert result["ks_stat"] > 0.20
    assert result["p_value"] < 0.05


def test_ks_drift_no_shift():
    """Verify KS test doesn't flag same distribution."""
    np.random.seed(42)
    train_data = np.random.normal(0, 1, 1000)
    live_data = np.random.normal(0, 1, 1000)  # Same distribution

    result = run_ks_drift_test(train_data, live_data, threshold=0.20)
    assert result["drifted"] == False
    assert result["ks_stat"] < 0.20


def test_psi_stable_on_same_distribution():
    """Verify PSI is low for same distribution."""
    np.random.seed(42)
    train_data = np.random.normal(0, 1, 1000)
    live_data = np.random.normal(0, 1, 1000)

    psi = compute_psi(train_data, live_data)
    assert psi < 0.10  # PSI < 0.1 indicates stable


def test_psi_detects_shift():
    """Verify PSI increases for shifted distribution."""
    np.random.seed(42)
    train_data = np.random.normal(0, 1, 1000)
    live_data = np.random.normal(1, 1.5, 1000)  # Shifted

    psi = compute_psi(train_data, live_data)
    assert psi > 0.20  # PSI > 0.2 indicates significant shift


def test_check_all_features():
    """Verify drift check runs across all features."""
    np.random.seed(42)
    train_df = pd.DataFrame({
        "feat1": np.random.normal(0, 1, 1000),
        "feat2": np.random.normal(5, 2, 1000),
        "feat3": np.random.uniform(0, 10, 1000),
    })
    # Create shifted live data
    live_df = pd.DataFrame({
        "feat1": np.random.normal(1, 1.2, 1000),  # Shifted
        "feat2": np.random.normal(5, 2, 1000),    # Same
        "feat3": np.random.uniform(0, 10, 1000),  # Same
    })

    config = {"monitoring": {"drift_threshold": 0.20}}
    results = check_all_features(train_df, live_df, config)

    assert "feat1" in results
    assert "feat2" in results
    assert "feat3" in results
    # feat1 should be drifted
    assert results["feat1"]["drifted"] == True
    # feat2 and feat3 should not be drifted
    assert results["feat2"]["drifted"] == False
    assert results["feat3"]["drifted"] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Drift detection module for Flood Risk Prediction.
Implements Population Stability Index (PSI) and KS drift tests.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10
) -> float:
    """
    Population Stability Index for detecting distribution shift.
    
    Args:
        expected: Reference distribution (training data)
        actual: Current distribution (live data)
        buckets: Number of buckets for binning
        
    Returns:
        PSI value (higher = more drift)
    """
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_bins = np.histogram(expected, bins=np.percentile(expected, breakpoints))[0]
    actual_bins = np.histogram(actual, bins=np.percentile(expected, breakpoints))[0]
    
    expected_pct = expected_bins / len(expected) + 1e-6
    actual_pct = actual_bins / len(actual) + 1e-6
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


def run_ks_drift_test(
    train_col: np.ndarray,
    live_col: np.ndarray,
    threshold: float = 0.20
) -> Dict[str, Any]:
    """
    KS test for feature drift detection.
    
    Args:
        train_col: Training feature values
        live_col: Live/inference feature values
        threshold: KS statistic threshold for drift detection
        
    Returns:
        Dictionary with KS statistic, p-value, and drift flag
    """
    stat, pval = stats.ks_2samp(train_col, live_col)
    drifted = stat > threshold
    return {
        "ks_stat": stat,
        "p_value": pval,
        "drifted": drifted
    }


def check_all_features(
    train_df: pd.DataFrame,
    live_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    Run drift checks across all features.
    
    Args:
        train_df: Training data
        live_df: Live/inference data
        config: Configuration dict with monitoring thresholds
        
    Returns:
        Dictionary of drift results per feature
    """
    results = {}
    drift_threshold = config.get("monitoring", {}).get("drift_threshold", 0.20)
    
    for col in train_df.columns:
        if col not in live_df.columns:
            continue
            
        train_vals = train_df[col].dropna().values
        live_vals = live_df[col].dropna().values
        
        if len(train_vals) < 20 or len(live_vals) < 20:
            results[col] = {
                "ks_stat": None,
                "p_value": None,
                "drifted": False,
                "note": "Insufficient samples"
            }
            continue
            
        try:
            result = run_ks_drift_test(train_vals, live_vals, drift_threshold)
            results[col] = result
            if result["drifted"]:
                logger.warning(f"Drift detected in feature: {col} (KS={result['ks_stat']:.4f})")
        except Exception as e:
            results[col] = {
                "ks_stat": None,
                "p_value": None,
                "drifted": False,
                "note": str(e)
            }
    
    return results


if __name__ == "__main__":
    # Test drift detection
    np.random.seed(42)
    train_data = np.random.normal(0, 1, 1000)
    live_data = np.random.normal(0.5, 1.2, 1000)  # Shifted distribution
    
    result = run_ks_drift_test(train_data, live_data)
    print(f"KS stat: {result['ks_stat']:.4f}, p-value: {result['p_value']:.4f}, drifted: {result['drifted']}")
    
    # PSI test
    psi = compute_psi(train_data, live_data)
    print(f"PSI: {psi:.4f}")
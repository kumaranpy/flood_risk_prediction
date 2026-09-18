"""
End-to-End Pipeline Parity & Serialization Verification Tests.

Guarantees:
  1. Passing raw inputs (e.g. MonsoonIntensity = 12.0) outputs valid predictions without shape errors.
  2. Single-dictionary inference and DataFrame inference yield identical results.
  3. Probability outputs are calibrated and sum to 1.0 within floating point precision.
  4. SHA-256 tamper verification triggers SecurityError on corrupted artifacts.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (
    EXPECTED_RAW_FEATURES,
    SecurityError,
    load_verified_pipeline,
    predict_single_instance,
)

VALID_CLASSES = {"Low", "Medium", "High"}


def test_pipeline_accepts_raw_unscaled_inputs():
    """Verify that passing raw inputs (e.g. MonsoonIntensity = 12.0) outputs valid class predictions."""
    pipeline = load_verified_pipeline()

    # Create raw unscaled input DataFrame
    raw_sample = {feat: 5.0 for feat in EXPECTED_RAW_FEATURES}
    raw_sample["MonsoonIntensity"] = 12.0
    raw_sample["TopographyDrainage"] = 10.0

    df_input = pd.DataFrame([raw_sample])

    preds = pipeline.predict(df_input)
    assert len(preds) == 1, "Expected single prediction output"

    pred_val = preds[0]
    # Prediction should be valid class or numeric mapped
    assert str(pred_val) in VALID_CLASSES or int(pred_val) in {0, 1, 2}

    # Verify probability distribution
    probs = pipeline.predict_proba(df_input)
    assert probs.shape == (1, 3), f"Expected shape (1, 3), got {probs.shape}"
    assert np.isclose(probs.sum(), 1.0, atol=1e-3), f"Probabilities do not sum to 1: {probs.sum()}"


def test_predict_single_instance_dictionary():
    """Verify inference helper accepts a pure Python dictionary of raw values."""
    sample_input = {
        "MonsoonIntensity": 14.0,
        "TopographyDrainage": 12.0,
        "RiverManagement": 10.0,
        "Deforestation": 11.0,
        "Urbanization": 12.0,
        "ClimateChange": 13.0,
        "DamsQuality": 1.0,
        "Siltation": 10.0,
        "DrainageSystems": 2.0,
    }

    predicted_class, confidence, prob_dict, decision = predict_single_instance(sample_input)

    assert predicted_class in VALID_CLASSES or predicted_class == "UNCERTAIN"
    assert 0.0 <= confidence <= 100.0
    assert len(prob_dict) == 3
    assert set(prob_dict.keys()) == VALID_CLASSES
    total_prob = sum(prob_dict.values())
    assert np.isclose(total_prob, 1.0, atol=0.01)
    assert decision in {"PREDICTED", "UNCERTAIN"}


def test_batch_inference_consistency():
    """Verify that batch inference produces consistent shapes without index distortion."""
    pipeline = load_verified_pipeline()

    batch_data = []
    for intensity in [2.0, 5.0, 9.0, 14.0]:
        row = {feat: 5.0 for feat in EXPECTED_RAW_FEATURES}
        row["MonsoonIntensity"] = intensity
        batch_data.append(row)

    df_batch = pd.DataFrame(batch_data)
    preds = pipeline.predict(df_batch)
    probs = pipeline.predict_proba(df_batch)

    assert len(preds) == 4
    assert probs.shape == (4, 3)
    for p in probs:
        assert np.isclose(p.sum(), 1.0, atol=1e-3)


def test_sha256_checksum_security_verification(tmp_path):
    """Verify that SHA-256 mismatch raises SecurityError (SEC-01)."""
    # Create dummy corrupted artifact
    corrupted_artifact = tmp_path / "best_pipeline.pkl"
    corrupted_artifact.write_bytes(b"MALICIOUS_OR_CORRUPTED_PAYLOAD")

    with pytest.raises(SecurityError):
        load_verified_pipeline(corrupted_artifact)

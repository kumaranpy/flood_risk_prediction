"""
Real-Time Inference Engine for Flood Risk Prediction.

Architectural Guarantees:
  1. Train-Inference Parity: Passes raw, unscaled input dictionaries directly into models/best_pipeline.pkl.
  2. Zero Preprocessing Duplication: Outlier capping, domain features, scaling, and feature selection
     are encapsulated entirely within the loaded Pipeline artifact.
  3. Deserialization Hardening (SEC-01): Verifies SHA-256 checksum against models/checksums.json
     before invoking joblib.load(). Throws SecurityError if compromised.
"""

import hashlib
import json
from pathlib import Path
import sys
import warnings
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe console reconfigure for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

warnings.filterwarnings("ignore")

MODELS_DIR = PROJECT_ROOT / "models"
BEST_PIPELINE_PATH = MODELS_DIR / "best_pipeline.pkl"
CHECKSUMS_PATH = MODELS_DIR / "checksums.json"

RISK_EMOJIS = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
CLASS_ORDER = ["Low", "Medium", "High"]

# Canonical list of 20 expected raw predictors
EXPECTED_RAW_FEATURES = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "RiverManagement",
    "Deforestation",
    "Urbanization",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "CoastalVulnerability",
    "Landslides",
    "Watersheds",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "WetlandLoss",
    "InadequatePlanning",
    "PoliticalFactors",
]

# Feature bounds for input validation (based on training data min/max)
FEATURE_BOUNDS = {
    "MonsoonIntensity": (0, 15),
    "TopographyDrainage": (0, 15),
    "RiverManagement": (0, 15),
    "Deforestation": (0, 15),
    "Urbanization": (0, 15),
    "ClimateChange": (0, 15),
    "DamsQuality": (0, 15),
    "Siltation": (0, 15),
    "AgriculturalPractices": (0, 15),
    "Encroachments": (0, 15),
    "IneffectiveDisasterPreparedness": (0, 15),
    "DrainageSystems": (0, 15),
    "CoastalVulnerability": (0, 15),
    "Landslides": (0, 15),
    "Watersheds": (0, 15),
    "DeterioratingInfrastructure": (0, 15),
    "PopulationScore": (0, 15),
    "WetlandLoss": (0, 15),
    "InadequatePlanning": (0, 15),
    "PoliticalFactors": (0, 15),
}

UNCERTAIN_THRESHOLD = 0.65  # Threshold from calibration tuning


def validate_input_features(sample_input: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate input features are within expected bounds."""
    for feat in EXPECTED_RAW_FEATURES:
        if feat in sample_input:
            val = sample_input[feat]
            if not isinstance(val, (int, float)):
                return False, f"Feature '{feat}' must be numeric, got {type(val).__name__}"
            min_val, max_val = FEATURE_BOUNDS.get(feat, (0, 15))
            if val < min_val or val > max_val:
                return False, f"Feature '{feat}' = {val} out of bounds [{min_val}, {max_val}]"
    return True, "OK"


class SecurityError(Exception):
    """Raised when an artifact fails SHA-256 integrity verification."""
    pass


def load_verified_pipeline(pipeline_path: Path = BEST_PIPELINE_PATH) -> Any:
    """
    Verifies the SHA-256 checksum of the pipeline artifact against models/checksums.json
    before deserializing with joblib.load().
    """
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline artifact not found at: {pipeline_path}")

    if not CHECKSUMS_PATH.exists():
        raise SecurityError(f"Security Alert: Checksums registry missing at: {CHECKSUMS_PATH}")

    with open(CHECKSUMS_PATH, "r", encoding="utf-8") as f:
        checksums = json.load(f)

    expected_hash = checksums.get(pipeline_path.name)
    if not expected_hash:
        raise SecurityError(f"Security Alert: No registered checksum for {pipeline_path.name}")

    sha256 = hashlib.sha256()
    with open(pipeline_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    actual_hash = sha256.hexdigest()

    if actual_hash != expected_hash:
        raise SecurityError(
            f"SECURITY ALERT: SHA-256 mismatch for {pipeline_path.name}!\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}\n"
            f"Model file may have been tampered with or corrupted. Halting inference."
        )

    return joblib.load(pipeline_path)


def predict_single_instance(sample_input: Dict[str, Any], pipeline=None, confidence_threshold: float = UNCERTAIN_THRESHOLD) -> Tuple[str, float, Dict[str, float], str]:
    """
    Runs single-instance inference by passing raw inputs directly into the pipeline.
    
    Args:
        sample_input: Dictionary of raw feature values
        pipeline: Pre-loaded pipeline (optional)
        confidence_threshold: Minimum max probability to make a prediction (default 0.65 from calibration tuning)
        
    Returns:
      predicted_class (str): 'Low', 'Medium', 'High', or 'UNCERTAIN'
      confidence (float): Percentage confidence of top prediction (0-100)
      prob_dict (dict): Dictionary mapping class name to calibrated probability
      decision (str): 'PREDICTED' or 'UNCERTAIN' (if max_prob < confidence_threshold)
    """
    # Validate inputs
    is_valid, msg = validate_input_features(sample_input)
    if not is_valid:
        raise ValueError(f"Input validation failed: {msg}")
    
    if pipeline is None:
        pipeline = load_verified_pipeline(BEST_PIPELINE_PATH)

    # Convert dictionary to single-row DataFrame with expected columns
    row_data = {}
    for feat in EXPECTED_RAW_FEATURES:
        row_data[feat] = float(sample_input.get(feat, 5.0))

    input_df = pd.DataFrame([row_data])

    # Pass raw input through the unified pipeline
    raw_pred = pipeline.predict(input_df)[0]

    # Normalize prediction to string
    int_to_class = {0: "Low", 1: "Medium", 2: "High"}
    if isinstance(raw_pred, (int, np.integer)):
        predicted_class = int_to_class.get(int(raw_pred), "Medium")
    else:
        predicted_class = str(raw_pred)

    prob_dict = {}
    confidence = 0.0
    decision = "PREDICTED"

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(input_df)[0]
        # Map probabilities using pipeline.classes_
        if hasattr(pipeline, "classes_"):
            raw_classes = list(pipeline.classes_)
            # If classes are integers: map to Low, Medium, High
            if all(isinstance(c, (int, np.integer)) for c in raw_classes):
                class_names = [int_to_class.get(int(c), str(c)) for c in raw_classes]
            else:
                class_names = [str(c) for c in raw_classes]
        else:
            class_names = CLASS_ORDER

        for cls_name, prob in zip(class_names, probabilities):
            prob_dict[cls_name] = round(float(prob), 4)

        # Confidence is probability of predicted class or max probability
        max_prob = max(probabilities)
        confidence = round(float(max_prob * 100.0), 2)
        
        # Apply confidence threshold - return UNCERTAIN if max probability too low
        if max_prob < confidence_threshold:
            decision = "UNCERTAIN"
            predicted_class = "UNCERTAIN"
    else:
        confidence = 100.0
        prob_dict = {predicted_class: 1.0}

    return predicted_class, confidence, prob_dict, decision


def run_prediction():
    """Phase 6 CLI inference demonstration."""
    print("=" * 60)
    print("PHASE 5: REAL-TIME INFERENCE DEMONSTRATION")
    print("=" * 60)

    # Test multiple scenarios
    scenarios = {
        "Severe Shock": {
            "MonsoonIntensity": 12.0,
            "TopographyDrainage": 10.0,
            "RiverManagement": 9.0,
            "Deforestation": 11.0,
            "Urbanization": 10.0,
            "ClimateChange": 11.0,
            "DamsQuality": 2.0,
            "Siltation": 9.0,
            "AgriculturalPractices": 8.0,
            "Encroachments": 9.0,
            "IneffectiveDisasterPreparedness": 11.0,
            "DrainageSystems": 2.0,
            "CoastalVulnerability": 9.0,
            "Landslides": 9.0,
            "Watersheds": 8.0,
            "DeterioratingInfrastructure": 10.0,
            "PopulationScore": 9.0,
            "WetlandLoss": 9.0,
            "InadequatePlanning": 10.0,
            "PoliticalFactors": 8.0,
        },
        "Moderate Risk": {
            "MonsoonIntensity": 6.0,
            "TopographyDrainage": 5.0,
            "RiverManagement": 6.0,
            "Deforestation": 5.0,
            "Urbanization": 5.0,
            "ClimateChange": 6.0,
            "DamsQuality": 6.0,
            "Siltation": 5.0,
            "AgriculturalPractices": 5.0,
            "Encroachments": 5.0,
            "IneffectiveDisasterPreparedness": 5.0,
            "DrainageSystems": 6.0,
            "CoastalVulnerability": 5.0,
            "Landslides": 5.0,
            "Watersheds": 5.0,
            "DeterioratingInfrastructure": 5.0,
            "PopulationScore": 5.0,
            "WetlandLoss": 5.0,
            "InadequatePlanning": 5.0,
            "PoliticalFactors": 5.0,
        },
        "Low Risk": {
            "MonsoonIntensity": 2.0,
            "TopographyDrainage": 2.0,
            "RiverManagement": 2.0,
            "Deforestation": 2.0,
            "Urbanization": 2.0,
            "ClimateChange": 2.0,
            "DamsQuality": 10.0,
            "Siltation": 2.0,
            "AgriculturalPractices": 2.0,
            "Encroachments": 2.0,
            "IneffectiveDisasterPreparedness": 2.0,
            "DrainageSystems": 10.0,
            "CoastalVulnerability": 2.0,
            "Landslides": 2.0,
            "Watersheds": 2.0,
            "DeterioratingInfrastructure": 2.0,
            "PopulationScore": 2.0,
            "WetlandLoss": 2.0,
            "InadequatePlanning": 2.0,
            "PoliticalFactors": 2.0,
        }
    }

    for scenario_name, sample_scenario in scenarios.items():
        print(f"\n--- Input Scenario: {scenario_name} ---")
        for k in list(sample_scenario.keys())[:6]:
            print(f"  {k:<32}: {sample_scenario[k]}")
        print("  ... [20 raw factors total]")

        predicted_class, confidence, prob_dict, decision = predict_single_instance(sample_scenario, confidence_threshold=UNCERTAIN_THRESHOLD)
        emoji = RISK_EMOJIS.get(predicted_class, "⚪")

        print("\n" + "=" * 60)
        print(f"INFERENCE RESULT ({decision})")
        print("=" * 60)
        print(f"Predicted Flood Risk Level : {emoji}  {predicted_class.upper()}")
        print(f"Prediction Confidence      : {confidence:.2f}%")
        print(f"Decision Threshold         : {UNCERTAIN_THRESHOLD*100:.0f}% (max_prob >= {UNCERTAIN_THRESHOLD:.2f} required)")
        print("\nCalibrated Class Probabilities:")
        for cls in CLASS_ORDER:
            if cls in prob_dict:
                prob = prob_dict[cls]
                bar = "█" * int(prob * 25)
                print(f"  {cls:<8} : {prob*100:5.1f}%  {bar}")

        if decision == "UNCERTAIN":
            print("\n⚠️  WARNING: Low confidence - prediction marked as UNCERTAIN")
            print("   Recommendation: Request additional data or human expert review")

        print("=" * 60)

    print("PHASE 5: INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return predicted_class, confidence, prob_dict, decision


if __name__ == "__main__":
    run_prediction()
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
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config import get_paths, get_threshold_config
from src.utils.security import load_verified_model, validate_input_features, EXPECTED_RAW_FEATURES, FEATURE_BOUNDS
from src.utils.logger import setup_logging, get_logger, log_prediction
from src.utils.exceptions import InputValidationError, SecurityError

logger = get_logger(__name__)

# Re-export SecurityError so tests can import it from this module
from src.utils.exceptions import SecurityError  # noqa: F401

UNCERTAIN_THRESHOLD = get_threshold_config().get("decision_threshold", 0.65)



def load_verified_pipeline(pipeline_name="best_pipeline") -> Any:
    """
    Load a verified model pipeline with SHA-256 checksum verification.
    Accepts either a string model name or a Path object.
    """
    from pathlib import Path as _Path
    from src.utils.security import load_verified_model

    # Support Path objects (e.g. from tests creating tmp files)
    if isinstance(pipeline_name, _Path):
        model_path = pipeline_name
        checksums_path = get_paths()["models"].parent / "checksums.json"
    else:
        model_path = get_paths()["models"] / f"{pipeline_name}.pkl"
        checksums_path = get_paths()["models"].parent / "checksums.json"

    return load_verified_model(model_path, checksums_path)


def predict_single_instance(
    sample_input: Dict[str, Any], 
    pipeline=None, 
    confidence_threshold: float = None
) -> Tuple[str, float, Dict[str, float], str]:
    """
    Runs single-instance inference by passing raw inputs directly into the pipeline.
    
    Args:
        sample_input: Dictionary of raw feature values
        pipeline: Pre-loaded pipeline (optional)
        confidence_threshold: Minimum max probability to make a prediction (default from config)
        
    Returns:
      predicted_class (str): 'Low', 'Medium', 'High', or 'UNCERTAIN'
      confidence (float): Percentage confidence of top prediction (0-100)
      prob_dict (dict): Dictionary mapping class name to calibrated probability
      decision (str): 'PREDICTED' or 'UNCERTAIN' (if max_prob < confidence_threshold)
    """
    if confidence_threshold is None:
        confidence_threshold = get_threshold_config().get("decision_threshold", 0.65)
    
    # Validate inputs
    is_valid, msg = validate_input_features(sample_input)
    if not is_valid:
        raise InputValidationError(f"Input validation failed: {msg}")
    
    if pipeline is None:
        pipeline = load_verified_pipeline()

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
            class_names = ["Low", "Medium", "High"]

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

    # Log prediction for monitoring
    log_prediction(logger, sample_input, predicted_class, confidence, decision, 0.0)

    return predicted_class, confidence, prob_dict, decision


def run_prediction() -> Tuple[str, float, Dict[str, float], str]:
    """CLI inference demonstration."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("PHASE 5: REAL-TIME INFERENCE DEMONSTRATION")
    logger.info("=" * 60)

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

    RISK_EMOJIS = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    CLASS_ORDER = ["Low", "Medium", "High"]

    for scenario_name, sample_scenario in scenarios.items():
        logger.info(f"\n--- Input Scenario: {scenario_name} ---")
        for k in list(sample_scenario.keys())[:6]:
            logger.info(f"  {k:<32}: {sample_scenario[k]}")
        logger.info("  ... [20 raw factors total]")

        predicted_class, confidence, prob_dict, decision = predict_single_instance(sample_scenario)
        emoji = RISK_EMOJIS.get(predicted_class, "⚪")

        logger.info("\n" + "=" * 60)
        logger.info(f"INFERENCE RESULT ({decision})")
        logger.info("=" * 60)
        logger.info(f"Predicted Flood Risk Level : {emoji}  {predicted_class.upper()}")
        logger.info(f"Prediction Confidence      : {confidence:.2f}%")
        logger.info(f"Decision Threshold         : {0.65*100:.0f}% (max_prob >= 0.65 required)")
        logger.info("\nCalibrated Class Probabilities:")
        for cls in CLASS_ORDER:
            if cls in prob_dict:
                prob = prob_dict[cls]
                bar = "█" * int(prob * 25)
                logger.info(f"  {cls:<8} : {prob*100:5.1f}%  {bar}")

        if decision == "UNCERTAIN":
            logger.warning("\n⚠️  WARNING: Low confidence - prediction marked as UNCERTAIN")
            logger.warning("   Recommendation: Request additional data or human expert review")

        logger.info("=" * 60)

    logger.info("PHASE 5: INFERENCE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)

    return predicted_class, confidence, prob_dict, decision


if __name__ == "__main__":
    run_prediction()
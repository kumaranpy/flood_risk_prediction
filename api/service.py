"""
Prediction service for Flood Risk Prediction API.
Loads model, validates input, returns calibrated predictions.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import get_paths
from src.models.predict import (
    load_verified_pipeline,
    predict_single_instance,
)
from src.utils.security import EXPECTED_RAW_FEATURES, validate_input_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Global pipeline cache
_pipeline = None


def get_pipeline() -> Any:
    """Get or load the verified pipeline (cached)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = load_verified_pipeline()
    return _pipeline


def predict_service(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction service function.
    
    Args:
        input_dict: Dictionary of raw feature values
        
    Returns:
        Dictionary with prediction results matching PredictionResponse schema
    """
    # Validate inputs
    is_valid, msg = validate_input_features(input_dict)
    if not is_valid:
        from src.utils.exceptions import InputValidationError
        raise InputValidationError(msg)
    
    # Get pipeline
    pipeline = get_pipeline()
    
    # Run prediction
    predicted_class, confidence, prob_dict, decision = predict_single_instance(
        input_dict, pipeline=pipeline
    )
    
    # Get model version
    registry_path = get_paths()["models"].parent / "registry.json"
    model_version = "unknown"
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)
        model_version = registry.get("model_version", "unknown")
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": prob_dict,
        "decision": decision,
        "model_version": model_version,
    }


def validate_and_predict(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate input and return prediction (for API endpoints).
    
    Raises:
        InputValidationError: If input validation fails
    """
    return predict_service(input_data)


if __name__ == "__main__":
    # Test service
    import json
    test_input = {
        "MonsoonIntensity": 10.0,
        "TopographyDrainage": 8.0,
        "RiverManagement": 7.0,
        "Deforestation": 8.0,
        "Urbanization": 9.0,
        "ClimateChange": 9.0,
        "DamsQuality": 4.0,
        "Siltation": 8.0,
        "AgriculturalPractices": 6.0,
        "Encroachments": 8.0,
        "IneffectiveDisasterPreparedness": 9.0,
        "DrainageSystems": 5.0,
        "CoastalVulnerability": 7.0,
        "Landslides": 6.0,
        "Watersheds": 7.0,
        "DeterioratingInfrastructure": 8.0,
        "PopulationScore": 8.0,
        "WetlandLoss": 7.0,
        "InadequatePlanning": 8.0,
        "PoliticalFactors": 6.0,
    }
    result = predict_service(test_input)
    logger.info(json.dumps(result, indent=2))
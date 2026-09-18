"""
Monitoring logger for Flood Risk Prediction.
Logs predictions with structured metadata.
"""

import logging
import json
import hashlib
from datetime import datetime
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger("flood_risk.monitoring")


def log_prediction(
    input_dict: dict,
    predicted_class: str,
    confidence: float,
    decision: str,
    latency_ms: float
) -> None:
    """
    Log a prediction with structured metadata.
    
    Args:
        input_dict: Input features
        predicted_class: Predicted risk class
        confidence: Prediction confidence (0-100)
        decision: "PREDICTED" or "UNCERTAIN"
        latency_ms: Inference latency in milliseconds
    """
    # Create a hash of input for traceability
    input_str = json.dumps(input_dict, sort_keys=True)
    input_hash = hashlib.sha256(input_str.encode()).hexdigest()[:16]
    
    extra = {
        "input_hash": input_hash,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "decision": decision,
        "latency_ms": latency_ms
    }
    
    logger.info(
        f"Prediction: {predicted_class} (conf={confidence:.1f}%, decision={decision}, latency={latency_ms:.1f}ms)",
        extra=extra
    )


def get_monitoring_logger() -> logging.Logger:
    """Get monitoring-specific logger."""
    return logging.getLogger("flood_risk.monitoring")
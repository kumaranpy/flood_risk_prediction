"""
Logging configuration for Flood Risk Prediction.
Structured logging with JSON output option.
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

from src.config import get_paths_config, get_monitoring_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, "input_hash"):
            log_obj["input_hash"] = record.input_hash
        if hasattr(record, "predicted_class"):
            log_obj["predicted_class"] = record.predicted_class
        if hasattr(record, "confidence"):
            log_obj["confidence"] = record.confidence
        if hasattr(record, "decision"):
            log_obj["decision"] = record.decision
        if hasattr(record, "latency_ms"):
            log_obj["latency_ms"] = record.latency_ms
        
        return json.dumps(log_obj, default=str)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    json_format: bool = False
) -> logging.Logger:
    """
    Set up structured logging for the pipeline.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        json_format: Whether to use JSON formatting
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("flood_risk")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(module)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
    
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(level)
        
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(levelname)s - %(module)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
            )
        
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "flood_risk") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def log_prediction(
    logger: logging.Logger,
    input_dict: dict,
    predicted_class: str,
    confidence: float,
    decision: str,
    latency_ms: float
) -> None:
    """
    Log a prediction with structured metadata.
    
    Args:
        logger: Logger instance
        input_dict: Input features
        predicted_class: Predicted risk class
        confidence: Prediction confidence (0-100)
        decision: "PREDICTED" or "UNCERTAIN"
        latency_ms: Inference latency in milliseconds
    """
    # Create a hash of input for traceability (without sensitive data)
    import hashlib
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
    return get_logger("flood_risk.monitoring")


if __name__ == "__main__":
    # Test logging setup
    logger = setup_logging(level=logging.INFO, json_format=True)
    logger.info("Test log message")
    log_prediction(logger, {"test": 1.0}, "High", 95.5, "PREDICTED", 12.3)
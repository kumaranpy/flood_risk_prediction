"""
Centralized exception classes for Flood Risk Prediction pipeline.
"""

class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class DataLeakageError(PipelineError):
    """Raised when data leakage is detected."""
    pass


class SecurityError(PipelineError):
    """Raised when security verification fails (e.g., SHA-256 mismatch)."""
    pass


class ModelNotFoundError(PipelineError):
    """Raised when model artifact is not found."""
    pass


class DriftDetectedError(PipelineError):
    """Raised when significant distribution drift is detected."""
    pass


class InputValidationError(PipelineError):
    """Raised when input validation fails."""
    pass


class CalibrationError(PipelineError):
    """Raised when model calibration fails."""
    pass


class FeatureEngineeringError(PipelineError):
    """Raised when feature engineering fails."""
    pass


class ConfigurationError(PipelineError):
    """Raised when configuration is invalid or missing."""
    pass
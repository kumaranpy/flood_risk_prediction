"""
Security utilities for Flood Risk Prediction.
SHA-256 checksum verification and secure model loading.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from src.utils.exceptions import SecurityError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Feature bounds for input validation (based on training data 1st/99th percentiles)
FEATURE_BOUNDS = {
    "MonsoonIntensity": (0.0, 15.0),
    "TopographyDrainage": (0.0, 15.0),
    "RiverManagement": (0.0, 15.0),
    "Deforestation": (0.0, 15.0),
    "Urbanization": (0.0, 15.0),
    "ClimateChange": (0.0, 15.0),
    "DamsQuality": (0.0, 15.0),
    "Siltation": (0.0, 15.0),
    "AgriculturalPractices": (0.0, 15.0),
    "Encroachments": (0.0, 15.0),
    "IneffectiveDisasterPreparedness": (0.0, 15.0),
    "DrainageSystems": (0.0, 15.0),
    "CoastalVulnerability": (0.0, 15.0),
    "Landslides": (0.0, 15.0),
    "Watersheds": (0.0, 15.0),
    "DeterioratingInfrastructure": (0.0, 15.0),
    "PopulationScore": (0.0, 15.0),
    "WetlandLoss": (0.0, 15.0),
    "InadequatePlanning": (0.0, 15.0),
    "PoliticalFactors": (0.0, 15.0),
}

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


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_checksum(filepath: Path, checksums_path: Path) -> bool:
    """
    Verify SHA-256 checksum of a file against checksums registry.
    
    Args:
        filepath: Path to the file to verify
        checksums_path: Path to checksums.json registry
        
    Returns:
        True if verification passes
        
    Raises:
        SecurityError: If verification fails
    """
    if not filepath.exists():
        raise SecurityError(f"File not found: {filepath}")
    
    if not checksums_path.exists():
        raise SecurityError(f"Checksums registry missing: {checksums_path}")
    
    with open(checksums_path, "r", encoding="utf-8") as f:
        checksums = json.load(f)
    
    expected_hash = checksums.get(filepath.name)
    if not expected_hash:
        raise SecurityError(f"No registered checksum for: {filepath.name}")
    
    actual_hash = compute_file_sha256(filepath)
    
    if actual_hash != expected_hash:
        raise SecurityError(
            f"SHA-256 mismatch for {filepath.name}!\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}\n"
            f"File may have been tampered with or corrupted."
        )
    
    return True


def load_verified_model(model_path: Path, checksums_path: Path) -> Any:
    """
    Load a model artifact after verifying its SHA-256 checksum.
    
    Args:
        model_path: Path to the model .pkl file
        checksums_path: Path to checksums.json registry
        
    Returns:
        Loaded model object
        
    Raises:
        SecurityError: If verification fails
        FileNotFoundError: If model file doesn't exist
    """
    verify_checksum(model_path, checksums_path)
    
    import joblib
    return joblib.load(model_path)


def update_checksums(filepaths: list[Path], checksums_path: Path) -> dict:
    """
    Update checksums.json with SHA-256 hashes of given files.
    
    Args:
        filepaths: List of file paths to hash
        checksums_path: Path to checksums.json registry
        
    Returns:
        Dictionary of filename -> hash
    """
    checksums = {}
    if checksums_path.exists():
        try:
            with open(checksums_path, "r", encoding="utf-8") as f:
                checksums = json.load(f)
        except Exception:
            checksums = {}
    
    for path in filepaths:
        if path.exists():
            checksums[path.name] = compute_file_sha256(path)
    
    with open(checksums_path, "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2)
    
    return checksums


if __name__ == "__main__":
    # Test security utilities
    from pathlib import Path
    test_path = Path(__file__)
    print(f"SHA-256 of {test_path.name}: {compute_file_sha256(test_path)}")
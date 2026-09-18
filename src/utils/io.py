"""
I/O utilities for Flood Risk Prediction.
File operations, model artifact management.
"""

import json
import joblib
import hashlib
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.security import compute_file_sha256, update_checksums
from src.utils.exceptions import ModelNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_model_dir() -> Path:
    """Get models directory."""
    return get_paths()["models"]


def get_splits_dir() -> Path:
    """Get data splits directory."""
    return get_paths()["splits"]


def get_outputs_dir() -> Path:
    """Get outputs directory."""
    return get_paths()["outputs"]


def get_plots_dir() -> Path:
    """Get plots directory."""
    return get_paths()["plots"]


def get_reports_dir() -> Path:
    """Get reports directory."""
    return get_paths()["reports"]


def get_raw_data_dir() -> Path:
    """Get raw data directory."""
    return get_paths()["raw_data"]


def get_paths() -> Dict[str, Path]:
    """Get all configured paths with resolved absolute paths."""
    from src.config import get_paths_config
    return get_paths_config()


def load_model(model_name: str = "best_pipeline") -> Any:
    """
    Load a model artifact with SHA-256 verification.
    
    Args:
        model_name: Name of model file (without .pkl extension)
        
    Returns:
        Loaded model object
    """
    model_dir = get_model_dir()
    model_path = model_dir / f"{model_name}.pkl"
    checksums_path = model_dir.parent / "checksums.json"
    
    if not model_path.exists():
        raise ModelNotFoundError(f"Model not found: {model_path}")
    
    from src.utils.security import load_verified_model
    return load_verified_model(model_path, checksums_path)


def save_model(model: Any, model_name: str = "best_pipeline") -> Path:
    """
    Save model artifact and update checksums.
    
    Args:
        model: Model object to save
        model_name: Name for the model file (without .pkl)
        
    Returns:
        Path to saved model file
    """
    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    
    # Update checksums
    checksums_path = model_dir.parent / "checksums.json"
    update_checksums([model_path], checksums_path)
    
    return model_path


def save_artifact(obj: Any, filename: str, subdir: str = "") -> Path:
    """
    Save any artifact (JSON, CSV, etc.) to outputs directory.
    
    Args:
        obj: Object to save (must be JSON serializable or have .to_csv())
        filename: Output filename
        subdir: Subdirectory under outputs/
        
    Returns:
        Path to saved file
    """
    output_dir = get_outputs_dir()
    if subdir:
        output_dir = output_dir / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    
    if filename.endswith(".json"):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
    elif filename.endswith(".csv"):
        if hasattr(obj, "to_csv"):
            obj.to_csv(filepath, index=False)
        else:
            raise ValueError("Object must have .to_csv() method for CSV output")
    else:
        with open(filepath, "wb") as f:
            joblib.dump(obj, f)
    
    return filepath


def load_json(filepath: Path) -> dict:
    """Load JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, filepath: Path) -> Path:
    """Save object as JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return filepath


def load_csv(filepath: Path) -> pd.DataFrame:
    """Load CSV file."""
    import pandas as pd
    return pd.read_csv(filepath)


def save_csv(df: pd.DataFrame, filepath: Path) -> Path:
    """Save DataFrame as CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    return filepath


# Import pandas at module level for type hints
try:
    import pandas as pd
except ImportError:
    pd = None


if __name__ == "__main__":
    # Test I/O utilities
    paths = get_paths()
    print("Configured paths:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
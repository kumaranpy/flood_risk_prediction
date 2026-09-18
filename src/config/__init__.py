"""
Configuration loader for Flood Risk Prediction.
Loads all settings from src/config/config.yaml.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "config.yaml"

_config: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    global _config
    if _config is not None:
        return _config
    
    path = config_path or CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)
    
    return _config


def get_config() -> Dict[str, Any]:
    """Get loaded config (loads if not already loaded)."""
    if _config is None:
        load_config()
    return _config


def get_model_config() -> Dict[str, Any]:
    """Get model configuration."""
    return get_config().get("model", {})


def get_training_config() -> Dict[str, Any]:
    """Get training configuration."""
    return get_config().get("training", {})


def get_threshold_config() -> Dict[str, Any]:
    """Get threshold configuration."""
    return get_config().get("threshold", {})


def get_features_config() -> Dict[str, Any]:
    """Get features configuration."""
    return get_config().get("features", {})


def get_monitoring_config() -> Dict[str, Any]:
    """Get monitoring configuration."""
    return get_config().get("monitoring", {})


def get_paths_config() -> Dict[str, Any]:
    """Get paths configuration with resolved absolute paths."""
    paths = get_config().get("paths", {})
    resolved = {}
    for key, path in paths.items():
        resolved[key] = (PROJECT_ROOT / path).resolve()
    return resolved


# Alias for backward compatibility
get_paths = get_paths_config


def get_project_root() -> Path:
    """Get project root directory."""
    return PROJECT_ROOT


if __name__ == "__main__":
    # Test config loading
    cfg = load_config()
    print("Config loaded successfully:")
    import json
    print(json.dumps(cfg, indent=2))
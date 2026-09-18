"""
Feature Engineering Module for Flood Risk Prediction.

This module provides:
  1. DomainFeatureAdder: A scikit-learn compatible transformer that constructs
     domain-specific sub-index features (meteorological, infrastructure, environmental).
     CRITICAL: No global row-wise means or sums across all predictors are permitted (F-01).
  2. Helper utilities for single-instance dictionary feature expansion during real-time inference.
  3. Feature correlation verification to guarantee no feature exceeds |r| >= 0.85 with the target.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Domain-specific feature definitions: each feature uses ONLY a restricted subset of factors
DOMAIN_FEATURE_DEFINITIONS = {
    "Environmental_Risk": ["MonsoonIntensity", "Landslides", "DeterioratingInfrastructure"],
    "Infrastructure_Vulnerability": ["Siltation", "DrainageSystems"],
    "Anthropogenic_Pressure": ["Urbanization", "Deforestation", "Encroachments", "WetlandLoss"],
    "Hydrometeorological_Risk": ["MonsoonIntensity", "ClimateChange", "CoastalVulnerability"],
}

# Key interaction features for flood prediction (domain-driven)
INTERACTION_FEATURES = [
    ("MonsoonIntensity", "Urbanization"),
    ("Deforestation", "RiverManagement"),
    ("ClimateChange", "DamsQuality"),
    ("Siltation", "AgriculturalPractices"),
    ("TopographyDrainage", "MonsoonIntensity"),
]


def compute_domain_features_dict(input_dict: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]]:
    """
    Computes domain-specific sub-index features for a single input dictionary.
    Used for real-time inference to ensure parity with batch transformations.
    Does NOT compute any global row-wise sums or averages.
    """
    output = dict(input_dict)

    # Environmental Risk
    env_cols = [c for c in DOMAIN_FEATURE_DEFINITIONS["Environmental_Risk"] if c in input_dict]
    if env_cols:
        output["Environmental_Risk"] = float(np.mean([input_dict[c] for c in env_cols]))

    # Infrastructure Vulnerability
    infra_cols = [c for c in DOMAIN_FEATURE_DEFINITIONS["Infrastructure_Vulnerability"] if c in input_dict]
    if infra_cols:
        output["Infrastructure_Vulnerability"] = float(np.mean([input_dict[c] for c in infra_cols]))

    # Anthropogenic Pressure
    anthro_cols = [c for c in DOMAIN_FEATURE_DEFINITIONS["Anthropogenic_Pressure"] if c in input_dict]
    if anthro_cols:
        output["Anthropogenic_Pressure"] = float(np.mean([input_dict[c] for c in anthro_cols]))

    # Hydrometeorological Risk
    hydro_cols = [c for c in DOMAIN_FEATURE_DEFINITIONS["Hydrometeorological_Risk"] if c in input_dict]
    if hydro_cols:
        output["Hydrometeorological_Risk"] = float(np.mean([input_dict[c] for c in hydro_cols]))

    # Domain-driven interaction features
    for feat1, feat2 in INTERACTION_FEATURES:
        if feat1 in input_dict and feat2 in input_dict:
            output[f"{feat1}_x_{feat2}"] = float(input_dict[feat1] * input_dict[feat2])

    # Ratio features (invariant to scale shifts) - key for distribution shift robustness
    if "MonsoonIntensity" in input_dict and "TopographyDrainage" in input_dict:
        output["Water_Stress"] = float(input_dict["MonsoonIntensity"] / (input_dict["TopographyDrainage"] + 1e-6))
    
    if "Urbanization" in input_dict and "DamsQuality" in input_dict:
        output["Infra_Gap"] = float(input_dict["Urbanization"] / (input_dict["DamsQuality"] + 1e-6))
    
    if "Deforestation" in input_dict and "ClimateChange" in input_dict:
        output["Eco_Damage"] = float(input_dict["Deforestation"] * input_dict["ClimateChange"])
    
    if "Siltation" in input_dict and "RiverManagement" in input_dict:
        output["Siltation_Pressure"] = float(input_dict["Siltation"] / (input_dict["RiverManagement"] + 1e-6))
    
    if "IneffectiveDisasterPreparedness" in input_dict and "DrainageSystems" in input_dict:
        output["Preparedness_Deficit"] = float(input_dict["IneffectiveDisasterPreparedness"] / (input_dict["DrainageSystems"] + 1e-6))

    return output


class DomainFeatureAdder(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that computes domain-specific sub-index features.
    
    Adheres strictly to F-01:
      - Does NOT compute any global row-wise aggregations (e.g. Aggregate_Hazard_Index).
      - Combines only logically bounded sub-domain predictors.
    """

    def __init__(self, include_all_subdomains: bool = True):
        self.include_all_subdomains = include_all_subdomains
        self.feature_names_in_: Optional[List[str]] = None
        self.output_features_: Optional[List[str]] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        elif hasattr(X, "shape") and self.feature_names_in_ is None:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Transforms input X into a DataFrame containing original columns plus domain features.
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, (np.ndarray, list)):
            cols = self.feature_names_in_ if self.feature_names_in_ is not None else [f"feature_{i}" for i in range(np.asarray(X).shape[1])]
            df = pd.DataFrame(X, columns=cols)
        else:
            raise TypeError(f"Unsupported input type for DomainFeatureAdder: {type(X)}")

        # Ensure no accidental global aggregator exists
        if "Aggregate_Hazard_Index" in df.columns:
            df = df.drop(columns=["Aggregate_Hazard_Index"])

        # Compute domain-specific features
        # 1. Environmental Risk
        env_cols = [c for c in DOMAIN_FEATURE_DEFINITIONS["Environmental_Risk"] if c in df.columns]
        if env_cols:
            df["Environmental_Risk"] = df[env_cols].mean(axis=1)

        # 2. Infrastructure Vulnerability
        infra_cols = [c for c in DOMAIN_FEATURE_DEFINITIONS["Infrastructure_Vulnerability"] if c in df.columns]
        if infra_cols:
            df["Infrastructure_Vulnerability"] = df[infra_cols].mean(axis=1)

        # 3. Anthropogenic Pressure
        anthro_cols = [c for c in DOMAIN_FEATURE_DEFINITIONS["Anthropogenic_Pressure"] if c in df.columns]
        if anthro_cols:
            df["Anthropogenic_Pressure"] = df[anthro_cols].mean(axis=1)

        # 4. Hydrometeorological Risk
        hydro_cols = [c for c in DOMAIN_FEATURE_DEFINITIONS["Hydrometeorological_Risk"] if c in df.columns]
        if hydro_cols:
            df["Hydrometeorological_Risk"] = df[hydro_cols].mean(axis=1)

        # Domain-driven interaction features
        for feat1, feat2 in INTERACTION_FEATURES:
            if feat1 in df.columns and feat2 in df.columns:
                df[f"{feat1}_x_{feat2}"] = df[feat1] * df[feat2]

        # Ratio features (invariant to scale shifts) - key for distribution shift robustness
        # water stress: rainfall intensity relative to drainage capacity
        if "MonsoonIntensity" in df.columns and "TopographyDrainage" in df.columns:
            df["Water_Stress"] = df["MonsoonIntensity"] / (df["TopographyDrainage"] + 1e-6)
        
        # infrastructure gap: urbanization relative to infrastructure quality
        if "Urbanization" in df.columns and "DamsQuality" in df.columns:
            df["Infra_Gap"] = df["Urbanization"] / (df["DamsQuality"] + 1e-6)
        
        # environmental degradation: deforestation * climate change impact
        if "Deforestation" in df.columns and "ClimateChange" in df.columns:
            df["Eco_Damage"] = df["Deforestation"] * df["ClimateChange"]
        
        # siltation pressure relative to river management
        if "Siltation" in df.columns and "RiverManagement" in df.columns:
            df["Siltation_Pressure"] = df["Siltation"] / (df["RiverManagement"] + 1e-6)
        
        # disaster preparedness deficit
        if "IneffectiveDisasterPreparedness" in df.columns and "DrainageSystems" in df.columns:
            df["Preparedness_Deficit"] = df["IneffectiveDisasterPreparedness"] / (df["DrainageSystems"] + 1e-6)

        self.output_features_ = list(df.columns)
        return df

    def get_feature_names_out(self, input_features=None):
        if self.output_features_ is not None:
            return np.array(self.output_features_, dtype=object)
        if input_features is not None:
            names = list(input_features)
            for k in DOMAIN_FEATURE_DEFINITIONS.keys():
                names.append(k)
            return np.array(names, dtype=object)
        return np.array(self.feature_names_in_, dtype=object) if self.feature_names_in_ else None


def verify_no_target_proxy_leakage(df: pd.DataFrame, target_col: str, max_allowed_corr: float = 0.85) -> None:
    """
    Asserts that no predictor in df has |r| >= max_allowed_corr with target_col.
    Raises ValueError if target proxy leakage is detected.
    """
    if target_col not in df.columns:
        return

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    target_series = df[target_col]

    # If target is numeric
    if np.issubdtype(target_series.dtype, np.number):
        correlations = df[numeric_cols].apply(lambda s: s.corr(target_series)).abs()
    else:
        # If categorical, map or encode temporarily for Pearson check
        codes = pd.Series(pd.factorize(target_series)[0], index=df.index)
        correlations = df[numeric_cols].apply(lambda s: s.corr(codes)).abs()

    violators = correlations[correlations >= max_allowed_corr]
    if not violators.empty:
        raise ValueError(
            f"TARGET PROXY LEAKAGE DETECTED (F-01)! Features with |r| >= {max_allowed_corr}: "
            f"{violators.to_dict()}"
        )
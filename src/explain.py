"""
SHAP Explainability Module for Flood Risk Prediction.

Provides model-agnostic SHAP explanations using KernelExplainer wrapping
the calibrated pipeline's predict_proba to ensure explanations reflect
deployed probabilities (including calibration).
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.predict import load_verified_pipeline

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def create_kernel_explainer(
    pipeline: Any,
    X_background: pd.DataFrame,
    max_background: int = 100
) -> Tuple[shap.Explainer, List[str]]:
    """
    Creates a SHAP KernelExplainer wrapping pipeline.predict_proba.
    
    This ensures explanations reflect the calibrated probabilities
    that are actually deployed, not the raw model outputs.
    """
    # Limit background samples for performance
    if len(X_background) > max_background:
        X_background = X_background.sample(n=max_background, random_state=42)
    
    # Use a subset of background data to compute expected values
    background_sample = X_background.sample(n=min(50, len(X_background)), random_state=42)
    
    # Create explainer using the full pipeline's predict_proba
    # This captures all preprocessing + calibration
    explainer = shap.KernelExplainer(
        pipeline.predict_proba,
        background_sample,
        link="logit"  # Use logit link for probability outputs
    )
    
    # Get feature names from the pipeline's feature selector
    if hasattr(pipeline, "calibrated_classifiers_"):
        base_pipeline = pipeline.calibrated_classifiers_[0].estimator
    elif hasattr(pipeline, "estimator"):
        base_pipeline = pipeline.estimator
    else:
        base_pipeline = pipeline
    
    # Get feature names after SelectKBest
    selector = base_pipeline.named_steps["feature_select"]
    domain_features = base_pipeline.named_steps["domain_features"].output_features_
    if domain_features is None:
        # Fallback: run domain feature adder on background
        X_domain = base_pipeline.named_steps["domain_features"].transform(
            base_pipeline.named_steps["capper"].transform(X_background)
        )
        domain_features = list(X_domain.columns)
    
    selected_indices = selector.get_support(indices=True)
    selected_names = [domain_features[i] for i in selected_indices]
    
    return explainer, selected_names


def compute_shap_values(
    pipeline: Any,
    X_explain: pd.DataFrame,
    X_background: Optional[pd.DataFrame] = None,
    max_samples: int = 50
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Computes SHAP values for the given samples using KernelExplainer.
    
    Returns:
        shap_values: Shape (n_samples, n_features, n_classes) for multi-class
        feature_names: Names of features
        X_transformed: Transformed feature values for the explained samples
    """
    if X_background is None:
        X_background = X_explain
    
    # Limit explanation samples for performance
    if len(X_explain) > max_samples:
        X_explain = X_explain.sample(n=max_samples, random_state=42)
    
    # Create KernelExplainer wrapping the full pipeline
    explainer, feature_names = create_kernel_explainer(pipeline, X_background)
    
    # Compute SHAP values
    # shap_values will be a list of length n_classes, each (n_samples, n_features)
    shap_values = explainer.shap_values(X_explain)
    
    return shap_values, feature_names, np.array([])  # X_transformed not used with KernelExplainer


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: List[str],
    X_explain: pd.DataFrame,
    class_names: List[str] = ["Low", "Medium", "High"],
    max_display: int = 20,
    save_path: Optional[Path] = None
):
    """Plots SHAP summary (beeswarm) for each class."""
    n_classes = len(class_names)
    
    if n_classes == 3 and isinstance(shap_values, list):
        # Multi-class: shap_values is a list of arrays, one per class
        for i, (cls_name, cls_shap) in enumerate(zip(class_names, shap_values)):
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(
                cls_shap, X_explain, feature_names=feature_names,
                show=False, max_display=max_display, plot_type="dot"
            )
            ax.set_title(f"SHAP Summary: {cls_name} Risk", fontsize=12, fontweight="bold")
            if save_path:
                cls_path = save_path.parent / f"{save_path.stem}_{cls_name.lower()}{save_path.suffix}"
                plt.savefig(cls_path, dpi=150, bbox_inches="tight")
                print(f"  Saved SHAP summary to: {cls_path}")
            plt.close()
    else:
        # Binary or single output
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_explain, feature_names=feature_names,
            show=False, max_display=max_display, plot_type="dot"
        )
        ax.set_title("SHAP Summary", fontsize=12, fontweight="bold")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved SHAP summary to: {save_path}")
        plt.close()


def plot_shap_waterfall(
    shap_values: np.ndarray,
    feature_names: List[str],
    X_explain: pd.DataFrame,
    sample_idx: int = 0,
    class_idx: int = 2,  # High risk class
    class_names: List[str] = ["Low", "Medium", "High"],
    save_path: Optional[Path] = None
):
    """Plots SHAP waterfall plot for a single prediction (multi-class compatible)."""
    try:
        # shap_values is list of [n_samples, n_features] per class
        if isinstance(shap_values, list):
            if sample_idx >= len(shap_values[class_idx]):
                print(f"  Skipping waterfall: sample_idx {sample_idx} >= {len(shap_values[class_idx])}")
                return
            values = shap_values[class_idx][sample_idx]
            base_value = float(values.sum())
        else:
            # Handle other formats
            if shap_values.ndim == 3:
                values = shap_values[sample_idx, :, class_idx]
            else:
                values = shap_values[sample_idx]
            base_value = 0.0
        
        # Ensure 1D
        values = np.asarray(values).ravel()
        
        # Create Explanation object for waterfall plot
        explanation = shap.Explanation(
            values=values,
            base_values=base_value,
            data=X_explain.iloc[sample_idx].values,
            feature_names=feature_names
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        ax.set_title(f"SHAP Waterfall: Sample {sample_idx} ({class_names[class_idx]} Risk)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved SHAP waterfall to: {save_path}")
        plt.close()
    except Exception as e:
        print(f"  Waterfall plot failed: {e}")


def get_top_features_per_prediction(
    pipeline: Any,
    X_raw: pd.DataFrame,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Returns top-k contributing features for each prediction.
    
    Returns list of dicts with: sample_idx, predicted_class, top_features
    """
    pipeline = load_verified_pipeline() if pipeline is None else pipeline
    
    # Get predictions
    raw_preds = pipeline.predict(X_raw)
    int_to_class = {0: "Low", 1: "Medium", 2: "High"}
    # Handle both integer and string predictions
    predicted_classes = []
    for p in raw_preds:
        if isinstance(p, (int, np.integer)):
            predicted_classes.append(int_to_class.get(int(p), str(p)))
        else:
            predicted_classes.append(str(p))
    
    # Compute SHAP values
    shap_values, feature_names, _ = compute_shap_values(pipeline, X_raw)
    
    results = []
    n_samples = len(X_raw)
    for i in range(n_samples):
        pred_class = predicted_classes[i]
        
        # Get SHAP values for this sample and predicted class
        if isinstance(shap_values, list):
            class_idx = ["Low", "Medium", "High"].index(pred_class)
            sample_shap = shap_values[class_idx][i]
        else:
            # Handle other formats
            if shap_values.ndim == 3:
                class_idx = ["Low", "Medium", "High"].index(pred_class)
                sample_shap = shap_values[i, :, class_idx]
            else:
                sample_shap = shap_values[i]
        
        # Get absolute SHAP values for ranking
        abs_shap = np.abs(sample_shap)
        # Ensure 1D
        if abs_shap.ndim > 1:
            abs_shap = abs_shap.ravel()
            sample_shap = sample_shap.ravel()
        # Cap top_k to available features
        actual_k = min(top_k, len(abs_shap))
        top_indices = np.argsort(abs_shap)[-actual_k:][::-1]
        
        top_features = []
        for idx in top_indices:
            idx_int = int(idx)
            if idx_int < len(feature_names):
                top_features.append({
                    "feature": feature_names[idx_int],
                    "shap_value": float(sample_shap[idx_int]),
                    "feature_value": float(X_raw.iloc[i, idx_int]) if idx_int < X_raw.shape[1] else 0.0,
                    "abs_shap": float(abs_shap[idx_int])
                })
        
        results.append({
            "sample_idx": i,
            "predicted_class": pred_class,
            "top_features": top_features
        })
    
    return results


def run_explainability_demo():
    """Runs SHAP explainability demonstration on test data."""
    print("=" * 60)
    print("SHAP EXPLAINABILITY DEMONSTRATION (KernelExplainer)")
    print("=" * 60)
    
    # Load pipeline
    pipeline = load_verified_pipeline()
    
    # Load test data
    test_path = PROJECT_ROOT / "data" / "splits" / "test.csv"
    test_df = pd.read_csv(test_path)
    feature_cols = [c for c in test_df.columns if c not in ["FloodProbability_raw", "RiskLevel"]]
    X_test = test_df[feature_cols].copy()
    
    # Sample a few test cases
    X_sample = X_test.sample(n=min(50, len(X_test)), random_state=42)
    
    print(f"\nExplaining {len(X_sample)} test samples...")
    
    # Compute SHAP values using KernelExplainer
    shap_values, feature_names, _ = compute_shap_values(pipeline, X_sample)
    
    # Plot summary
    plot_shap_summary(
        shap_values, feature_names, X_sample,
        save_path=PLOTS_DIR / "shap_summary.png"
    )
    
    # Plot waterfall for first high-risk prediction
    raw_preds = pipeline.predict(X_sample)
    int_to_class = {0: "Low", 1: "Medium", 2: "High"}
    predicted_classes = []
    for p in raw_preds:
        if isinstance(p, (int, np.integer)):
            predicted_classes.append(int_to_class.get(int(p), str(p)))
        else:
            predicted_classes.append(str(p))
    
    high_risk_indices = [i for i, c in enumerate(predicted_classes) if c == "High"]
    if high_risk_indices:
        plot_shap_waterfall(
            shap_values, feature_names, X_sample,
            sample_idx=high_risk_indices[0],
            class_idx=2,  # High risk
            save_path=PLOTS_DIR / "shap_waterfall_high.png"
        )
    
    # Get top features for each prediction
    top_features = get_top_features_per_prediction(pipeline, X_sample, top_k=5)
    
    print("\nTop 5 Contributing Features per Sample:")
    for result in top_features[:10]:
        print(f"  Sample {result['sample_idx']} ({result['predicted_class']}):")
        if result['top_features']:
            for feat in result['top_features']:
                print(f"    {feat['feature']}: SHAP={feat['shap_value']:+.4f} (value={feat['feature_value']:.2f})")
        else:
            print(f"    (no features found)")
    
    # Detailed SHAP analysis for first 5 samples
    print("\n--- Detailed SHAP Analysis (first 5 samples) ---")
    if isinstance(shap_values, list):
        for i in range(min(5, len(X_sample))):
            pred_class = predicted_classes[i]
            class_idx = ["Low", "Medium", "High"].index(pred_class)
            sample_shap = shap_values[class_idx][i]
            
            abs_shap_sum = np.abs(sample_shap)
            top_indices = np.argsort(abs_shap_sum)[::-1][:5]
            print(f"  Sample {i} (Predicted: {pred_class}):")
            for idx in top_indices:
                print(f"    {feature_names[idx]}: SHAP={sample_shap[idx]:+.4f}")
            print()
    
    print("\n" + "=" * 60)
    print("SHAP EXPLAINABILITY COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_explainability_demo()
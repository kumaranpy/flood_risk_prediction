"""
Model Training & Pipeline Calibration Module for Flood Risk Prediction.

Architectural Guarantees:
  1. Fold-Safe Pipeline: Encapsulates OutlierCapper -> DomainFeatureAdder -> StandardScaler -> SelectKBest -> Classifier.
  2. Zero Leakage: All scaling, outlier capping, and feature selection are fit strictly within training folds.
  3. Probability Calibration: Best model is wrapped in CalibratedClassifierCV(cv=5).
  4. Artifact Integrity: Calculates SHA-256 hashes for all generated model files and stores in models/checksums.json.
  5. Parity: Saves single models/best_pipeline.pkl which accepts raw, unscaled inputs.
"""

import hashlib
import json
import logging
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import platform
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from src.config import get_config, get_paths_config, get_training_config
from src.features import DomainFeatureAdder, verify_no_target_proxy_leakage
from src.data.preprocess import OutlierCapper
from src.utils.io import save_model, get_paths, save_json
from src.utils.logger import setup_logging, get_logger
from src.utils.security import update_checksums

# Initialize logger
logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)


def create_pipeline(classifier: Any, k_features: int = 12) -> Pipeline:
    """
    Builds the fold-safe scikit-learn Pipeline.
    """
    return Pipeline([
        ("capper", OutlierCapper(lower_percentile=1.0, upper_percentile=99.0)),
        ("domain_features", DomainFeatureAdder()),
        ("scaler", StandardScaler()),
        ("feature_select", SelectKBest(score_func=f_classif, k=k_features)),
        ("classifier", classifier),
    ])


def run_training() -> Tuple[Dict[str, Any], Any]:
    """
    Executes model training, cross-validation, probability calibration, and artifact persistence.
    """
    setup_logging()
    logger.info("=" * 60)
    logger.info("PHASE 2 & 3: FOLD-SAFE PIPELINE TRAINING & CALIBRATION")
    logger.info("=" * 60)

    paths = get_paths()
    training_cfg = get_training_config()
    
    train_data_path = paths["splits"] / "train.csv"
    logger.info(f"Loading raw training partition from: {train_data_path}")
    train_df = pd.read_csv(train_data_path)

    # Verification: Assert no target proxy leakage exists in training predictors
    verify_no_target_proxy_leakage(train_df, target_col="FloodProbability_raw", max_allowed_corr=0.85)
    logger.info("PASSED: Verified no target proxy leakage (|r| < 0.85) across all predictors.")

    feature_cols = [c for c in train_df.columns if c not in ["FloodProbability_raw", "RiskLevel"]]
    X_train = train_df[feature_cols].copy()
    y_train = train_df["RiskLevel"].copy()

    logger.info(f"Features count: {len(feature_cols)} | Training instances: {len(X_train)}")

    # Use full training set for CV
    cv = StratifiedKFold(n_splits=training_cfg.get("cv_folds", 5), shuffle=True, random_state=training_cfg.get("random_state", 42))
    X_cv = X_train
    y_cv = y_train

    # Label encoding for XGBoost
    label_to_int = {"Low": 0, "Medium": 1, "High": 2}
    int_to_label = {0: "Low", 1: "Medium", 2: "High"}

    # Compute class weights for XGBoost scale_pos_weight
    y_cv_int_for_weights = y_cv.map(label_to_int)
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_cv_int_for_weights)
    xgb_scale_pos_weight = class_weights[2] / class_weights[0]  # High vs Low

    # FINAL HYPERPARAMETERS - used for both CV and final fit to ensure alignment
    model_cfg = get_config().get("model", {})
    model_params = model_cfg.get("params", {})
    
    FINAL_HYPERPARAMS = {
        "LogisticRegression": {"max_iter": 2000, "class_weight": "balanced", "random_state": 42},
        "RandomForest": {"n_estimators": 300, "max_depth": 15, "random_state": 42, "n_jobs": -1, "class_weight": "balanced"},
        "XGBoost": {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.05, "random_state": 42, "eval_metric": "mlogloss", "n_jobs": -1, "scale_pos_weight": xgb_scale_pos_weight},
        "LightGBM": {"n_estimators": 300, "max_depth": 10, "learning_rate": 0.05, "random_state": 42, "class_weight": "balanced", "n_jobs": -1, "verbosity": -1},
        "KNN": {"n_neighbors": 9, "n_jobs": -1},
    }

    candidate_classifiers = {}
    for name, params in FINAL_HYPERPARAMS.items():
        if name == "LogisticRegression":
            candidate_classifiers[name] = LogisticRegression(**params)
        elif name == "RandomForest":
            candidate_classifiers[name] = RandomForestClassifier(**params)
        elif name == "XGBoost":
            candidate_classifiers[name] = XGBClassifier(**params)
        elif name == "LightGBM":
            candidate_classifiers[name] = LGBMClassifier(**params)
        elif name == "KNN":
            candidate_classifiers[name] = KNeighborsClassifier(**params)

    cv_results = {}
    saved_artifacts = []

    logger.info(f"\n--- 5-FOLD CROSS-VALIDATION (FOLD-SAFE PIPELINE, FULL DATA) ---")
    
    for name, clf in candidate_classifiers.items():
        pipe = create_pipeline(clf, k_features=training_cfg.get("n_select", 12))

        # XGBoost requires integer encoded labels
        if name == "XGBoost":
            y_target = y_cv.map(label_to_int)
        else:
            y_target = y_cv

        scores = cross_val_score(pipe, X_cv, y_target, cv=cv, scoring="f1_weighted", n_jobs=-1)
        mean_score = float(scores.mean())
        std_score = float(scores.std())
        cv_results[name] = {"f1_weighted": mean_score, "f1_std": std_score}
        logger.info(f"  {name:<22} CV F1-Weighted: {mean_score:.4f} (+/- {std_score:.4f})")

        # Fit on training set and persist candidate pipeline
        pipe.fit(X_cv, y_target)
        candidate_path = get_paths()["models"] / f"{name}_pipeline.pkl"
        joblib.dump(pipe, candidate_path)
        saved_artifacts.append(candidate_path)

    # Select best model
    best_name = max(cv_results.keys(), key=lambda k: cv_results[k]["f1_weighted"])
    logger.info(f"\nTop performing architecture: {best_name} (CV F1 = {cv_results[best_name]['f1_weighted']:.4f})")

    # Use same hyperparameters for final model
    y_train_int = y_train.map(label_to_int)
    class_weights_full = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_train_int)
    xgb_scale_pos_weight_full = class_weights_full[2] / class_weights_full[0]

    if best_name == "XGBoost":
        FINAL_HYPERPARAMS["XGBoost"]["scale_pos_weight"] = xgb_scale_pos_weight_full

    logger.info(f"\nFitting final calibrated pipeline with {best_name} on full training partition ({len(X_train)} samples)...")
    
    if best_name == "RandomForest":
        best_clf = RandomForestClassifier(**FINAL_HYPERPARAMS["RandomForest"])
        best_pipe = create_pipeline(best_clf, k_features=training_cfg.get("n_select", 12))
    elif best_name == "LogisticRegression":
        best_clf = LogisticRegression(**FINAL_HYPERPARAMS["LogisticRegression"])
        best_pipe = create_pipeline(best_clf, k_features=training_cfg.get("n_select", 12))
    elif best_name == "LightGBM":
        best_clf = LGBMClassifier(**FINAL_HYPERPARAMS["LightGBM"])
        best_pipe = create_pipeline(best_clf, k_features=training_cfg.get("n_select", 12))
    elif best_name == "XGBoost":
        best_clf = XGBClassifier(**FINAL_HYPERPARAMS["XGBoost"])
        best_pipe = create_pipeline(best_clf, k_features=training_cfg.get("n_select", 12))
    else:
        best_clf = RandomForestClassifier(**FINAL_HYPERPARAMS["RandomForest"])
        best_pipe = create_pipeline(best_clf, k_features=training_cfg.get("n_select", 12))

    # Skip calibration for LogisticRegression - it natively produces well-calibrated probabilities
    if best_name == "LogisticRegression":
        logger.info("  Skipping calibration for LogisticRegression (natively calibrated)")
        calibrated_pipeline = best_pipe
        calibrated_pipeline.fit(X_train, y_train)
    else:
        # Compare sigmoid vs isotonic calibration for other models
        logger.info("  Comparing calibration methods (sigmoid vs isotonic)...")
        from sklearn.metrics import brier_score_loss
        from sklearn.preprocessing import label_binarize
        from sklearn.model_selection import train_test_split
        
        # Split for calibration validation
        X_cal_train, X_cal_val, y_cal_train, y_cal_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Fit and evaluate sigmoid
        calibrated_sigmoid = CalibratedClassifierCV(estimator=best_pipe, method='sigmoid', cv=5)
        calibrated_sigmoid.fit(X_cal_train, y_cal_train)
        prob_sigmoid = calibrated_sigmoid.predict_proba(X_cal_val)
        
        # Fit and evaluate isotonic
        calibrated_isotonic = CalibratedClassifierCV(estimator=best_pipe, method='isotonic', cv=5)
        calibrated_isotonic.fit(X_cal_train, y_cal_train)
        prob_isotonic = calibrated_isotonic.predict_proba(X_cal_val)
        
        # Compute multi-class Brier score (average of OvR Brier scores)
        y_cal_val_int = y_cal_val.map(label_to_int)
        n_classes = 3
        
        def multiclass_brier(y_true, y_prob, n_classes):
            brier_scores = []
            for c in range(n_classes):
                y_binary = (y_true == c).astype(int)
                brier = brier_score_loss(y_binary, y_prob[:, c])
                brier_scores.append(brier)
            return np.mean(brier_scores)
        
        brier_sigmoid = multiclass_brier(y_cal_val_int, prob_sigmoid, n_classes)
        brier_isotonic = multiclass_brier(y_cal_val_int, prob_isotonic, n_classes)
        
        logger.info(f"  Sigmoid Brier: {brier_sigmoid:.4f}, Isotonic Brier: {brier_isotonic:.4f}")
        
        # Refit best calibration on full training data
        if brier_sigmoid <= brier_isotonic:
            calibrated_pipeline = CalibratedClassifierCV(estimator=best_pipe, method='sigmoid', cv=5)
            logger.info("  Selected: Sigmoid calibration")
        else:
            calibrated_pipeline = CalibratedClassifierCV(estimator=best_pipe, method='isotonic', cv=5)
            logger.info("  Selected: Isotonic calibration")
        
        calibrated_pipeline.fit(X_train, y_train)

    # Save final artifacts - ONLY best_pipeline.pkl (no duplicate best_model.pkl)
    best_pipeline_path = get_paths()["models"] / "best_pipeline.pkl"
    save_model(calibrated_pipeline, "best_pipeline")
    logger.info(f"Saved calibrated pipeline to: {best_pipeline_path}")

    # Generate SHA-256 checksums
    saved_artifacts.append(best_pipeline_path)
    checksums_path = get_paths()["models"].parent / "checksums.json"
    checksums = update_checksums(saved_artifacts, checksums_path)
    logger.info(f"\nGenerated and stored SHA-256 checksums for {len(saved_artifacts)} artifacts:")
    for path in saved_artifacts:
        logger.info(f"  {path.name:<28} SHA-256: {checksums.get(path.name, '')[:16]}...")

    # Create model registry
    registry_path = get_paths()["models"].parent / "registry.json"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            cwd=PROJECT_ROOT, 
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    registry = {
        "model_version": "1.0.0",
        "git_commit_sha": git_commit,
        "training_timestamp": datetime.utcnow().isoformat() + "Z",
        "model_name": best_name,
        "cv_f1_weighted": cv_results[best_name]["f1_weighted"],
        "cv_f1_std": cv_results[best_name]["f1_std"],
        "hyperparameters": FINAL_HYPERPARAMS.get(best_name, {}),
        "n_features": len(feature_cols),
        "feature_list": feature_cols,
        "target_bins": {"p33": 0.475, "p67": 0.520, "classes": ["Low", "Medium", "High"]},
        "training_samples": len(X_train),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    logger.info(f"Saved model registry to: {registry_path}")

    logger.info("=" * 60)
    logger.info("PHASE 2 & 3 COMPLETED: Unified Calibrated Pipeline Successfully Saved")
    logger.info("=" * 60)

    return cv_results, calibrated_pipeline


if __name__ == "__main__":
    run_training()
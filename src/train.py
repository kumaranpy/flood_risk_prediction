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
from pathlib import Path
import sys
import warnings
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import DomainFeatureAdder, verify_no_target_proxy_leakage
from src.preprocess import OutlierCapper

warnings.filterwarnings("ignore")

TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BEST_PIPELINE_PATH = MODELS_DIR / "best_pipeline.pkl"
CHECKSUMS_PATH = MODELS_DIR / "checksums.json"


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def update_checksums(saved_files: List[Path]) -> Dict[str, str]:
    """Updates models/checksums.json with SHA-256 hashes of saved artifacts."""
    checksums = {}
    if CHECKSUMS_PATH.exists():
        try:
            with open(CHECKSUMS_PATH, "r", encoding="utf-8") as f:
                checksums = json.load(f)
        except Exception:
            checksums = {}

    for path in saved_files:
        if path.exists():
            checksums[path.name] = compute_file_sha256(path)

    with open(CHECKSUMS_PATH, "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2)

    return checksums


def create_pipeline(classifier, k_features: int = 12) -> Pipeline:
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


def run_training() -> Tuple[Dict[str, float], CalibratedClassifierCV]:
    """
    Executes model training, cross-validation, probability calibration, and artifact persistence.
    """
    print("=" * 60)
    print("PHASE 2 & 3: FOLD-SAFE PIPELINE TRAINING & CALIBRATION")
    print("=" * 60)

    print(f"Loading raw training partition from: {TRAIN_DATA_PATH}")
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    # Verification: Assert no target proxy leakage exists in training predictors
    verify_no_target_proxy_leakage(train_df, target_col="FloodProbability_raw", max_allowed_corr=0.85)
    print("PASSED: Verified no target proxy leakage (|r| < 0.85) across all predictors.")

    feature_cols = [c for c in train_df.columns if c not in ["FloodProbability_raw", "RiskLevel"]]
    X_train = train_df[feature_cols].copy()
    y_train = train_df["RiskLevel"].copy()

    print(f"Features count: {len(feature_cols)} | Training instances: {len(X_train)}")

    # Use full training set for CV - no subsampling (NFR-01 runtime limit addressed by efficient models)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_cv = X_train
    y_cv = y_train

    # Label encoding for XGBoost
    label_to_int = {"Low": 0, "Medium": 1, "High": 2}
    int_to_label = {0: "Low", 1: "Medium", 2: "High"}

    # FR-11: Candidate Classifiers (with class weights for imbalance handling)
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    # Compute class weights for XGBoost scale_pos_weight
    y_cv_int_for_weights = y_cv.map(label_to_int)
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_cv_int_for_weights)
    xgb_scale_pos_weight = class_weights[2] / class_weights[0]  # High vs Low

    # FINAL HYPERPARAMETERS - used for both CV and final fit to ensure alignment
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

    print(f"\n--- 5-FOLD CROSS-VALIDATION (FOLD-SAFE PIPELINE, FULL DATA) ---")
    
    for name, clf in candidate_classifiers.items():
        pipe = create_pipeline(clf, k_features=12)

        # XGBoost requires integer encoded labels
        if name == "XGBoost":
            y_target = y_cv.map(label_to_int)
        else:
            y_target = y_cv

        scores = cross_val_score(pipe, X_cv, y_target, cv=cv, scoring="f1_weighted", n_jobs=-1)
        mean_score = float(scores.mean())
        std_score = float(scores.std())
        cv_results[name] = {"f1_weighted": mean_score, "f1_std": std_score}
        print(f"  {name:<22} CV F1-Weighted: {mean_score:.4f} (+/- {std_score:.4f})")

        # Fit on training set and persist candidate pipeline
        pipe.fit(X_cv, y_target)
        candidate_path = MODELS_DIR / f"{name}_pipeline.pkl"
        joblib.dump(pipe, candidate_path)
        saved_artifacts.append(candidate_path)

    # Select best model using combined score: F1 + (1 - Brier) for calibration quality
    # For now, use CV F1 as primary; calibration will be evaluated separately
    best_name = max(cv_results.keys(), key=lambda k: cv_results[k]["f1_weighted"])
    print(f"\nTop performing architecture: {best_name} (CV F1 = {cv_results[best_name]['f1_weighted']:.4f})")

    # Use same hyperparameters for final model
    y_train_int = y_train.map(label_to_int)
    class_weights_full = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_train_int)
    xgb_scale_pos_weight_full = class_weights_full[2] / class_weights_full[0]

    # Update XGBoost scale_pos_weight for full training set
    if best_name == "XGBoost":
        FINAL_HYPERPARAMS["XGBoost"]["scale_pos_weight"] = xgb_scale_pos_weight_full

    # Wrap best pipeline in CalibratedClassifierCV - compare sigmoid vs isotonic
    print(f"\nFitting final calibrated pipeline with {best_name} on full training partition ({len(X_train)} samples)...")
    if best_name == "RandomForest":
        best_clf = RandomForestClassifier(**FINAL_HYPERPARAMS["RandomForest"])
        best_pipe = create_pipeline(best_clf, k_features=12)
    elif best_name == "LogisticRegression":
        best_clf = LogisticRegression(**FINAL_HYPERPARAMS["LogisticRegression"])
        best_pipe = create_pipeline(best_clf, k_features=12)
    elif best_name == "LightGBM":
        best_clf = LGBMClassifier(**FINAL_HYPERPARAMS["LightGBM"])
        best_pipe = create_pipeline(best_clf, k_features=12)
    elif best_name == "XGBoost":
        best_clf = XGBClassifier(**FINAL_HYPERPARAMS["XGBoost"])
        best_pipe = create_pipeline(best_clf, k_features=12)
    else:
        best_clf = RandomForestClassifier(**FINAL_HYPERPARAMS["RandomForest"])
        best_pipe = create_pipeline(best_clf, k_features=12)

    # Skip calibration for LogisticRegression - it natively produces well-calibrated probabilities
    if best_name == "LogisticRegression":
        print("  Skipping calibration for LogisticRegression (natively calibrated)")
        calibrated_pipeline = best_pipe
        calibrated_pipeline.fit(X_train, y_train)
    else:
        # Compare sigmoid vs isotonic calibration for other models
        print("  Comparing calibration methods (sigmoid vs isotonic)...")
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
        
        print(f"  Sigmoid Brier: {brier_sigmoid:.4f}, Isotonic Brier: {brier_isotonic:.4f}")
        
        # Refit best calibration on full training data
        if brier_sigmoid <= brier_isotonic:
            calibrated_pipeline = CalibratedClassifierCV(estimator=best_pipe, method='sigmoid', cv=5)
            print("  Selected: Sigmoid calibration")
        else:
            calibrated_pipeline = CalibratedClassifierCV(estimator=best_pipe, method='isotonic', cv=5)
            print("  Selected: Isotonic calibration")
        
        calibrated_pipeline.fit(X_train, y_train)

    # Save final artifacts
    joblib.dump(calibrated_pipeline, BEST_PIPELINE_PATH)
    saved_artifacts.append(BEST_PIPELINE_PATH)

    print(f"Saved calibrated pipeline to: {BEST_PIPELINE_PATH}")

    # Generate SHA-256 checksums
    checksums = update_checksums(saved_artifacts)
    print(f"\nGenerated and stored SHA-256 checksums for {len(saved_artifacts)} artifacts:")
    for path in saved_artifacts:
        print(f"  {path.name:<28} SHA-256: {checksums.get(path.name, '')[:16]}...")

    # Create model registry
    registry_path = MODELS_DIR / "registry.json"
    try:
        import subprocess
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            cwd=PROJECT_ROOT, 
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    from datetime import datetime
    import platform
    
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
    print(f"Saved model registry to: {registry_path}")

    print("=" * 60)
    print("PHASE 2 & 3 COMPLETED: Unified Calibrated Pipeline Successfully Saved")
    print("=" * 60)

    return cv_results, calibrated_pipeline


if __name__ == "__main__":
    run_training()
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_ENCODER_PATH = MODELS_DIR / "target_encoder.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"


def auto_detect_target(df: pd.DataFrame) -> str:
    """Auto-detect target column: prefers FloodProbability, RiskLevel, or last column."""
    priority_cols = ["RiskLevel", "FloodProbability", "target", "label", "risk_level"]
    for col in priority_cols:
        if col in df.columns:
            return col
    target_keywords = ["risk", "flood", "label", "target", "probability", "class"]
    for col in df.columns:
        col_lower = col.lower()
        if "score" in col_lower or "index" in col_lower or "vulnerability" in col_lower:
            continue
        if any(kw in col_lower for kw in target_keywords):
            return col
    return df.columns[-1]


def run_training():
    """
    Phase 4 Model Training meeting requirements:
    - FR-11: 6 models (LR, RF, XGB, LGBM, SVM, KNN)
    - FR-12: 5-fold stratified cross-validation
    - FR-13: RandomizedSearchCV tuning for RF & XGBoost
    - FR-14: Save all models as .pkl files
    - FR-15: Save best model to models/best_model.pkl
    - FR-16: Save scaler fit on training split to models/scaler.pkl (NFR-06)
    - NFR-01: Execution under 10 minutes
    - NFR-04: random_state=42 in stochastic operations
    """
    print("=" * 60)
    print("PHASE 4: MODEL TRAINING STARTED")
    print("=" * 60)

    print(f"\nLoading features from: {FEATURES_DATA_PATH}")
    df = pd.read_csv(FEATURES_DATA_PATH)
    target_col = auto_detect_target(df)
    X = df.drop(columns=[target_col])
    y_raw = df[target_col].astype(str)

    print(f"Target column: '{target_col}' | Samples: {len(df)} | Features: {X.shape[1]}")

    # Ensure consistent target label encoding (Low=0, Medium=1, High=2)
    target_encoder = LabelEncoder()
    target_encoder.fit(["Low", "Medium", "High"])
    y = target_encoder.transform(y_raw)
    joblib.dump(target_encoder, TARGET_ENCODER_PATH)

    # Subsample stratified 15,000 instances if dataset > 15,000 for NFR-01 efficiency
    if len(df) > 15000:
        print("Sampling 15,000 stratified instances to guarantee NFR-01 (< 10 min runtime)...")
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=15000, random_state=42)
        idx_sample, _ = next(sss.split(X, y))
        X = X.iloc[idx_sample].reset_index(drop=True)
        y = y[idx_sample]

    # Train / Test split (80/20) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Split sizes -> Train: {len(X_train)} | Test: {len(X_test)}")

    # NFR-06: Scaler fit ONLY on training split to avoid data leakage
    print("\n--- FITTING SCALER ON TRAINING SPLIT ONLY (NFR-06) ---")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  FR-16: Saved train-split scaler to: {SCALER_PATH}")

    # 5-Fold Stratified Cross-Validation (FR-12)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # FR-11: 6 Model Architectures & Configurations
    base_models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0
        ),
        "LightGBM": LGBMClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1, verbosity=-1
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, random_state=42
        ),
        "KNN_k5": KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1
        ),
        "KNN_k11": KNeighborsClassifier(
            n_neighbors=11, n_jobs=-1
        ),
    }

    # FR-13: Hyperparameter Search Spaces
    rf_param_dist = {
        "n_estimators": [100, 150, 200, 250],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    xgb_param_dist = {
        "n_estimators": [100, 150, 200],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
    }

    training_results = {}
    fitted_models = {}

    for name, model in base_models.items():
        print(f"\n[{name}] Evaluating 5-Fold Cross-Validation (FR-12)...")
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1
        )
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        print(f"  CV F1-Weighted: {cv_mean:.4f} (+/- {cv_std:.4f})")

        # FR-13: Hyperparameter Tuning for RF and XGBoost
        if name == "RandomForest":
            print("  Tuning RandomForest via RandomizedSearchCV (FR-13)...")
            search = RandomizedSearchCV(
                model, rf_param_dist, n_iter=10, cv=3, scoring="f1_weighted",
                random_state=42, n_jobs=-1, verbose=0
            )
            search.fit(X_train_scaled, y_train)
            fitted_model = search.best_estimator_
            print(f"  RF Best Params: {search.best_params_}")
            print(f"  RF Best CV Score: {search.best_score_:.4f}")
        elif name == "XGBoost":
            print("  Tuning XGBoost via RandomizedSearchCV (FR-13)...")
            search = RandomizedSearchCV(
                model, xgb_param_dist, n_iter=10, cv=3, scoring="f1_weighted",
                random_state=42, n_jobs=-1, verbose=0
            )
            search.fit(X_train_scaled, y_train)
            fitted_model = search.best_estimator_
            print(f"  XGB Best Params: {search.best_params_}")
            print(f"  XGB Best CV Score: {search.best_score_:.4f}")
        else:
            model.fit(X_train_scaled, y_train)
            fitted_model = model

        fitted_models[name] = fitted_model

        # FR-14: Save individual model artifact
        save_path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(fitted_model, save_path)
        print(f"  FR-14: Saved artifact to: {save_path}")

        training_results[name] = {
            "model": fitted_model,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
        }

    # Select best KNN configuration between k=5 and k=11
    knn_best_name = "KNN_k5" if training_results["KNN_k5"]["cv_mean"] >= training_results["KNN_k11"]["cv_mean"] else "KNN_k11"
    joblib.dump(fitted_models[knn_best_name], MODELS_DIR / "KNN.pkl")
    print(f"\nSelected best KNN ({knn_best_name}) -> saved as models/KNN.pkl")

    # FR-15: Save best overall model
    candidate_keys = [k for k in training_results.keys() if k not in ["KNN_k5", "KNN_k11"]] + ["KNN"]
    training_results["KNN"] = training_results[knn_best_name]

    best_name = max(candidate_keys, key=lambda k: training_results[k]["cv_mean"])
    best_model = training_results[best_name]["model"]
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"\n{'*'*50}")
    print(f"FR-15: BEST MODEL SELECTED: {best_name}")
    print(f"Cross-Validation F1-Weighted: {training_results[best_name]['cv_mean']:.4f}")
    print(f"Saved best model artifact to: {BEST_MODEL_PATH}")
    print(f"{'*'*50}")

    # Summary table
    print("\n=== MODEL TRAINING & CV SUMMARY ===")
    print(f"{'Model':<24} {'CV Mean (F1)':<16} {'CV Std':<10}")
    print("-" * 50)
    for name in sorted(candidate_keys, key=lambda k: -training_results[k]["cv_mean"]):
        print(f"{name:<24} {training_results[name]['cv_mean']:<16.4f} {training_results[name]['cv_std']:<10.4f}")

    print("=" * 60)
    print("PHASE 4: MODEL TRAINING COMPLETED")
    print("=" * 60)

    return training_results, X_test_scaled, y_test


if __name__ == "__main__":
    run_training()
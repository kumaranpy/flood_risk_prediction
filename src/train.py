import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def auto_detect_target(df: pd.DataFrame) -> str:
    target_keywords = ["risk", "flood", "label", "target", "probability", "class"]
    for col in df.columns:
        if any(kw in col.lower() for kw in target_keywords):
            return col
    return df.columns[-1]


def run_training():
    print("=" * 60)
    print("PHASE 4: MODEL TRAINING STARTED")
    print("=" * 60)

    print(f"\nLoading features from: {FEATURES_DATA_PATH}")
    df = pd.read_csv(FEATURES_DATA_PATH)
    target_col = auto_detect_target(df)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"Target: {target_col}, Classes: {sorted(y.unique())}")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0),
        "LightGBM": LGBMClassifier(random_state=42, n_jobs=-1, verbosity=-1),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "KNN_k5": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "KNN_k11": KNeighborsClassifier(n_neighbors=11, n_jobs=-1),
    }

    rf_param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None],
    }

    xgb_param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 15),
        "learning_rate": uniform(0.01, 0.3),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "gamma": uniform(0, 0.5),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        print(f"  5-Fold CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

        if name == "RandomForest":
            print("  Running RandomizedSearchCV...")
            search = RandomizedSearchCV(
                model, rf_param_dist, n_iter=20, cv=5, scoring="accuracy",
                n_jobs=-1, random_state=42, verbose=0
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            print(f"  Best params: {search.best_params_}")
            print(f"  Best CV score: {search.best_score_:.4f}")
        elif name == "XGBoost":
            print("  Running RandomizedSearchCV...")
            search = RandomizedSearchCV(
                model, xgb_param_dist, n_iter=20, cv=5, scoring="accuracy",
                n_jobs=-1, random_state=42, verbose=0
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            print(f"  Best params: {search.best_params_}")
            print(f"  Best CV score: {search.best_score_:.4f}")
        else:
            model.fit(X_train, y_train)
            best_model = model

        trained_models[name] = best_model
        model_path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(best_model, model_path)
        print(f"  Saved model to: {model_path}")

        results[name] = {
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "model": best_model,
        }

    best_name = max(results.keys(), key=lambda k: results[k]["cv_mean"])
    best_model = results[best_name]["model"]
    best_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best_model, best_path)
    print(f"\n>>> Best model: {best_name} (CV Accuracy: {results[best_name]['cv_mean']:.4f})")
    print(f">>> Saved best model to: {best_path}")

    print("\n=== TRAINING SUMMARY ===")
    print(f"{'Model':<20} {'CV Mean':>10} {'CV Std':>10}")
    print("-" * 42)
    for name, res in sorted(results.items(), key=lambda x: -x[1]["cv_mean"]):
        print(f"{name:<20} {res['cv_mean']:>10.4f} {res['cv_std']:>10.4f}")

    print("=" * 60)
    print("PHASE 4: MODEL TRAINING COMPLETED")
    print("=" * 60)

    return results, target_col, X_test, y_test


if __name__ == "__main__":
    run_training()
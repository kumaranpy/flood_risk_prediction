import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def auto_detect_target(df: pd.DataFrame) -> str:
    target_keywords = ["risk", "flood", "label", "target", "probability", "class"]
    for col in df.columns:
        if any(kw in col.lower() for kw in target_keywords):
            return col
    return df.columns[-1]


def run_evaluation():
    print("=" * 60)
    print("PHASE 5: MODEL EVALUATION STARTED")
    print("=" * 60)

    print(f"\nLoading test data from: {FEATURES_DATA_PATH}")
    df = pd.read_csv(FEATURES_DATA_PATH)
    target_col = auto_detect_target(df)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classes = sorted(y.unique())
    n_classes = len(classes)
    class_labels = [str(c) for c in classes]

    print(f"Target: {target_col}, Classes: {class_labels}")

    model_files = list(MODELS_DIR.glob("*.pkl"))
    model_files = [f for f in model_files if f.name not in ["scaler.pkl", "encoders.pkl", "best_model.pkl"]]
    print(f"\nFound {len(model_files)} models to evaluate")

    all_results = []

    for model_path in model_files:
        name = model_path.stem
        print(f"\n--- Evaluating {name} ---")
        
        model = joblib.load(model_path)
        
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)

        report = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True, zero_division=0)
        print(classification_report(y_test, y_pred, target_names=class_labels, zero_division=0))

        cm = confusion_matrix(y_test, y_pred, labels=classes)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        cm_path = PLOTS_DIR / f"cm_{name}.png"
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved confusion matrix: {cm_path}")

        roc_auc = None
        if y_proba is not None:
            if n_classes == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
                plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve: {name}")
                plt.legend()
                plt.tight_layout()
            else:
                y_test_bin = label_binarize(y_test, classes=classes)
                roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")
                plt.figure(figsize=(6, 5))
                for i, cls in enumerate(classes):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    cls_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f"Class {class_labels[i]} (AUC = {cls_auc:.3f})")
                plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curves (OvR): {name}")
                plt.legend(fontsize=8)
                plt.tight_layout()
            
            roc_path = PLOTS_DIR / f"roc_{name}.png"
            plt.savefig(roc_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved ROC curve: {roc_path}")

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_names = X.columns
            indices = np.argsort(importances)[::-1]
            top_n = min(15, len(feat_names))
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(top_n), importances[indices[:top_n]], align="center")
            plt.xticks(range(top_n), [feat_names[i] for i in indices[:top_n]], rotation=45, ha="right")
            plt.title(f"Feature Importances: {name}")
            plt.xlabel("Feature")
            plt.ylabel("Importance")
            plt.tight_layout()
            fi_path = PLOTS_DIR / f"feature_importance_{name}.png"
            plt.savefig(fi_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved feature importance: {fi_path}")

        accuracy = report["accuracy"]
        f1_macro = report["macro avg"]["f1-score"]
        f1_weighted = report["weighted avg"]["f1-score"]
        
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        all_results.append({
            "Model": name,
            "Accuracy": accuracy,
            "F1_Macro": f1_macro,
            "F1_Weighted": f1_weighted,
            "ROC_AUC": roc_auc if roc_auc else 0.0,
            "CV_Mean": cv_mean,
            "CV_Std": cv_std,
        })

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    
    report_path = REPORTS_DIR / "model_comparison.csv"
    results_df.to_csv(report_path, index=False)
    print(f"\n=== MODEL COMPARISON ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved comparison to: {report_path}")

    best_row = results_df.iloc[0]
    print(f"\n>>> BEST MODEL: {best_row['Model']}")
    print(f"    Accuracy: {best_row['Accuracy']:.4f}")
    print(f"    F1 Macro: {best_row['F1_Macro']:.4f}")
    print(f"    F1 Weighted: {best_row['F1_Weighted']:.4f}")
    print(f"    ROC AUC: {best_row['ROC_AUC']:.4f}")
    print(f"    CV Mean: {best_row['CV_Mean']:.4f} ± {best_row['CV_Std']:.4f}")

    print("=" * 60)
    print("PHASE 5: MODEL EVALUATION COMPLETED")
    print("=" * 60)

    return results_df


if __name__ == "__main__":
    run_evaluation()
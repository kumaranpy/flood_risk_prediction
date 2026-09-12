import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned.csv"
FEATURES_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"


def run_feature_engineering():
    print("=" * 60)
    print("PHASE 2: FEATURE ENGINEERING STARTED")
    print("=" * 60)

    print(f"\nLoading cleaned data from: {CLEANED_DATA_PATH}")
    df = pd.read_csv(CLEANED_DATA_PATH)

    target_keywords = ["risk", "flood", "label", "target", "probability", "class"]
    target_col = None
    for col in df.columns:
        if any(kw in col.lower() for kw in target_keywords):
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]

    print(f"Target column: '{target_col}'")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"\n--- CORRELATION ANALYSIS ---")
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(col, row, corr_matrix.loc[row, col]) 
                       for col in upper_tri.columns 
                       for row in upper_tri.index 
                       if upper_tri.loc[row, col] > 0.90]

    print(f"Found {len(high_corr_pairs)} pairs with correlation > 0.90")
    to_drop = set()
    for col1, col2, corr_val in high_corr_pairs:
        if col1 not in to_drop and col2 not in to_drop:
            to_drop.add(col2)
            print(f"  Dropping '{col2}' (corr={corr_val:.4f} with '{col1}')")

    X = X.drop(columns=list(to_drop))
    print(f"Features after correlation filtering: {X.shape[1]}")

    print(f"\n--- ENGINEERING FLOOD_VULNERABILITY_SCORE ---")
    weights = {
        "MonsoonIntensity": 0.35,
        "TopographyDrainage": 0.25,
        "RiverManagement": 0.20,
        "Deforestation": 0.20,
    }
    available_weights = {k: v for k, v in weights.items() if k in X.columns}
    if available_weights:
        total_weight = sum(available_weights.values())
        normalized_weights = {k: v / total_weight for k, v in available_weights.items()}
        X["Flood_Vulnerability_Score"] = sum(
            X[col] * w for col, w in normalized_weights.items()
        )
        print(f"  Created Flood_Vulnerability_Score using: {list(normalized_weights.keys())}")
        print(f"  Normalized weights: {normalized_weights}")
    else:
        print("  Warning: No weight columns found, skipping vulnerability score")

    print(f"\n--- FEATURE SELECTION (SelectKBest, k=12) ---")
    selector = SelectKBest(score_func=f_classif, k=min(12, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    feature_scores = selector.scores_[selected_mask]

    print(f"Selected {len(selected_features)} features:")
    for feat, score in sorted(zip(selected_features, feature_scores), key=lambda x: -x[1]):
        print(f"  {feat}: F-score = {score:.4f}")

    X_final = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    final_df = X_final.copy()
    final_df[target_col] = y.values

    final_df.to_csv(FEATURES_DATA_PATH, index=False)
    print(f"\nSaved final features to: {FEATURES_DATA_PATH}")
    print(f"Final shape: {final_df.shape}")

    print("=" * 60)
    print("PHASE 2: FEATURE ENGINEERING COMPLETED")
    print("=" * 60)

    return final_df, target_col, selected_features


if __name__ == "__main__":
    run_feature_engineering()
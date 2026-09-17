from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned.csv"
FEATURES_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
SELECTED_FEATURES_PATH = MODELS_DIR / "selected_features.pkl"
FEATURE_SELECTOR_PATH = MODELS_DIR / "feature_selector.pkl"


def auto_detect_target(df: pd.DataFrame) -> str:
    """Auto-detect target column from dataframe."""
    target_keywords = ["risk", "flood", "label", "target", "probability", "class"]
    for col in df.columns:
        if any(kw in col.lower() for kw in target_keywords):
            return col
    return df.columns[-1]


def compute_flood_vulnerability_score(df: pd.DataFrame) -> pd.Series:
    """
    FR-08: Compute composite Flood Vulnerability Score using domain-specified weights:
      - MonsoonIntensity: 0.35
      - TopographyDrainage: 0.25
      - RiverManagement: 0.20
      - Deforestation: 0.20
    """
    weights = {
        "MonsoonIntensity": 0.35,
        "TopographyDrainage": 0.25,
        "RiverManagement": 0.20,
        "Deforestation": 0.20,
    }
    available_weights = {k: v for k, v in weights.items() if k in df.columns}
    if not available_weights:
        return pd.Series(0.0, index=df.index)

    total_weight = sum(available_weights.values())
    normalized_weights = {k: v / total_weight for k, v in available_weights.items()}
    score = sum(df[col] * w for col, w in normalized_weights.items())
    return score


def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Flood_Vulnerability_Score (FR-08) as well as infrastructure,
    environmental, and hazard aggregate indices to capture multidimensional risk.
    """
    df_feat = df.copy()

    # FR-08: Primary Flood Vulnerability Score
    df_feat["Flood_Vulnerability_Score"] = compute_flood_vulnerability_score(df_feat)

    # Domain indices
    infra_cols = ["DamsQuality", "DrainageSystems", "DeterioratingInfrastructure", "IneffectiveDisasterPreparedness"]
    avail_infra = [c for c in infra_cols if c in df_feat.columns]
    if avail_infra:
        df_feat["Infrastructure_Deficit_Score"] = df_feat[avail_infra].mean(axis=1)

    env_cols = ["Urbanization", "ClimateChange", "AgriculturalPractices", "Encroachments", "WetlandLoss"]
    avail_env = [c for c in env_cols if c in df_feat.columns]
    if avail_env:
        df_feat["Environmental_Stress_Score"] = df_feat[avail_env].mean(axis=1)

    # Aggregate hazard index over numeric input factors
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df_feat["Aggregate_Hazard_Index"] = df[numeric_cols].mean(axis=1)

    return df_feat


def run_feature_engineering():
    """
    Executes Phase 2 Feature Engineering meeting FR-07, FR-08, FR-09, and FR-10.
    """
    print("=" * 60)
    print("PHASE 2: FEATURE ENGINEERING STARTED")
    print("=" * 60)

    print(f"\nLoading cleaned data from: {CLEANED_DATA_PATH}")
    df = pd.read_csv(CLEANED_DATA_PATH)
    target_col = auto_detect_target(df)
    print(f"Target column: '{target_col}'")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # FR-07: Drop collinear features with Pearson correlation r > 0.90
    print("\n--- CORRELATION FILTERING (FR-07) ---")
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()

    for col in upper_tri.columns:
        for row in upper_tri.index:
            if upper_tri.loc[row, col] > 0.90:
                corr_val = upper_tri.loc[row, col]
                if col not in to_drop and row not in to_drop:
                    to_drop.add(col)
                    print(f"  Dropping collinear feature '{col}' (r = {corr_val:.4f} with '{row}')")

    if to_drop:
        X = X.drop(columns=list(to_drop))
        print(f"  Dropped {len(to_drop)} features. Remaining: {X.shape[1]}")
    else:
        print("  No features exceeded the 0.90 correlation threshold.")

    # FR-08: Feature Engineering (Composite Flood Vulnerability Score & Domain Indices)
    print("\n--- COMPUTING FLOOD VULNERABILITY SCORE & INDICES (FR-08) ---")
    X = engineer_domain_features(X)
    print(f"  Generated features: {X.columns.tolist()[-4:]}")
    print(f"  Total features after engineering: {X.shape[1]}")

    # FR-09: Select top 12 features using SelectKBest (score_func=f_classif, k=12)
    k_features = min(12, X.shape[1])
    print(f"\n--- SELECTING TOP {k_features} FEATURES (SelectKBest, k=12) (FR-09) ---")
    selector = SelectKBest(score_func=f_classif, k=k_features)
    X_selected = selector.fit_transform(X, y)
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    scores = selector.scores_[selected_mask]

    print(f"Selected {len(selected_features)} features:")
    for feat, score in sorted(zip(selected_features, scores), key=lambda x: -x[1]):
        print(f"  - {feat:<30} F-Score: {score:.2f}")

    # Save feature names and selector artifact for inference
    joblib.dump(selected_features, SELECTED_FEATURES_PATH)
    joblib.dump(selector, FEATURE_SELECTOR_PATH)
    print(f"  Saved selected features list to: {SELECTED_FEATURES_PATH}")

    # FR-10: Save final feature matrix to data/processed/features.csv
    final_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    final_df[target_col] = y.values
    final_df.to_csv(FEATURES_DATA_PATH, index=False)
    print(f"\nFR-10: Saved final feature matrix to: {FEATURES_DATA_PATH}")
    print(f"Final shape: {final_df.shape}")

    print("=" * 60)
    print("PHASE 2: FEATURE ENGINEERING COMPLETED")
    print("=" * 60)

    return final_df, target_col, selected_features


if __name__ == "__main__":
    run_feature_engineering()
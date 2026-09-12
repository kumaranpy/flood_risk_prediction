import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "flood_factors.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned.csv"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
ENCODERS_PATH = PROJECT_ROOT / "models" / "encoders.pkl"


def auto_detect_target(df: pd.DataFrame) -> str:
    """Auto-detect target column: last column or column with risk/flood/label in name."""
    target_keywords = ["risk", "flood", "label", "target", "probability", "class"]
    for col in df.columns:
        if any(kw in col.lower() for kw in target_keywords):
            return col
    return df.columns[-1]


def run_preprocessing():
    print("=" * 60)
    print("PHASE 1: PREPROCESSING STARTED")
    print("=" * 60)

    print(f"\nLoading data from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)

    print(f"\n--- INITIAL DATA INFO ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"Null counts:\n{df.isnull().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    target_col = auto_detect_target(df)
    print(f"\nAuto-detected target column: '{target_col}'")

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")

    print("\n--- HANDLING MISSING VALUES ---")
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    for col in numeric_cols:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            print(f"  Filled {col} missing with median: {median_val:.4f}")

    for col in categorical_cols:
        if X[col].isnull().any():
            mode_val = X[col].mode()[0] if not X[col].mode().empty else "unknown"
            X[col].fillna(mode_val, inplace=True)
            print(f"  Filled {col} missing with mode: {mode_val}")

    print("\n--- CAPPING OUTLIERS (IQR 1.5x) ---")
    for col in numeric_cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers_before = ((X[col] < lower) | (X[col] > upper)).sum()
        if outliers_before > 0:
            X[col] = X[col].clip(lower, upper)
            print(f"  {col}: capped {outliers_before} outliers to [{lower:.4f}, {upper:.4f}]")

    print("\n--- ENCODING CATEGORICAL COLUMNS ---")
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        print(f"  Label encoded: {col} ({len(le.classes_)} classes)")

    if len(categorical_cols) == 0:
        print("  No categorical columns to encode.")

    print("\n--- SCALING NUMERIC FEATURES ---")
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    print(f"  Applied StandardScaler to {len(numeric_cols)} numeric columns")

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"  Saved scaler to: {SCALER_PATH}")
    print(f"  Saved encoders to: {ENCODERS_PATH}")

    cleaned_df = X.copy()
    cleaned_df[target_col] = y.values
    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"\n--- PREPROCESSING COMPLETE ---")
    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {cleaned_df.shape}")
    print(f"Null count after: {cleaned_df.isnull().sum().sum()}")
    print(f"Saved cleaned data to: {PROCESSED_DATA_PATH}")

    print("=" * 60)
    print("PHASE 1: PREPROCESSING COMPLETED")
    print("=" * 60)

    return cleaned_df, target_col


if __name__ == "__main__":
    run_preprocessing()
import csv
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "flood_factors.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODERS_PATH = MODELS_DIR / "encoders.pkl"
TARGET_ENCODER_PATH = MODELS_DIR / "target_encoder.pkl"


def detect_file_properties(file_path: Path):
    """
    FR-01: Auto-detect dataset delimiter and file encoding.
    Attempts standard encodings and uses csv.Sniffer for delimiter detection.
    """
    candidate_encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    detected_encoding = "utf-8"
    detected_delimiter = ","

    for enc in candidate_encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                sample = f.read(8192)
                detected_encoding = enc
                try:
                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(sample, delimiters=[",", ";", "\t", "|"])
                    detected_delimiter = dialect.delimiter
                except Exception:
                    # Fallback to comma if sniffing dialect fails
                    detected_delimiter = ","
                break
        except (UnicodeDecodeError, UnicodeError):
            continue

    return detected_delimiter, detected_encoding


def auto_detect_target(df: pd.DataFrame) -> str:
    """Auto-detect target column: last column or column matching target keywords."""
    target_keywords = ["risk", "flood", "label", "target", "probability", "class"]
    for col in df.columns:
        if any(kw in col.lower() for kw in target_keywords):
            return col
    return df.columns[-1]


def discretize_target(y_series: pd.Series) -> pd.Series:
    """
    Converts target into discrete risk classes: Low, Medium, High.
    If continuous (float), bins into balanced tertiles:
      - Low: <= 0.475 (~33rd percentile)
      - Medium: 0.475 to 0.520
      - High: > 0.520 (~67th percentile)
    If already categorical/string, normalizes casing.
    """
    if np.issubdtype(y_series.dtype, np.floating) or np.issubdtype(y_series.dtype, np.integer):
        if y_series.nunique() > 10:
            bins = [-float("inf"), 0.475, 0.520, float("inf")]
            labels = ["Low", "Medium", "High"]
            discretized = pd.cut(y_series, bins=bins, labels=labels)
            return discretized.astype(str)
    return y_series.astype(str)


def run_preprocessing():
    """
    End-to-end Phase 1 preprocessing function meeting requirements FR-01 through FR-06.
    """
    print("=" * 60)
    print("PHASE 1: PREPROCESSING STARTED")
    print("=" * 60)

    # FR-01: Auto-detect delimiter and encoding
    delimiter, encoding = detect_file_properties(RAW_DATA_PATH)
    print(f"FR-01: Detected Delimiter: '{delimiter}' | Encoding: '{encoding}'")
    print(f"Loading data from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH, delimiter=delimiter, encoding=encoding)

    print("\n--- INITIAL DATA INFO ---")
    print(f"Shape: {df.shape}")
    print(f"Null counts total: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    target_col = auto_detect_target(df)
    print(f"\nAuto-detected target column: '{target_col}'")

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()
    raw_y = df[target_col].copy()

    # Discretize continuous target to Low, Medium, High
    y_discrete = discretize_target(raw_y)
    target_encoder = LabelEncoder()
    # Explicitly fit with standard ordering: Low=0, Medium=1, High=2
    target_encoder.fit(["Low", "Medium", "High"])
    print(f"Target classes: {target_encoder.classes_}")
    print(f"Target class distribution:\n{y_discrete.value_counts(normalize=True)}")

    joblib.dump(target_encoder, TARGET_ENCODER_PATH)

    # FR-02: Handling missing values (median for numeric, mode for categorical)
    print("\n--- HANDLING MISSING VALUES (FR-02) ---")
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in numeric_cols:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            print(f"  Imputed {col} missing with median: {median_val:.4f}")

    for col in categorical_cols:
        if X[col].isnull().any():
            mode_val = X[col].mode()[0] if not X[col].mode().empty else "unknown"
            X[col] = X[col].fillna(mode_val)
            print(f"  Imputed {col} missing with mode: {mode_val}")

    # FR-03: Capping outliers using IQR 1.5x rule
    print("\n--- CAPPING OUTLIERS (IQR 1.5x) (FR-03) ---")
    capped_count = 0
    for col in numeric_cols:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((X[col] < lower) | (X[col] > upper)).sum()
        if outliers > 0:
            X[col] = X[col].clip(lower, upper)
            capped_count += outliers
            print(f"  {col}: capped {outliers} outliers to [{lower:.2f}, {upper:.2f}]")
    if capped_count == 0:
        print("  No extreme outliers beyond 1.5x IQR.")

    # FR-04: Auto-encode categorical columns
    print("\n--- ENCODING CATEGORICAL COLUMNS (FR-04) ---")
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        print(f"  Encoded {col} ({len(le.classes_)} classes)")
    joblib.dump(encoders, ENCODERS_PATH)

    # FR-05: Apply StandardScaler to numeric features
    print("\n--- SCALING NUMERIC FEATURES (FR-05) ---")
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    joblib.dump(scaler, SCALER_PATH)
    print(f"  Saved fitted scaler to: {SCALER_PATH}")

    # FR-06: Save cleaned data
    cleaned_df = X.copy()
    cleaned_df[target_col] = y_discrete.values
    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\nFR-06: Saved cleaned dataset to: {PROCESSED_DATA_PATH}")
    print(f"Cleaned shape: {cleaned_df.shape}")

    print("=" * 60)
    print("PHASE 1: PREPROCESSING COMPLETED")
    print("=" * 60)

    return cleaned_df, target_col


if __name__ == "__main__":
    run_preprocessing()
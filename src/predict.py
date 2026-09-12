import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODERS_PATH = MODELS_DIR / "encoders.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"


def auto_detect_target(df: pd.DataFrame) -> str:
    target_keywords = ["risk", "flood", "label", "target", "probability", "class"]
    for col in df.columns:
        if any(kw in col.lower() for kw in target_keywords):
            return col
    return df.columns[-1]


def run_prediction():
    print("=" * 60)
    print("PHASE 6: PREDICTION SCRIPT")
    print("=" * 60)

    print(f"\nLoading best model from: {BEST_MODEL_PATH}")
    model = joblib.load(BEST_MODEL_PATH)
    print(f"Model type: {type(model).__name__}")

    print(f"Loading scaler from: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    
    encoders = {}
    if ENCODERS_PATH.exists():
        encoders = joblib.load(ENCODERS_PATH)
        print(f"Loaded encoders for: {list(encoders.keys())}")

    df = pd.read_csv(FEATURES_DATA_PATH)
    target_col = auto_detect_target(df)
    feature_cols = [c for c in df.columns if c != target_col]
    print(f"Expected features ({len(feature_cols)}): {feature_cols}")

    sample_input = {
        "MonsoonIntensity": 8,
        "TopographyDrainage": 7,
        "RiverManagement": 5,
        "Deforestation": 6,
        "Urbanization": 7,
        "ClimateChange": 8,
        "DamsQuality": 4,
        "Siltation": 5,
        "AgriculturalPractices": 6,
        "Encroachments": 5,
        "IneffectiveDisasterPreparedness": 7,
        "DrainageSystems": 6,
        "CoastalVulnerability": 5,
        "Landslides": 4,
        "Watersheds": 5,
        "DeterioratingInfrastructure": 6,
        "PopulationScore": 7,
        "WetlandLoss": 5,
        "InadequatePlanning": 6,
        "PoliticalFactors": 5,
    }

    print("\n--- Sample Input ---")
    for k, v in sample_input.items():
        print(f"  {k}: {v}")

    input_df = pd.DataFrame([sample_input])
    
    missing_cols = set(feature_cols) - set(input_df.columns)
    extra_cols = set(input_df.columns) - set(feature_cols)
    if missing_cols:
        print(f"Warning: Missing columns (will use defaults): {missing_cols}")
        for col in missing_cols:
            input_df[col] = 0
    if extra_cols:
        print(f"Warning: Extra columns (will be dropped): {extra_cols}")
        input_df = input_df.drop(columns=list(extra_cols))
    
    input_df = input_df[feature_cols]

    if encoders:
        for col, encoder in encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = encoder.transform(input_df[col].astype(str))
                except ValueError:
                    print(f"  Warning: Unknown category in {col}, using first class")
                    input_df[col] = 0

    numeric_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        print(f"  Applied scaler to: {numeric_cols}")

    prediction = model.predict(input_df)[0]
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_
    else:
        probabilities = None
        classes = None

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Predicted Risk Level: {prediction}")
    
    if probabilities is not None:
        max_idx = np.argmax(probabilities)
        confidence = probabilities[max_idx] * 100
        print(f"Confidence: {confidence:.1f}%")
        print("\nClass Probabilities:")
        for cls, prob in zip(classes, probabilities):
            bar = "█" * int(prob * 20)
            print(f"  {cls}: {prob*100:6.1f}% {bar}")

    risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    pred_str = str(prediction).upper()
    emoji = risk_emoji.get(pred_str, "⚪")
    print(f"\n{emoji}  Risk Level: {prediction}  (confidence: {confidence:.1f}%)")

    print("=" * 60)
    print("PHASE 6: PREDICTION COMPLETED")
    print("=" * 60)

    return prediction, probabilities


if __name__ == "__main__":
    run_prediction()
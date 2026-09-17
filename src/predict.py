from pathlib import Path
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
TARGET_ENCODER_PATH = MODELS_DIR / "target_encoder.pkl"
SELECTED_FEATURES_PATH = MODELS_DIR / "selected_features.pkl"

RISK_EMOJIS = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}


def compute_engineered_features(input_dict: dict) -> dict:
    """
    Computes composite indicators from raw factor inputs:
    - Flood_Vulnerability_Score
    - Infrastructure_Deficit_Score
    - Environmental_Stress_Score
    - Aggregate_Hazard_Index
    """
    row = dict(input_dict)

    # 1. Flood Vulnerability Score (FR-08)
    weights = {
        "MonsoonIntensity": 0.35,
        "TopographyDrainage": 0.25,
        "RiverManagement": 0.20,
        "Deforestation": 0.20,
    }
    avail_weights = {k: v for k, v in weights.items() if k in row}
    if avail_weights:
        norm_w = {k: v / sum(avail_weights.values()) for k, v in avail_weights.items()}
        row["Flood_Vulnerability_Score"] = sum(row[k] * w for k, w in norm_w.items())
    else:
        row["Flood_Vulnerability_Score"] = 5.0

    # 2. Infrastructure Deficit Score
    infra = ["DamsQuality", "DrainageSystems", "DeterioratingInfrastructure", "IneffectiveDisasterPreparedness"]
    avail_infra = [row[c] for c in infra if c in row]
    row["Infrastructure_Deficit_Score"] = float(np.mean(avail_infra)) if avail_infra else 5.0

    # 3. Environmental Stress Score
    env = ["Urbanization", "ClimateChange", "AgriculturalPractices", "Encroachments", "WetlandLoss"]
    avail_env = [row[c] for c in env if c in row]
    row["Environmental_Stress_Score"] = float(np.mean(avail_env)) if avail_env else 5.0

    # 4. Aggregate Hazard Index
    all_numeric_vals = [v for k, v in row.items() if isinstance(v, (int, float))]
    row["Aggregate_Hazard_Index"] = float(np.mean(all_numeric_vals)) if all_numeric_vals else 5.0

    return row


def predict_single_instance(sample_input: dict):
    """
    Executes single-instance inference end-to-end.
    """
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {BEST_MODEL_PATH}. Run training first.")

    model = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    target_encoder = joblib.load(TARGET_ENCODER_PATH) if TARGET_ENCODER_PATH.exists() else None
    selected_features = joblib.load(SELECTED_FEATURES_PATH)

    # Calculate engineered features
    enriched_input = compute_engineered_features(sample_input)
    input_df = pd.DataFrame([enriched_input])

    # Ensure all 12 selected features are present
    for feat in selected_features:
        if feat not in input_df.columns:
            input_df[feat] = 5.0

    input_df = input_df[selected_features]

    # Transform with training scaler (no leakage)
    scaled_array = scaler.transform(input_df)
    scaled_input = pd.DataFrame(scaled_array, columns=selected_features)

    # Predict class & probabilities
    pred_idx = model.predict(scaled_input)[0]

    if target_encoder is not None:
        predicted_class = target_encoder.inverse_transform([pred_idx])[0]
        class_names = list(target_encoder.classes_)
    else:
        mapping = {0: "Low", 1: "Medium", 2: "High"}
        predicted_class = mapping.get(pred_idx, str(pred_idx))
        class_names = ["Low", "Medium", "High"]

    probabilities = None
    confidence = 0.0
    prob_dict = {}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_input)[0]
        max_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[max_idx] * 100.0)
        for cls_name, prob in zip(class_names, probabilities):
            prob_dict[cls_name] = float(prob)

    return predicted_class, confidence, prob_dict


def run_prediction():
    """
    Phase 6 CLI demonstration inference.
    """
    print("=" * 60)
    print("PHASE 6: INFERENCE PIPELINE DEMONSTRATION")
    print("=" * 60)

    sample_scenario = {
        "MonsoonIntensity": 9.0,
        "TopographyDrainage": 8.0,
        "RiverManagement": 7.0,
        "Deforestation": 8.0,
        "Urbanization": 8.0,
        "ClimateChange": 9.0,
        "DamsQuality": 3.0,
        "Siltation": 7.0,
        "AgriculturalPractices": 7.0,
        "Encroachments": 7.0,
        "IneffectiveDisasterPreparedness": 8.0,
        "DrainageSystems": 4.0,
        "CoastalVulnerability": 7.0,
        "Landslides": 6.0,
        "Watersheds": 7.0,
        "DeterioratingInfrastructure": 8.0,
        "PopulationScore": 8.0,
        "WetlandLoss": 7.0,
        "InadequatePlanning": 8.0,
        "PoliticalFactors": 6.0,
    }

    print("\n--- Input Scenario (Severe Monsoon & Poor Infrastructure) ---")
    for k, v in list(sample_scenario.items())[:8]:
        print(f"  {k:<32}: {v}")
    print("  ... [20 factors total]")

    predicted_class, confidence, prob_dict = predict_single_instance(sample_scenario)
    emoji = RISK_EMOJIS.get(predicted_class, "⚪")

    print("\n" + "=" * 60)
    print("INFERENCE RESULT")
    print("=" * 60)
    print(f"Predicted Flood Risk Level : {emoji}  {predicted_class.upper()}")
    print(f"Prediction Confidence      : {confidence:.2f}%")
    print("\nClass Probability Distribution:")
    for cls, prob in prob_dict.items():
        bar = "█" * int(prob * 25)
        print(f"  {cls:<8} : {prob*100:5.1f}%  {bar}")

    print("=" * 60)
    print("PHASE 6: INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return predicted_class, confidence, prob_dict


if __name__ == "__main__":
    run_prediction()
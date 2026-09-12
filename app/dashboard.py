import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODERS_PATH = MODELS_DIR / "encoders.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"


@st.cache_resource
def load_artifacts():
    model = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH) if ENCODERS_PATH.exists() else {}
    df = pd.read_csv(FEATURES_DATA_PATH)
    target_keywords = ["risk", "flood", "label", "target", "probability", "class"]
    target_col = None
    for col in df.columns:
        if any(kw in col.lower() for kw in target_keywords):
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]
    return model, scaler, encoders, feature_cols, target_col


def main():
    st.set_page_config(page_title="Flood Risk Prediction System", page_icon="🌊", layout="wide")
    
    st.title("🌊 Flood Risk Prediction System")
    st.markdown("---")

    model, scaler, encoders, feature_cols, target_col = load_artifacts()
    
    with st.sidebar:
        st.header("📊 Input Features")
        st.markdown("Adjust the sliders for each feature:")
        
        input_values = {}
        
        for feat in feature_cols:
            min_val = 0.0
            max_val = 10.0
            default_val = 5.0
            step = 0.5
            
            if "Intensity" in feat or "Urbanization" in feat or "Population" in feat:
                max_val = 12.0
                default_val = 6.0
            elif "Quality" in feat or "Systems" in feat or "Preparedness" in feat:
                max_val = 11.0
                default_val = 5.0
            elif "Vulnerability" in feat:
                max_val = 1.0
                default_val = 0.5
                step = 0.05
            
            input_values[feat] = st.slider(
                feat,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                step=float(step),
                help=f"Range: {min_val} - {max_val}"
            )
        
        predict_btn = st.button("🔮 Predict Risk", type="primary", use_container_width=True)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Prediction Result")
        
        if predict_btn:
            input_df = pd.DataFrame([input_values])
            
            for col, encoder in encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = encoder.transform(input_df[col].astype(str))
                    except ValueError:
                        input_df[col] = 0
            
            numeric_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
            
            input_df = input_df[feature_cols]
            
            prediction = model.predict(input_df)[0]
            
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = model.classes_
                
                max_idx = np.argmax(probabilities)
                confidence = probabilities[max_idx] * 100
                pred_class = classes[max_idx]
                
                risk_colors = {"HIGH": "#FF4444", "MEDIUM": "#FFAA00", "LOW": "#44AA44"}
                risk_emojis = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                
                pred_str = str(pred_class).upper()
                color = risk_colors.get(pred_str, "#888888")
                emoji = risk_emojis.get(pred_str, "⚪")
                
                st.markdown(
                    f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: {color}20; border: 2px solid {color};">
                        <h2 style="margin: 0; color: {color};">{emoji} Predicted Risk Level: {pred_class}</h2>
                        <h3 style="margin: 10px 0 0 0; color: {color};">Confidence: {confidence:.1f}%</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown("### Class Probabilities")
                for cls, prob in zip(classes, probabilities):
                    prob_pct = prob * 100
                    cls_str = str(cls).upper()
                    bar_color = risk_colors.get(cls_str, "#888888")
                    st.progress(prob, text=f"{cls}: {prob_pct:.1f}%")
            else:
                st.error("Model does not support probability predictions")
    
    with col2:
        st.subheader("🎯 Feature Importances")
        
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_n = min(12, len(feature_cols))
            
            imp_df = pd.DataFrame({
                "Feature": [feature_cols[i] for i in indices[:top_n]],
                "Importance": importances[indices[:top_n]]
            })
            
            st.bar_chart(imp_df.set_index("Feature"))
        else:
            st.info("Feature importance not available for this model type")


if __name__ == "__main__":
    main()
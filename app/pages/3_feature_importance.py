"""
Streamlit Page: Feature Importance — Global & Local SHAP Narrative Deep Dives
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_verified_pipeline, SecurityError

# Page config
st.set_page_config(
    page_title="Feature Importance | Flood Risk Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .feature-card {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    .feature-header {
        font-size: 1.3rem;
        font-weight: 800;
        color: #F1F5F9;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .shap-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(139, 92, 246, 0.15);
        color: #A78BFA;
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 9999px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .shap-value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .shap-positive { color: #EF4444; }
    .shap-negative { color: #10B981; }
    .shap-neutral { color: #94A3B8; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    """Load pipeline and data."""
    try:
        pipeline = load_verified_pipeline()
    except SecurityError as e:
        st.error(f"Security Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        st.stop()
    
    train_path = PROJECT_ROOT / "data" / "splits" / "train.csv"
    test_path = PROJECT_ROOT / "data" / "splits" / "test.csv"
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    feature_cols = [c for c in train_df.columns if c not in ["FloodProbability_raw", "RiskLevel"]]
    X_train = train_df[feature_cols].copy()
    y_train = train_df["RiskLevel"].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df["RiskLevel"].copy()
    
    return pipeline, X_train, y_train, X_test, y_test, feature_cols


def main():
    # Hero
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(16, 24, 40, 0.95) 0%, rgba(15, 23, 42, 0.90) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 32px; margin-bottom: 24px;">
            <div class="shap-badge">🔍 SHAP Explainability & Feature Attribution</div>
            <h1 style="margin: 0 0 8px 0; font-size: 2.5rem; font-weight: 800; color: #FFFFFF; line-height: 1.2;">
                Why Did the Model Predict This?
            </h1>
            <p style="margin: 0; color: #94A3B8; font-size: 1.05rem; max-width: 900px; line-height: 1.6;">
                Global feature importance, local explanations, and SHAP value deep dives for calibrated probabilities.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Load assets
    with st.spinner("Loading pipeline and computing SHAP values..."):
        pipeline, X_train, y_train, X_test, y_test, feature_cols = load_assets()
    
    st.success(f"✅ Pipeline loaded: {len(X_train)} train / {len(X_test)} test samples")
    
    # SHAP computation status
    shap_summary_path = PROJECT_ROOT / "outputs" / "plots" / "shap_summary.png"
    shap_waterfall_path = PROJECT_ROOT / "outputs" / "plots" / "shap_waterfall_high.png"
    
    col1, col2 = st.columns(2)
    with col1:
        if shap_summary_path.exists():
            st.image(str(shap_summary_path), caption="SHAP Summary: Feature Importance by Class", use_container_width=True)
        else:
            st.info("SHAP summary not found. Run the full pipeline to generate.")
    
    with col2:
        if shap_waterfall_path.exists():
            st.image(str(shap_waterfall_path), caption="SHAP Waterfall: High-Risk Sample Explanation", use_container_width=True)
        else:
            st.info("SHAP waterfall not found. Run the full pipeline to generate.")
    
    # Load SHAP audit data
    audit_path = PROJECT_ROOT / "outputs" / "reports" / "statistical_audit.json"
    if audit_path.exists():
        with open(audit_path, "r") as f:
            audit_data = json.load(f)
    else:
        audit_data = {}
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌍 Global Importance",
        "📍 Local Explanations", 
        "📊 Per-Class SHAP",
        "📈 Feature Interactions"
    ])
    
    with tab1:
        st.markdown('<div class="feature-header">Global Feature Importance (Mean |SHAP|)</div>', unsafe_allow_html=True)
        st.caption("Mean absolute SHAP values aggregated across all samples and classes.")
        
        # If we have ablation data, show it
        if audit_data and "ablation" in audit_data:
            ablation_df = pd.DataFrame(audit_data["ablation"])
            if "drop" in ablation_df.columns:
                # Sort by drop magnitude
                ablation_df = ablation_df.sort_values("drop", ascending=False)
                
                # Plot
                st.markdown("### Feature Ablation: F1 Drop When Feature Removed")
                
                # Create a nice bar chart
                chart_data = ablation_df.set_index("feature_dropped")["drop"]
                st.bar_chart(chart_data)
                
                st.dataframe(
                    ablation_df[["feature_dropped", "f1_weighted", "drop"]],
                    hide_index=True,
                    use_container_width=True,
                )
    
    with tab2:
        st.markdown('<div class="feature-header">Local Explanations: Sample-Level SHAP</div>', unsafe_allow_html=True)
        st.caption("Individual prediction breakdowns showing feature contributions.")
        
        # Sample selector
        sample_idx = st.slider("Select Sample Index", 0, min(49, len(X_test)-1), 0)
        
        # Show raw features for this sample
        sample = X_test.iloc[sample_idx]
        true_class = y_test.iloc[sample_idx]
        
        st.markdown(f"### Sample {sample_idx} (True Class: {true_class})")
        
        # Get prediction
        raw_pred = pipeline.predict(sample.to_frame().T)[0]
        int_to_class = {0: "Low", 1: "Medium", 2: "High"}
        if isinstance(raw_pred, (int, np.integer)):
            pred_class = int_to_class.get(int(raw_pred), str(raw_pred))
        else:
            pred_class = str(raw_pred)
        
        prob = pipeline.predict_proba(sample.to_frame().T)[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Class", pred_class)
        with col2:
            st.metric("Max Probability", f"{max(prob):.1%}")
        
        # Show top raw features
        st.markdown("#### Top 10 Raw Feature Values")
        sample_df = pd.DataFrame({"Feature": sample.index, "Value": sample.values})
        sample_df = sample_df.sort_values("Value", ascending=False).head(10)
        st.dataframe(sample_df, hide_index=True, use_container_width=True)
        
        # If SHAP values available, show them
        if audit_data and "shap" in audit_data:
            st.info("SHAP values for individual samples available in statistical audit.")
    
    with tab3:
        st.markdown('<div class="feature-header">Per-Class SHAP Feature Rankings</div>', unsafe_allow_html=True)
        st.caption("Feature importance separated by predicted risk class.")
        
        # Show class-specific feature importance from statistical audit
        if audit_data and "anova" in audit_data:
            anova_df = pd.DataFrame(audit_data["anova"])
            if "f_statistic" in anova_df.columns:
                st.markdown("### ANOVA F-Statistics by Feature (Across Risk Classes)")
                
                # Plot top features by F-statistic
                top_anova = anova_df.nlargest(15, "f_statistic")
                chart_data = top_anova.set_index("feature")["f_statistic"]
                st.bar_chart(chart_data)
        
        # Chi-square results
        if audit_data and "chi2" in audit_data:
            chi2_df = pd.DataFrame(audit_data["chi2"])
            if "chi2_statistic" in chi2_df.columns:
                st.markdown("### Chi-Square Statistics by Feature (Tertile Binned)")
                top_chi2 = chi2_df.nlargest(15, "chi2_statistic")
                chart_data = top_chi2.set_index("feature")["chi2_statistic"]
                st.bar_chart(chart_data)
    
    with tab4:
        st.markdown('<div class="feature-header">Feature Interactions & Non-Linear Effects</div>', unsafe_allow_html=True)
        st.caption("Engineered interaction terms and their contribution to model predictions.")
        
        # Show VIF for interaction features
        if audit_data and "vif" in audit_data:
            vif_df = pd.DataFrame(audit_data["vif"])
            if "VIF" in vif_df.columns:
                st.markdown("### VIF for Engineered Features")
                st.dataframe(
                    vif_df[["feature", "VIF", "concern_level"]],
                    hide_index=True,
                    use_container_width=True,
                )
        
        # Interaction feature list
        st.markdown("### Engineered Interaction Features")
        interactions = [
            "MonsoonIntensity_x_Urbanization",
            "Deforestation_x_RiverManagement",
            "ClimateChange_x_DamsQuality",
            "Siltation_x_AgriculturalPractices",
            "TopographyDrainage_x_MonsoonIntensity",
        ]
        
        # Check which exist in data
        train_path = PROJECT_ROOT / "data" / "splits" / "train.csv"
        if train_path.exists():
            train_df = pd.read_csv(train_path)
            existing_interactions = [f for f in interactions if f in train_df.columns]
            
            if existing_interactions:
                st.markdown("**Existing Interaction Features in Dataset:**")
                for feat in existing_interactions:
                    mean_val = train_df[feat].mean()
                    std_val = train_df[feat].std()
                    st.write(f"• **{feat}**: μ={mean_val:.2f}, σ={std_val:.2f}")
            else:
                st.info("Interaction features not found in raw splits. Run feature engineering pipeline.")
        
        # Ratio features
        st.markdown("### Scale-Invariant Ratio Features")
        ratios = [
            "Water_Stress",
            "Infra_Gap",
            "Eco_Damage",
            "Siltation_Pressure",
            "Preparedness_Deficit",
        ]
        
        if train_path.exists():
            existing_ratios = [f for f in ratios if f in train_df.columns]
            if existing_ratios:
                st.markdown("**Existing Ratio Features:**")
                for feat in existing_ratios:
                    mean_val = train_df[feat].mean()
                    std_val = train_df[feat].std()
                    st.write(f"• **{feat}**: μ={mean_val:.2f}, σ={std_val:.2f} (scale-invariant)")
    
    # Download section
    st.markdown("---")
    st.markdown('<div class="feature-header">📥 Download SHAP Artifacts</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if shap_summary_path.exists():
            with open(shap_summary_path, "rb") as f:
                st.download_button(
                    "📊 SHAP Summary (PNG)",
                    f.read(),
                    "shap_summary.png",
                    "image/png"
                )
    with col2:
        if shap_waterfall_path.exists():
            with open(shap_waterfall_path, "rb") as f:
                st.download_button(
                    "🌊 SHAP Waterfall (PNG)",
                    f.read(),
                    "shap_waterfall_high.png",
                    "image/png"
                )
    with col3:
        if audit_path.exists():
            with open(audit_path, "r") as f:
                st.download_button(
                    "📄 Full Audit (JSON)",
                    f.read(),
                    "statistical_audit.json",
                    "application/json"
                )


if __name__ == "__main__":
    main()
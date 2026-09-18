"""
Streamlit Page: Statistical EDA & Hypothesis Testing
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

from src.statistical_audit import (
    anova_test_by_class,
    chi2_test_by_class,
    pearson_significance_matrix,
    compute_vif,
    feature_ablation_study,
    ks_distribution_drift,
    gaussian_noise_stress_test,
)
from src.predict import load_verified_pipeline, SecurityError


# Page config
st.set_page_config(
    page_title="Statistical EDA | Flood Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS
st.markdown(
    """
    <style>
    .stat-card {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
    }
    .stat-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .significant-yes { color: #10B981; font-weight: 700; }
    .significant-no { color: #64748B; font-weight: 500; }
    .vif-severe { color: #EF4444; font-weight: 700; }
    .vif-moderate { color: #F59E0B; font-weight: 700; }
    .vif-low { color: #10B981; font-weight: 600; }
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


@st.cache_data
def run_anova(X_train, y_train):
    return anova_test_by_class(X_train, y_train)


@st.cache_data
def run_chi2(X_train, y_train):
    return chi2_test_by_class(X_train, y_train)


@st.cache_data
def run_pearson(X_train, y_train):
    return pearson_significance_matrix(X_train, y_train)


@st.cache_data
def run_vif(X_train):
    return compute_vif(X_train)


@st.cache_data
def run_ablation(pipeline, X_test, y_test, domain_features):
    return feature_ablation_study(pipeline, X_test, y_test, domain_features)


@st.cache_data
def run_ks_drift(X_train, X_test):
    return ks_distribution_drift(X_train, X_test)


@st.cache_data
def run_noise_stress(pipeline, X_test, y_test):
    return gaussian_noise_stress_test(pipeline, X_test, y_test)


def format_p_value(p):
    if pd.isna(p):
        return "N/A"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def render_significance_badge(p_val, alpha=0.05):
    if pd.isna(p_val):
        return '<span class="significant-no">❓ Unknown</span>'
    if p_val < alpha:
        return '<span class="significant-yes">✅ Significant</span>'
    return '<span class="significant-no">❌ Not Significant</span>'


def render_vif_badge(vif):
    if pd.isna(vif):
        return '<span class="vif-severe">❓ Error</span>'
    if vif > 10:
        return '<span class="vif-severe">🔴 SEVERE</span>'
    if vif > 5:
        return '<span class="vif-moderate">🟡 MODERATE</span>'
    return '<span class="vif-low">🟢 LOW</span>'


def main():
    # Hero
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(16, 24, 40, 0.95) 0%, rgba(15, 23, 42, 0.90) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 24px 32px; margin-bottom: 20px;">
            <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(56, 189, 248, 0.15);
                        color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 9999px;
                        padding: 4px 14px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em;
                        text-transform: uppercase; margin-bottom: 10px;">
                📊 Statistical Inference & Feature Validation
            </div>
            <h1 style="margin: 0 0 6px 0; font-size: 2rem; font-weight: 800; color: #FFFFFF;">
                Rigorous Feature Science
            </h1>
            <p style="margin: 0; color: #94A3B8; font-size: 0.95rem; max-width: 900px;">
                ANOVA · Chi-Square · VIF · Feature Ablation · Distribution Drift · Stress Testing
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Load assets
    with st.spinner("Loading pipeline and data..."):
        pipeline, X_train, y_train, X_test, y_test, feature_cols = load_assets()
    
    st.success(f"✅ Loaded: {len(X_train)} train / {len(X_test)} test samples, {len(feature_cols)} features")
    
    # Domain features for ablation
    domain_features = [
        "Environmental_Risk", "Infrastructure_Vulnerability",
        "Anthropogenic_Pressure", "Hydrometeorological_Risk",
        "Water_Stress", "Infra_Gap", "Eco_Damage",
        "Siltation_Pressure", "Preparedness_Deficit"
    ]
    # Filter to only those that exist
    domain_features = [f for f in domain_features if f in X_train.columns]
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 ANOVA",
        "🔲 Chi-Square",
        "🔗 Pearson",
        "🔍 VIF",
        "✂️ Ablation",
        "📊 KS Drift",
        "🌊 Noise Stress"
    ])
    
    # --- ANOVA ---
    with tab1:
        st.markdown('<div class="stat-header">One-Way ANOVA: Feature Means Across Risk Classes</div>', unsafe_allow_html=True)
        st.caption("Tests whether feature means differ significantly across Low/Medium/High risk classes.")
        
        with st.spinner("Running ANOVA..."):
            anova_df = run_anova(X_train, y_train)
        
        # Summary metrics
        n_sig = anova_df["significant"].sum()
        n_total = len(anova_df)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Features Tested", n_total)
        with col2:
            st.metric("Significant (p<0.05)", int(n_sig))
        with col3:
            st.metric("Not Significant", n_total - int(n_sig))
        
        # Display table
        display_df = anova_df.copy()
        display_df["p_value"] = display_df["p_value"].apply(format_p_value)
        display_df["f_statistic"] = display_df["f_statistic"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        display_df["significant"] = display_df["p_value"].apply(lambda p: "✅" if p != "N/A" and float(p) < 0.05 else "❌" if p != "N/A" else "❓")
        
        st.dataframe(
            display_df[["feature", "f_statistic", "p_value", "significant"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "feature": "Feature",
                "f_statistic": "F-Statistic",
                "p_value": "P-Value",
                "significant": "Sig."
            }
        )
        
        # Download
        csv = anova_df.to_csv(index=False)
        st.download_button("📥 Download ANOVA Results", csv, "anova_results.csv", "text/csv")
    
    # --- Chi-Square ---
    with tab2:
        st.markdown('<div class="stat-header">Chi-Square: Feature-Class Independence (Tertile Binned)</div>', unsafe_allow_html=True)
        st.caption("Continuous features binned into tertiles; tests independence from risk class.")
        
        with st.spinner("Running Chi-Square tests..."):
            chi2_df = run_chi2(X_train, y_train)
        
        n_sig = chi2_df["significant"].sum()
        n_total = len(chi2_df)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Features Tested", n_total)
        with col2:
            st.metric("Significant (p<0.05)", int(n_sig))
        with col3:
            st.metric("Not Significant", n_total - int(n_sig))
        
        display_df = chi2_df.copy()
        display_df["p_value"] = display_df["p_value"].apply(format_p_value)
        display_df["chi2_statistic"] = display_df["chi2_statistic"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        display_df["significant"] = display_df["p_value"].apply(lambda p: "✅" if p != "N/A" and float(p) < 0.05 else "❌" if p != "N/A" else "❓")
        
        st.dataframe(
            display_df[["feature", "chi2_statistic", "p_value", "dof", "significant"]],
            hide_index=True,
            use_container_width=True,
        )
        
        csv = chi2_df.to_csv(index=False)
        st.download_button("📥 Download Chi-Square Results", csv, "chi2_results.csv", "text/csv")
    
    # --- Pearson ---
    with tab3:
        st.markdown('<div class="stat-header">Pearson Correlation Significance Matrix</div>', unsafe_allow_html=True)
        st.caption("Feature-target correlations with statistical significance (p < 0.05).")
        
        with st.spinner("Computing correlations..."):
            pearson_df = run_pearson(X_train, y_train)
        
        n_sig = pearson_df["significant"].sum()
        n_total = len(pearson_df)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Features Tested", n_total)
        with col2:
            st.metric("Significant (p<0.05)", int(n_sig))
        with col3:
            max_corr = pearson_df["correlation"].abs().max()
            st.metric("Max |Correlation|", f"{max_corr:.4f}")
        
        display_df = pearson_df.copy()
        display_df["correlation"] = display_df["correlation"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        display_df["p_value"] = display_df["p_value"].apply(format_p_value)
        display_df["significant"] = display_df["significant"].apply(lambda x: "✅" if x else "❌")
        
        st.dataframe(
            display_df[["feature", "correlation", "p_value", "significant"]],
            hide_index=True,
            use_container_width=True,
        )
        
        csv = pearson_df.to_csv(index=False)
        st.download_button("📥 Download Pearson Results", csv, "pearson_results.csv", "text/csv")
    
    # --- VIF ---
    with tab4:
        st.markdown('<div class="stat-header">Variance Inflation Factor (VIF) — Multi-collinearity Check</div>', unsafe_allow_html=True)
        st.caption("VIF > 10: Severe multi-collinearity | VIF > 5: Moderate concern | VIF < 5: Low concern")
        
        with st.spinner("Computing VIF..."):
            vif_df = run_vif(X_train)
        
        severe = (vif_df["VIF"] > 10).sum()
        moderate = ((vif_df["VIF"] > 5) & (vif_df["VIF"] <= 10)).sum()
        low = (vif_df["VIF"] <= 5).sum()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Features", len(vif_df))
        with col2:
            st.metric("🔴 Severe (>10)", int(severe))
        with col3:
            st.metric("🟡 Moderate (5-10)", int(moderate))
        with col4:
            st.metric("🟢 Low (<5)", int(low))
        
        display_df = vif_df.copy()
        display_df["VIF"] = display_df["VIF"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        display_df["concern"] = display_df["VIF"].apply(render_vif_badge)
        
        st.dataframe(
            display_df[["feature", "VIF", "concern_level"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "feature": "Feature",
                "VIF": "VIF Score",
                "concern_level": "Concern Level"
            }
        )
        
        csv = vif_df.to_csv(index=False)
        st.download_button("📥 Download VIF Results", csv, "vif_results.csv", "text/csv")
    
    # --- Ablation ---
    with tab5:
        st.markdown('<div class="stat-header">Feature Ablation Study: Domain Feature Importance</div>', unsafe_allow_html=True)
        st.caption("Drops each domain feature iteratively and measures F1 drop on test set.")
        
        with st.spinner("Running feature ablation..."):
            ablation_df = run_ablation(pipeline, X_test, y_test, domain_features)
        
        # Summary
        baseline_row = ablation_df[ablation_df["feature_dropped"] == "NONE (baseline)"]
        if not baseline_row.empty:
            baseline = baseline_row["f1_weighted"].values[0]
            st.metric("Baseline F1-Weighted", f"{baseline:.4f}")
        
        display_df = ablation_df.copy()
        display_df["f1_weighted"] = display_df["f1_weighted"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        display_df["drop"] = display_df["drop"].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "N/A")
        
        st.dataframe(
            display_df[["feature_dropped", "f1_weighted", "drop"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "feature_dropped": "Feature Dropped",
                "f1_weighted": "F1 Score",
                "drop": "Δ F1"
            }
        )
        
        csv = ablation_df.to_csv(index=False)
        st.download_button("📥 Download Ablation Results", csv, "ablation_results.csv", "text/csv")
    
    # --- KS Drift ---
    with tab6:
        st.markdown('<div class="stat-header">Kolmogorov-Smirnov: Train vs Test Distribution Drift</div>', unsafe_allow_html=True)
        st.caption("Detects significant distribution shifts between training and test splits.")
        
        with st.spinner("Running KS tests..."):
            ks_df = run_ks_drift(X_train, X_test)
        
        n_drift = ks_df["drift_detected"].sum()
        n_total = len(ks_df)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Features Tested", n_total)
        with col2:
            st.metric("🔴 Drift Detected", int(n_drift))
        with col3:
            st.metric("🟢 No Drift", n_total - int(n_drift))
        
        display_df = ks_df.copy()
        display_df["p_value"] = display_df["p_value"].apply(format_p_value)
        display_df["ks_statistic"] = display_df["ks_statistic"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        display_df["drift"] = display_df["drift_detected"].apply(lambda x: "🔴 YES" if x else "🟢 NO")
        
        st.dataframe(
            display_df[["feature", "ks_statistic", "p_value", "drift"]],
            hide_index=True,
            use_container_width=True,
        )
        
        csv = ks_df.to_csv(index=False)
        st.download_button("📥 Download KS Drift Results", csv, "ks_drift_results.csv", "text/csv")
    
    # --- Noise Stress ---
    with tab7:
        st.markdown('<div class="stat-header">Gaussian Noise Stress Test: Robustness to Perturbations</div>', unsafe_allow_html=True)
        st.caption("Applies noise clipped to training percentiles; measures F1 degradation.")
        
        with st.spinner("Running noise stress test..."):
            noise_df = run_noise_stress(pipeline, X_test, y_test)
        
        baseline_f1 = noise_df["f1_weighted_mean"].iloc[0] if len(noise_df) > 0 else 0
        st.metric("Baseline F1-Weighted", f"{baseline_f1:.4f}")
        
        display_df = noise_df.copy()
        display_df["f1_weighted_mean"] = display_df["f1_weighted_mean"].apply(lambda x: f"{x:.4f}")
        display_df["f1_weighted_std"] = display_df["f1_weighted_std"].apply(lambda x: f"{x:.4f}")
        display_df["degradation"] = (baseline_f1 - display_df["f1_weighted_mean"]).apply(lambda x: f"{x:+.4f}")
        
        st.dataframe(
            display_df[["noise_level", "f1_weighted_mean", "f1_weighted_std", "degradation"]],
            hide_index=True,
            use_container_width=True,
        )
        
        # Chart
        st.markdown("### Performance Degradation Curve")
        chart_data = noise_df.set_index("noise_level")["f1_weighted_mean"]
        st.line_chart(chart_data)
        
        csv = noise_df.to_csv(index=False)
        st.download_button("📥 Download Noise Stress Results", csv, "noise_stress_results.csv", "text/csv")
    
    # --- Combined Report ---
    st.markdown("---")
    st.markdown('<div class="stat-header">📄 Combined Statistical Audit Report</div>', unsafe_allow_html=True)
    
    report_path = PROJECT_ROOT / "outputs" / "reports" / "statistical_audit.json"
    if report_path.exists():
        with open(report_path, "r") as f:
            audit_data = json.load(f)
        
        st.json(audit_data, expanded=False)
        
        with open(report_path, "r") as f:
            st.download_button(
                "📥 Download Full Audit (JSON)",
                f.read(),
                "statistical_audit.json",
                "application/json"
            )
    else:
        st.info("Run the full pipeline to generate the combined audit report.")


if __name__ == "__main__":
    main()
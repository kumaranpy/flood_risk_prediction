"""
Streamlit Page: Data Story — Domain Context & Problem Framing
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Page config
st.set_page_config(
    page_title="Data Story | Flood Risk Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .story-card {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    .story-header {
        font-size: 1.3rem;
        font-weight: 800;
        color: #F1F5F9;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-highlight {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(16, 185, 129, 0.1) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        font-weight: 600;
        margin-top: 4px;
    }
    .quote-box {
        background: rgba(139, 92, 246, 0.12);
        border-left: 4px solid #8B5CF6;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 16px 0;
        font-style: italic;
        color: #D8B4FE;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    # Hero
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(16, 24, 40, 0.95) 0%, rgba(15, 23, 42, 0.90) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 32px; margin-bottom: 24px;">
            <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(139, 92, 246, 0.15);
                        color: #A78BFA; border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 9999px;
                        padding: 4px 14px; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em;
                        text-transform: uppercase; margin-bottom: 12px;">
                📖 Data Story & Domain Context
            </div>
            <h1 style="margin: 0 0 8px 0; font-size: 2.5rem; font-weight: 800; color: #FFFFFF; line-height: 1.2;">
                Why Flood Risk Intelligence Matters
            </h1>
            <p style="margin: 0; color: #94A3B8; font-size: 1.05rem; max-width: 900px; line-height: 1.6;">
                Understanding the hydrological, infrastructural, and anthropogenic drivers of flood vulnerability
                through data science and machine learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Load data for stats
    train_path = PROJECT_ROOT / "data" / "splits" / "train.csv"
    if train_path.exists():
        train_df = pd.read_csv(train_path)
        feature_cols = [c for c in train_df.columns if c not in ["FloodProbability_raw", "RiskLevel"]]
        X = train_df[feature_cols].copy()
        y = train_df["RiskLevel"].copy()
    else:
        X, y = None, None
    
    # --- Section 1: The Problem ---
    st.markdown(
        """
        <div class="story-card">
            <div class="story-header">🌍 The Global Flood Crisis</div>
            <p style="color: #E2E8F0; font-size: 1.05rem; line-height: 1.7;">
                Floods are the most frequent and costly natural disasters worldwide, affecting over
                <strong>2 billion people</strong> and causing <strong>$1.3 trillion</strong> in economic losses 
                in the past two decades (UN Office for Disaster Risk Reduction, 2023). Climate change is 
                intensifying precipitation extremes, while rapid urbanization expands impervious surfaces 
                that reduce natural infiltration. Aging infrastructure — dams, levees, drainage networks — 
                compounds vulnerability in densely populated floodplains.
            </p>
            <div class="quote-box">
                "The question is not whether a flood will occur, but whether our systems are prepared 
                to predict, mitigate, and respond when it does."
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # --- Section 2: Key Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="metric-highlight">
                <div class="metric-value">2.3B</div>
                <div class="metric-label">People at Risk Globally</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="metric-highlight">
                <div class="metric-value">$1.3T</div>
                <div class="metric-label">Economic Losses (2000-2023)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-highlight">
                <div class="metric-value">70%</div>
                <div class="metric-label">Urban Population in Floodplains</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="metric-highlight">
                <div class="metric-value">2.5×</div>
                <div class="metric-label">Extreme Precipitation Increase</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # --- Section 3: The Dataset ---
    st.markdown(
        """
        <div class="story-card">
            <div class="story-header">📊 The Kaggle Playground S4E5 Synthetic Benchmark</div>
            <p style="color: #E2E8F0; font-size: 1.05rem; line-height: 1.7;">
                This project uses the <strong>Kaggle Playground Series Season 4 Episode 5 (S4E5)</strong> dataset — 
                a synthetic benchmark designed to simulate regional flood risk classification. While not real-world 
                operational data, it provides a controlled environment to develop and validate ML pipelines for 
                multi-class risk stratification.
            </p>
            
            <h4 style="color: #F1F5F9; margin-top: 16px; margin-bottom: 8px;">Dataset Characteristics</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if X is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class="story-card">
                    <div class="story-header">📈 Training Data Profile</div>
                    <ul style="color: #E2E8F0; line-height: 1.8;">
                        <li><strong>Samples:</strong> {len(X):,} (80% of 50,000)</li>
                        <li><strong>Features:</strong> {len(X.columns)} raw environmental & infrastructure factors</li>
                        <li><strong>Target:</strong> FloodProbability (continuous) → Discretized tertiles</li>
                        <li><strong>Classes:</strong> Low / Medium / High risk</li>
                        <li><strong>Class Balance:</strong> ~33% each (by design)</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            # Feature categories
            categories = {
                "🌧️ Meteorology": ["MonsoonIntensity", "ClimateChange", "CoastalVulnerability"],
                "🏔️ Topography & Hydrology": ["TopographyDrainage", "RiverManagement", "Watersheds", "Landslides"],
                "🏗️ Infrastructure": ["DamsQuality", "DrainageSystems", "DeterioratingInfrastructure", "PoliticalFactors"],
                "🌱 Land Use & Pressures": ["Deforestation", "Urbanization", "Siltation", "AgriculturalPractices", "Encroachments", "WetlandLoss", "PopulationScore", "InadequatePlanning"],
                "🚨 Preparedness": ["IneffectiveDisasterPreparedness"],
            }
            
            st.markdown(
                """
                <div class="story-card">
                    <div class="story-header">🎯 Feature Categories (20 Raw Factors)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            for cat, feats in categories.items():
                with st.expander(cat):
                    for f in feats:
                        if f in X.columns:
                            st.write(f"• **{f}**: {X[f].mean():.1f} ± {X[f].std():.1f} (range: {X[f].min():.0f}–{X[f].max():.0f})")
    
    # --- Section 4: Engineering Approach ---
    st.markdown(
        """
        <div class="story-card">
            <div class="story-header">⚙️ From Raw Factors to Risk Intelligence</div>
            <p style="color: #E2E8F0; font-size: 1.05rem; line-height: 1.7;">
                Raw environmental factors alone cannot capture the complex, non-linear interactions that 
                drive flood risk. Our pipeline transforms 20 raw factors into <strong>15 engineered features</strong> 
                across four categories:
            </p>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px; margin-top: 16px;">
                <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 16px;">
                    <div style="font-weight: 700; color: #38BDF8; margin-bottom: 8px;">🌧️ Domain Indices (4)</div>
                    <div style="color: #94A3B8; font-size: 0.9rem;">
                        Environmental_Risk, Infrastructure_Vulnerability,<br>
                        Anthropogenic_Pressure, Hydrometeorological_Risk
                    </div>
                </div>
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 16px;">
                    <div style="font-weight: 700; color: #10B981; margin-bottom: 8px;">🔗 Interactions (5)</div>
                    <div style="color: #94A3B8; font-size: 0.9rem;">
                        MonsoonIntensity × Urbanization<br>
                        Deforestation × RiverManagement<br>
                        ClimateChange × DamsQuality<br>
                        Siltation × AgriculturalPractices<br>
                        TopographyDrainage × MonsoonIntensity
                    </div>
                </div>
                <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 12px; padding: 16px;">
                    <div style="font-weight: 700; color: #F59E0B; margin-bottom: 8px;">⚖️ Ratio Features (5)</div>
                    <div style="color: #94A3B8; font-size: 0.9rem;">
                        Water_Stress, Infra_Gap, Eco_Damage,<br>
                        Siltation_Pressure, Preparedness_Deficit
                    </div>
                </div>
                <div style="background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 12px; padding: 16px;">
                    <div style="font-weight: 700; color: #A78BFA; margin-bottom: 8px;">📊 Row Statistics (1)</div>
                    <div style="color: #94A3B8; font-size: 0.9rem;">
                        Row_Mean (captures S4E5 linear target structure)
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # --- Section 5: Model Philosophy ---
    st.markdown(
        """
        <div class="story-card">
            <div class="story-header">🧠 Modeling Philosophy: Inductive Bias Alignment</div>
            <p style="color: #E2E8F0; font-size: 1.05rem; line-height: 1.7;">
                The S4E5 synthetic target is a <strong>linear combination of the 20 input features</strong>. 
                This means the true decision boundary is a hyperplane in feature space — a classic case where 
                <strong>regularized linear models (Logistic Regression) outperform complex tree ensembles</strong> 
                (XGBoost, LightGBM) because their inductive bias matches the data-generating process.
            </p>
            
            <h4 style="color: #F1F5F9; margin-top: 16px; margin-bottom: 8px;">Why This Matters</h4>
            <ul style="color: #E2E8F0; line-height: 1.8;">
                <li><strong>Linear models</strong> naturally learn the correct decision boundary with minimal overfitting</li>
                <li><strong>Tree ensembles</strong> waste capacity learning axis-aligned splits to approximate diagonal boundaries</li>
                <li><strong>Engineered interactions & ratios</strong> give linear models the non-linear expressiveness they need without tree overfitting</li>
                <li><strong>Result:</strong> Logistic Regression achieves 98.24% accuracy with excellent calibration (Brier = 0.0355)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # --- Section 6: Limitations ---
    st.markdown(
        """
        <div class="story-card" style="border-color: rgba(239, 68, 68, 0.4);">
            <div class="story-header">⚠️ Limitations & Honest Assessment</div>
            <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 16px;">
                <h4 style="color: #FCA5A5; margin-bottom: 12px;">🔴 Critical Limitations</h4>
                <ul style="color: #FCA5A5; line-height: 1.8;">
                    <li><strong>Synthetic Data Only:</strong> Trained on Kaggle S4E5 benchmark — NOT real hydrological observations</li>
                    <li><strong>Scale-Shift Fragility:</strong> 1.5× feature scaling drops F1 from 0.98 to 0.18 — model learns magnitudes, not relationships</li>
                    <li><strong>No Temporal Dynamics:</strong> Static snapshot; real floods evolve over hours/days</li>
                    <li><strong>No Spatial Dependencies:</strong> Regions treated independently; ignores watershed connectivity</li>
                    <li><strong>Class Balance Assumption:</strong> Real-world flood events are rare (severe class imbalance)</li>
                </ul>
            </div>
            <p style="color: #E2E8F0; margin-top: 16px; font-size: 1.05rem; line-height: 1.7;">
                <strong>This model is for educational, research, and benchmarking purposes only.</strong>
                <br><strong>It must NOT be used for operational disaster management or life-safety decisions.</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # --- Section 7: Navigation ---
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <h3 style="color: #F1F5F9; margin-bottom: 16px;">Navigate the Intelligence System</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    nav_cols = st.columns(5)
    pages = [
        ("🏠", "Main Dashboard", "🏠 Main"),
        ("📖", "Data Story", "📖 Story"),
        ("📊", "Statistical EDA", "📊 EDA"),
        ("🔍", "Feature Importance", "🔍 Features"),
        ("🔄", "What-If Analysis", "🔄 What-If"),
        ("💰", "Business Impact", "💰 ROI"),
    ]
    
    for i, (icon, title, page) in enumerate(pages):
        with nav_cols[i % 5]:
            st.markdown(
                f"""
                <div style="text-align: center; padding: 16px; background: rgba(17, 24, 39, 0.8); 
                            border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px;">
                    <div style="font-size: 2rem;">{icon}</div>
                    <div style="font-weight: 600; color: #F1F5F9; margin-top: 8px;">{title}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
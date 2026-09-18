"""
Interactive Flood Risk Assessment Dashboard.

Architectural Guarantees:
  1. Parity: Passes raw user slider inputs directly into models/best_pipeline.pkl.
  2. Security: Verifies SHA-256 checksum against models/checksums.json before loading.
  3. Safety & Compliance (F-10): Prominently renders academic simulation disclaimer;
     prohibits operational directives (no evacuation or spillway orders).
  4. Centralized Feature Logic (F-09): Derives domain indices strictly via src.features.
"""

import hashlib
import json
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import DOMAIN_FEATURE_DEFINITIONS, compute_domain_features_dict
from src.predict import EXPECTED_RAW_FEATURES, load_verified_pipeline, predict_single_instance, SecurityError

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
BEST_PIPELINE_PATH = MODELS_DIR / "best_pipeline.pkl"
COMPARISON_CSV_PATH = REPORTS_DIR / "model_comparison.csv"
CHECKSUMS_PATH = MODELS_DIR / "checksums.json"

# Predefined input features with their display names and descriptions
ALL_INPUT_FACTORS = {
    "MonsoonIntensity": ("Monsoon Intensity", "Precipitation volume, duration, and peak storm surge intensity", 0.0, 15.0, 6.0),
    "TopographyDrainage": ("Topography & Drainage", "Slope gradient, natural runoff channels, and elevation variance", 0.0, 15.0, 5.0),
    "RiverManagement": ("River Management", "Condition of river embankments, levee maintenance, and canal dredging", 0.0, 15.0, 5.0),
    "Deforestation": ("Deforestation Rate", "Loss of vegetative soil cover, root binding, and canopy interception", 0.0, 15.0, 5.0),
    "Urbanization": ("Urbanization & Impervious Area", "Paved surface density reducing natural infiltration and groundwater percolation", 0.0, 15.0, 6.0),
    "ClimateChange": ("Climate Change Anomaly", "Deviation in historical precipitation regimes and extreme weather events", 0.0, 15.0, 6.0),
    "DamsQuality": ("Dam & Reservoir Integrity", "Structural condition, spillway capacity, and silt accumulation in reservoirs", 0.0, 15.0, 5.0),
    "Siltation": ("Riverbed Siltation", "Sediment buildup reducing hydrological conveyance capacity", 0.0, 15.0, 5.0),
    "AgriculturalPractices": ("Agricultural Land Use", "Terracing, soil tilling intensity, and floodplain agriculture", 0.0, 15.0, 5.0),
    "Encroachments": ("Floodplain Encroachment", "Unregulated settlements and industrial structures within floodways", 0.0, 15.0, 5.0),
    "IneffectiveDisasterPreparedness": ("Preparedness Deficit", "Gaps in early warning dissemination, evacuation routes, and disaster response", 0.0, 15.0, 5.0),
    "DrainageSystems": ("Urban Drainage Capacity", "Culvert dimensions, storm sewer networks, and pumping station readiness", 0.0, 15.0, 5.0),
    "CoastalVulnerability": ("Coastal Vulnerability", "Susceptibility to tidal surges, storm tides, and sea-level rise", 0.0, 15.0, 5.0),
    "Landslides": ("Landslide Hazard", "Unstable slopes contributing debris dams and sudden flash flood surges", 0.0, 15.0, 4.0),
    "Watersheds": ("Watershed Degradation", "Degraded catchment areas unable to retain seasonal surface runoff", 0.0, 15.0, 5.0),
    "DeterioratingInfrastructure": ("Aging Infrastructure", "Lifespan fatigue on bridges, retaining walls, and drainage barriers", 0.0, 15.0, 5.0),
    "PopulationScore": ("Population Density in Flood Zone", "Concentration of communities in low-lying hazard zones", 0.0, 15.0, 6.0),
    "WetlandLoss": ("Wetland & Floodplain Loss", "Destruction of natural retention basins and marsh buffer areas", 0.0, 15.0, 5.0),
    "InadequatePlanning": ("Inadequate Urban Planning", "Zoning failures and lack of sustainable storm drainage Master Plans", 0.0, 15.0, 5.0),
    "PoliticalFactors": ("Policy & Governance Gaps", "Delayed maintenance funding and fragmented disaster management coordination", 0.0, 15.0, 5.0),
}

# Quick scenario presets
PRESETS = {
    "🚨 Severe Storm & Infrastructure Stress": {
        "MonsoonIntensity": 12.0, "TopographyDrainage": 10.0, "RiverManagement": 9.0, "Deforestation": 9.0,
        "Urbanization": 10.0, "ClimateChange": 11.0, "DamsQuality": 11.0, "Siltation": 10.0,
        "AgriculturalPractices": 8.0, "Encroachments": 9.0, "IneffectiveDisasterPreparedness": 10.0,
        "DrainageSystems": 9.0, "CoastalVulnerability": 8.0, "Landslides": 8.0, "Watersheds": 9.0,
        "DeterioratingInfrastructure": 10.0, "PopulationScore": 10.0, "WetlandLoss": 9.0,
        "InadequatePlanning": 10.0, "PoliticalFactors": 8.0
    },
    "🌊 Urban Drainage Saturation": {
        "MonsoonIntensity": 9.0, "TopographyDrainage": 7.0, "RiverManagement": 6.0, "Deforestation": 5.0,
        "Urbanization": 11.0, "ClimateChange": 8.0, "DamsQuality": 5.0, "Siltation": 6.0,
        "AgriculturalPractices": 4.0, "Encroachments": 8.0, "IneffectiveDisasterPreparedness": 6.0,
        "DrainageSystems": 9.0, "CoastalVulnerability": 5.0, "Landslides": 3.0, "Watersheds": 5.0,
        "DeterioratingInfrastructure": 7.0, "PopulationScore": 9.0, "WetlandLoss": 8.0,
        "InadequatePlanning": 9.0, "PoliticalFactors": 5.0
    },
    "☀️ Nominal Seasonal Containment": {
        "MonsoonIntensity": 3.0, "TopographyDrainage": 2.0, "RiverManagement": 2.0, "Deforestation": 2.0,
        "Urbanization": 3.0, "ClimateChange": 2.0, "DamsQuality": 2.0, "Siltation": 2.0,
        "AgriculturalPractices": 2.0, "Encroachments": 1.0, "IneffectiveDisasterPreparedness": 2.0,
        "DrainageSystems": 2.0, "CoastalVulnerability": 2.0, "Landslides": 1.0, "Watersheds": 2.0,
        "DeterioratingInfrastructure": 2.0, "PopulationScore": 2.0, "WetlandLoss": 2.0,
        "InadequatePlanning": 2.0, "PoliticalFactors": 2.0
    }
}


@st.cache_resource
def load_ml_assets():
    """Securely loads and verifies the trained ML pipeline artifact."""
    pipeline = load_verified_pipeline(BEST_PIPELINE_PATH)
    comparison_df = pd.read_csv(COMPARISON_CSV_PATH) if COMPARISON_CSV_PATH.exists() else None
    return pipeline, comparison_df


def inject_custom_css():
    """Injects modern dark-mode styles and glassmorphism."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .hero-banner {
            background: linear-gradient(135deg, rgba(16, 24, 40, 0.95) 0%, rgba(15, 23, 42, 0.90) 50%, rgba(30, 41, 59, 0.85) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 24px 32px;
            margin-bottom: 20px;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(12px);
        }

        .badge-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(56, 189, 248, 0.15);
            color: #38BDF8;
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 9999px;
            padding: 4px 14px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .disclaimer-banner {
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.35);
            border-radius: 12px;
            padding: 14px 20px;
            margin-bottom: 24px;
            color: #FCA5A5;
            font-size: 0.88rem;
            line-height: 1.45;
        }

        .glass-card {
            background: rgba(17, 24, 39, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 18px 20px;
            backdrop-filter: blur(8px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        }

        .metric-val {
            font-size: 2.0rem;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            line-height: 1.1;
        }

        .metric-lbl {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94A3B8;
            font-weight: 600;
            margin-top: 4px;
        }

        .risk-banner-high {
            background: linear-gradient(135deg, rgba(127, 29, 29, 0.85) 0%, rgba(185, 28, 28, 0.75) 100%);
            border: 1px solid rgba(239, 68, 68, 0.4);
            border-radius: 16px;
            padding: 22px 26px;
            box-shadow: 0 10px 30px rgba(239, 68, 68, 0.25);
        }

        .risk-banner-medium {
            background: linear-gradient(135deg, rgba(120, 53, 15, 0.85) 0%, rgba(180, 83, 9, 0.75) 100%);
            border: 1px solid rgba(245, 158, 11, 0.4);
            border-radius: 16px;
            padding: 22px 26px;
            box-shadow: 0 10px 30px rgba(245, 158, 11, 0.25);
        }

        .risk-banner-low {
            background: linear-gradient(135deg, rgba(6, 78, 59, 0.85) 0%, rgba(16, 185, 129, 0.65) 100%);
            border: 1px solid rgba(16, 185, 129, 0.4);
            border-radius: 16px;
            padding: 22px 26px;
            box-shadow: 0 10px 30px rgba(16, 185, 129, 0.25);
        }

        .prob-bar-container {
            background: rgba(255, 255, 255, 0.06);
            border-radius: 999px;
            height: 8px;
            overflow: hidden;
            margin-top: 6px;
            margin-bottom: 12px;
        }

        .prob-bar-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.4s ease;
        }

        .prob-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.86rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Flood Risk Intelligence Dashboard",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # Load Assets with Security Error Handling
    try:
        pipeline, comparison_df = load_ml_assets()
    except SecurityError as sec_err:
        st.error(f"🚨 Security Error: {sec_err}")
        st.stop()
    except Exception as err:
        st.error(f"Failed to load ML pipeline: {err}")
        st.stop()

    # Hero Banner
    st.markdown(
        """
        <div class="hero-banner">
            <div class="badge-pill">🛡️ Academic AI Benchmark & Simulation</div>
            <h1 style="margin: 0 0 6px 0; font-size: 2.2rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">
                Regional Flood Risk Intelligence System
            </h1>
            <p style="margin: 0; color: #94A3B8; font-size: 0.95rem; max-width: 900px; line-height: 1.5;">
                Real-time ML risk assessment classifying regions into Low, Medium, or High categories
                using calibrated ensemble pipelines trained on synthetic benchmark environmental factors.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # MANDATORY DISCLAIMER BANNER (F-10 / Operational Safety)
    st.markdown(
        """
        <div class="disclaimer-banner">
            <b style="color: #F87171; text-transform: uppercase; font-size: 0.86rem; letter-spacing: 0.05em;">⚠️ SIMULATION ONLY:</b>
            This system is an academic demonstration trained on synthetic benchmark data (Kaggle Playground s4e5).
            It must <b>NOT</b> be used for operational disaster management, early-warning deployment, or life-safety decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize Session State
    if "input_state" not in st.session_state:
        st.session_state.input_state = {k: v[4] for k, v in ALL_INPUT_FACTORS.items()}

    # Sidebar: Scenario Presets
    with st.sidebar:
        st.markdown("<h3 style='color:#F1F5F9; font-weight:700; margin-bottom:8px;'>⚡ Scenario Presets</h3>", unsafe_allow_html=True)
        st.caption("Apply standard simulated profiles to explore model sensitivity.")

        for preset_name, preset_vals in PRESETS.items():
            if st.button(preset_name, use_container_width=True):
                for k, v in preset_vals.items():
                    st.session_state.input_state[k] = v
                st.rerun()

        st.markdown("---")
        st.markdown("<h4 style='color:#E2E8F0; font-weight:600;'>System & Pipeline Info</h4>", unsafe_allow_html=True)
        st.markdown(
            """
            - **Pipeline**: `CalibratedClassifierCV`
            - **Target**: Discretized Tertiles (`Low`, `Medium`, `High`)
            - **Integrity**: SHA-256 Verified (`checksums.json`)
            - **Preprocessing**: Fold-Safe In-Pipeline Scaling
            """
        )

    # Main Grid: Inputs (Left) and Live Assessment (Right)
    col_inputs, col_pred = st.columns([1.15, 1.0], gap="large")

    with col_inputs:
        st.markdown("<h3 style='color:#F1F5F9; font-weight:700; margin-bottom:12px;'>🎛️ Regional Factor Inputs</h3>", unsafe_allow_html=True)
        st.caption("Adjust factor values (Scale: 0.0 - 15.0) to simulate environmental and infrastructure conditions.")

        tab1, tab2, tab3 = st.tabs([
            "🌧️ Meteorology & Catchment",
            "🏗️ Infrastructure & Defenses",
            "🌱 Land Use & Pressures",
        ])

        tab_grouping = {
            tab1: ["MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Watersheds", "CoastalVulnerability", "Landslides"],
            tab2: ["DamsQuality", "DrainageSystems", "DeterioratingInfrastructure", "IneffectiveDisasterPreparedness", "PoliticalFactors"],
            tab3: ["Urbanization", "Deforestation", "ClimateChange", "Siltation", "AgriculturalPractices", "Encroachments", "PopulationScore", "WetlandLoss", "InadequatePlanning"],
        }

        user_inputs = {}
        for tab, factor_keys in tab_grouping.items():
            with tab:
                for key in factor_keys:
                    label, desc, min_val, max_val, _ = ALL_INPUT_FACTORS[key]
                    curr_val = float(st.session_state.input_state.get(key, 5.0))
                    val = st.slider(
                        label=label,
                        min_value=min_val,
                        max_value=max_val,
                        value=curr_val,
                        step=0.5,
                        help=desc,
                        key=f"slider_{key}",
                    )
                    user_inputs[key] = val
                    st.session_state.input_state[key] = val

    # Ensure all 20 raw features present
    for k in ALL_INPUT_FACTORS:
        if k not in user_inputs:
            user_inputs[k] = float(st.session_state.input_state.get(k, 5.0))

    # Real-Time Inference Execution via Pipeline Parity
    with col_pred:
        st.markdown("<h3 style='color:#F1F5F9; font-weight:700; margin-bottom:12px;'>📊 Live Risk Assessment</h3>", unsafe_allow_html=True)

        t_start = time.time()
        pred_class, confidence, prob_dict, decision = predict_single_instance(user_inputs, pipeline=pipeline)
        inference_latency_ms = (time.time() - t_start) * 1000.0

        # Centralized Domain Feature Calculations (from src.features)
        domain_feats = compute_domain_features_dict(user_inputs)

        # Style Banner by Risk Level
        risk_configs = {
            "High": {
                "class_name": "risk-banner-high",
                "emoji": "🔴",
                "color": "#EF4444",
                "tag": "HIGH RISK SIMULATION",
                "summary": "Simulated multi-factor conditions indicate critical vulnerability under hypothetical stress.",
                "note": "For academic simulation analysis only. Not an authorized emergency directive."
            },
            "Medium": {
                "class_name": "risk-banner-medium",
                "emoji": "🟡",
                "color": "#F59E0B",
                "tag": "MODERATE RISK SIMULATION",
                "summary": "Simulated conditions exhibit elevated parameter pressures approaching thresholds.",
                "note": "Elevated parameter scores reflect heightened simulated sensitivity."
            },
            "Low": {
                "class_name": "risk-banner-low",
                "emoji": "🟢",
                "color": "#10B981",
                "tag": "NOMINAL RISK SIMULATION",
                "summary": "Simulated parameters fall within standard baseline containment bounds.",
                "note": "Parameters reflect typical seasonal stability."
            },
        }
        cfg = risk_configs.get(pred_class, risk_configs["Medium"])

        # Render Risk Banner
        st.markdown(
            f"""
            <div class="{cfg['class_name']}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="background: {cfg['color']}; color: #FFFFFF; font-weight: 800; font-size: 0.78rem; padding: 4px 12px; border-radius: 999px; letter-spacing: 0.06em;">
                        {cfg['tag']}
                    </span>
                    <span style="color: #E2E8F0; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">
                        ⚡ Latency: {inference_latency_ms:.1f}ms
                    </span>
                </div>
                <div style="display: flex; align-items: baseline; gap: 12px;">
                    <h2 style="margin: 0; font-size: 2.3rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">
                        {cfg['emoji']} {pred_class.upper()} RISK
                    </h2>
                    <span style="font-size: 1.4rem; font-weight: 700; color: {cfg['color']}; font-family: 'JetBrains Mono', monospace;">
                        {confidence:.1f}%
                    </span>
                </div>
                <p style="margin: 10px 0 0 0; color: #F1F5F9; font-size: 0.95rem; line-height: 1.45;">
                    {cfg['summary']}
                </p>
                <div style="margin-top: 14px; padding: 10px 14px; background: rgba(0, 0, 0, 0.25); border-radius: 8px; border-left: 3px solid {cfg['color']};">
                    <span style="color: #CBD5E1; font-size: 0.85rem;">ℹ️ {cfg['note']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Calibrated Probability Breakdown
        st.markdown("<h4 style='color:#E2E8F0; font-weight:700; margin-bottom:12px;'>🎯 Calibrated Probabilities</h4>", unsafe_allow_html=True)
        bar_colors = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}
        for cls in ["High", "Medium", "Low"]:
            prob_val = prob_dict.get(cls, 0.0)
            pct_val = prob_val * 100.0
            fill_color = bar_colors.get(cls, "#38BDF8")
            st.markdown(
                f"""
                <div class="prob-row">
                    <span style="color: #F1F5F9;">{cls} Risk</span>
                    <span style="font-family: 'JetBrains Mono', monospace; color: {fill_color};">{pct_val:5.1f}%</span>
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill" style="width: {pct_val}%; background: {fill_color};"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Domain Sub-Indices (Zero Global Row Aggregators)
        st.markdown("<h4 style='color:#E2E8F0; font-weight:700; margin: 18px 0 10px 0;'>🧪 Sub-Domain Risk Indices</h4>", unsafe_allow_html=True)
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center;">
                    <div class="metric-val" style="color: #38BDF8;">{domain_feats.get('Environmental_Risk', 5.0):.2f}</div>
                    <div class="metric-lbl">Environmental</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col2:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center;">
                    <div class="metric-val" style="color: #F59E0B;">{domain_feats.get('Infrastructure_Vulnerability', 5.0):.2f}</div>
                    <div class="metric-lbl">Infrastructure</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col3:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center;">
                    <div class="metric-val" style="color: #A855F7;">{domain_feats.get('Anthropogenic_Pressure', 5.0):.2f}</div>
                    <div class="metric-lbl">Anthropogenic</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Bottom Section: Model Benchmarks & Integrity
    st.markdown("<h3 style='color:#F1F5F9; font-weight:700; margin-bottom:16px;'>📈 Pipeline Benchmarking & Validation</h3>", unsafe_allow_html=True)
    bench_col1, bench_col2 = st.columns([1.0, 1.0], gap="large")

    with bench_col1:
        st.markdown("<h4 style='color:#E2E8F0; font-weight:700;'>📋 Held-out Test Partition Metrics</h4>", unsafe_allow_html=True)
        st.caption("Benchmark evaluated strictly on 10,000 held-out samples without test leakage.")

        if comparison_df is not None:
            styled_df = comparison_df.copy()
            styled_df["Accuracy"] = (styled_df["Accuracy"] * 100).map("{:.2f}%".format)
            styled_df["F1_Weighted"] = styled_df["F1_Weighted"].map("{:.4f}".format)
            styled_df["ROC_AUC"] = styled_df["ROC_AUC"].map("{:.4f}".format)
            styled_df["Brier_Score"] = styled_df["Brier_Score"].map("{:.4f}".format)

            st.dataframe(
                styled_df[["Model", "Accuracy", "F1_Weighted", "ROC_AUC", "Brier_Score"]],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.warning("Model comparison data not found. Run evaluate.py to generate benchmarks.")

    with bench_col2:
        st.markdown("<h4 style='color:#E2E8F0; font-weight:700;'>🛡️ Integrity & Calibration Verification</h4>", unsafe_allow_html=True)
        st.caption("Verification guarantees enforcing production reliability.")

        st.markdown(
            """
            - **Zero Target Proxy Leakage**: No row-wise global aggregations ($|r| < 0.85$).
            - **Fold-Safe Pipeline**: Preprocessing & scaling fit strictly within training folds.
            - **Calibrated Probabilities**: Sigmoid/isotonic calibration prevents overconfidence.
            - **Cryptographic Guardrails**: SHA-256 verified deserialization blocks untrusted pickles.
            """
        )


if __name__ == "__main__":
    main()
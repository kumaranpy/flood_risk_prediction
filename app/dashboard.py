import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
TARGET_ENCODER_PATH = MODELS_DIR / "target_encoder.pkl"
SELECTED_FEATURES_PATH = MODELS_DIR / "selected_features.pkl"
COMPARISON_CSV_PATH = REPORTS_DIR / "model_comparison.csv"

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
    "🚨 Severe Monsoon & Dam Failure": {
        "MonsoonIntensity": 12.0, "TopographyDrainage": 10.0, "RiverManagement": 9.0, "Deforestation": 9.0,
        "Urbanization": 10.0, "ClimateChange": 11.0, "DamsQuality": 11.0, "Siltation": 10.0,
        "AgriculturalPractices": 8.0, "Encroachments": 9.0, "IneffectiveDisasterPreparedness": 10.0,
        "DrainageSystems": 9.0, "CoastalVulnerability": 8.0, "Landslides": 8.0, "Watersheds": 9.0,
        "DeterioratingInfrastructure": 10.0, "PopulationScore": 10.0, "WetlandLoss": 9.0,
        "InadequatePlanning": 10.0, "PoliticalFactors": 8.0
    },
    "🌊 Urban Flash Flood": {
        "MonsoonIntensity": 9.0, "TopographyDrainage": 7.0, "RiverManagement": 6.0, "Deforestation": 5.0,
        "Urbanization": 11.0, "ClimateChange": 8.0, "DamsQuality": 5.0, "Siltation": 6.0,
        "AgriculturalPractices": 4.0, "Encroachments": 8.0, "IneffectiveDisasterPreparedness": 6.0,
        "DrainageSystems": 9.0, "CoastalVulnerability": 5.0, "Landslides": 3.0, "Watersheds": 5.0,
        "DeterioratingInfrastructure": 7.0, "PopulationScore": 9.0, "WetlandLoss": 8.0,
        "InadequatePlanning": 9.0, "PoliticalFactors": 5.0
    },
    "☀️ Normal Seasonal / Low Risk": {
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
    """Caches and loads all ML models, encoders, scalers, and metadata."""
    model = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    target_encoder = joblib.load(TARGET_ENCODER_PATH) if TARGET_ENCODER_PATH.exists() else None
    selected_features = joblib.load(SELECTED_FEATURES_PATH)
    comparison_df = pd.read_csv(COMPARISON_CSV_PATH) if COMPARISON_CSV_PATH.exists() else None
    return model, scaler, target_encoder, selected_features, comparison_df


def compute_derived_indicators(inputs: dict) -> dict:
    """Computes real-time composite indices matching features.py."""
    data = dict(inputs)

    # 1. Flood Vulnerability Score (FR-08)
    weights = {
        "MonsoonIntensity": 0.35,
        "TopographyDrainage": 0.25,
        "RiverManagement": 0.20,
        "Deforestation": 0.20,
    }
    avail = {k: v for k, v in weights.items() if k in data}
    if avail:
        norm_w = {k: v / sum(avail.values()) for k, v in avail.items()}
        data["Flood_Vulnerability_Score"] = sum(data[k] * w for k, w in norm_w.items())
    else:
        data["Flood_Vulnerability_Score"] = 5.0

    # 2. Infrastructure Deficit Score
    infra = ["DamsQuality", "DrainageSystems", "DeterioratingInfrastructure", "IneffectiveDisasterPreparedness"]
    avail_infra = [data[c] for c in infra if c in data]
    data["Infrastructure_Deficit_Score"] = float(np.mean(avail_infra)) if avail_infra else 5.0

    # 3. Environmental Stress Score
    env = ["Urbanization", "ClimateChange", "AgriculturalPractices", "Encroachments", "WetlandLoss"]
    avail_env = [data[c] for c in env if c in data]
    data["Environmental_Stress_Score"] = float(np.mean(avail_env)) if avail_env else 5.0

    # 4. Aggregate Hazard Index
    numeric_vals = [v for k, v in data.items() if isinstance(v, (int, float))]
    data["Aggregate_Hazard_Index"] = float(np.mean(numeric_vals)) if numeric_vals else 5.0

    return data


def inject_custom_css():
    """Injects high-end glassmorphism and modern dark-mode styles."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Top Hero Header Banner */
        .hero-banner {
            background: linear-gradient(135deg, rgba(16, 24, 40, 0.95) 0%, rgba(15, 23, 42, 0.90) 50%, rgba(30, 41, 59, 0.85) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 28px 36px;
            margin-bottom: 24px;
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
            margin-bottom: 12px;
        }

        /* Glassmorphism Metric Cards */
        .glass-card {
            background: rgba(17, 24, 39, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        .glass-card:hover {
            border-color: rgba(56, 189, 248, 0.3);
            transform: translateY(-2px);
        }

        /* Risk Banners */
        .risk-banner-high {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.25) 0%, rgba(185, 28, 28, 0.35) 100%);
            border: 2px solid #EF4444;
            border-radius: 18px;
            padding: 24px;
            box-shadow: 0 0 30px rgba(239, 68, 68, 0.3);
            animation: pulse-red 2.5s infinite;
        }

        .risk-banner-medium {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.25) 0%, rgba(180, 83, 9, 0.35) 100%);
            border: 2px solid #F59E0B;
            border-radius: 18px;
            padding: 24px;
            box-shadow: 0 0 30px rgba(245, 158, 11, 0.3);
            animation: pulse-amber 2.5s infinite;
        }

        .risk-banner-low {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(4, 120, 87, 0.35) 100%);
            border: 2px solid #10B981;
            border-radius: 18px;
            padding: 24px;
            box-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
            animation: pulse-green 2.5s infinite;
        }

        @keyframes pulse-red {
            0%, 100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.25); }
            50% { box-shadow: 0 0 35px rgba(239, 68, 68, 0.55); }
        }
        @keyframes pulse-amber {
            0%, 100% { box-shadow: 0 0 20px rgba(245, 158, 11, 0.25); }
            50% { box-shadow: 0 0 35px rgba(245, 158, 11, 0.55); }
        }
        @keyframes pulse-green {
            0%, 100% { box-shadow: 0 0 20px rgba(16, 185, 129, 0.25); }
            50% { box-shadow: 0 0 35px rgba(16, 185, 129, 0.55); }
        }

        /* Custom Progress Bar Styling */
        .prob-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 0.92rem;
            font-weight: 600;
        }
        .prob-bar-container {
            height: 10px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 999px;
            overflow: hidden;
            margin-bottom: 16px;
        }
        .prob-bar-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Clean metric numbers */
        .metric-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.8rem;
            font-weight: 700;
            color: #F8FAFC;
        }
        .metric-lbl {
            font-size: 0.82rem;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Flood Risk Prediction System | Early Warning AI",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()
    model, scaler, target_encoder, selected_features, comparison_df = load_ml_assets()

    # Hero Banner
    st.markdown(
        """
        <div class="hero-banner">
            <span class="badge-pill">⚡ Early Warning System &bull; Production ML Pipeline</span>
            <h1 style="margin: 0 0 8px 0; font-size: 2.2rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">
                🌊 Regional Flood Risk Intelligence Platform
            </h1>
            <p style="margin: 0; color: #94A3B8; font-size: 1.05rem; max-width: 850px; line-height: 1.5;">
                Real-time multi-hazard classification engine categorizing regional vulnerabilities into
                <b>High</b>, <b>Medium</b>, and <b>Low</b> flood risk tiers using ensemble machine learning and hydrometeorological indicators.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize Session State for Inputs
    if "input_state" not in st.session_state:
        st.session_state.input_state = {k: v[4] for k, v in ALL_INPUT_FACTORS.items()}

    # Preset Selector in Top Row
    preset_cols = st.columns([1.2, 1, 1, 1])
    with preset_cols[0]:
        st.markdown("<p style='font-weight:700; color:#E2E8F0; margin-top:8px;'>⚡ Load Scenario Preset:</p>", unsafe_allow_html=True)
    for i, (name, preset_vals) in enumerate(PRESETS.items()):
        with preset_cols[i + 1]:
            if st.button(name, use_container_width=True):
                st.session_state.input_state.update(preset_vals)
                st.rerun()

    # Main Layout: 2 Columns (Inputs Sidebar/Tabs on Left, Real-Time Prediction & Insights on Right)
    col_input, col_pred = st.columns([1.1, 1.0], gap="large")

    with col_input:
        st.markdown("<h3 style='color:#F1F5F9; font-weight:700; margin-bottom:12px;'>🎛️ Regional Factor Inputs (FR-21)</h3>", unsafe_allow_html=True)
        st.caption("Adjust the sliders below to simulate environmental, meteorological, and infrastructure conditions (Scale: 0 - 15).")

        tab1, tab2, tab3 = st.tabs([
            "🌧️ Meteorology & Watershed",
            "🏗️ Infrastructure & Defense",
            "🌱 Land Use & Urban Pressures",
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

    # Fill in any missing factor defaults
    for k in ALL_INPUT_FACTORS:
        if k not in user_inputs:
            user_inputs[k] = float(st.session_state.input_state.get(k, 5.0))

    # Real-Time Inference Execution
    with col_pred:
        st.markdown("<h3 style='color:#F1F5F9; font-weight:700; margin-bottom:12px;'>📊 Live Risk Assessment</h3>", unsafe_allow_html=True)

        t_start = time.time()
        enriched = compute_derived_indicators(user_inputs)
        input_row = pd.DataFrame([enriched])

        # Guarantee all 12 selected features in exact model order
        for feat in selected_features:
            if feat not in input_row.columns:
                input_row[feat] = 5.0
        input_row = input_row[selected_features]

        # Scaler transformation (safeguards against data leakage)
        scaled_array = scaler.transform(input_row)
        scaled_input = pd.DataFrame(scaled_array, columns=selected_features)

        # Predict risk class & probabilities
        pred_raw = model.predict(scaled_input)[0]
        if target_encoder is not None:
            pred_class = target_encoder.inverse_transform([pred_raw])[0]
            class_order = list(target_encoder.classes_)
        else:
            map_cls = {0: "Low", 1: "Medium", 2: "High"}
            pred_class = map_cls.get(pred_raw, str(pred_raw))
            class_order = ["Low", "Medium", "High"]

        probabilities = model.predict_proba(scaled_input)[0]
        inference_latency_ms = (time.time() - t_start) * 1000.0

        # Style Banner by Risk Level
        risk_configs = {
            "High": {
                "class_name": "risk-banner-high",
                "emoji": "🔴",
                "color": "#EF4444",
                "tag": "CRITICAL RISK",
                "summary": "Severe regional flood hazard detected. Impending breach of hydrological thresholds.",
                "action": "Trigger emergency flood alarms, open auxiliary spillways, and initiate evacuation protocols for low-lying zones."
            },
            "Medium": {
                "class_name": "risk-banner-medium",
                "emoji": "🟡",
                "color": "#F59E0B",
                "tag": "MODERATE RISK",
                "summary": "Elevated water volume approaching bankfull stage with localized drainage bottlenecks.",
                "action": "Place disaster management rapid-response units on standby. Clear culverts and monitor upstream rainfall gauges."
            },
            "Low": {
                "class_name": "risk-banner-low",
                "emoji": "🟢",
                "color": "#10B981",
                "tag": "NOMINAL RISK",
                "summary": "Hydrological parameters within normal seasonal containment capacities.",
                "action": "Maintain routine catchment monitoring. Carry out standard seasonal levee and canal maintenance."
            },
        }
        cfg = risk_configs.get(pred_class, risk_configs["Medium"])
        prob_dict = {cls: prob for cls, prob in zip(class_order, probabilities)}
        curr_confidence = prob_dict.get(pred_class, 0.0) * 100.0

        # Render Main Risk Banner (FR-22)
        st.markdown(
            f"""
            <div class="{cfg['class_name']}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="background: {cfg['color']}; color: #FFFFFF; font-weight: 800; font-size: 0.78rem; padding: 4px 12px; border-radius: 999px; letter-spacing: 0.06em;">
                        {cfg['tag']}
                    </span>
                    <span style="color: #E2E8F0; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;">
                        ⚡ Inference: {inference_latency_ms:.1f}ms
                    </span>
                </div>
                <div style="display: flex; align-items: baseline; gap: 12px;">
                    <h2 style="margin: 0; font-size: 2.3rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">
                        {cfg['emoji']} {pred_class.upper()} RISK
                    </h2>
                    <span style="font-size: 1.4rem; font-weight: 700; color: {cfg['color']}; font-family: 'JetBrains Mono', monospace;">
                        {curr_confidence:.1f}%
                    </span>
                </div>
                <p style="margin: 10px 0 0 0; color: #F1F5F9; font-size: 0.98rem; line-height: 1.45;">
                    {cfg['summary']}
                </p>
                <div style="margin-top: 14px; padding: 12px 16px; background: rgba(0, 0, 0, 0.25); border-radius: 10px; border-left: 3px solid {cfg['color']};">
                    <b style="color: #F8FAFC; font-size: 0.85rem; text-transform: uppercase;">Recommended Response Protocol:</b>
                    <div style="color: #CBD5E1; font-size: 0.9rem; margin-top: 4px;">{cfg['action']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Confidence Progress Bars per Class (FR-23)
        st.markdown("<h4 style='color:#E2E8F0; font-weight:700; margin-bottom:12px;'>🎯 Class Confidence Probabilities (FR-23)</h4>", unsafe_allow_html=True)

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

        # Composite Indicator Metrics Card
        st.markdown("<h4 style='color:#E2E8F0; font-weight:700; margin: 18px 0 10px 0;'>🧪 Engineered Composite Indices (FR-08)</h4>", unsafe_allow_html=True)
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center;">
                    <div class="metric-val" style="color: #38BDF8;">{enriched['Flood_Vulnerability_Score']:.2f}</div>
                    <div class="metric-lbl">Vulnerability Score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col2:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center;">
                    <div class="metric-val" style="color: #F59E0B;">{enriched['Infrastructure_Deficit_Score']:.2f}</div>
                    <div class="metric-lbl">Infrastructure Deficit</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col3:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center;">
                    <div class="metric-val" style="color: #A855F7;">{enriched['Aggregate_Hazard_Index']:.2f}</div>
                    <div class="metric-lbl">Hazard Index</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Bottom Section: Feature Importance (FR-24) & Multi-Model Benchmark Comparison
    st.markdown("<h3 style='color:#F1F5F9; font-weight:700; margin-bottom:16px;'>📈 Model Intelligence & Benchmarking</h3>", unsafe_allow_html=True)
    bench_col1, bench_col2 = st.columns([1.1, 1.0], gap="large")

    with bench_col1:
        st.markdown("<h4 style='color:#E2E8F0; font-weight:700;'>🏆 Top Predictive Feature Importances (FR-24)</h4>", unsafe_allow_html=True)
        st.caption("Gini relative importance for the top selected predictors in the best ensemble model.")

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_df = pd.DataFrame({
                "Feature": selected_features,
                "Importance": importances
            }).sort_values(by="Importance", ascending=True)

            st.bar_chart(feat_df.set_index("Feature"), color="#38BDF8", use_container_width=True)
        else:
            st.info("Feature importances not directly exposed for this model type.")

    with bench_col2:
        st.markdown("<h4 style='color:#E2E8F0; font-weight:700;'>📋 6-Model Benchmark Comparison (FR-20)</h4>", unsafe_allow_html=True)
        st.caption("Sorted strictly by F1-Weighted score on 3,000 held-out test samples.")

        if comparison_df is not None:
            # Format dataframe for display
            styled_df = comparison_df.copy()
            styled_df["Accuracy"] = (styled_df["Accuracy"] * 100).map("{:.2f}%".format)
            styled_df["F1_Weighted"] = styled_df["F1_Weighted"].map("{:.4f}".format)
            styled_df["ROC_AUC"] = styled_df["ROC_AUC"].map("{:.4f}".format)
            styled_df["CV_Mean"] = styled_df["CV_Mean"].map("{:.4f}".format)

            st.dataframe(
                styled_df[["Model", "F1_Weighted", "Accuracy", "ROC_AUC", "CV_Mean"]],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.warning("Model comparison data not found. Run evaluate.py to generate benchmarks.")


if __name__ == "__main__":
    main()
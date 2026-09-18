"""
Streamlit Page: What-If Counterfactual Scenario Planner
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_verified_pipeline, SecurityError, EXPECTED_RAW_FEATURES
from src.features import compute_domain_features_dict

# Page config
st.set_page_config(
    page_title="What-If Analysis | Flood Risk Intelligence",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .whatif-card {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    .whatif-header {
        font-size: 1.3rem;
        font-weight: 800;
        color: #F1F5F9;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .scenario-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 9999px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .delta-positive { color: #EF4444; font-weight: 700; }
    .delta-negative { color: #10B981; font-weight: 700; }
    .delta-neutral { color: #94A3B8; font-weight: 600; }
    .slider-label {
        font-size: 0.85rem;
        color: #E2E8F0;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline():
    try:
        pipeline = load_verified_pipeline()
    except SecurityError as e:
        st.error(f"Security Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        st.stop()
    return pipeline


# Factor descriptions and bounds
FACTOR_INFO = {
    "MonsoonIntensity": ("Monsoon Intensity", "Precipitation volume, duration, and peak storm surge intensity", 0.0, 15.0),
    "TopographyDrainage": ("Topography & Drainage", "Slope gradient, natural runoff channels, and elevation variance", 0.0, 15.0),
    "RiverManagement": ("River Management", "Condition of river embankments, levee maintenance, and canal dredging", 0.0, 15.0),
    "Deforestation": ("Deforestation Rate", "Loss of vegetative soil cover, root binding, and canopy interception", 0.0, 15.0),
    "Urbanization": ("Urbanization & Impervious Area", "Paved surface density reducing natural infiltration and groundwater percolation", 0.0, 15.0),
    "ClimateChange": ("Climate Change Anomaly", "Deviation in historical precipitation regimes and extreme weather events", 0.0, 15.0),
    "DamsQuality": ("Dam & Reservoir Integrity", "Structural condition, spillway capacity, and silt accumulation in reservoirs", 0.0, 15.0),
    "Siltation": ("Riverbed Siltation", "Sediment buildup reducing hydrological conveyance capacity", 0.0, 15.0),
    "AgriculturalPractices": ("Agricultural Land Use", "Terracing, soil tilling intensity, and floodplain agriculture", 0.0, 15.0),
    "Encroachments": ("Floodplain Encroachment", "Unregulated settlements and industrial structures within floodways", 0.0, 15.0),
    "IneffectiveDisasterPreparedness": ("Preparedness Deficit", "Gaps in early warning dissemination, evacuation routes, and disaster response", 0.0, 15.0),
    "DrainageSystems": ("Urban Drainage Capacity", "Culvert dimensions, storm sewer networks, and pumping station readiness", 0.0, 15.0),
    "CoastalVulnerability": ("Coastal Vulnerability", "Susceptibility to tidal surges, storm tides, and sea-level rise", 0.0, 15.0),
    "Landslides": ("Landslide Hazard", "Unstable slopes contributing debris dams and sudden flash flood surges", 0.0, 15.0),
    "Watersheds": ("Watershed Degradation", "Degraded catchment areas unable to retain seasonal surface runoff", 0.0, 15.0),
    "DeterioratingInfrastructure": ("Aging Infrastructure", "Lifespan fatigue on bridges, retaining walls, and drainage barriers", 0.0, 15.0),
    "PopulationScore": ("Population Density in Flood Zone", "Concentration of communities in low-lying hazard zones", 0.0, 15.0),
    "WetlandLoss": ("Wetland & Floodplain Loss", "Destruction of natural retention basins and marsh buffer areas", 0.0, 15.0),
    "InadequatePlanning": ("Inadequate Urban Planning", "Zoning failures and lack of sustainable storm drainage Master Plans", 0.0, 15.0),
    "PoliticalFactors": ("Policy & Governance Gaps", "Delayed maintenance funding and fragmented disaster management coordination", 0.0, 15.0),
}

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
    },
}

INTERVENTIONS = {
    "Reforestation": {"Deforestation": -3.0, "WetlandLoss": -2.0, "Eco_Damage": -0.5},
    "River Restoration": {"RiverManagement": +3.0, "Siltation": -2.0, "Siltation_Pressure": -0.5},
    "Dam Upgrade": {"DamsQuality": +4.0, "Infra_Gap": -0.3},
    "Urban Drainage Expansion": {"DrainageSystems": +4.0, "Preparedness_Deficit": -0.5, "Water_Stress": -0.3},
    "Climate Adaptation": {"ClimateChange": -2.0, "MonsoonIntensity": -1.0, "Eco_Damage": -0.3},
    "Early Warning System": {"IneffectiveDisasterPreparedness": -4.0, "Preparedness_Deficit": -0.5},
    "Wetland Restoration": {"WetlandLoss": -3.0, "Water_Stress": -0.3, "Eco_Damage": -0.3},
    "Zoning Reform": {"Urbanization": -2.0, "Encroachments": -2.0, "InadequatePlanning": -2.0},
}


def render_slider(key, label, min_val, max_val, default, help_text):
    st.markdown(f'<div class="slider-label">{label}</div>', unsafe_allow_html=True)
    return st.slider(
        label="",
        min_value=min_val,
        max_value=max_val,
        value=default,
        step=0.5,
        help=help_text,
        key=f"slider_{key}",
        label_visibility="collapsed",
    )


def format_delta(delta):
    if delta > 0:
        return f'<span class="delta-positive">+{delta:.2f} pp</span>'
    elif delta < 0:
        return f'<span class="delta-negative">{delta:.2f} pp</span>'
    return f'<span class="delta-neutral">{delta:.2f} pp</span>'


def main():
    # Hero
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(16, 24, 40, 0.95) 0%, rgba(15, 23, 42, 0.90) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 32px; margin-bottom: 24px;">
            <div class="scenario-badge">🔄 Counterfactual Policy Intervention Planner</div>
            <h1 style="margin: 0 0 8px 0; font-size: 2.5rem; font-weight: 800; color: #FFFFFF; line-height: 1.2;">
                What If We Changed the Conditions?
            </h1>
            <p style="margin: 0; color: #94A3B8; font-size: 1.05rem; max-width: 900px; line-height: 1.6;">
                Simulate infrastructure investments, nature-based solutions, and policy reforms to quantify 
                their impact on predicted flood risk probabilities.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Load pipeline
    with st.spinner("Loading calibrated pipeline..."):
        pipeline = load_pipeline()
    
    st.success("✅ Pipeline loaded — ready for counterfactual simulation")
    
    # Initialize session state
    if "baseline_inputs" not in st.session_state:
        st.session_state.baseline_inputs = {k: v[4] if len(v) > 4 else 5.0 for k, v in FACTOR_INFO.items()}
    if "scenario_inputs" not in st.session_state:
        st.session_state.scenario_inputs = st.session_state.baseline_inputs.copy()
    
    # --- Layout: Baseline (Left) vs Scenario (Right) ---
    col_baseline, col_scenario = st.columns([1, 1], gap="large")
    
    with col_baseline:
        st.markdown(
            """
            <div class="whatif-card">
                <div class="whatif-header">📍 Baseline Regional Profile</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Preset selector
        preset = st.selectbox(
            "Load Baseline Preset",
            list(PRESETS.keys()),
            index=1,
            key="baseline_preset"
        )
        
        if st.button("Apply Baseline Preset", use_container_width=True):
            for k, v in PRESETS[preset].items():
                st.session_state.baseline_inputs[k] = v
            st.rerun()
        
        # Baseline sliders
        baseline_tabs = st.tabs(["🌧️ Meteorology", "🏗️ Infrastructure", "🌱 Land Use"])
        tab_groups = {
            "🌧️ Meteorology": ["MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Watersheds", "CoastalVulnerability", "Landslides"],
            "🏗️ Infrastructure": ["DamsQuality", "DrainageSystems", "DeterioratingInfrastructure", "IneffectiveDisasterPreparedness", "PoliticalFactors"],
            "🌱 Land Use": ["Urbanization", "Deforestation", "ClimateChange", "Siltation", "AgriculturalPractices", "Encroachments", "PopulationScore", "WetlandLoss", "InadequatePlanning"],
        }
        
        for tab, factors in tab_groups.items():
            with baseline_tabs[list(tab_groups.keys()).index(tab)]:
                for key in factors:
                    label, desc, min_val, max_val = FACTOR_INFO[key]
                    st.session_state.baseline_inputs[key] = render_slider(
                        key + "_base", label, min_val, max_val,
                        st.session_state.baseline_inputs.get(key, 5.0), desc
                    )
    
    with col_scenario:
        st.markdown(
            """
            <div class="whatif-card">
                <div class="whatif-header">🎯 Intervention Scenario</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Start with baseline
        if st.button("🔄 Reset Scenario to Baseline", use_container_width=True):
            st.session_state.scenario_inputs = st.session_state.baseline_inputs.copy()
            st.rerun()
        
        # Intervention presets
        st.markdown("**Quick Interventions**")
        intervention_cols = st.columns(4)
        for i, (intv_name, changes) in enumerate(INTERVENTIONS.items()):
            with intervention_cols[i % 4]:
                if st.button(intv_name, use_container_width=True):
                    for feat, delta in changes.items():
                        if feat in st.session_state.scenario_inputs:
                            new_val = st.session_state.scenario_inputs[feat] + delta
                            min_val, max_val = 0.0, 15.0
                            if feat in FACTOR_INFO:
                                min_val, max_val = FACTOR_INFO[feat][2], FACTOR_INFO[feat][3]
                            st.session_state.scenario_inputs[feat] = np.clip(new_val, min_val, max_val)
                    st.rerun()
        
        # Scenario sliders
        st.markdown("**Fine-Tune Scenario**")
        scenario_tabs = st.tabs(["🌧️ Meteorology", "🏗️ Infrastructure", "🌱 Land Use"])
        for tab, factors in tab_groups.items():
            with scenario_tabs[list(tab_groups.keys()).index(tab)]:
                for key in factors:
                    label, desc, min_val, max_val = FACTOR_INFO[key]
                    st.session_state.scenario_inputs[key] = render_slider(
                        key + "_scen", label, min_val, max_val,
                        st.session_state.scenario_inputs.get(key, 5.0), desc
                    )
    
    # --- Run Predictions ---
    st.markdown("---")
    st.markdown(
        """
        <div class="whatif-header">📊 Comparative Risk Assessment</div>
        """,
        unsafe_allow_html=True,
    )
    
    # Predict baseline
    baseline_df = pd.DataFrame([st.session_state.baseline_inputs])
    baseline_pred = pipeline.predict(baseline_df)[0]
    baseline_prob = pipeline.predict_proba(baseline_df)[0]
    int_to_class = {0: "Low", 1: "Medium", 2: "High"}
    if isinstance(baseline_pred, (int, np.integer)):
        baseline_class = int_to_class.get(int(baseline_pred), str(baseline_pred))
    else:
        baseline_class = str(baseline_pred)
    baseline_high_prob = baseline_prob[2]  # High risk index
    
    # Predict scenario
    scenario_df = pd.DataFrame([st.session_state.scenario_inputs])
    scenario_pred = pipeline.predict(scenario_df)[0]
    scenario_prob = pipeline.predict_proba(scenario_df)[0]
    if isinstance(scenario_pred, (int, np.integer)):
        scenario_class = int_to_class.get(int(scenario_pred), str(scenario_pred))
    else:
        scenario_class = str(scenario_pred)
    scenario_high_prob = scenario_prob[2]
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Baseline
        risk_colors = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}
        base_color = risk_colors.get(baseline_class, "#94A3B8")
        st.markdown(
            f"""
            <div class="whatif-card" style="border-left: 4px solid {base_color};">
                <h4 style="color: {base_color}; margin-bottom: 8px;">📍 Baseline</h4>
                <h2 style="margin: 0; color: #FFFFFF;">{baseline_class.upper()}</h2>
                <p style="margin: 8px 0 0 0; color: #94A3B8;">P(High Risk) = {baseline_high_prob:.1%}</p>
                <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.9rem;">
                    Low: {baseline_prob[0]:.1%} | Med: {baseline_prob[1]:.1%} | High: {baseline_prob[2]:.1%}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        # Scenario
        scen_color = risk_colors.get(scenario_class, "#94A3B8")
        st.markdown(
            f"""
            <div class="whatif-card" style="border-left: 4px solid {scen_color};">
                <h4 style="color: {scen_color}; margin-bottom: 8px;">🎯 Scenario</h4>
                <h2 style="margin: 0; color: #FFFFFF;">{scenario_class.upper()}</h2>
                <p style="margin: 8px 0 0 0; color: #94A3B8;">P(High Risk) = {scenario_high_prob:.1%}</p>
                <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.9rem;">
                    Low: {scenario_prob[0]:.1%} | Med: {scenario_prob[1]:.1%} | High: {scenario_prob[2]:.1%}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col3:
        # Delta
        delta_high = (scenario_high_prob - baseline_high_prob) * 100
        delta_class = "delta-positive" if delta_high > 0 else "delta-negative" if delta_high < 0 else "delta-neutral"
        delta_label = "INCREASE" if delta_high > 0 else "REDUCTION" if delta_high < 0 else "NO CHANGE"
        
        st.markdown(
            f"""
            <div class="whatif-card" style="border-left: 4px solid #38BDF8;">
                <h4 style="color: #38BDF8; margin-bottom: 8px;">📈 Impact (Δ)</h4>
                <h2 style="margin: 0; color: #FFFFFF;">{delta_high:+.2f} pp</h2>
                <p style="margin: 8px 0 0 0; color: #94A3B8;">P(High Risk) {delta_label}</p>
                <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.9rem;">
                    Risk Class: {baseline_class} → {scenario_class}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # --- Detailed Delta Breakdown ---
    st.markdown(
        """
        <div class="whatif-header">🔬 Feature-Level Change Analysis</div>
        """,
        unsafe_allow_html=True,
    )
    
    # Show top feature changes
    changes = []
    for key in EXPECTED_RAW_FEATURES:
        base_val = st.session_state.baseline_inputs.get(key, 0)
        scen_val = st.session_state.scenario_inputs.get(key, 0)
        diff = scen_val - base_val
        if abs(diff) > 0.01:
            changes.append({
                "Feature": key,
                "Baseline": f"{base_val:.1f}",
                "Scenario": f"{scen_val:.1f}",
                "Δ": f"{diff:+.1f}",
            })
    
    if changes:
        changes_df = pd.DataFrame(changes).sort_values("Δ", key=abs, ascending=False).head(10)
        st.dataframe(changes_df, hide_index=True, use_container_width=True)
    
    # --- Domain Features Comparison ---
    st.markdown(
        """
        <div class="whatif-header">🧪 Domain Feature Impact</div>
        """,
        unsafe_allow_html=True,
    )
    
    base_domain = compute_domain_features_dict(st.session_state.baseline_inputs)
    scen_domain = compute_domain_features_dict(st.session_state.scenario_inputs)
    
    domain_keys = ["Environmental_Risk", "Infrastructure_Vulnerability", "Anthropogenic_Pressure", "Hydrometeorological_Risk",
                   "Water_Stress", "Infra_Gap", "Eco_Damage", "Siltation_Pressure", "Preparedness_Deficit"]
    
    domain_data = []
    for key in domain_keys:
        if key in base_domain and key in scen_domain:
            base_val = base_domain[key]
            scen_val = scen_domain[key]
            diff = scen_val - base_val
            domain_data.append({
                "Domain Feature": key,
                "Baseline": f"{base_val:.2f}",
                "Scenario": f"{scen_val:.2f}",
                "Δ": f"{diff:+.2f}",
            })
    
    if domain_data:
        domain_df = pd.DataFrame(domain_data).sort_values("Δ", key=abs, ascending=False)
        st.dataframe(domain_df, hide_index=True, use_container_width=True)
    
    # --- Intervention Summary ---
    st.markdown("---")
    st.markdown(
        """
        <div class="whatif-header">📋 Intervention Summary</div>
        """,
        unsafe_allow_html=True,
    )
    
    # Identify which interventions were applied
    applied = []
    for intv_name, intv_changes in INTERVENTIONS.items():
        match = True
        for feat, delta in intv_changes.items():
            base_val = st.session_state.baseline_inputs.get(feat, 0)
            scen_val = st.session_state.scenario_inputs.get(feat, 0)
            if abs((scen_val - base_val) - delta) > 0.5:
                match = False
                break
        if match:
            applied.append(intv_name)
    
    if applied:
        st.markdown(f"**Detected Interventions:** {', '.join(applied)}")
    else:
        st.markdown("*Custom scenario — no standard intervention template matched*")
    
    # Export
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        scenario_json = pd.DataFrame([st.session_state.scenario_inputs]).to_json(orient="records", indent=2)
        st.download_button(
            "📥 Download Scenario (JSON)",
            scenario_json,
            "whatif_scenario.json",
            "application/json"
        )
    with col2:
        # Create comparison report
        report = f"""# What-If Analysis Report

## Baseline Profile
- Predicted Class: {baseline_class}
- P(High Risk): {baseline_high_prob:.2%}

## Intervention Scenario
- Predicted Class: {scenario_class}
- P(High Risk): {scenario_high_prob:.2%}

## Impact
- Δ P(High Risk): {delta_high:+.2f} pp
- Risk Class Change: {baseline_class} → {scenario_class}

## Top Feature Changes
{changes_df.to_string(index=False) if changes else 'No changes'}

## Domain Feature Impact
{domain_df.to_string(index=False) if domain_data else 'N/A'}
"""
        st.download_button(
            "📄 Download Report (Markdown)",
            report,
            "whatif_report.md",
            "text/markdown"
        )


if __name__ == "__main__":
    main()
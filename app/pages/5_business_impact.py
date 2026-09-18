"""
Streamlit Page: Business Impact & ROI Calculator
"""

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
    page_title="Business Impact | Flood Risk Intelligence",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .roi-card {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    .roi-header {
        font-size: 1.3rem;
        font-weight: 800;
        color: #F1F5F9;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .roi-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 9999px;
        padding: 4px 14px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .money-positive { color: #10B981; font-weight: 700; }
    .money-negative { color: #EF4444; font-weight: 700; }
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        font-weight: 600;
        margin-top: 4px;
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


def compute_expected_annual_loss(
    population: int,
    avg_property_value: float,
    historical_flood_freq: float,  # per year
    high_risk_prob: float,
    mitigation_effectiveness: float = 0.40,
    vulnerability_factor: float = 0.15  # fraction of property value lost per flood
) -> dict:
    """
    Computes expected annual loss with and without early warning.
    
    Assumptions:
    - Each flood event damages a fraction of total property value
    - Early warning reduces losses by mitigation_effectiveness (default 40%)
    - High-risk probability scales the baseline flood frequency
    """
    total_asset_value = population * avg_property_value
    
    # Baseline annual flood probability (calibrated to historical frequency)
    # If region is high risk, scale up the frequency
    baseline_annual_flood_prob = historical_flood_freq
    adjusted_flood_prob = baseline_annual_flood_prob * (1 + high_risk_prob)  # Simple scaling
    
    # Expected loss per flood event
    loss_per_event = total_asset_value * vulnerability_factor
    
    # Annual expected loss without early warning
    annual_loss_no_warning = adjusted_flood_prob * loss_per_event
    
    # Annual expected loss WITH early warning (40% mitigation)
    annual_loss_with_warning = annual_loss_no_warning * (1 - mitigation_effectiveness)
    
    # Annual savings
    annual_savings = annual_loss_no_warning - annual_loss_with_warning
    
    # NPV over 10 years at 5% discount rate
    discount_rate = 0.05
    years = 10
    npv_savings = sum(annual_savings / (1 + discount_rate)**t for t in range(1, years + 1))
    
    return {
        "total_asset_value": total_asset_value,
        "adjusted_annual_flood_prob": adjusted_flood_prob,
        "loss_per_event": loss_per_event,
        "annual_loss_no_warning": annual_loss_no_warning,
        "annual_loss_with_warning": annual_loss_with_warning,
        "annual_savings": annual_savings,
        "npv_10yr_savings": npv_savings,
        "mitigation_effectiveness": mitigation_effectiveness,
    }


def main():
    # Hero
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(16, 24, 40, 0.95) 0%, rgba(15, 23, 42, 0.90) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 32px; margin-bottom: 24px;">
            <div class="roi-badge">💰 Business Impact & ROI Calculator</div>
            <h1 style="margin: 0 0 8px 0; font-size: 2.5rem; font-weight: 800; color: #FFFFFF; line-height: 1.2;">
                Financial Case for Early Warning Investment
            </h1>
            <p style="margin: 0; color: #94A3B8; font-size: 1.05rem; max-width: 900px; line-height: 1.6;">
                Quantify the return on investment for flood early warning systems using regional risk profiles 
                and calibrated probability outputs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Load pipeline
    with st.spinner("Loading calibrated pipeline..."):
        pipeline = load_pipeline()
    
    st.success("✅ Pipeline loaded")
    
    # --- Input Section ---
    st.markdown(
        """
        <div class="roi-card">
            <div class="roi-header">📍 Regional Profile & Economic Inputs</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics**")
        population = st.number_input(
            "Population in Flood Zone",
            min_value=1000,
            max_value=10_000_000,
            value=50_000,
            step=1000,
            help="Number of people residing in the flood-prone area"
        )
        
        avg_property_value = st.number_input(
            "Average Property Value per Capita ($)",
            min_value=10_000,
            max_value=1_000_000,
            value=75_000,
            step=5000,
            help="Average value of residential + commercial assets per person"
        )
    
    with col2:
        st.markdown("**Hydrological Risk**")
        historical_flood_freq = st.number_input(
            "Historical Flood Frequency (events/year)",
            min_value=0.01,
            max_value=5.0,
            value=0.3,
            step=0.05,
            format="%.2f",
            help="Average number of flood events per year historically"
        )
        
        # Get current model prediction for high-risk probability
        # Use a sample high-risk scenario
        sample_high = {
            "MonsoonIntensity": 10.0, "TopographyDrainage": 8.0, "RiverManagement": 7.0,
            "Deforestation": 8.0, "Urbanization": 9.0, "ClimateChange": 9.0,
            "DamsQuality": 4.0, "Siltation": 8.0, "AgriculturalPractices": 6.0,
            "Encroachments": 8.0, "IneffectiveDisasterPreparedness": 9.0,
            "DrainageSystems": 5.0, "CoastalVulnerability": 7.0, "Landslides": 6.0,
            "Watersheds": 7.0, "DeterioratingInfrastructure": 8.0, "PopulationScore": 8.0,
            "WetlandLoss": 7.0, "InadequatePlanning": 8.0, "PoliticalFactors": 6.0,
        }
        prob = pipeline.predict_proba(pd.DataFrame([sample_high]))[0]
        model_high_prob = prob[2]
        
        st.metric("Model P(High Risk) - Sample Scenario", f"{model_high_prob:.1%}")
        
        # Allow override
        use_model_prob = st.checkbox("Use model probability for risk scaling", value=True)
        if not use_model_prob:
            manual_high_prob = st.slider(
                "Manual High-Risk Probability",
                0.0, 1.0, 0.5, 0.05,
                help="Override model probability for scenario analysis"
            )
        else:
            manual_high_prob = model_high_prob
    
    with col3:
        st.markdown("**Economic Parameters**")
        mitigation_effectiveness = st.slider(
            "Early Warning Mitigation Effectiveness",
            0.10, 0.80, 0.40, 0.05,
            help="Fraction of losses prevented by early warning (literature: 30-50%)"
        )
        
        vulnerability_factor = st.slider(
            "Asset Vulnerability per Flood Event",
            0.05, 0.50, 0.15, 0.01,
            help="Fraction of total asset value damaged per flood event"
        )
        
        discount_rate = st.slider(
            "Discount Rate for NPV (%)",
            1.0, 10.0, 5.0, 0.5,
            help="Annual discount rate for NPV calculation"
        ) / 100.0
        
        analysis_years = st.slider(
            "Analysis Horizon (years)",
            5, 30, 10, 1,
            help="Investment evaluation period"
        )
    
    # --- Compute ROI ---
    results = compute_expected_annual_loss(
        population=population,
        avg_property_value=avg_property_value,
        historical_flood_freq=historical_flood_freq,
        high_risk_prob=manual_high_prob if use_model_prob else manual_high_prob,
        mitigation_effectiveness=mitigation_effectiveness,
        vulnerability_factor=vulnerability_factor,
    )
    
    # Adjust for custom discount rate and years
    annual_savings = results["annual_savings"]
    npv_savings = sum(annual_savings / (1 + discount_rate)**t for t in range(1, analysis_years + 1))
    
    # --- KPI Dashboard ---
    st.markdown("---")
    st.markdown(
        """
        <div class="roi-header">📊 Key Financial Metrics</div>
        """,
        unsafe_allow_html=True,
    )
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 16px; background: rgba(16, 185, 129, 0.1); 
                        border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px;">
                <div class="kpi-value money-positive">${results['annual_savings']:,.0f}</div>
                <div class="kpi-label">Annual Savings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with kpi_col2:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 16px; background: rgba(16, 185, 129, 0.1); 
                        border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px;">
                <div class="kpi-value money-positive">${npv_savings:,.0f}</div>
                <div class="kpi-label">{analysis_years}-Year NPV Savings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with kpi_col3:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 16px; background: rgba(56, 189, 248, 0.1); 
                        border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px;">
                <div class="kpi-value">${results['annual_loss_no_warning']:,.0f}</div>
                <div class="kpi-label">Annual Loss (No Warning)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with kpi_col4:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 16px; background: rgba(16, 185, 129, 0.1); 
                        border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px;">
                <div class="kpi-value">${results['annual_loss_with_warning']:,.0f}</div>
                <div class="kpi-label">Annual Loss (With Warning)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # --- Detailed Breakdown ---
    st.markdown("---")
    st.markdown('<div class="roi-header">💸 Loss Breakdown</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            <div class="roi-card">
                <div class="roi-header">Without Early Warning System</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.metric("Total Regional Asset Value", f"${results['total_asset_value']:,.0f}")
        st.metric("Adjusted Annual Flood Probability", f"{results['adjusted_annual_flood_prob']:.2%}")
        st.metric("Expected Loss per Flood Event", f"${results['loss_per_event']:,.0f}")
        st.metric("Annual Expected Loss", f"${results['annual_loss_no_warning']:,.0f}")
        
        # Bar chart
        loss_data = pd.DataFrame({
            "Scenario": ["No Warning", "With Warning"],
            "Annual Loss": [results['annual_loss_no_warning'], results['annual_loss_with_warning']]
        })
        st.bar_chart(loss_data.set_index("Scenario"))
    
    with col2:
        st.markdown(
            """
            <div class="roi-card">
                <div class="roi-header">With Early Warning System</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.metric("Mitigation Effectiveness", f"{mitigation_effectiveness:.0%}")
        st.metric("Annual Expected Loss", f"${results['annual_loss_with_warning']:,.0f}")
        st.metric("Annual Savings", f"${results['annual_savings']:,.0f}")
        st.metric(f"{analysis_years}-Year NPV (at {discount_rate:.1%})", f"${npv_savings:,.0f}")
        
        # Payback analysis
        system_cost = st.number_input(
            "Early Warning System Deployment Cost ($)",
            min_value=10_000,
            max_value=100_000_000,
            value=500_000,
            step=50_000,
            help="Estimated cost to deploy and operate the early warning system"
        )
        
        if results['annual_savings'] > 0:
            payback_years = system_cost / results['annual_savings']
            roi_pct = (results['annual_savings'] * analysis_years - system_cost) / system_cost * 100
            
            st.metric("System Deployment Cost", f"${system_cost:,.0f}")
            st.metric("Payback Period", f"{payback_years:.1f} years")
            st.metric(f"{analysis_years}-Year ROI", f"{roi_pct:.0f}%")
            
            if payback_years <= analysis_years:
                st.success(f"✅ Investment pays back within {analysis_years} years")
            else:
                st.warning(f"⚠️ Payback exceeds {analysis_years}-year horizon")
    
    # --- Scenario Comparison Table ---
    st.markdown("---")
    st.markdown('<div class="roi-header">📈 Scenario Sensitivity Analysis</div>', unsafe_allow_html=True)
    
    # Vary key parameters
    scenarios = []
    for freq in [0.1, 0.3, 0.5, 1.0]:
        for vuln in [0.10, 0.15, 0.25]:
            for mitigate in [0.30, 0.40, 0.50]:
                r = compute_expected_annual_loss(
                    population=population,
                    avg_property_value=avg_property_value,
                    historical_flood_freq=freq,
                    high_risk_prob=manual_high_prob if use_model_prob else manual_high_prob,
                    mitigation_effectiveness=mitigate,
                    vulnerability_factor=vuln,
                )
                npv = sum(r['annual_savings'] / (1 + discount_rate)**t for t in range(1, analysis_years + 1))
                scenarios.append({
                    "Flood Freq/yr": freq,
                    "Vulnerability": f"{vuln:.0%}",
                    "Mitigation": f"{mitigate:.0%}",
                    "Annual Savings": r['annual_savings'],
                    f"{analysis_years}yr NPV": npv,
                })
    
    scenario_df = pd.DataFrame(scenarios)
    scenario_df = scenario_df.sort_values(f"{analysis_years}yr NPV", ascending=False).head(20)
    
    # Format for display
    display_df = scenario_df.copy()
    display_df["Annual Savings"] = display_df["Annual Savings"].apply(lambda x: f"${x:,.0f}")
    display_df[f"{analysis_years}yr NPV"] = display_df[f"{analysis_years}yr NPV"].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    # --- Investment Decision Framework ---
    st.markdown("---")
    st.markdown('<div class="roi-header">🎯 Investment Decision Framework</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="roi-card">
            <h4 style="color: #F1F5F9; margin-bottom: 12px;">Decision Rule</h4>
            <p style="color: #E2E8F0; line-height: 1.7;">
                Deploy early warning system if:<br>
                <code style="color: #38BDF8;">NPV_Savings > System_Cost</code> AND <code style="color: #38BDF8;">Payback ≤ Analysis_Horizon</code>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if npv_savings > system_cost and payback_years <= analysis_years:
        st.success(
            f"""
            ✅ **RECOMMEND DEPLOYMENT**
            
            - NPV Savings (${npv_savings:,.0f}) > System Cost (${system_cost:,.0f})
            - Payback Period ({payback_years:.1f} years) ≤ Analysis Horizon ({analysis_years} years)
            - Expected {analysis_years}-Year ROI: {roi_pct:.0f}%
            """
        )
    else:
        reasons = []
        if npv_savings <= system_cost:
            reasons.append(f"NPV Savings (${npv_savings:,.0f}) ≤ System Cost (${system_cost:,.0f})")
        if payback_years > analysis_years:
            reasons.append(f"Payback ({payback_years:.1f} years) > Horizon ({analysis_years} years)")
        
        st.error(
            f"""
            ❌ **DO NOT DEPLOY** (under current assumptions)
            
            {'; '.join(reasons)}
            
            Consider: higher mitigation effectiveness, longer horizon, or lower deployment cost.
            """
        )
    
    # --- Assumptions & Methodology ---
    st.markdown("---")
    with st.expander("📋 Methodology & Assumptions"):
        st.markdown(
            f"""
            ### Loss Model
            - **Total Asset Value** = Population × Avg Property Value per Capita
            - **Adjusted Flood Probability** = Historical Frequency × (1 + Model P(High Risk))
            - **Loss per Event** = Total Asset Value × Vulnerability Factor
            - **Annual Expected Loss** = Adjusted Flood Probability × Loss per Event
            
            ### Mitigation
            - Early warning reduces losses by **{mitigation_effectiveness:.0%}** (configurable)
            - Based on literature: 30-50% loss reduction from early warning systems
            
            ### Financial
            - Discount Rate: **{discount_rate:.1%}**
            - Analysis Horizon: **{analysis_years} years**
            - NPV = Σ Annual_Savings / (1 + r)^t
            
            ### Limitations
            - Simplified linear loss model (real losses are non-linear)
            - Does not account for indirect losses (business interruption, health, ecology)
            - Assumes stationary flood frequency (climate change may increase)
            - Model probability used as risk multiplier (calibration dependent)
            - Single-region analysis (no spatial portfolio effects)
            """
        )
    
    # Export
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        report = f"""# Flood Early Warning ROI Analysis Report

## Regional Profile
- Population at Risk: {population:,}
- Avg Property Value: ${avg_property_value:,.0f}
- Historical Flood Freq: {historical_flood_freq:.2f}/year

## Model Risk Assessment
- P(High Risk): {manual_high_prob:.2%}
- Source: {'Calibrated Model' if use_model_prob else 'Manual Override'}

## Economic Parameters
- Vulnerability Factor: {vulnerability_factor:.1%}
- Mitigation Effectiveness: {mitigation_effectiveness:.0%}
- Discount Rate: {discount_rate:.1%}
- Horizon: {analysis_years} years
- System Cost: ${system_cost:,.0f}

## Results
- Total Asset Value: ${results['total_asset_value']:,.0f}
- Annual Loss (No Warning): ${results['annual_loss_no_warning']:,.0f}
- Annual Loss (With Warning): ${results['annual_loss_with_warning']:,.0f}
- Annual Savings: ${results['annual_savings']:,.0f}
- {analysis_years}-Year NPV: ${npv_savings:,.0f}
- Payback Period: {payback_years:.1f} years
- {analysis_years}-Year ROI: {roi_pct:.0f}%

## Decision
{'✅ DEPLOY RECOMMENDED' if npv_savings > system_cost and payback_years <= analysis_years else '❌ DO NOT DEPLOY'}

---
*Analysis based on calibrated ML probabilities from Kaggle S4E5 synthetic benchmark.
Not for operational use. For educational purposes only.*
"""
        st.download_button(
            "📄 Download Full Report (Markdown)",
            report,
            "roi_analysis_report.md",
            "text/markdown"
        )
    
    with col2:
        scenario_json = pd.DataFrame([{
            "population": population,
            "avg_property_value": avg_property_value,
            "historical_flood_freq": historical_flood_freq,
            "high_risk_prob": manual_high_prob,
            "mitigation_effectiveness": mitigation_effectiveness,
            "vulnerability_factor": vulnerability_factor,
            "discount_rate": discount_rate,
            "analysis_years": analysis_years,
            "system_cost": system_cost,
        }]).to_json(orient="records", indent=2)
        st.download_button(
            "📥 Download Inputs (JSON)",
            scenario_json,
            "roi_inputs.json",
            "application/json"
        )


if __name__ == "__main__":
    main()
"""
Shared Utilities for Flood Risk Intelligence Dashboard Pages
"""

import streamlit as st
from pathlib import Path


def inject_global_css():
    """Inject global CSS styles for all pages."""
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
            border-radius: 9999px;
            padding: 4px 14px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        
        .badge-simulation { background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); }
        .badge-statistics { background: rgba(139, 92, 246, 0.15); color: #A78BFA; border: 1px solid rgba(139, 92, 246, 0.3); }
        .badge-features { background: rgba(139, 92, 246, 0.15); color: #A78BFA; border: 1px solid rgba(139, 92, 246, 0.3); }
        .badge-whatif { background: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.3); }
        .badge-roi { background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); }
        .badge-story { background: rgba(139, 92, 246, 0.15); color: #A78BFA; border: 1px solid rgba(139, 92, 246, 0.3); }
        
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
        
        .section-header {
            font-size: 1.2rem;
            font-weight: 800;
            color: #F1F5F9;
            margin: 20px 0 12px 0;
            text-transform: uppercase;
            letter-spacing: 0.04em;
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
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: rgba(17, 24, 39, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 8px 16px;
            color: #94A3B8;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background: rgba(56, 189, 248, 0.2);
            border-color: rgba(56, 189, 248, 0.4);
            color: #38BDF8;
        }
        
        .download-btn {
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 8px;
            padding: 8px 16px;
            color: #38BDF8;
            font-weight: 600;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-right: 8px;
            margin-bottom: 8px;
        }
        
        .download-btn:hover {
            background: rgba(56, 189, 248, 0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, badge_text: str, badge_class: str = "badge-simulation"):
    """Render the standard hero banner for all pages."""
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="badge-pill {badge_class}">{badge_text}</div>
            <h1 style="margin: 0 0 6px 0; font-size: 2.2rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">
                {title}
            </h1>
            <p style="margin: 0; color: #94A3B8; font-size: 0.95rem; max-width: 900px; line-height: 1.5;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer():
    """Render the mandatory simulation disclaimer."""
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


def render_sidebar_navigation():
    """Render sidebar with page navigation and system info."""
    with st.sidebar:
        st.markdown("<h3 style='color:#F1F5F9; font-weight:700; margin-bottom:8px;'>🧭 Navigation</h3>", unsafe_allow_html=True)
        
        pages = [
            ("🏠", "Real-Time Inference", "Home"),
            ("📖", "Data Story & Domain Context", "1_data_story"),
            ("📊", "Statistical EDA & Hypothesis Tests", "2_statistical_eda"),
            ("🔍", "SHAP Feature Importance", "3_feature_importance"),
            ("🔄", "What-If Counterfactual Planner", "4_what_if_analysis"),
            ("💰", "ROI & Business Impact", "5_business_impact"),
        ]
        
        for icon, label, _ in pages:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 10px; padding: 10px 12px; 
                            background: rgba(17, 24, 39, 0.6); border-radius: 8px; margin: 4px 0;
                            color: #E2E8F0; font-weight: 500;">
                    <span style="font-size: 1.2rem;">{icon}</span>
                    <span>{label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.markdown("---")
        st.markdown("<h4 style='color:#E2E8F0; font-weight:600;'>System & Pipeline Info</h4>", unsafe_allow_html=True)
        st.markdown(
            """
            - **Pipeline**: `CalibratedClassifierCV` (LogisticRegression)
            - **Target**: Discretized Tertiles (`Low`, `Medium`, `High`)
            - **Integrity**: SHA-256 Verified (`checksums.json`)
            - **Preprocessing**: Fold-Safe In-Pipeline Scaling
            - **Calibration**: Native (LogReg) / Sigmoid-Isotonic
            - **Decision Threshold**: 65% (UNCERTAIN if max_prob < 0.65)
            """
        )
        
        # Model registry info
        st.markdown("---")
        st.markdown("<h4 style='color:#E2E8F0; font-weight:600;'>📋 Model Registry</h4>", unsafe_allow_html=True)
        try:
            import json
            registry_path = Path(__file__).resolve().parents[1] / "models" / "registry.json"
            if registry_path.exists():
                with open(registry_path, "r") as f:
                    registry = json.load(f)
                st.markdown(f"- **Version**: {registry.get('model_version', 'N/A')}")
                st.markdown(f"- **Model**: {registry.get('model_name', 'N/A')}")
                st.markdown(f"- **CV F1**: {registry.get('cv_f1_weighted', 0):.4f}")
                st.markdown(f"- **Test F1**: {registry.get('test_f1_weighted', 0):.4f}")
                st.markdown(f"- **Train Samples**: {registry.get('training_samples', 0):,}")
        except Exception:
            st.markdown("- Registry not available")


def get_page_config():
    """Standard page config for all pages."""
    return {
        "page_title": "Flood Risk Intelligence System",
        "page_icon": "🌊",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
    }


# Page metadata for navigation
PAGES = [
    ("🏠", "Real-Time Inference", "Home", "app/dashboard.py"),
    ("📖", "Data Story & Domain Context", "Data Story", "app/pages/1_data_story.py"),
    ("📊", "Statistical EDA & Hypothesis Tests", "Statistical EDA", "app/pages/2_statistical_eda.py"),
    ("🔍", "SHAP Feature Importance", "Feature Importance", "app/pages/3_feature_importance.py"),
    ("🔄", "What-If Counterfactual Planner", "What-If Analysis", "app/pages/4_what_if_analysis.py"),
    ("💰", "ROI & Business Impact", "Business Impact", "app/pages/5_business_impact.py"),
]

PAGE_ICONS = {path: icon for icon, _, _, path in PAGES}
PAGE_TITLES = {path: title for _, title, _, path in PAGES}
PAGE_SHORT = {path: short for _, _, short, path in PAGES}
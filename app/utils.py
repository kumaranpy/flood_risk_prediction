"""
AquaSense: Shared Theme & Component Utilities
Premium Deep-Ocean Dark Theme with Glassmorphism, Responsive Components, and Plotly Styling.
"""

from pathlib import Path
import json
import base64
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_SVG_PATH = ASSETS_DIR / "logo.svg"


def get_logo_svg_inline(width: int = 42, height: int = 42) -> str:
    """Return inline SVG markup for the AquaSense logo."""
    if LOGO_SVG_PATH.exists():
        try:
            content = LOGO_SVG_PATH.read_text(encoding="utf-8")
            # Inject custom width/height attributes if desired
            return f'<div style="display:inline-flex; align-items:center; justify-content:center; width:{width}px; height:{height}px;">{content}</div>'
        except Exception:
            pass
    return f'<span style="font-size:{width//2}px;">🌊</span>'


def get_logo_base64() -> str:
    """Return base64 data URI of the SVG logo for use in img tags."""
    if LOGO_SVG_PATH.exists():
        try:
            b64 = base64.b64encode(LOGO_SVG_PATH.read_bytes()).decode("utf-8")
            return f"data:image/svg+xml;base64,{b64}"
        except Exception:
            pass
    return ""


def inject_theme():
    """
    Inject AquaSense Deep-Ocean Dark Theme CSS.
    Palette:
      - Background: #050B18 -> #0A1A2F
      - Glass Card: #0F2236 (70% opacity, 16px blur)
      - Primary: #22D3EE (Aqua)
      - Secondary: #3B82F6 (Ocean Blue)
      - Success: #34D399 (Emerald)
      - Warning: #FBBF24 (Amber)
      - Danger: #F43F5E (Coral Red)
      - Text: #E6F1FF, Muted: #8BA3BF
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

        /* Root Typography & Accessibility */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #E6F1FF;
        }

        h1, h2, h3, h4 {
            font-family: 'Sora', sans-serif;
            letter-spacing: -0.02em;
            color: #FFFFFF;
        }

        /* Deep-Ocean App Canvas */
        .stApp {
            background: radial-gradient(1400px 700px at 15% -5%, #0e2b4a 0%, transparent 65%),
                        radial-gradient(1000px 500px at 85% 10%, #071f38 0%, transparent 55%),
                        linear-gradient(180deg, #050B18 0%, #0A1A2F 100%);
            background-attachment: fixed;
        }

        /* Hide Streamlit default branding */
        #MainMenu, footer {
            visibility: hidden;
            display: none !important;
        }

        /* Custom Sleek Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #050B18;
        }
        ::-webkit-scrollbar-thumb {
            background: #153A5B;
            border-radius: 999px;
            border: 2px solid #050B18;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #22D3EE;
        }

        /* Glassmorphism Cards */
        .glass, .glass-card {
            background: rgba(15, 34, 54, 0.70);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(34, 211, 238, 0.18);
            border-radius: 20px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45);
            transition: border-color 0.25s ease, box-shadow 0.25s ease;
        }
        .glass:hover, .glass-card:hover {
            border-color: rgba(34, 211, 238, 0.35);
            box-shadow: 0 16px 44px rgba(4, 20, 36, 0.6), 0 0 20px rgba(34, 211, 238, 0.1);
        }

        /* Premium AquaSense Hero Banner */
        .hero, .hero-banner {
            position: relative;
            background: linear-gradient(125deg, rgba(8, 145, 178, 0.95) 0%, rgba(37, 99, 235, 0.90) 55%, rgba(30, 58, 138, 0.92) 100%);
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 24px;
            padding: 2.2rem 2.6rem;
            color: #FFFFFF;
            box-shadow: 0 14px 45px rgba(14, 165, 233, 0.25), 0 0 30px rgba(34, 211, 238, 0.15);
            margin-bottom: 1.5rem;
            overflow: hidden;
        }
        .hero::before {
            content: "";
            position: absolute;
            top: -50%;
            right: -20%;
            width: 320px;
            height: 320px;
            background: radial-gradient(circle, rgba(34, 211, 238, 0.25) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
            animation: pulse-glow 6s ease-in-out infinite alternate;
        }
        @keyframes pulse-glow {
            0% { transform: scale(0.9) translateY(0px); opacity: 0.6; }
            100% { transform: scale(1.15) translateY(-20px); opacity: 0.95; }
        }
        .hero h1 {
            margin: 0;
            font-size: 2.35rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .hero p {
            margin: 0.5rem 0 0 0;
            font-size: 1.05rem;
            color: #E0F2FE;
            line-height: 1.55;
            max-width: 920px;
        }

        /* Pill Badges */
        .badge, .badge-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 0.3rem 0.95rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.8rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .badge-simulation {
            background: rgba(34, 211, 238, 0.15);
            color: #22D3EE;
            border: 1px solid rgba(34, 211, 238, 0.35);
        }
        .badge-low {
            background: rgba(52, 211, 153, 0.16);
            color: #34D399;
            border: 1px solid rgba(52, 211, 153, 0.35);
        }
        .badge-med {
            background: rgba(251, 191, 36, 0.16);
            color: #FBBF24;
            border: 1px solid rgba(251, 191, 36, 0.35);
        }
        .badge-high {
            background: rgba(244, 63, 94, 0.16);
            color: #F43F5E;
            border: 1px solid rgba(244, 63, 94, 0.35);
        }
        .badge-uncertain {
            background: rgba(168, 85, 247, 0.18);
            color: #C084FC;
            border: 1px solid rgba(168, 85, 247, 0.35);
        }

        /* Mandatory Simulation Disclaimer */
        .disclaimer-banner {
            background: rgba(244, 63, 94, 0.10);
            border: 1px solid rgba(244, 63, 94, 0.35);
            border-left: 4px solid #F43F5E;
            border-radius: 14px;
            padding: 1rem 1.4rem;
            margin-bottom: 1.6rem;
            color: #FECDD3;
            font-size: 0.88rem;
            line-height: 1.5;
            box-shadow: 0 4px 18px rgba(244, 63, 94, 0.12);
        }
        .disclaimer-banner b {
            color: #FFA4B2;
            letter-spacing: 0.04em;
        }

        /* Metrics Overrides */
        div[data-testid="stMetric"] {
            background: rgba(15, 34, 54, 0.70);
            border: 1px solid rgba(34, 211, 238, 0.18);
            border-radius: 18px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
            transition: all 0.2s ease;
        }
        div[data-testid="stMetric"]:hover {
            border-color: rgba(34, 211, 238, 0.4);
            transform: translateY(-2px);
        }
        div[data-testid="stMetric"] label {
            color: #8BA3BF !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-weight: 700 !important;
            font-size: 1.95rem !important;
        }

        /* Buttons with Vibrant Ocean Glow */
        .stButton > button {
            background: linear-gradient(90deg, #22D3EE 0%, #3B82F6 100%);
            color: #04101F;
            font-weight: 700;
            font-size: 0.92rem;
            border: 0;
            border-radius: 12px;
            padding: 0.65rem 1.5rem;
            letter-spacing: 0.02em;
            transition: all 0.22s ease-in-out;
            box-shadow: 0 4px 16px rgba(34, 211, 238, 0.25);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            color: #020817;
            box-shadow: 0 6px 24px rgba(34, 211, 238, 0.55), 0 0 15px rgba(59, 130, 246, 0.4);
        }
        .stButton > button:active {
            transform: translateY(0);
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background: rgba(8, 20, 36, 0.96) !important;
            border-right: 1px solid rgba(34, 211, 238, 0.15) !important;
            backdrop-filter: blur(20px);
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
            padding-top: 10px;
        }

        /* Preset Scenario Cards */
        .preset-card {
            background: rgba(15, 34, 54, 0.6);
            border: 1px solid rgba(34, 211, 238, 0.2);
            border-radius: 14px;
            padding: 12px 14px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .preset-card:hover {
            background: rgba(15, 34, 54, 0.9);
            border-color: #22D3EE;
            transform: translateY(-2px);
        }

        /* Custom Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background: rgba(8, 20, 36, 0.6);
            padding: 6px;
            border-radius: 14px;
            border: 1px solid rgba(34, 211, 238, 0.12);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border: 0;
            border-radius: 10px;
            padding: 10px 18px;
            color: #8BA3BF;
            font-weight: 600;
            font-size: 0.9rem;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #E6F1FF;
            background: rgba(34, 211, 238, 0.08);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, rgba(34, 211, 238, 0.22) 0%, rgba(59, 130, 246, 0.22) 100%) !important;
            color: #22D3EE !important;
            border: 1px solid rgba(34, 211, 238, 0.45) !important;
            box-shadow: 0 4px 14px rgba(34, 211, 238, 0.15);
        }

        /* Risk assessment banners */
        .risk-banner {
            border-radius: 20px;
            padding: 1.6rem 2rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 10px 35px rgba(0,0,0,0.4);
            border-width: 1px;
            border-style: solid;
        }
        .risk-banner-high {
            background: linear-gradient(135deg, rgba(159, 18, 57, 0.85) 0%, rgba(225, 29, 72, 0.75) 100%);
            border-color: rgba(244, 63, 94, 0.5);
            box-shadow: 0 10px 30px rgba(244, 63, 94, 0.25);
        }
        .risk-banner-medium {
            background: linear-gradient(135deg, rgba(180, 83, 9, 0.85) 0%, rgba(217, 119, 6, 0.75) 100%);
            border-color: rgba(251, 191, 36, 0.5);
            box-shadow: 0 10px 30px rgba(251, 191, 36, 0.25);
        }
        .risk-banner-low {
            background: linear-gradient(135deg, rgba(6, 95, 70, 0.85) 0%, rgba(16, 185, 129, 0.75) 100%);
            border-color: rgba(52, 211, 153, 0.5);
            box-shadow: 0 10px 30px rgba(52, 211, 153, 0.25);
        }
        .risk-banner-uncertain {
            background: linear-gradient(135deg, rgba(88, 28, 135, 0.85) 0%, rgba(126, 34, 206, 0.75) 100%);
            border-color: rgba(192, 132, 252, 0.5);
            box-shadow: 0 10px 30px rgba(192, 132, 252, 0.25);
        }

        /* Subtle animated probability bars */
        .prob-bar-track {
            background: rgba(8, 20, 36, 0.8);
            border-radius: 999px;
            height: 12px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin: 6px 0 12px 0;
        }
        .prob-bar-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Footer */
        .aquasense-footer {
            margin-top: 3.5rem;
            padding: 1.8rem 1rem;
            border-top: 1px solid rgba(34, 211, 238, 0.15);
            text-align: center;
            color: #8BA3BF;
            font-size: 0.85rem;
            line-height: 1.6;
        }
        .aquasense-footer a {
            color: #22D3EE;
            text-decoration: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(
    title: str = "AquaSense",
    subtitle: str = "Intelligent Flood Risk Intelligence",
    badge_text: str = "🛡️ AI Simulation Benchmark",
    badge_class: str = "badge-simulation",
):
    """Render the standard AquaSense hero banner."""
    logo_svg = get_logo_svg_inline(width=48, height=48)
    st.markdown(
        f"""
        <div class="hero">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span class="badge {badge_class}">{badge_text}</span>
                <span style="font-size:0.75rem; letter-spacing:0.08em; text-transform:uppercase; color:rgba(255,255,255,0.7); font-weight:700;">
                    v1.0.0 · Production Ready
                </span>
            </div>
            <div style="display:flex; align-items:center; gap:16px;">
                {logo_svg}
                <div>
                    <h1 style="margin:0;">{title}</h1>
                    <p style="margin:4px 0 0 0;">{subtitle}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Alias for backward compatibility across all existing pages
inject_global_css = inject_theme
render_hero = hero


def render_disclaimer():
    """Render the mandatory operational safety disclaimer."""
    st.markdown(
        """
        <div class="disclaimer-banner">
            <b>⚠️ SIMULATION & BENCHMARK DISCLAIMER (Operational Safety):</b><br>
            AquaSense is an academic research and engineering demonstration trained on the 
            synthetic <b>Kaggle Playground s4e5 benchmark</b> dataset. It does <b>NOT</b> ingest real-time geospatial feeds, 
            radar telemetry, or hydrological sensor streams. It must <b>NOT</b> be deployed for real-world evacuation dispatch, 
            emergency disaster response, or life-safety directives.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    """Render the standardized AquaSense footer."""
    st.markdown(
        """
        <div class="aquasense-footer">
            <div style="font-weight:700; color:#E6F1FF; margin-bottom:4px;">
                🌊 AquaSense · Intelligent Flood Risk Intelligence
            </div>
            <div>
                Built with <b>scikit-learn 1.5.1</b>, <b>Streamlit</b>, and <b>FastAPI</b>. 
                Cryptographically verified models (SHA-256). Zero proxy target leakage.
            </div>
            <div style="margin-top:6px; font-size:0.78rem; color:#64748B;">
                Academic simulation demonstration · Not for operational disaster management or life-safety decisions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_navigation():
    """Render enhanced AquaSense sidebar with branding, metrics, and navigation."""
    with st.sidebar:
        logo_svg = get_logo_svg_inline(width=36, height=36)
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:12px; padding:8px 4px 16px 4px; border-bottom:1px solid rgba(34,211,238,0.15); margin-bottom:14px;">
                {logo_svg}
                <div>
                    <div style="font-family:'Sora',sans-serif; font-size:1.25rem; font-weight:800; color:#FFFFFF; letter-spacing:-0.02em;">AquaSense</div>
                    <div style="font-size:0.75rem; color:#22D3EE; font-weight:600; text-transform:uppercase; letter-spacing:0.06em;">Flood Intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<p style='color:#8BA3BF; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; font-weight:700; margin-bottom:6px;'>Modules & Pages</p>", unsafe_allow_html=True)
        pages = [
            ("🏠", "Real-Time Inference", "Home"),
            ("📖", "Data Story & Domain Context", "1_data_story"),
            ("📊", "Statistical EDA & Tests", "2_statistical_eda"),
            ("🔍", "SHAP Feature Attribution", "3_feature_importance"),
            ("🔄", "What-If Scenario Planner", "4_what_if_analysis"),
            ("💰", "ROI & Business Impact", "5_business_impact"),
        ]
        for icon, label, _ in pages:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:10px; padding:8px 12px; 
                            background:rgba(15,34,54,0.55); border:1px solid rgba(34,211,238,0.08); 
                            border-radius:10px; margin:4px 0; color:#E6F1FF; font-size:0.86rem; font-weight:500;">
                    <span>{icon}</span>
                    <span>{label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("<p style='color:#8BA3BF; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; font-weight:700; margin-bottom:6px;'>Verified ML Pipeline</p>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="font-size:0.82rem; color:#CBD5E1; line-height:1.7;">
                • <b>Model</b>: Calibrated LogisticRegression<br>
                • <b>CV F1-Score</b>: <code>0.7130</code> (Leakage-Free)<br>
                • <b>Accuracy</b>: <code>71.34%</code> on held-out test<br>
                • <b>ROC-AUC</b>: <code>0.8781</code> (OvR)<br>
                • <b>Guardrail</b>: SHA-256 hash verified<br>
                • <b>Safety Layer</b>: UNCERTAIN when max prob &lt; 0.65
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Model registry info
        st.markdown("---")
        registry_path = PROJECT_ROOT / "models" / "registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, "r") as f:
                    registry = json.load(f)
                st.markdown(
                    f"""
                    <div style="font-size:0.78rem; color:#8BA3BF; line-height:1.5;">
                        <span style="color:#22D3EE; font-weight:700;">REGISTRY:</span> v{registry.get('model_version', '1.0.0')}<br>
                        Trained: {registry.get('training_samples', 40000):,} samples<br>
                        Checksum: <code>9bb391b...</code>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception:
                pass


def get_plotly_dark_template() -> dict:
    """Return consistent Plotly dark theme layout specifications."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E6F1FF", size=12),
        margin=dict(l=30, r=30, t=40, b=30),
        xaxis=dict(
            gridcolor="rgba(34, 211, 238, 0.08)",
            zerolinecolor="rgba(34, 211, 238, 0.15)",
            tickfont=dict(color="#8BA3BF"),
        ),
        yaxis=dict(
            gridcolor="rgba(34, 211, 238, 0.08)",
            zerolinecolor="rgba(34, 211, 238, 0.15)",
            tickfont=dict(color="#8BA3BF"),
        ),
    )


def create_risk_gauge_chart(score: float, pred_class: str, confidence: float) -> go.Figure:
    """
    Create a large, beautiful Plotly risk gauge chart with color zones.
    score: 0.0 to 100.0 risk score
    pred_class: 'Low', 'Medium', 'High', or 'UNCERTAIN'
    """
    bar_color = "#34D399" if pred_class == "Low" else ("#FBBF24" if pred_class == "Medium" else ("#F43F5E" if pred_class == "High" else "#C084FC"))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%", "font": {"family": "JetBrains Mono, monospace", "size": 42, "color": "#FFFFFF"}},
            title={"text": f"<b>{pred_class.upper()} RISK ASSESSMENT</b>", "font": {"family": "Sora, sans-serif", "size": 17, "color": bar_color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#8BA3BF", "tickfont": {"color": "#8BA3BF", "size": 11}},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(15, 34, 54, 0.6)",
                "borderwidth": 1,
                "bordercolor": "rgba(34, 211, 238, 0.2)",
                "steps": [
                    {"range": [0, 33.3], "color": "rgba(52, 211, 153, 0.18)"},
                    {"range": [33.3, 66.6], "color": "rgba(251, 191, 36, 0.18)"},
                    {"range": [66.6, 100], "color": "rgba(244, 63, 94, 0.18)"},
                ],
                "threshold": {
                    "line": {"color": "#FFFFFF", "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )
    layout = get_plotly_dark_template()
    layout.update(height=260, margin=dict(l=25, r=25, t=50, b=15))
    fig.update_layout(**layout)
    return fig


def get_page_config(page_title: str = "AquaSense · Flood Intelligence", page_icon: str = "🌊") -> dict:
    """Standard page config for all pages."""
    return {
        "page_title": page_title,
        "page_icon": page_icon,
        "layout": "wide",
        "initial_sidebar_state": "expanded",
    }


# Page metadata
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
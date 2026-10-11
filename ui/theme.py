"""theme — single visual language (tokens + global stylesheet).

Institutional terminal aesthetic: dark slate surfaces, IBM Plex Mono,
one restrained blue accent. Semantic colours (positive/negative/warning)
are reserved for values that need them — never decoration.
"""

from __future__ import annotations

import streamlit as st

TOKENS = {
    "background": "#0b0f14",
    "surface": "#11161d",
    "surface_raised": "#161d27",
    "border": "#232c38",
    "text_primary": "#e6ebf2",
    "text_secondary": "#9aa7b8",
    "text_muted": "#64707f",
    "accent": "#58a6ff",
    "accent_dim": "#1f3a5f",
    "positive": "#3fb950",
    "negative": "#f85149",
    "warning": "#d29922",
    "info": "#58a6ff",
    "radius_sm": "6px",
    "radius_md": "10px",
    "space_xs": "4px",
    "space_sm": "8px",
    "space_md": "16px",
    "space_lg": "24px",
    "font": "'IBM Plex Mono', ui-monospace, monospace",
}

CHART_COLORS = ["#58a6ff", "#3fb950", "#f0883e", "#bc8cff", "#f85149",
                "#79c0ff", "#ffa657", "#7ee787"]

CHART_THEME = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Mono, monospace", color="#c9d1d9"),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    margin=dict(l=60, r=30, t=60, b=60),
)

_GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Mono', ui-monospace, monospace; }
.stApp { background: #0b0f14; color: #e6ebf2; }
section[data-testid="stSidebar"] { background: #0e1319; border-right: 1px solid #232c38; }
section[data-testid="stSidebar"] .stMarkdown h2 { font-size: 13px; letter-spacing: 2px; color: #9aa7b8; }
div[data-testid="stMetric"] { background: #11161d; border: 1px solid #232c38; border-radius: 10px; padding: 12px 14px; }
div[data-testid="stMetric"] label { color: #9aa7b8 !important; }
div[data-testid="stMetricValue"] { color: #58a6ff; }
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] > div {
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
}
.stButton>button { border-radius: 6px; font-weight: 600; }
.stButton>button[kind="primary"] { background: #1f3a5f; border: 1px solid #58a6ff; color: #e6ebf2; }
div[data-testid="stTabs"] button { color: #9aa7b8; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: #e6ebf2; border-bottom: 2px solid #58a6ff; }
.stDataFrame { border: 1px solid #232c38; border-radius: 10px; }
details.stExpander { border: 1px solid #232c38; border-radius: 10px; background: #11161d; }
.stAlert { border-radius: 10px; }
hr { border-color: #232c38; }
.pc-card { background: #11161d; border: 1px solid #232c38; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; }
.pc-kicker { font-size: 11px; letter-spacing: 2px; color: #64707f; margin-bottom: 4px; }
.pc-pill { display: inline-block; padding: 2px 10px; border-radius: 999px; border: 1px solid #232c38; font-size: 12px; color: #9aa7b8; }
.pc-pill-good { border-color: #1f5c38; color: #3fb950; }
.pc-pill-bad { border-color: #6e2b28; color: #f85149; }
.pc-pill-warn { border-color: #6b5416; color: #d29922; }
@media (max-width: 900px) {
  div[data-testid="stMetric"] { padding: 8px 10px; }
  .pc-card { padding: 10px 12px; }
}
"""


def inject_theme() -> None:
    """Apply page config + global stylesheet (idempotent)."""
    try:
        st.set_page_config(
            page_title="Portfolio Calc — Research & Risk",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass  # set_page_config is first-call-only; safe to skip on reruns
    st.markdown(f"<style>{_GLOBAL_CSS}</style>", unsafe_allow_html=True)


def tokens() -> dict:
    """Design tokens (copy — callers must not mutate the registry)."""
    return dict(TOKENS)

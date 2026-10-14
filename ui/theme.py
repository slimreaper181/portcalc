"""theme — blue institutional design system (light + dark).

Premium-fintech visual language: soft neutral page, white/dark cards with
large radii and restrained shadows, strong type hierarchy (eyebrow / title /
metric), segmented navigation, pill statuses, blue-led charts.

Design rules enforced here:

* ONE token registry per appearance (``LIGHT`` / ``DARK``); views must not
  hardcode colours. Backwards-compatible names (``TOKENS``,
  ``CHART_COLORS``, ``CHART_THEME``, ``tokens()``, ``inject_theme()``) are
  preserved; ``CHART_THEME`` / ``CHART_COLORS`` are refreshed **in place**
  per run so every ``**CHART_THEME`` call site picks up the active mode.
* Positive/negative stay green/red (economic meaning); blue is brand/accent.
* CSS targets stable Streamlit test-ids and our own ``pc-*`` classes only —
  no brittle deep-DOM selectors.
"""

from __future__ import annotations

import html

import streamlit as st

# ---------------------------------------------------------------------------
# Appearance state (session-persistent, display-only)
# ---------------------------------------------------------------------------

APPEARANCE_KEY = "appearance"
APPEARANCES = ("light", "dark")
_APPEARANCE_LABELS = {"light": "Light", "dark": "Dark"}


def get_appearance() -> str:
    """Active appearance ('light' default). Safe outside a session."""
    try:
        mode = st.session_state.get(APPEARANCE_KEY, "light")
    except Exception:
        return "light"
    return mode if mode in APPEARANCES else "light"


def set_appearance(mode: str) -> str:
    """Validate + persist an appearance choice. Returns the active mode."""
    mode = str(mode or "").strip().lower()
    if mode not in APPEARANCES:
        raise ValueError(f"Unknown appearance {mode!r}.")
    try:
        st.session_state[APPEARANCE_KEY] = mode
    except Exception:
        pass
    return mode


def appearance_selector() -> str:
    """Sidebar Light/Dark control. New key; existing keys untouched.

    Display-only: changing appearance never clears result caches (those are
    keyed by analytical inputs, never by theme).
    """
    try:
        current = get_appearance()
    except Exception:
        current = "light"
    try:
        choice = st.radio(
            "Appearance",
            options=list(APPEARANCES),
            index=list(APPEARANCES).index(current),
            format_func=lambda m: ("Light" if m == "light" else "Dark"),
            key=APPEARANCE_KEY,
            horizontal=True,
            help="Light or dark workstation theme. Display only — "
                 "analytical results are never recomputed by this choice.",
        )
        return choice if choice in APPEARANCES else current
    except Exception:
        return current


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

FONT_STACK = ("Inter, 'SF Pro Display', 'Segoe UI', system-ui, "
              "-apple-system, sans-serif")
FONT_MONO = ("ui-monospace, 'SF Mono', 'Cascadia Code', Menlo, "
             "Consolas, monospace")

LIGHT: dict = {
    # surfaces
    "background": "#F5F7FA",
    "surface": "#FFFFFF",
    "surface_elevated": "#FFFFFF",
    "surface_subtle": "#EFF6FF",
    "surface_raised": "#FFFFFF",  # back-compat alias
    # text
    "text_primary": "#111827",
    "text_secondary": "#64748B",
    "text_muted": "#94A3B8",
    # borders
    "border": "#E5E7EB",
    "border_strong": "#CBD5E1",
    # brand blue
    "accent": "#2563EB",
    "accent_hover": "#1D4ED8",
    "accent_soft": "#DBEAFE",
    "accent_strong": "#1E3A8A",
    "accent_dim": "#DBEAFE",  # back-compat alias
    # semantics (economic meaning, NOT brand)
    "success": "#16A34A",
    "positive": "#16A34A",  # back-compat alias
    "warning": "#D97706",
    "danger": "#DC2626",
    "negative": "#DC2626",  # back-compat alias
    "info": "#2563EB",
    # elevation
    "shadow_sm": "0 1px 2px rgba(17, 24, 39, 0.05)",
    "shadow_md": "0 4px 16px rgba(17, 24, 39, 0.07)",
    # shape / space / type
    "radius_sm": "8px",
    "radius_md": "12px",
    "radius_lg": "18px",
    "radius_xl": "24px",
    "space_xs": "4px",
    "space_sm": "8px",
    "space_md": "16px",
    "space_lg": "24px",
    "font": FONT_STACK,
}

DARK: dict = {
    "background": "#0B1020",
    "surface": "#111827",
    "surface_elevated": "#172033",
    "surface_subtle": "#172554",
    "surface_raised": "#172033",  # back-compat alias
    "text_primary": "#F8FAFC",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
    "border": "rgba(148, 163, 184, 0.15)",
    "border_strong": "rgba(148, 163, 184, 0.30)",
    "accent": "#3B82F6",
    "accent_hover": "#60A5FA",
    "accent_soft": "#172554",
    "accent_strong": "#93C5FD",
    "accent_dim": "#172554",  # back-compat alias
    "success": "#34D399",
    "positive": "#34D399",  # back-compat alias
    "warning": "#FBBF24",
    "danger": "#F87171",
    "negative": "#F87171",  # back-compat alias
    "info": "#60A5FA",
    "shadow_sm": "0 1px 2px rgba(0, 0, 0, 0.40)",
    "shadow_md": "0 6px 20px rgba(0, 0, 0, 0.45)",
    "radius_sm": "8px",
    "radius_md": "12px",
    "radius_lg": "18px",
    "radius_xl": "24px",
    "space_xs": "4px",
    "space_sm": "8px",
    "space_md": "16px",
    "space_lg": "24px",
    "font": FONT_STACK,
}

#: Back-compat registry (light defaults). Prefer ``tokens()``.
TOKENS: dict = dict(LIGHT)

_MODES = {"light": LIGHT, "dark": DARK}


def tokens(mode: str | None = None) -> dict:
    """Design tokens for ``mode`` (default: active appearance).

    Returns a copy — callers must not mutate the registry.
    """
    want = str(mode or "").strip().lower() if mode else get_appearance()
    return dict(_MODES.get(want, LIGHT))


# ---------------------------------------------------------------------------
# Charts (mode-aware, refreshed in place per run)
# ---------------------------------------------------------------------------

CHART_COLORS: list = [
    "#2563EB", "#64748B", "#0EA5E9", "#8B5CF6", "#F59E0B", "#10B981",
    "#EF4444", "#06B6D3",
]

CHART_THEME: dict = dict(
    paper_bgcolor="rgba(0, 0, 0, 0)",
    plot_bgcolor="rgba(0, 0, 0, 0)",
    font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif",
              color="#111827", size=12),
    xaxis=dict(gridcolor="#E5E7EB", zerolinecolor="#E5E7EB",
               title_font=dict(size=12), tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#E5E7EB", zerolinecolor="#E5E7EB",
               title_font=dict(size=12), tickfont=dict(size=11)),
    hoverlabel=dict(bgcolor="#FFFFFF", font=dict(color="#111827", size=12),
                    bordercolor="#E5E7EB"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right",
                x=1, font=dict(size=11)),
    margin=dict(l=60, r=30, t=60, b=60),
)

_CHART_COLORS_LIGHT = list(CHART_COLORS)
_CHART_COLORS_DARK = [
    "#60A5FA", "#94A3B8", "#38BDF8", "#A78BFA", "#FBBF24", "#34D399",
    "#F87171", "#22D3EE",
]


def _chart_theme_for(mode: str) -> dict:
    """Full Plotly layout for ``mode`` (transparent over cards)."""
    if mode == "dark":
        return dict(
            paper_bgcolor="rgba(0, 0, 0, 0)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif",
                      color="#F8FAFC", size=12),
            xaxis=dict(gridcolor="rgba(148, 163, 184, 0.18)",
                       zerolinecolor="rgba(148, 163, 184, 0.30)",
                       title_font=dict(size=12), tickfont=dict(size=11)),
            yaxis=dict(gridcolor="rgba(148, 163, 184, 0.18)",
                       zerolinecolor="rgba(148, 163, 184, 0.30)",
                       title_font=dict(size=12), tickfont=dict(size=11)),
            hoverlabel=dict(bgcolor="#172033",
                            font=dict(color="#F8FAFC", size=12),
                            bordercolor="rgba(148, 163, 184, 0.30)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, font=dict(size=11)),
            margin=dict(l=60, r=30, t=60, b=60),
        )
    return dict(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif",
                  color="#111827", size=12),
        xaxis=dict(gridcolor="#E5E7EB", zerolinecolor="#E5E7EB",
                   title_font=dict(size=12), tickfont=dict(size=11)),
        yaxis=dict(gridcolor="#E5E7EB", zerolinecolor="#E5E7EB",
                   title_font=dict(size=12), tickfont=dict(size=11)),
        hoverlabel=dict(bgcolor="#FFFFFF",
                        font=dict(color="#111827", size=12),
                        bordercolor="#E5E7EB"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
        margin=dict(l=60, r=30, t=60, b=60),
    )


def _apply_chart_mode(mode: str) -> None:
    """Refresh chart globals in place (existing imports keep working)."""
    CHART_THEME.clear()
    CHART_THEME.update(_chart_theme_for(mode))
    CHART_COLORS[:] = (_CHART_COLORS_DARK if mode == "dark"
                       else _CHART_COLORS_LIGHT)


# ---------------------------------------------------------------------------
# Global stylesheet
# ---------------------------------------------------------------------------

def _css(t: dict, mode: str) -> str:
    """Stylesheet for one token set (stable selectors + pc-* classes).

    Root-cause design: EVERY text-bearing widget selector gets an explicit
    foreground/background/border per mode, plus ``color-scheme`` on the
    root, so rendering never depends on Streamlit's base chrome or the
    browser/OS colour scheme. Selectors stay at the test-id / baseweb /
    ARIA-role level already used by this codebase — no deep-DOM hacks.
    """
    scheme = "dark" if mode == "dark" else "light"
    return f"""
html {{ color-scheme: {scheme}; }}
html, body, [class*="css"] {{
  font-family: {t['font']};
  font-feature-settings: "cv11", "ss01";
}}
.stApp {{ background: {t['background']}; color: {t['text_primary']}; }}
.block-container {{ padding-top: 1.5rem; max-width: 1400px; }}

/* ---------- base text (explicit so browser chrome never wins) ---------- */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {{
  color: {t['text_primary']};
}}
div[data-testid="stCaptionContainer"] {{
  color: {t['text_secondary']};
}}
div[data-testid="stCaptionContainer"] p {{ color: {t['text_secondary']}; }}

/* ---------- sidebar (full inheritance audit) ---------- */
section[data-testid="stSidebar"] {{
  background: {t['surface']}; border-right: 1px solid {t['border']};
  color: {t['text_primary']};
}}
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] li {{
  color: {t['text_primary']};
}}
section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] p {{
  color: {t['text_secondary']};
}}
section[data-testid="stSidebar"] .stMarkdown h2 {{
  font-size: 13px; letter-spacing: 2px; color: {t['text_secondary']};
}}

/* ---------- widget labels / help ---------- */
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p {{
  color: {t['text_secondary']};
}}
div[data-testid="stWidgetLabel"] {{
  color: {t['text_secondary']};
}}
[data-testid="stTooltipIcon"], [data-testid="stHelpIcon"] {{
  color: {t['text_muted']};
}}

/* ---------- segmented tab navigation ----------
   Multi-variant selectors (test-id + baseweb + ARIA role) so the active
   state never falls through to Streamlit's default red primary. Selected
   uses our own background + accent text + blue underline; the native
   underline/shadow is neutralised. */
div[data-testid="stTabs"] div[role="tablist"],
div[data-testid="stTabs"] div[data-baseweb="tab-list"] {{
  background: {t['surface']}; border: 1px solid {t['border']};
  border-radius: 999px; padding: 4px; gap: 2px;
  box-shadow: {t['shadow_sm']};
}}
div[data-testid="stTabs"] button[role="tab"],
div[data-testid="stTabs"] div[data-baseweb="tab"] {{
  color: {t['text_secondary']}; border-radius: 999px; font-weight: 600;
  padding: 6px 14px; box-shadow: none;
  border-bottom: 2px solid transparent;
}}
div[data-testid="stTabs"] button[role="tab"]:hover,
div[data-testid="stTabs"] div[data-baseweb="tab"]:hover {{
  color: {t['text_primary']}; background: {t['surface_subtle']};
}}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
div[data-testid="stTabs"] div[data-baseweb="tab"][aria-selected="true"] {{
  color: {t['accent_hover']}; background: {t['accent_soft']};
  box-shadow: none; border-bottom: 2px solid {t['accent']};
}}

/* ---------- metric cards + summary strips ---------- */
div[data-testid="stMetric"] {{
  background: {t['surface']}; border: 1px solid {t['border']};
  border-radius: {t['radius_lg']}; padding: 14px 16px;
  box-shadow: {t['shadow_sm']};
}}
div[data-testid="stMetric"] label {{
  color: {t['text_secondary']} !important; font-size: 11px;
  letter-spacing: 1.5px; text-transform: uppercase; font-weight: 700;
}}
div[data-testid="stMetricValue"] {{
  color: {t['text_primary']}; font-weight: 750; font-size: 27px;
  letter-spacing: -0.01em; font-variant-numeric: tabular-nums;
}}
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] > div {{
  white-space: normal !important; overflow: visible !important;
  text-overflow: clip !important;
}}
div[data-testid="stMetricDelta"] {{ font-variant-numeric: tabular-nums; }}

/* ---------- buttons ---------- */
.stButton>button, .stDownloadButton>button {{
  border-radius: 10px; font-weight: 650;
  border: 1px solid {t['border']};
  color: {t['text_primary']};
}}
.stButton>button[kind="primary"], .stDownloadButton>button[kind="primary"] {{
  background: {t['accent']}; border-color: {t['accent']}; color: #FFFFFF;
}}
.stButton>button[kind="primary"]:hover,
.stDownloadButton>button[kind="primary"]:hover {{
  background: {t['accent_hover']}; border-color: {t['accent_hover']};
  color: #FFFFFF;
}}
.stButton>button[kind="secondary"], .stDownloadButton>button {{
  background: {t['surface']}; color: {t['accent_hover']};
  border-color: {t['border_strong']};
}}
.stButton>button[kind="tertiary"] {{
  background: transparent; border-color: transparent;
  color: {t['text_muted']}; font-weight: 600;
}}
.stButton>button[kind="tertiary"]:hover {{
  color: {t['danger']}; background: {t['surface_subtle']};
  border-color: {t['border']};
}}
.stButton>button:disabled, .stDownloadButton>button:disabled {{
  opacity: 0.65; color: {t['text_muted']};
}}

/* ---------- inputs (explicit per-mode surfaces + text) ---------- */
div[data-baseweb="select"] > div {{
  background: {t['surface']}; border-color: {t['border_strong']};
  border-radius: 10px; color: {t['text_primary']};
}}
div[data-baseweb="select"] span {{
  color: {t['text_primary']};
}}
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stDateInput"] input {{
  background: {t['surface']}; color: {t['text_primary']};
  border-color: {t['border_strong']};
  border-radius: 10px !important;
}}
input::placeholder, textarea::placeholder {{
  color: {t['text_muted']}; opacity: 1;
}}
div[role="listbox"], div[data-baseweb="popover"], div[data-baseweb="menu"] {{
  background: {t['surface_elevated']}; color: {t['text_primary']};
}}
div[data-testid="stSlider"] {{
  padding-top: 6px; color: {t['text_primary']};
}}
div[data-testid="stRadio"] label, div[data-testid="stCheckbox"] label {{
  color: {t['text_primary']};
}}
div[data-testid="stRadio"] {{
  color: {t['text_primary']};
}}
input:disabled, textarea:disabled, [aria-disabled="true"] {{
  color: {t['text_muted']}; opacity: 0.8;
}}

/* ---------- expanders / alerts / tables / dividers ---------- */
details.stExpander {{
  border: 1px solid {t['border']}; border-radius: {t['radius_lg']};
  background: {t['surface']}; box-shadow: {t['shadow_sm']};
  color: {t['text_primary']};
}}
details.stExpander summary {{
  color: {t['text_primary']}; font-weight: 650;
}}
.stAlert {{ border-radius: {t['radius_md']}; }}
.stDataFrame {{
  border: 1px solid {t['border']}; border-radius: 14px;
  box-shadow: {t['shadow_sm']}; background: {t['surface']};
  color: {t['text_primary']};
}}
.stDataFrame th {{
  color: {t['text_secondary']}; font-weight: 700;
}}
.stDataFrame [role="gridcell"] {{
  color: {t['text_primary']};
}}
hr {{ border-color: {t['border']}; }}

/* ---------- workstation primitives ---------- */
.pc-card {{
  background: {t['surface']}; border: 1px solid {t['border']};
  border-radius: {t['radius_lg']}; padding: 18px 20px; margin-bottom: 14px;
  box-shadow: {t['shadow_sm']};
}}
.pc-card-accent {{
  background: linear-gradient(135deg, {t['accent_soft']}, {t['surface']});
  border: 1px solid {t['border']};
  border-radius: {t['radius_xl']}; padding: 20px 22px; margin-bottom: 16px;
  box-shadow: {t['shadow_md']};
}}
.pc-eyebrow {{
  font-size: 11px; font-weight: 800; letter-spacing: 2px;
  text-transform: uppercase; color: {t['accent_hover']}; margin-bottom: 2px;
}}
.pc-kicker {{
  font-size: 11px; letter-spacing: 2px; color: {t['text_secondary']};
  margin-bottom: 4px; font-weight: 700;
}}
.pc-title {{
  font-size: 27px; font-weight: 750; letter-spacing: -0.02em;
  color: {t['text_primary']}; margin: 0 0 2px 0; line-height: 1.15;
}}
.pc-subtitle {{
  font-size: 14px; color: {t['text_secondary']}; margin: 0 0 12px 0;
}}
.pc-strip {{
  background: {t['surface']}; border: 1px solid {t['border']};
  border-radius: {t['radius_xl']}; box-shadow: {t['shadow_md']};
  padding: 18px 6px 14px 6px; margin-bottom: 16px;
}}
.pc-strip-grid {{ display: flex; flex-wrap: wrap; }}
.pc-strip-item {{
  flex: 1 1 0; min-width: 150px; padding: 2px 20px;
  border-left: 1px solid {t['border']};
}}
.pc-strip-item:first-child {{ border-left: none; }}
.pc-strip-eyebrow {{
  font-size: 10.5px; font-weight: 800; letter-spacing: 1.6px;
  text-transform: uppercase; color: {t['text_secondary']}; margin-bottom: 4px;
}}
.pc-strip-value {{
  font-size: 27px; font-weight: 750; letter-spacing: -0.015em;
  color: {t['text_primary']}; font-variant-numeric: tabular-nums;
  line-height: 1.2; overflow-wrap: anywhere;
}}
.pc-strip-sub {{
  font-size: 12.5px; color: {t['text_secondary']}; margin-top: 3px;
}}
.pc-pill {{
  display: inline-block; padding: 2px 12px; border-radius: 999px;
  border: 1px solid {t['border_strong']}; font-size: 12px; font-weight: 650;
  color: {t['text_secondary']}; background: {t['surface']};
  margin: 0 4px 4px 0; white-space: nowrap;
}}
.pc-pill-accent {{
  border-color: {t['accent']}; color: {t['accent_hover']};
  background: {t['accent_soft']};
}}
.pc-pill-good {{
  border-color: {t['success']}; color: {t['success']}; background: {t['surface']};
}}
.pc-pill-bad {{
  border-color: {t['danger']}; color: {t['danger']}; background: {t['surface']};
}}
.pc-pill-warn {{
  border-color: {t['warning']}; color: {t['warning']}; background: {t['surface']};
}}
.pc-pill-info {{
  border-color: {t['info']}; color: {t['info']}; background: {t['surface']};
}}
.pc-tag {{
  display: inline-block; padding: 1px 10px; border-radius: 999px;
  font-size: 11.5px; font-weight: 650; color: {t['accent_hover']};
  background: {t['accent_soft']}; margin: 0 4px 4px 0; white-space: nowrap;
}}
.pc-muted {{ color: {t['text_secondary']}; }}
.pc-mono {{ font-family: {FONT_MONO}; font-size: 12.5px; }}
@media (max-width: 900px) {{
  div[data-testid="stMetric"] {{ padding: 10px 12px; }}
  div[data-testid="stMetricValue"] {{ font-size: 22px; }}
  .pc-card, .pc-card-accent {{ padding: 14px 14px; }}
  .pc-strip-item {{ min-width: 45%; border-left: none; margin-bottom: 10px; }}
  .pc-title {{ font-size: 23px; }}
}}
"""


def inject_theme() -> None:
    """Apply page config + mode stylesheet (idempotent, per-run)."""
    try:
        st.set_page_config(
            page_title="Portfolio Calc — Research & Risk",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass  # set_page_config is first-call-only; safe to skip on reruns
    mode = get_appearance()
    try:
        if APPEARANCE_KEY not in st.session_state:
            st.session_state[APPEARANCE_KEY] = mode
    except Exception:
        pass
    _apply_chart_mode(mode)
    st.markdown(f"<style>{_css(_MODES[mode], mode)}</style>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Presentation helpers (markdown-only: no widgets, no state, no new keys)
# ---------------------------------------------------------------------------

def _esc(text) -> str:
    return html.escape("" if text is None else str(text), quote=True)


def page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    """Consistent view header: eyebrow / title / subtitle (no widgets)."""
    st.markdown(
        f"<div class='pc-eyebrow'>{_esc(eyebrow)}</div>"
        f"<div class='pc-title'>{_esc(title)}</div>"
        + (f"<p class='pc-subtitle'>{_esc(subtitle)}</p>" if subtitle else ""),
        unsafe_allow_html=True,
    )


def metric_strip(items: list) -> None:
    """One broad summary panel: [(eyebrow, value, sub)] (all pre-formatted).

    Visual grouping only — values render verbatim, so numbers cannot drift
    from their ``st.metric`` equivalents.
    """
    cells = []
    for row in items or []:
        label, value, sub = (list(row) + ["", "", ""])[:3]
        cells.append(
            "<div class='pc-strip-item'>"
            f"<div class='pc-strip-eyebrow'>{_esc(label)}</div>"
            f"<div class='pc-strip-value'>{_esc(value)}</div>"
            + (f"<div class='pc-strip-sub'>{_esc(sub)}</div>" if sub else "")
            + "</div>")
    st.markdown(
        "<div class='pc-strip'><div class='pc-strip-grid'>"
        + "".join(cells) + "</div></div>",
        unsafe_allow_html=True,
    )


_PILL_KINDS = {
    "live": "pc-pill-accent", "streaming": "pc-pill-accent",
    "accent": "pc-pill-accent", "good": "pc-pill-good",
    "pass": "pc-pill-good", "filled": "pc-pill-good",
    "bad": "pc-pill-bad", "failed": "pc-pill-bad",
    "rejected": "pc-pill-bad", "stale": "pc-pill-warn",
    "warn": "pc-pill-warn", "warning": "pc-pill-warn",
    "delayed": "pc-pill-warn", "fallback": "pc-pill-info",
    "info": "pc-pill-info", "closed": "pc-pill",
    "neutral": "pc-pill", "sim": "pc-pill-warn",
}


def pill(text: str, kind: str = "neutral") -> str:
    """Inline pill-badge HTML (returns string; caller renders it)."""
    cls = _PILL_KINDS.get(str(kind or "").strip().lower(), "pc-pill")
    return f"<span class='{cls}'>{_esc(text)}</span>"


def pills_row(pills: list) -> None:
    """Render ``[(text, kind)]`` as one row of pill badges (no widgets)."""
    st.markdown(
        "".join(pill(t, k) for t, k in (pills or [])),
        unsafe_allow_html=True,
    )


def status_pill(status: str) -> str:
    """Map a market-data status word to its pill (meaning unchanged)."""
    s = str(status or "").strip().upper()
    kind = {"LIVE": "live", "STREAMING": "live", "DELAYED": "delayed",
            "STALE": "stale", "FALLBACK": "fallback",
            "CLOSED": "closed", "POLLING": "info"}.get(s, "neutral")
    return pill(s or "?", kind)


def tag_pill(tag: str) -> str:
    """Research-library tag chip HTML."""
    return f"<span class='pc-tag'>{_esc(tag)}</span>"


def tags_row(tags: list) -> None:
    """Render research tags as chips (plain-text tags, escaped)."""
    if not tags:
        return
    st.markdown("".join(tag_pill(t) for t in tags), unsafe_allow_html=True)

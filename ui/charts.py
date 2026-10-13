"""charts — one Plotly design system (pure helpers + thin renderers).

Sizes: compact (320px) / standard (480px) / large (640px). TradingView
embed heights (650/900) stay with the TradingView component — out of scope.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .theme import CHART_COLORS, CHART_THEME

SIZES = {"compact": 320, "standard": 480, "large": 640}


def palette(i: int) -> str:
    """Deterministic colour-cycle entry."""
    return CHART_COLORS[int(i) % len(CHART_COLORS)]


def base_layout(title: str = "", ytitle: str = "", size: str = "standard",
                height: int | None = None) -> dict:
    """Shared Plotly layout (pure — returns a layout dict)."""
    layout = dict(CHART_THEME)
    layout.update(
        title=dict(text=title, font=dict(size=15)) if title else {},
        xaxis_title="",
        margin=dict(l=60, r=20, t=50 if title else 20, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        height=int(height) if height else SIZES.get(size, 480),
    )
    if ytitle:
        layout["yaxis_title"] = ytitle
    return layout


def line_chart(df: pd.DataFrame, title: str = "", ytitle: str = "",
               size: str = "standard", percent: bool = False) -> go.Figure:
    """Multi-series line chart from a DataFrame (columns = series)."""
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col].values, mode="lines", name=str(col),
            line=dict(color=palette(i), width=2.0)))
    fig.update_layout(**base_layout(title, ytitle, size))
    if percent:
        fig.update_layout(yaxis_tickformat=".1%")
    return fig


def stacked_area(df: pd.DataFrame, title: str = "", ytitle: str = "Weight",
                 size: str = "standard") -> go.Figure:
    """Stacked-area chart (allocation-through-time, weight drift)."""
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col].values, mode="lines", name=str(col),
            stackgroup="one", line=dict(color=palette(i), width=0.5)))
    fig.update_layout(**base_layout(title, ytitle, size))
    fig.update_layout(yaxis_tickformat=".0%")
    return fig


def horizontal_bars(labels: list, values: list, title: str = "",
                    size: str = "compact") -> go.Figure:
    """Ranked horizontal bars (allocation, exposures)."""
    order = sorted(range(len(values)), key=lambda i: values[i],
                   reverse=True)
    fig = go.Figure(go.Bar(
        x=[values[i] for i in order], y=[str(labels[i]) for i in order],
        orientation="h", marker_color=[palette(i) for i in range(len(order))]))
    fig.update_layout(**base_layout(title, "", size))
    return fig


def centered_bars(labels: list, values: list, title: str = "",
                  size: str = "standard") -> go.Figure:
    """Zero-centred horizontal bars (factor exposures — sign matters)."""
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in values]
    fig = go.Figure(go.Bar(x=list(values), y=[str(l) for l in labels],
                           orientation="h", marker_color=colors))
    fig.update_layout(**base_layout(title, "", size))
    fig.add_vline(x=0, line_color="#64707f", line_width=1)
    return fig


def show(fig: go.Figure, key: str | None = None) -> None:
    """Render a chart full-width (key reserved for future use)."""
    st.plotly_chart(fig, width="stretch")

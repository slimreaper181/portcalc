"""bl — Black-Litterman UI inputs + pipeline runner (no maths here).

Wraps :mod:`analytics.allocation` for the Optimise / OOS / Walk-Forward
tabs: reference/delta/tau/view editors and posterior→allocation assembly.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from analytics.allocation import (
    AllocationResult,
    black_litterman_allocation,
    black_litterman_posterior,
)

def _bl_views_editor(prefix: str, tickers: list[str],
                     ref_default: str | None = None):
    """Render Black-Litterman reference/delta/tau/views inputs.

    Returns ``(config, error)`` where config is None when inputs are
    invalid. Config: ``{"ref", "delta", "tau", "views"}``. ``ref_default``
    optionally preselects the reference prior (walk-forward defaults to
    the Equal Weights proxy prior for genuine OOS).
    """
    _ref_options = ["Current Portfolio — proxy prior",
                    "Equal Weights — proxy prior"]
    _ref_index = (_ref_options.index(ref_default) if ref_default in _ref_options
                  else 0)
    ref_choice = st.selectbox(
        "Reference weights (prior)",
        _ref_options,
        index=_ref_index,
        key=f"{prefix}_ref",
        help="Used to reverse-engineer implied equilibrium returns.",
    )
    st.caption(
        "These weights are used to reverse-engineer implied equilibrium "
        "returns. They are not necessarily true market-cap equilibrium weights."
    )
    delta = st.slider(
        "Risk aversion (δ)", 0.5, 10.0, 2.5, 0.1, format="%.1f",
        key=f"{prefix}_delta",
        help="Higher δ means more risk-averse posterior portfolios.",
    )
    with st.expander("Advanced: prior uncertainty (τ)", expanded=False):
        tau = st.slider(
            "Tau (τ)", 0.01, 0.30, 0.05, 0.005, format="%.3f",
            key=f"{prefix}_tau",
            help="τ controls uncertainty in the prior expected returns.",
        )
        st.caption("τ controls uncertainty in the prior expected returns.")
    n_views_raw = st.number_input(
        "Number of views", min_value=0, max_value=4, value=0, step=1,
        key=f"{prefix}_nviews",
        help="Absolute and/or relative views, each with its own confidence.",
    )
    n_views = int(n_views_raw)
    views = []
    for i in range(n_views):
        with st.expander(f"View {i + 1}", expanded=(i == 0)):
            vtype = st.selectbox("Type", ["Absolute", "Relative"],
                                 key=f"{prefix}_vtype_{i}")
            conf_pct = st.slider(
                "Confidence (%)", 5, 95, 70, 5,
                key=f"{prefix}_conf_{i}",
                help="Higher confidence pulls the posterior harder toward "
                     "the view.")
            conf = float(conf_pct) / 100.0
            if vtype == "Absolute":
                tick = st.selectbox("Asset", tickers,
                                    key=f"{prefix}_tick_{i}")
                ret_pct = st.number_input(
                    "Expected annual return (%)", value=10.0, step=0.5,
                    key=f"{prefix}_ret_{i}")
                views.append({"type": "absolute", "ticker": tick,
                              "return": float(ret_pct) / 100.0,
                              "confidence": conf})
            else:
                if len(tickers) < 2:
                    return None, "Relative views need at least two holdings."
                long = st.selectbox("Outperform (long)", tickers,
                                    key=f"{prefix}_long_{i}")
                short = st.selectbox(
                    "Underperform (short)", tickers,
                    index=1 if len(tickers) > 1 else 0,
                    key=f"{prefix}_short_{i}")
                spread = st.number_input(
                    "Expected outperformance, annual (%)", value=3.0, step=0.5,
                    key=f"{prefix}_spread_{i}")
                views.append({"type": "relative", "long": long,
                              "short": short,
                              "return": float(spread) / 100.0,
                              "confidence": conf})
    return ({"ref": ref_choice, "delta": float(delta), "tau": float(tau),
             "views": views}, None)

def _bl_reference_weights(ref_choice: str, live_weights, tickers):
    """Resolve Current/Equal reference weights for Black-Litterman."""
    if ref_choice.startswith("Current"):
        return np.asarray(live_weights, dtype=float)
    return np.ones(len(tickers)) / len(tickers)

def _run_bl_pipeline(tickers, cov_df, ref_choice, live_weights, delta, tau,
                     views, rf_rate, min_w, max_w, constraints=None):
    """Run BL posterior + utility allocation. Returns (posterior, alloc)."""
    ref_w = _bl_reference_weights(ref_choice, live_weights, tickers)
    post = black_litterman_posterior(tickers, cov_df, ref_w, delta, tau, views)
    if not post.success or post.posterior is None:
        alloc = AllocationResult(
            method="Black-Litterman", success=False, weights=None,
            expected_return=float("nan"), volatility=float("nan"),
            sharpe=float("nan"), diversification_ratio=float("nan"),
            risk_contributions=None, message=post.message)
        return post, alloc
    alloc = black_litterman_allocation(
        tickers, post.posterior, cov_df, delta, rf_rate, min_w, max_w,
        constraints=constraints)
    return post, alloc

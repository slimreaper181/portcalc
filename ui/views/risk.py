"""views.risk — Risk tab (rendered by the app shell).

Extracted from the application shell; analytics live in
``analytics/``, fetching in ``data/``. Receives the shared
context dict and renders into the active tab.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from analytics.performance import historical_cvar
from analytics.var import (
    historical_var,
    mean_to_daily,
    monte_carlo_var,
    parametric_var,
    to_daily,
)
from ui.formatting import fmt_usd
from ui.theme import CHART_THEME


def render(ctx):
    """Render the Risk tab from the shared context."""
    cov = ctx["cov"]
    mu = ctx["mu"]
    port_daily_returns = ctx["port_daily_returns"]
    port_vol = ctx["port_vol"]
    portfolio_value = ctx["portfolio_value"]
    weights = ctx["weights"]

    st.markdown("## Risk Analysis")

    r_col1, r_col2 = st.columns([1, 3])

    with r_col1:
        var_horizon = st.number_input(
            "VaR Horizon (days)", min_value=1, max_value=30, value=1, step=1
        )
        st.caption("VaR confidence fixed at 95% (industry standard)")

    st.markdown("---")

    # Daily parameters (log-return space)
    port_std_daily  = to_daily(port_vol)
    mu_daily        = mean_to_daily(mu.values)
    cov_daily       = cov.values / 252

    try:
        var_param = parametric_var(port_std_daily, portfolio_value, 0.95, int(var_horizon))
    except ValueError as e:
        var_param = None
        st.error(f"Parametric VaR error: {e}")
    try:
        var_hist = historical_var(
            port_daily_returns, portfolio_value, 0.95, int(var_horizon)
        )
    except ValueError as e:
        var_hist = None
        hist_err = str(e)
    try:
        var_mc, mc_sim_rets = monte_carlo_var(
            weights, mu_daily, cov_daily, portfolio_value, 0.95, int(var_horizon), 10_000
        )
    except ValueError as e:
        var_mc, mc_sim_rets = None, None
        st.error(f"Monte Carlo VaR error: {e}")

    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown("#### Parametric VaR")
        st.metric("95% VaR", fmt_usd(var_param) if var_param is not None else "n/a")
        st.caption(
            "Assumes normally distributed returns. "
            f"At 95% confidence over {int(var_horizon)}d, potential loss ≤ "
            f"{fmt_usd(var_param) if var_param is not None else 'n/a'}."
        )
    with v2:
        st.markdown("#### Historical VaR")
        st.metric("95% VaR", fmt_usd(var_hist) if var_hist is not None else "n/a")
        if var_hist is None:
            st.caption(f"Historical VaR unavailable: {hist_err}")
        else:
            n_win = len(port_daily_returns) - int(var_horizon) + 1
            st.caption(
                "Uses genuine rolling "
                f"{int(var_horizon)}-day historical returns ({n_win:,} overlapping "
                "windows). No normality or √T-scaling assumption."
            )
    with v3:
        st.markdown("#### Monte Carlo VaR")
        st.metric("95% VaR", fmt_usd(var_mc) if var_mc is not None else "n/a")
        st.caption(
            "10,000 simulated return paths using multivariate normal. "
            "Log returns are converted to simple P&L via exp(r)−1. "
            "Captures correlation structure."
        )

    # Historical CVaR (Expected Shortfall) at the same horizon
    st.markdown("---")
    st.markdown("### Historical CVaR (Expected Shortfall)")
    try:
        cvar_out = historical_cvar(
            port_daily_returns, portfolio_value, 0.95, int(var_horizon)
        )
    except ValueError as e:
        cvar_out = None
        st.warning(f"Historical CVaR unavailable: {e}")

    if cvar_out is not None:
        cc1, cc2 = st.columns(2)
        with cc1:
            st.metric(
                f"95% Historical VaR ({int(var_horizon)}d)",
                fmt_usd(cvar_out["var_currency"]),
                help="Loss threshold exceeded in the worst 5% of historical periods.",
            )
            st.caption(f"Cutoff return: {cvar_out['var_pct']:.2%}")
        with cc2:
            st.metric(
                f"95% Historical CVaR ({int(var_horizon)}d)",
                fmt_usd(cvar_out["cvar_currency"]),
                help="Average loss once the VaR threshold has been exceeded.",
            )
            st.caption(
                f"Average tail loss: {cvar_out['cvar_pct']:.2%} "
                f"over {cvar_out['n_tail']:,} of {cvar_out['n_windows']:,} "
                "genuine rolling windows."
            )
        st.caption(
            "VaR estimates the loss threshold exceeded in the worst 5% of periods. "
            "CVaR estimates the average loss once that threshold has been exceeded."
        )

    st.markdown("---")

    # Return distribution
    st.markdown("### Daily Return Distribution")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=port_daily_returns,
        nbinsx=60,
        name="Daily Returns",
        marker_color="rgba(88, 166, 255, 0.6)",
        marker_line=dict(color="rgba(88,166,255,0.9)", width=0.5),
    ))
    var_pct_5 = np.percentile(port_daily_returns, 5)
    fig_dist.add_vline(x=var_pct_5, line=dict(color="#f85149", dash="dash", width=2))
    fig_dist.add_annotation(
        x=var_pct_5, text="5th pct",
        showarrow=True, arrowcolor="#f85149",
        font=dict(color="#f85149"), yref="paper", y=0.9,
    )
    fig_dist.update_layout(
        **CHART_THEME,
        title="Portfolio Daily Return Distribution",
        xaxis_title="Daily Return", yaxis_title="Count",
    )
    st.plotly_chart(fig_dist, width="stretch")

    # MC simulation return histogram
    if mc_sim_rets is not None:
        st.markdown("### Monte Carlo Simulated Returns")
        fig_mc = go.Figure(go.Histogram(
            x=mc_sim_rets,
            nbinsx=60,
            marker_color="rgba(240, 136, 62, 0.6)",
            marker_line=dict(color="rgba(240,136,62,0.9)", width=0.5),
        ))
        var_mc_pct = np.percentile(mc_sim_rets, 5)
        fig_mc.add_vline(x=var_mc_pct, line=dict(color="#f85149", dash="dash", width=2))
        fig_mc.add_annotation(
            x=var_mc_pct, text="VaR 5th pct",
            showarrow=True, arrowcolor="#f85149",
            font=dict(color="#f85149"), yref="paper", y=0.9,
        )
        fig_mc.update_layout(
            **CHART_THEME,
            title=f"MC Simulated {int(var_horizon)}-Day Portfolio Returns (simple returns)",
            xaxis_title="Portfolio Return", yaxis_title="Count",
        )
        st.plotly_chart(fig_mc, width="stretch")


# ============================================================

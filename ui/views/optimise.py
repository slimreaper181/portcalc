"""views.optimise — Optimise tab (rendered by the app shell).

Extracted from the application shell; analytics live in
``analytics/``, fetching in ``data/``. Receives the shared
context dict and renders into the active tab.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from analytics.allocation import (
    AllocationResult,
    allocation_summary,
    annualised_covariance_simple,
    annualised_mean_returns_simple,
    equal_risk_contribution,
    maximum_diversification,
    simple_returns_from_prices,
)
from analytics.currency import currency_symbol
from analytics.optimisation import (
    efficient_frontier,
    feasible_return_range,
    feasible_volatility_range,
    max_sharpe,
    min_variance,
    rebalance_trades,
    target_return,
    target_volatility,
)
from analytics.validation import validate_weight_bounds
from ui.bl import _bl_reference_weights, _bl_views_editor, _run_bl_pipeline
from ui.formatting import fmt_pct
from ui.theme import CHART_THEME


def _optim_to_alloc(method, res, tickers, mu_s, cov_df, rf):
    """Convert an OptimResult to AllocationResult form for comparison."""
    nan = float("nan")
    if res is None or not res.success or res.weights is None:
        reason = res.message if res is not None and res.message else "Unavailable"
        return AllocationResult(
            method=method, success=False, weights=None,
            expected_return=nan, volatility=nan, sharpe=nan,
            diversification_ratio=nan, risk_contributions=None,
            message=str(reason))
    try:
        summary = allocation_summary(res.weights, mu_s, cov_df, rf, tickers)
    except ValueError as e:
        return AllocationResult(
            method=method, success=False, weights=None,
            expected_return=nan, volatility=nan, sharpe=nan,
            diversification_ratio=nan, risk_contributions=None, message=str(e))
    return AllocationResult(
        method=method, success=True,
        weights=pd.Series(np.asarray(res.weights, dtype=float), index=tickers),
        expected_return=summary["expected_return"],
        volatility=summary["volatility"],
        sharpe=summary["sharpe"],
        diversification_ratio=summary["diversification_ratio"],
        risk_contributions=summary["risk_contributions"],
        message="Converged")


def _snapshot_alloc(method, w, tickers, mu_s, cov_df, rf, message):
    """Wrap fixed weights (Current / Equal) in AllocationResult form."""
    nan = float("nan")
    try:
        summary = allocation_summary(w, mu_s, cov_df, rf, tickers)
    except ValueError as e:
        return AllocationResult(
            method=method, success=False, weights=None,
            expected_return=nan, volatility=nan, sharpe=nan,
            diversification_ratio=nan, risk_contributions=None, message=str(e))
    return AllocationResult(
        method=method, success=True,
        weights=pd.Series(np.asarray(w, dtype=float), index=tickers),
        expected_return=summary["expected_return"],
        volatility=summary["volatility"],
        sharpe=summary["sharpe"],
        diversification_ratio=summary["diversification_ratio"],
        risk_contributions=summary["risk_contributions"],
        message=message)


def render(ctx):
    """Render the Optimise tab from the shared context."""
    base_ccy = ctx["base_ccy"]
    cov = ctx["cov"]
    mu = ctx["mu"]
    port_return = ctx["port_return"]
    port_vol = ctx["port_vol"]
    portfolio_value = ctx["portfolio_value"]
    prices = ctx["prices"]
    rf_rate = ctx["rf_rate"]
    tickers = ctx["tickers"]
    weights = ctx["weights"]

    st.markdown("## Portfolio Optimisation")

    mu_arr  = mu.values
    cov_arr = cov.values

    # Constraints
    st.markdown("### Constraints")
    oc1, oc2 = st.columns(2)
    with oc1:
        min_w = st.slider("Min weight per asset", 0.0, 0.20, 0.0, 0.01, format="%.2f")
    with oc2:
        max_w = st.slider("Max weight per asset", 0.20, 1.0, 1.0, 0.01, format="%.2f")

    try:
        validate_weight_bounds(len(tickers), float(min_w), float(max_w))
        bounds_ok = True
    except ValueError as e:
        bounds_ok = False
        st.error(f"Invalid weight bounds: {e}")

    st.markdown("---")

    if not bounds_ok:
        st.warning("Fix the weight bounds above to run optimisations.")
    else:
        # Run optimisations (failures render as errors in their result cards;
        # failed results carry no weights and are never shown as solutions).
        res_mv = min_variance(mu_arr, cov_arr, rf_rate, min_w, max_w)
        res_ms = max_sharpe(mu_arr, cov_arr, rf_rate, min_w, max_w)

        # Target return slider — bounded by the feasible return range.
        try:
            r_lo, r_hi = feasible_return_range(mu_arr, min_w, max_w)
        except ValueError as e:
            st.error(f"Cannot determine feasible returns: {e}")
            r_lo, r_hi = None, None

        if r_lo is None:
            res_tr = None
            target_r_val = None
        else:
            r_min_bound = float(np.clip(r_lo, -0.30, 0.50))
            r_max_bound = float(np.clip(r_hi, r_lo + 0.01, 1.0))
            if "target_r_val" not in st.session_state:
                st.session_state.target_r_val = float(
                    np.clip(port_return, r_min_bound, r_max_bound))
            # Clamp a stale session value into the current bounds.
            st.session_state.target_r_val = float(
                np.clip(st.session_state.target_r_val, r_min_bound, r_max_bound))
            # No `value=` here: the widget reads the clamped session value
            # (passing both logs a Streamlit session-state warning).
            target_r_val = st.slider(
                "Target annual return",
                min_value=r_min_bound, max_value=r_max_bound, step=0.005,
                format="%.1f%%",
                key="target_r_val",
            )
            res_tr = target_return(mu_arr, cov_arr, target_r_val, rf_rate, min_w, max_w)

        # Target volatility slider — bounded by feasible portfolio volatility
        # (from the covariance structure), NOT by return dispersion.
        try:
            v_lo, v_hi = feasible_volatility_range(mu_arr, cov_arr, min_w, max_w)
        except ValueError as e:
            st.error(f"Cannot determine feasible volatility range: {e}")
            v_lo, v_hi = None, None

        if v_lo is None:
            res_tv = None
            target_v_val = None
        else:
            v_min_bound = float(max(v_lo, 0.005))
            v_max_bound = float(max(v_hi, v_min_bound + 0.005))
            if "target_v_val" not in st.session_state:
                st.session_state.target_v_val = float(
                    np.clip(port_vol, v_min_bound, v_max_bound))
            st.session_state.target_v_val = float(
                np.clip(st.session_state.target_v_val, v_min_bound, v_max_bound))
            st.caption(
                f"Feasible volatility under current bounds: "
                f"{v_lo:.1%} – {v_hi:.1%}."
            )
            target_v_val = st.slider(
                "Target annual volatility",
                min_value=v_min_bound, max_value=v_max_bound, step=0.005,
                format="%.1f%%",
                key="target_v_val",
            )
            res_tv = target_volatility(mu_arr, cov_arr, target_v_val, rf_rate, min_w, max_w)

        st.markdown("---")
        st.markdown("### Optimisation Results")

        def _result_card(label: str, res, tickers: list[str]) -> None:
            if res is None or not res.success or res.weights is None:
                reason = ""
                if res is not None and res.message:
                    reason = f": {res.message}"
                st.error(f"❌ {label} — optimisation failed{reason}")
                st.caption(
                    "Your current portfolio is unchanged; no fallback weights "
                    "are shown as a solution."
                )
                return
            cols = st.columns(4)
            cols[0].metric(f"✅ {label} — Return", fmt_pct(res.expected_return))
            cols[1].metric("Volatility", fmt_pct(res.volatility))
            cols[2].metric("Sharpe", f"{res.sharpe:.2f}")
            w_df = pd.DataFrame({"Ticker": tickers, "Weight": res.weights}).set_index("Ticker")
            cols[3].dataframe(w_df.style.format({"Weight": "{:.1%}"}), height=160)

        _result_card("Min Variance", res_mv, tickers)
        st.markdown("---")
        _result_card("Max Sharpe", res_ms, tickers)
        st.markdown("---")
        _result_card(
            f"Target Return {target_r_val:.1%}" if target_r_val is not None else "Target Return",
            res_tr, tickers)
        st.markdown("---")
        _result_card(
            f"Target Vol {target_v_val:.1%}" if target_v_val is not None else "Target Vol",
            res_tv, tickers)

        st.markdown("---")

        # Efficient frontier
        st.markdown("### Efficient Frontier")
        with st.spinner("Tracing efficient frontier…"):
            ef_df = efficient_frontier(mu_arr, cov_arr, rf_rate, 60, min_w, max_w)

        if ef_df.empty:
            st.warning(
                "Efficient frontier could not be traced under the current "
                "constraints (no target-return point converged)."
            )
        else:
            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(
                x=ef_df["volatility"], y=ef_df["return"],
                mode="lines",
                name="Efficient Frontier",
                line=dict(color="#58a6ff", width=2),
            ))
            # Mark current portfolio
            fig_ef.add_trace(go.Scatter(
                x=[port_vol], y=[port_return],
                mode="markers", name="Current",
                marker=dict(color="#f85149", size=12, symbol="star"),
            ))
            # Mark max Sharpe
            if (res_ms.success and res_ms.weights is not None
                    and np.isfinite(res_ms.volatility) and res_ms.volatility > 0):
                fig_ef.add_trace(go.Scatter(
                    x=[res_ms.volatility], y=[res_ms.expected_return],
                    mode="markers", name="Max Sharpe",
                    marker=dict(color="#3fb950", size=12, symbol="diamond"),
                ))
                # CML
                cml_vol = np.linspace(0, ef_df["volatility"].max() * 1.2, 50)
                cml_ret = rf_rate + (res_ms.expected_return - rf_rate) / res_ms.volatility * cml_vol
                fig_ef.add_trace(go.Scatter(
                    x=cml_vol, y=cml_ret,
                    mode="lines", name="Capital Market Line",
                    line=dict(color="#f0883e", width=1.5, dash="dash"),
                ))
            fig_ef.update_layout(
                **CHART_THEME,
                title="Efficient Frontier with CML",
                xaxis_title="Volatility (σ)", yaxis_title="Expected Return",
                xaxis_tickformat=".0%", yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig_ef, width="stretch")

        # Rebalancing trades
        st.markdown("---")
        st.markdown("### Rebalancing Trades")

        target_choice = st.selectbox(
            "Rebalance to",
            ["Max Sharpe", "Min Variance", "Target Return", "Target Volatility"],
            key="rebalance_target",
        )
        target_map = {
            "Max Sharpe": res_ms,
            "Min Variance": res_mv,
            "Target Return": res_tr,
            "Target Volatility": res_tv,
        }
        chosen = target_map[target_choice]

        if (chosen is not None and chosen.success
                and chosen.weights is not None
                and np.all(np.isfinite(chosen.weights))):
            try:
                trades_df = rebalance_trades(weights, chosen.weights, tickers, portfolio_value)
                trades_df = trades_df.rename(
                    columns={"Trade ($)": f"Trade ({base_ccy})"})
            except ValueError as e:
                st.warning(f"Could not compute rebalancing trades: {e}")
            else:
                def _colour_action(val):
                    if val == "BUY":
                        return "color: #3fb950; font-weight: bold"
                    if val == "SELL":
                        return "color: #f85149; font-weight: bold"
                    return "color: #8b949e"

                st.dataframe(
                    trades_df.style
                    .format({
                        "Current Weight": "{:.1%}", "Target Weight": "{:.1%}",
                        "Δ Weight": "{:+.1%}",
                        f"Trade ({base_ccy})":
                            currency_symbol(base_ccy) + "{:+,.2f}",
                    })
                    .map(_colour_action, subset=["Action"]),
                    width="stretch",
                )
        else:
            detail = ""
            if chosen is not None and not chosen.success:
                detail = f" ({chosen.message})"
            st.warning(f"Optimisation did not converge for selected target.{detail}")

        st.markdown("---")
        st.markdown("### Institutional Allocation")
        st.caption(
            "Risk parity, maximum diversification and Black-Litterman on "
            "**simple-return** statistics (mean × 252, covariance × 252) so "
            "all methods are comparable. Minimum Variance / Maximum Sharpe "
            "are re-solved here on the same simple basis (their cards above "
            "use log-return inputs)."
        )
        try:
            simp_rets = simple_returns_from_prices(prices, tickers)
            mu_simp = annualised_mean_returns_simple(simp_rets)
            cov_simp = annualised_covariance_simple(simp_rets)
        except ValueError as e:
            st.error(f"Cannot build simple-return statistics: {e}")
            simp_ok = False
        else:
            simp_ok = True

        if simp_ok:
            inst_methods = st.multiselect(
                "Allocation methods",
                ["Current", "Equal Weight", "Min Variance", "Max Sharpe",
                 "Risk Parity", "Max Diversification", "Black-Litterman"],
                default=["Current", "Equal Weight", "Risk Parity",
                         "Max Diversification"],
                key="inst_methods",
                help="Methods to compare side by side.")
            bl_cfg, bl_err = None, None
            if "Black-Litterman" in inst_methods:
                with st.expander("Black-Litterman inputs", expanded=True):
                    bl_cfg, bl_err = _bl_views_editor("inst_bl", tickers)
                if bl_err:
                    st.error(f"Black-Litterman inputs: {bl_err}")

            inst_allocs: dict[str, AllocationResult] = {}
            if "Current" in inst_methods:
                inst_allocs["Current"] = _snapshot_alloc(
                    "Current", weights, tickers, mu_simp.values, cov_simp,
                    rf_rate, "Current portfolio mix.")
            if "Equal Weight" in inst_methods:
                inst_allocs["Equal Weight"] = _snapshot_alloc(
                    "Equal Weight", np.ones(len(tickers)) / len(tickers),
                    tickers, mu_simp.values, cov_simp, rf_rate,
                    "Equal weights (1/N).")
            if "Min Variance" in inst_methods:
                inst_allocs["Min Variance"] = _optim_to_alloc(
                    "Min Variance",
                    min_variance(mu_simp.values, cov_simp.values, rf_rate,
                                 min_w, max_w),
                    tickers, mu_simp.values, cov_simp, rf_rate)
            if "Max Sharpe" in inst_methods:
                inst_allocs["Max Sharpe"] = _optim_to_alloc(
                    "Max Sharpe",
                    max_sharpe(mu_simp.values, cov_simp.values, rf_rate,
                               min_w, max_w),
                    tickers, mu_simp.values, cov_simp, rf_rate)
            if "Risk Parity" in inst_methods:
                inst_allocs["Risk Parity"] = equal_risk_contribution(
                    tickers, mu_simp.values, cov_simp, rf_rate,
                    min_w, max_w)
            if "Max Diversification" in inst_methods:
                inst_allocs["Max Diversification"] = maximum_diversification(
                    tickers, mu_simp.values, cov_simp, rf_rate,
                    min_w, max_w)
            bl_post = None
            if "Black-Litterman" in inst_methods:
                if bl_err or bl_cfg is None:
                    inst_allocs["Black-Litterman"] = AllocationResult(
                        method="Black-Litterman", success=False, weights=None,
                        expected_return=float("nan"), volatility=float("nan"),
                        sharpe=float("nan"),
                        diversification_ratio=float("nan"),
                        risk_contributions=None,
                        message=bl_err or "Black-Litterman inputs unavailable.")
                else:
                    bl_post, bl_alloc = _run_bl_pipeline(
                        tickers, cov_simp, bl_cfg["ref"], weights,
                        bl_cfg["delta"], bl_cfg["tau"], bl_cfg["views"],
                        rf_rate, min_w, max_w)
                    inst_allocs["Black-Litterman"] = bl_alloc

            for _m, _a in inst_allocs.items():
                if not _a.success:
                    st.error(f"❌ {_m} — {_a.message}")
            ok_allocs = {m: a for m, a in inst_allocs.items() if a.success}
            if ok_allocs:
                st.markdown("#### Weight Comparison")
                wtab = pd.DataFrame(
                    {m: a.weights for m, a in ok_allocs.items()})
                st.dataframe(wtab.style.format("{:.1%}"),
                             width="stretch")

                st.markdown("#### Allocation Metrics")
                mrows = ["Expected Return", "Volatility", "Sharpe",
                         "Diversification Ratio", "Largest Position",
                         "Effective Holdings", "RC Dispersion"]
                mtab = pd.DataFrame(index=mrows)
                for m, a in ok_allocs.items():
                    s = allocation_summary(a.weights.values, mu_simp.values,
                                           cov_simp, rf_rate, tickers)
                    mtab[m] = [
                        f"{s['expected_return']:.2%}",
                        f"{s['volatility']:.2%}",
                        f"{s['sharpe']:.2f}",
                        f"{s['diversification_ratio']:.2f}",
                        f"{s['largest_position']:.1%}",
                        f"{s['effective_holdings']:.2f}",
                        f"{s['rc_dispersion']:.2%}",
                    ]
                st.dataframe(mtab, width="stretch")
                st.caption(
                    "RC Dispersion is the std of percentage risk contributions "
                    "(near zero = evenly spread risk, the risk-parity ideal).")

                st.markdown("#### Weight Comparison Chart")
                fig_w = go.Figure()
                palette = px.colors.qualitative.Plotly
                for i, (m, a) in enumerate(ok_allocs.items()):
                    fig_w.add_trace(go.Bar(
                        x=tickers, y=a.weights.values, name=m,
                        marker_color=palette[i % len(palette)]))
                fig_w.update_layout(
                    **CHART_THEME, barmode="group",
                    title="Target Weights by Method",
                    xaxis_title="Ticker", yaxis_title="Weight",
                    yaxis_tickformat=".0%",
                )
                st.plotly_chart(fig_w, width="stretch")

                rc_methods = [m for m in (
                    "Current", "Risk Parity", "Max Diversification",
                    "Black-Litterman") if m in ok_allocs]
                if rc_methods:
                    st.markdown("#### Risk Contribution Comparison")
                    fig_rc = go.Figure()
                    for i, m in enumerate(rc_methods):
                        fig_rc.add_trace(go.Bar(
                            x=tickers,
                            y=ok_allocs[m].risk_contributions.values, name=m,
                            marker_color=palette[i % len(palette)]))
                    fig_rc.update_layout(
                        **CHART_THEME, barmode="group",
                        title="Percentage Risk Contribution by Method",
                        xaxis_title="Ticker", yaxis_title="Risk Contribution",
                        yaxis_tickformat=".0%",
                    )
                    st.plotly_chart(fig_rc, width="stretch")
                    st.caption(
                        "Risk parity should show approximately equal bars "
                        "when successful.")

            if ("Black-Litterman" in ok_allocs and bl_post is not None
                    and bl_post.success):
                st.markdown("#### Black-Litterman Prior vs Posterior")
                post_df = pd.DataFrame({
                    "Ticker": tickers,
                    "Historical Expected Return":
                        [f"{v:.2%}" for v in mu_simp.values],
                    "BL Prior Return":
                        [f"{v:.2%}" for v in bl_post.prior.values],
                    "BL Posterior Return":
                        [f"{v:.2%}" for v in bl_post.posterior.values],
                    "Change": [f"{(b - a):+.2%}" for a, b in zip(
                        bl_post.prior.values, bl_post.posterior.values)],
                }).set_index("Ticker")
                st.dataframe(post_df, width="stretch")
                _bl_ref_w = _bl_reference_weights(
                    bl_cfg["ref"], weights, tickers)
                _prior_ret = float(np.dot(_bl_ref_w, bl_post.prior.values))
                _post_ret = float(np.dot(
                    ok_allocs["Black-Litterman"].weights.values,
                    bl_post.posterior.values))
                d1, d2, d3 = st.columns(3)
                d1.metric("Risk Aversion (δ)", f"{bl_cfg['delta']:.2f}")
                d2.metric("Tau (τ)", f"{bl_cfg['tau']:.3f}")
                d3.metric("Views", f"{bl_post.n_views}")
                e1, e2, e3 = st.columns(3)
                e1.metric("Prior Expected Return", f"{_prior_ret:.2%}",
                          help="Reference weights × prior returns.")
                e2.metric("Posterior Expected Return", f"{_post_ret:.2%}",
                          help="BL weights × posterior returns.")
                e3.metric("BL Sharpe",
                          f"{ok_allocs['Black-Litterman'].sharpe:.2f}")


# ============================================================

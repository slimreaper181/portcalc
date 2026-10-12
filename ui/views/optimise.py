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
from analytics.constraints import (
    OptimizationConstraints,
    check_constraints,
    max_single_name_shift,
    turnover_fraction,
)
from analytics.estimation import (
    ESTIMATOR_LABELS,
    ESTIMATORS,
    correlation_stability,
    covariance_diagnostics,
    estimate_covariance,
    estimate_mean_returns,
    estimation_window_slice,
    ledoit_wolf_shrinkage,
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
from ui.formatting import fmt_pct, fmt_usd
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
    ccy = ctx["ccy"]
    cov = ctx["cov"]
    mu = ctx["mu"]
    port_return = ctx["port_return"]
    port_vol = ctx["port_vol"]
    portfolio_value = ctx["portfolio_value"]
    prices = ctx["prices"]
    rf_rate = ctx["rf_rate"]
    tickers = ctx["tickers"]
    weights = ctx["weights"]
    live_snapshot = ctx.get("live_snapshot", {})
    ledger_cash = ctx.get("ledger_cash", {})
    instruments = ctx.get("instruments", {})
    current_prices = ctx.get("current_prices", {})

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

    opt_mode = st.selectbox(
        "Optimisation view",
        ["Classical", "Risk-Based", "Black-Litterman", "Compare", "Plan"],
        index=0, key="opt_mode",
        help="Classical solves mean-variance portfolios. Risk-Based runs "
             "risk parity and maximum diversification only. "
             "Black-Litterman runs the views-based allocation only. "
             "Compare evaluates methods side by side. Plan turns a "
             "target into cash-aware trades (research-only).")

    if opt_mode == "Classical":
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

    if opt_mode not in ("Classical", "Plan"):
            st.markdown("---")
            st.markdown("### Institutional Allocation")
            st.caption(
                "Risk parity, maximum diversification and Black-Litterman on "
                "**simple-return** statistics (mean × 252, covariance × 252) so "
                "all methods are comparable. Minimum Variance / Maximum Sharpe "
                "are re-solved here on the same simple basis (their cards above "
                "use log-return inputs)."
            )
            r1, r2, r3 = st.columns(3)
            with r1:
                risk_model = st.selectbox(
                    "Risk model",
                    list(ESTIMATORS), index=0, key="inst_risk_model",
                    format_func=lambda m: ESTIMATOR_LABELS.get(m, m),
                    help="Sample preserves all historical results. EWMA reacts "
                         "faster to recent volatility. Ledoit-Wolf shrinks "
                         "toward the scaled identity (stable in small samples).")
            with r2:
                ewma_lam = st.slider(
                    "EWMA decay λ", 0.80, 0.99, 0.94, 0.01, key="inst_ewma",
                    help="Higher λ = longer memory. Applies to EWMA only.",
                    disabled=(risk_model != "ewma"))
            with r3:
                est_window = st.selectbox(
                    "Estimation window",
                    ["expanding", "1y", "2y", "3y", "5y"], index=0,
                    key="inst_window",
                    help="Trailing window of Base Currency history. Strictly "
                         "past data only.")
            try:
                _px = estimation_window_slice(
                    prices, prices.index[-1] + pd.Timedelta(days=1),
                    est_window)
                simp_rets = simple_returns_from_prices(_px, tickers)
                mu_simp = estimate_mean_returns(simp_rets)
                cov_simp = estimate_covariance(
                    simp_rets, method=risk_model, ewma_lambda=float(ewma_lam))
            except ValueError as e:
                st.error(f"Cannot build simple-return statistics: {e}")
                simp_ok = False
            else:
                simp_ok = True
                _diag = covariance_diagnostics(cov_simp.values)
                _dcols = st.columns(4)
                _cond_txt = ("∞" if _diag["cond"] == float("inf")
                             else ("n/a" if not np.isfinite(_diag["cond"])
                                   else f"{_diag['cond']:.1f}"))
                _dcols[0].metric("Condition №", _cond_txt,
                                 help="Largest/smallest eigenvalue. High = "
                                      "unstable minimum-variance weights.")
                _dcols[1].metric("Min eigenvalue", f"{_diag['eig_min']:.2e}")
                _dcols[2].metric("Max eigenvalue", f"{_diag['eig_max']:.2e}")
                _dcols[3].metric("Effective rank",
                                 f"{_diag['effective_rank']:.2f}",
                                 help="Participation ratio: 1 = concentrated, "
                                      "N = perfectly spread risk.")
                _psd_note = ("PSD check passed."
                             if _diag["psd_ok"] else
                             "NOT positive semidefinite — allocations below "
                             "may be unreliable.")
                if risk_model == "ledoit_wolf":
                    try:
                        _sh = ledoit_wolf_shrinkage(simp_rets)
                        st.caption(f"Risk model: Ledoit-Wolf "
                                   f"(shrinkage δ={_sh:.2f}) · "
                                   f"window {est_window} · {_psd_note}")
                    except ValueError:
                        st.caption(f"Risk model: Ledoit-Wolf · "
                                   f"window {est_window} · {_psd_note}")
                else:
                    st.caption(f"Risk model: {ESTIMATOR_LABELS[risk_model]} · "
                               f"window {est_window} · {_psd_note}")

            if simp_ok:
                if opt_mode == "Risk-Based":
                    inst_methods = ["Risk Parity", "Max Diversification"]
                elif opt_mode == "Black-Litterman":
                    inst_methods = ["Black-Litterman"]
                else:
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

                # ---- Real-world constraints (Milestone 5) ----
                _ccy_of = {}
                for _t in tickers:
                    _inst = (instruments or {}).get(_t)
                    _nc = getattr(_inst, "native_currency", None)
                    if _nc:
                        _ccy_of[_t] = str(_nc).upper()
                _ccys = sorted(set(_ccy_of.values()))
                with st.expander("Real-world constraints", expanded=False):
                    st.caption(
                        "Practical portfolio design: per-asset bounds, "
                        "turnover limit (one-way Σ|Δ|/2), cost-averse planning "
                        "penalty, quote-currency caps, cash target and custom "
                        "ERC risk budgets. Off = classic theoretical solves.")
                    use_c = st.checkbox(
                        "Apply constraints to institutional solves",
                        value=False, key="inst_use_constraints")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        to_limit = st.slider(
                            "Turnover limit (one-way)", 0.0, 1.0, 0.25, 0.01,
                            format="%.0f%%", key="inst_turnover",
                            help="Relative to current weights.",
                            disabled=not use_c)
                    with c2:
                        cost_bps = st.number_input(
                            "Cost-aversion (bps)", min_value=0.0,
                            max_value=500.0, value=0.0, step=5.0,
                            key="inst_cost",
                            help="Planning penalty per unit turnover — not "
                                 "the exact backtest accounting.",
                            disabled=not use_c)
                    with c3:
                        cash_t = st.slider(
                            "Cash target", 0.0, 0.5, 0.0, 0.01, format="%.0f%%",
                            key="inst_cash",
                            help="Risky weights + cash = 100%.",
                            disabled=not use_c)
                    _bdf = pd.DataFrame({
                        "Ticker": tickers,
                        "Min": [float(min_w)] * len(tickers),
                        "Max": [float(max_w)] * len(tickers)})
                    bounds_ed = st.data_editor(
                        _bdf, key="inst_bounds", width="stretch",
                        hide_index=True, disabled=not use_c,
                        help="Per-asset weight bounds (validated globally).")
                    ccy_caps: dict = {}
                    if _ccys and use_c:
                        st.markdown("**Quote-currency caps** (Base-valued "
                                    "risky weights)")
                        _ccols = st.columns(max(len(_ccys), 1))
                        for _i, _c in enumerate(_ccys):
                            with _ccols[_i % max(len(_ccols), 1)]:
                                _cap = st.slider(
                                    f"{_c} ≤", 0.0, 1.0, 1.0, 0.05,
                                    format="%.0f%%", key=f"inst_ccy_{_c}")
                                if _cap < 1.0 - 1e-12:
                                    ccy_caps[_c] = float(_cap)
                    budgets = None
                    if "Risk Parity" in inst_methods:
                        _eq = 100.0 / len(tickers)
                        _rdf = pd.DataFrame({
                            "Ticker": tickers,
                            "Budget %": [_eq] * len(tickers)})
                        rb_ed = st.data_editor(
                            _rdf, key="inst_budgets", width="stretch",
                            hide_index=True, disabled=not use_c,
                            help="Custom share of total risk per asset "
                                 "(must sum to 100%). Equal = classic ERC.")
                        try:
                            _bv = pd.to_numeric(
                                rb_ed["Budget %"],
                                errors="coerce").values.astype(float) / 100.0
                            if use_c and np.all(np.isfinite(_bv)) and not np.allclose(
                                    _bv, 1.0 / len(tickers)):
                                budgets = dict(zip(tickers, _bv))
                        except (KeyError, ValueError, TypeError):
                            budgets = None
                _inst_c: OptimizationConstraints | None = None
                if use_c:
                    try:
                        _pmin = dict(zip(
                            bounds_ed["Ticker"].astype(str),
                            pd.to_numeric(bounds_ed["Min"],
                                          errors="coerce").astype(float)))
                        _pmax = dict(zip(
                            bounds_ed["Ticker"].astype(str),
                            pd.to_numeric(bounds_ed["Max"],
                                          errors="coerce").astype(float)))
                    except (KeyError, ValueError, TypeError):
                        _pmin, _pmax = {}, {}
                    _inst_c = OptimizationConstraints(
                        per_asset_min=_pmin, per_asset_max=_pmax,
                        turnover_limit=float(to_limit),
                        turnover_reference=np.asarray(weights, dtype=float),
                        transaction_cost_bps=float(cost_bps),
                        cash_target=float(cash_t),
                        asset_currencies=[_ccy_of.get(t) for t in tickers],
                        currency_caps=ccy_caps or None,
                        risk_budgets=budgets)
                    _cprobs = check_constraints(tickers, _inst_c, min_w, max_w)
                    for _p in _cprobs:
                        st.error(f"Constraint conflict — {_p}")
                    if _cprobs:
                        st.caption("Solves below still attempt honestly; "
                                   "infeasible methods fail with messages, "
                                   "never fallback weights.")

                def _solve_inst(_m: str, _c):
                    if _m == "Min Variance":
                        return _optim_to_alloc(
                            "Min Variance",
                            min_variance(mu_simp.values, cov_simp.values,
                                         rf_rate, min_w, max_w,
                                         constraints=_c, tickers=tickers),
                            tickers, mu_simp.values, cov_simp, rf_rate)
                    if _m == "Max Sharpe":
                        return _optim_to_alloc(
                            "Max Sharpe",
                            max_sharpe(mu_simp.values, cov_simp.values,
                                       rf_rate, min_w, max_w,
                                       constraints=_c, tickers=tickers),
                            tickers, mu_simp.values, cov_simp, rf_rate)
                    if _m == "Risk Parity":
                        return equal_risk_contribution(
                            tickers, mu_simp.values, cov_simp, rf_rate,
                            min_w, max_w, constraints=_c)
                    if _m == "Max Diversification":
                        return maximum_diversification(
                            tickers, mu_simp.values, cov_simp, rf_rate,
                            min_w, max_w, constraints=_c)
                    return None

                inst_allocs: dict[str, AllocationResult] = {}
                inst_plain: dict[str, AllocationResult] = {}
                if "Current" in inst_methods:
                    inst_allocs["Current"] = _snapshot_alloc(
                        "Current", weights, tickers, mu_simp.values, cov_simp,
                        rf_rate, "Current portfolio mix.")
                if "Equal Weight" in inst_methods:
                    inst_allocs["Equal Weight"] = _snapshot_alloc(
                        "Equal Weight", np.ones(len(tickers)) / len(tickers),
                        tickers, mu_simp.values, cov_simp, rf_rate,
                        "Equal weights (1/N).")
                for _m in ("Min Variance", "Max Sharpe", "Risk Parity",
                           "Max Diversification"):
                    if _m in inst_methods:
                        inst_allocs[_m] = _solve_inst(_m, _inst_c)
                        if use_c:
                            inst_plain[_m] = _solve_inst(_m, None)
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
                            rf_rate, min_w, max_w, constraints=_inst_c)
                        inst_allocs["Black-Litterman"] = bl_alloc
                if use_c and inst_plain:
                    st.markdown("#### Before / After Constraints")
                    _w0 = np.asarray(weights, dtype=float)
                    _brows = []
                    for _m, _pa in inst_plain.items():
                        _ca = inst_allocs.get(_m)
                        if _pa is None or not _pa.success \
                                or _ca is None or not _ca.success:
                            _brows.append({
                                "Method": _m,
                                "Unconstrained Vol": (
                                    f"{_pa.volatility:.2%}"
                                    if _pa is not None and _pa.success
                                    else "failed"),
                                "Constrained Vol": (
                                    f"{_ca.volatility:.2%}"
                                    if _ca is not None and _ca.success
                                    else "failed"),
                                "Turnover vs Current": "n/a",
                                "Max Shift": "n/a"})
                            continue
                        _cw = _ca.weights.values.astype(float)
                        _brows.append({
                            "Method": _m,
                            "Unconstrained Vol": f"{_pa.volatility:.2%}",
                            "Constrained Vol": f"{_ca.volatility:.2%}",
                            "Turnover vs Current":
                                f"{turnover_fraction(_cw, _w0):.2%}",
                            "Max Shift":
                                f"{max_single_name_shift(_cw, _w0):.2%}"})
                    st.dataframe(pd.DataFrame(_brows).set_index("Method"),
                                 width="stretch")

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

                    with st.expander("Estimator comparison (same portfolio)",
                                     expanded=False):
                        st.caption(
                            "Same Base Currency returns, three risk models: "
                            "portfolio volatility plus MinVar / ERC / MaxDiv "
                            "weights re-solved per estimator. Means are "
                            "identical (sample mean throughout).")
                        try:
                            _cmp_rows: dict = {}
                            _cmp_w: dict = {}
                            for _em in ESTIMATORS:
                                _cc = estimate_covariance(
                                    simp_rets, method=_em,
                                    ewma_lambda=float(ewma_lam))
                                _pv = float(np.sqrt(max(
                                    float(weights @ _cc.values @ weights),
                                    0.0)))
                                _mv = min_variance(
                                    mu_simp.values, _cc.values, rf_rate,
                                    min_w, max_w)
                                _er = equal_risk_contribution(
                                    tickers, mu_simp.values, _cc, rf_rate,
                                    min_w, max_w)
                                _md = maximum_diversification(
                                    tickers, mu_simp.values, _cc, rf_rate,
                                    min_w, max_w)
                                _cmp_rows[ESTIMATOR_LABELS[_em]] = {
                                    "Portfolio Vol":
                                        f"{_pv:.2%}",
                                    "MinVar Vol":
                                        (f"{_mv.volatility:.2%}"
                                         if _mv.success else "failed"),
                                    "ERC Vol":
                                        (f"{_er.volatility:.2%}"
                                         if _er.success else "failed"),
                                    "MaxDiv Vol":
                                        (f"{_md.volatility:.2%}"
                                         if _md.success else "failed"),
                                }
                                if _mv.success and _mv.weights is not None:
                                    _cmp_w[ESTIMATOR_LABELS[_em]] = (
                                        pd.Series(
                                            np.asarray(_mv.weights,
                                                       dtype=float),
                                            index=tickers))
                            st.dataframe(pd.DataFrame(_cmp_rows).T,
                                         width="stretch")
                            if _cmp_w:
                                st.caption("MinVar weights by estimator")
                                st.dataframe(
                                    pd.DataFrame(_cmp_w).T.style.format(
                                        "{:.1%}"),
                                    width="stretch")
                        except ValueError as e:
                            st.warning(f"Estimator comparison unavailable: {e}")

                        try:
                            if len(simp_rets) >= 520:
                                _stab = correlation_stability(
                                    simp_rets.iloc[-252:], simp_rets)
                                st.caption(
                                    "Correlation stability — trailing 1y vs "
                                    "full window: Frobenius "
                                    f"{_stab['frob']:.3f} · max |Δ| "
                                    f"{_stab['max_abs']:.3f} · mean |Δ| "
                                    f"{_stab['mean_abs']:.3f}. Descriptive "
                                    "only — not a recommendation.")
                        except ValueError:
                            pass

                    if opt_mode == "Compare":
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

                if (opt_mode in ("Black-Litterman", "Compare")
                        and "Black-Litterman" in ok_allocs and bl_post is not None
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

    # ---- Rebalance Plan (research-only: targets → trades, no execution) ----
    if opt_mode == "Plan":
        from analytics.rebalance_plan import (
            plan_diagnostics as _plan_diag,
            plan_trades as _plan_tr,
            resolve_planner_targets as _plan_tgt)
        st.markdown("### Rebalance Plan")
        st.caption("Research-only: proposes trades toward a target. Nothing "
                   "is executed; no orders leave this app.")
        p1, p2 = st.columns(2)
        with p1:
            _pm = st.selectbox(
                "Target",
                ["Equal Weight", "Minimum Variance", "Maximum Sharpe",
                 "ERC", "Maximum Diversification", "Custom"],
                key="plan_method",
                help="Optimiser targets reuse the tested engines on current "
                     "simple-return statistics. For BL, enter its weights "
                     "as Custom.")
        with p2:
            _fx_now = ccy.get("fx_now", {}) or {}
            _cash_base = 0.0
            for _ccy, _bal in (ctx.get("ledger_cash", {}) or {}).items():
                _r = _fx_now.get(_ccy)
                if _r is not None and np.isfinite(float(_r)) \
                        and float(_r) > 0:
                    _cash_base += float(_bal) * float(_r)
            _pcash = st.number_input(
                "Cash available (base)", min_value=0.0,
                value=float(_cash_base),
                step=100.0, key="plan_cash",
                help="Ledger cash converted at current FX. Deployable "
                     "after the buffer below.")
        _cw = np.asarray(weights, dtype=float)
        try:
            _srets = simple_returns_from_prices(prices, tickers)
            _smu = annualised_mean_returns_simple(_srets)
            _scov = annualised_covariance_simple(_srets)
        except ValueError as e:
            st.error(f"Planning statistics unavailable: {e}")
            _smu = _scov = None
        if _smu is not None:
            _key = {"Equal Weight": "equal_weight",
                    "Minimum Variance": "min_variance",
                    "Maximum Sharpe": "max_sharpe", "ERC": "erc",
                    "Maximum Diversification": "max_diversification",
                    "Custom": "custom"}[_pm]
            _custom = None
            if _key == "custom":
                st.caption("Custom target weights (must sum to 1).")
                _ccols = st.columns(max(1, len(tickers)))
                _custom = []
                for _ci, _t in zip(_ccols, tickers):
                    with _ci:
                        _custom.append(st.number_input(
                            _t, min_value=0.0, max_value=1.0, value=round(
                                1.0 / len(tickers), 4), step=0.01,
                            key=f"plan_w_{_t}"))
                _custom = np.asarray(_custom, dtype=float)
                if abs(float(_custom.sum()) - 1.0) > 1e-6:
                    st.warning("Custom weights must sum to 1 to plan.")
                    _custom = None
            if _key != "custom" or _custom is not None:
                try:
                    _tname, _tw = _plan_tgt(
                        _key, tickers, _smu.values, _scov, rf_rate,
                        float(min_w), float(max_w),
                        custom=_custom)
                except ValueError as e:
                    st.error(f"Target estimation failed: {e}")
                    _tw = None
                if _tw is not None:
                    q1, q2, q3 = st.columns(3)
                    with q1:
                        _buf = st.number_input(
                            "Min cash buffer (base)", min_value=0.0,
                            value=0.0, step=100.0, key="plan_buffer")
                    with q2:
                        _minv = st.number_input(
                            "Min trade value (base)", min_value=0.0,
                            value=0.0, step=10.0, key="plan_minv")
                    with q3:
                        _cbps = st.number_input(
                            "Cost (bps)", min_value=0.0, max_value=1000.0,
                            value=10.0, step=1.0, key="plan_costs")
                    _share_px = {}
                    for _t in tickers:
                        _mk = (live_snapshot.get(_t, {}) or {}).get("mark")
                        _sc = 1.0
                        try:
                            _sc = float(instruments[_t].quote_scale or 1.0)
                        except (KeyError, AttributeError, TypeError):
                            pass
                        if _mk is not None and np.isfinite(float(_mk)):
                            _share_px[_t] = float(_mk) * _sc
                        else:
                            _share_px[_t] = float(
                                current_prices.get(_t, float("nan")))
                    try:
                        _plan = _plan_tr(
                            _cw, _tw.values, tickers, float(portfolio_value),
                            _share_px, cash_available=float(_pcash),
                            min_cash_buffer=float(_buf),
                            min_trade_value=float(_minv), cost_bps=float(_cbps))
                    except ValueError as e:
                        st.error(f"Planning failed: {e}")
                        _plan = None
                    if _plan is not None:
                        st.markdown("#### Proposed Trades")
                        st.dataframe(_plan.trades, width="stretch",
                                     hide_index=True)
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Turnover",
                                  f"{_plan.gross_turnover:.2%}")
                        m2.metric("Est. Cost",
                                  fmt_usd(_plan.estimated_cost))
                        m3.metric("Cash Residual",
                                  fmt_usd(_plan.cash_remaining))
                        m4.metric("Shortfall",
                                  fmt_usd(_plan.cash_shortfall)
                                  if _plan.cash_shortfall > 0 else "—")
                        if _plan.cash_shortfall > 0:
                            st.warning(
                                "Plan needs more cash than available — "
                                "raise cash, lower the buffer, or shrink "
                                "targets. Nothing was executed.")
                        if _plan.skipped_small:
                            st.caption(
                                "Held (below threshold): "
                                + ", ".join(_plan.skipped_small) + ".")
                        try:
                            _dg = _plan_diag(
                                _cw, _tw.values, tickers, _smu.values,
                                _scov, rf_rate)
                            st.markdown("#### Before / After")
                            _drows = []
                            for _lbl, _side in (("Current", _dg["current"]),
                                                ("Target", _dg["target"])):
                                _drows.append({
                                    "Leg": _lbl,
                                    "Exp Return":
                                        f"{_side['expected_return']:.2%}",
                                    "Vol":
                                        f"{_side['volatility']:.2%}",
                                    "Sharpe": f"{_side['sharpe']:.2f}",
                                    "Largest":
                                        f"{_side['largest_position']:.1%}",
                                    "Eff N":
                                        f"{_side['effective_holdings']:.2f}",
                                })
                            st.dataframe(pd.DataFrame(_drows),
                                         width="stretch", hide_index=True)
                        except ValueError as e:
                            st.caption(f"Before/after unavailable: {e}")
                        st.caption(
                            "Order intents are descriptive research objects "
                            f"({len(_plan.intents)} legs). No execution, no "
                            "broker calls.")


    # ============================================================

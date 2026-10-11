"""views.backtest — Backtest tab (rendered by the app shell).

Extracted from the application shell; analytics live in
``analytics/``, fetching in ``data/``. Receives the shared
context dict and renders into the active tab.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.allocation import (
    annualised_covariance_simple,
    annualised_mean_returns_simple,
    equal_risk_contribution,
    estimate_risk_aversion_from_benchmark,
    estimation_window,
    maximum_diversification,
    simple_returns_from_prices,
)
from analytics.backtest import (
    backtest_benchmark,
    backtest_buy_and_hold,
    backtest_rebalanced,
    compare_backtests,
    plot_backtest_growth,
    plot_weight_drift,
    prepare_backtest_data,
)
from analytics.currency import currency_symbol
from analytics.optimisation import max_sharpe, min_variance
from analytics.performance import (
    plot_drawdown,
    rolling_annualised_return,
    rolling_sharpe,
    rolling_volatility,
)
from analytics.validation import validate_ticker_symbol
from analytics.walkforward import (
    BL_HINDSIGHT_WARNING,
    HINDSIGHT_WARNING,
    RF_CONSTANT_NOTE,
    SURVIVORSHIP_WARNING,
    WALK_FORWARD_LABELS,
    WalkForwardConfig,
    ex_ante_realised_frame,
    run_walk_forward,
    turnover_diagnostics,
)
from data.pipeline import (
    base_benchmark_series,
    base_price_panel,
    load_backtest_data,
    load_benchmark_data,
)
from ui.bl import _bl_views_editor, _run_bl_pipeline
from ui.formatting import fmt_pct, fmt_usd
from ui.tables import render_backtest_comparison as _render_backtest_comparison
from ui.theme import CHART_THEME


def _fmt_bt_date(d) -> str:
    """Format a drawdown date for display (NaT → 'not yet recovered')."""
    try:
        ts = pd.Timestamp(d)
        if pd.isna(ts):
            return "not yet recovered"
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return "n/a"


def render(ctx):
    """Render the Backtest tab from the shared context."""
    base_ccy = ctx["base_ccy"]
    ccy = ctx["ccy"]
    instruments = ctx["instruments"]
    rf_rate = ctx["rf_rate"]
    tickers = ctx["tickers"]
    weights = ctx["weights"]

    st.markdown("## Historical Backtest")
    st.caption(
        "Reconstructs this portfolio from historical **adjusted** Yahoo Finance "
        "prices only — no live/Alpaca prices, no look-ahead. Adjusted closes "
        "reflect splits/dividends only to the extent Yahoo's adjusted data "
        "provides them. Fractional shares assumed; target weights are static. "
        "Scheduled rebalances are modelled at the closing price of the first "
        "eligible trading day on/after each calendar date (end-of-day "
        "execution approximation)."
    )

    # ---- Long-history data through the same cached yfinance pipeline ----
    try:
        bt_full = load_backtest_data(tickers)
    except ValueError as e:
        st.error(f"Backtest data error: {e}")
        return  # section-local return (was st.stop()): sibling tabs keep rendering
    except Exception:
        st.error("Could not load backtest history (network or API error). "
                 "Check your connection and try again.")
        return  # section-local return (was st.stop()): sibling tabs keep rendering

    bt_common = bt_full.dropna(how="any")
    if len(bt_common) < 2:
        st.error("No overlapping history available for backtesting.")
        return  # section-local return (was st.stop()): sibling tabs keep rendering
    bt_min_date = bt_common.index[0].date()
    bt_max_date = bt_common.index[-1].date()

    # ---- Backtest Controls ----
    st.markdown("### Backtest Controls")
    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        bt_start = st.date_input(
            "Start Date", value=bt_min_date,
            min_value=bt_min_date, max_value=bt_max_date, key="bt_start")
    with bc2:
        bt_end = st.date_input(
            "End Date", value=bt_max_date,
            min_value=bt_min_date, max_value=bt_max_date, key="bt_end")
    with bc3:
        bt_capital = st.number_input(
            f"Initial Capital ({currency_symbol(base_ccy).strip() or base_ccy})",
            min_value=100.0, max_value=100_000_000.0,
            value=10_000.0, step=1_000.0, key="bt_capital")
    with bc4:
        bt_costs = st.number_input(
            "Transaction Cost (bps)", min_value=0.0, max_value=1000.0,
            value=10.0, step=1.0, key="bt_costs",
            help="10 bps = 0.10% of traded notional at each rebalance.")
    bc5, bc6 = st.columns(2)
    with bc5:
        if "bt_benchmark" not in st.session_state:
            st.session_state.bt_benchmark = "SPY"
        st.text_input(
            "Benchmark", key="bt_benchmark",
            help="Any Yahoo Finance ticker, e.g. SPY, QQQ, ^GSPC.")
    with bc6:
        freq_label = st.selectbox(
            "Rebalancing Frequency",
            ["Monthly", "Quarterly", "Semi-annual", "Annual"],
            index=1, key="bt_freq",
            help="Rebalance on the first trading day on or after each "
                 "calendar period start, at that day's close "
                 "(end-of-day execution approximation).")

    st.markdown("#### Strategies")
    _retro_help = (
        "Today's portfolio weights are frozen as the fixed historical "
        "target. This is a retrospective hypothetical — not proof those "
        "weights would have been selected at the historical start date."
    )
    bs1, bs2, bs3 = st.columns(3)
    with bs1:
        show_cwbh = st.checkbox("Current Weights — Retrospective (Buy & Hold)",
                                value=False, key="bt_s_cwbh", help=_retro_help)
        show_cwr = st.checkbox("Current Weights — Retrospective (Rebalanced)",
                               value=True, key="bt_s_cwr", help=_retro_help)
    with bs2:
        show_ewbh = st.checkbox("Equal Weight — Buy & Hold", value=False,
                                key="bt_s_ewbh")
        show_ewr = st.checkbox("Equal Weight — Rebalanced", value=True,
                               key="bt_s_ewr")
    with bs3:
        show_bench = st.checkbox("Benchmark", value=True, key="bt_s_bench")
    st.caption(
        "“Current Weights — Retrospective” applies today's chosen portfolio "
        "weights across the whole historical window. The result shows what "
        "those weights *would have* earned — not that they would have been "
        "chosen back then."
    )
    st.caption(
        "Backtests use the securities currently supplied by the user. They "
        "do not reconstruct historical index membership and may therefore "
        "be subject to survivorship and selection bias."
    )


    # Benchmark symbol (shared by static, OOS and walk-forward).
    try:
        bench_sym_bt = validate_ticker_symbol(st.session_state.bt_benchmark)
    except ValueError as e:
        st.error(f"Invalid benchmark ticker: {e}")
        bench_sym_bt = None

    # ---- Walk-Forward Optimisation (estimate → invest → re-estimate) ----
    st.markdown("---")
    st.markdown("### Walk-Forward Optimisation")
    st.caption(
        "Each rebalance re-estimates its target portfolio using only data "
        "dated **strictly before** the execution date — then trades to that "
        "target at the execution-day close. This answers: had this "
        "construction process actually been run through time, how would it "
        "have performed? Unlike the static comparison above, nothing here "
        "is frozen: targets evolve as estimation windows roll forward."
    )
    st.caption(SURVIVORSHIP_WARNING)
    st.caption(HINDSIGHT_WARNING)

    wf1, wf2, wf3, wf4 = st.columns(4)
    with wf1:
        _wf_start_default = max(
            bt_min_date,
            (pd.Timestamp(bt_max_date) - pd.DateOffset(years=5)).date())
        wf_start = st.date_input(
            "Test Start", value=_wf_start_default,
            min_value=bt_min_date, max_value=bt_max_date, key="wf_start",
            help="Estimation for the initial allocation uses only data "
                 "strictly before this date.")
        wf_end = st.date_input(
            "Test End", value=bt_max_date,
            min_value=bt_min_date, max_value=bt_max_date, key="wf_end",
            help="Last date of the simulated test window.")
    with wf2:
        wf_lookback = st.selectbox(
            "Estimation Lookback", [1, 2, 3, 5], index=2, key="wf_lookback",
            format_func=lambda y: f"{y} Year{'s' if y > 1 else ''} (~{y * 252} obs)",
            help="Calendar years of history ending strictly before each "
                 "execution date (rolling), or ignored in expanding mode.")
        wf_window = st.selectbox(
            "Window Type", ["Rolling", "Expanding"], index=0, key="wf_window",
            help="Rolling uses only the trailing lookback; expanding uses "
                 "all history available before each execution date.")
    with wf3:
        wf_freq_label = st.selectbox(
            "Rebalance Frequency",
            ["Monthly", "Quarterly", "Semi-annual", "Annual"],
            index=1, key="wf_freq",
            help="Scheduled dates map to the first trading day on or after "
                 "each calendar mark; training always ends before that day.")
        wf_costs = st.number_input(
            "Transaction Cost (bps)", min_value=0.0, max_value=1000.0,
            value=10.0, step=1.0, key="wf_costs")
    with wf4:
        wf_minw = st.slider(
            "Min weight per asset", 0.0, 0.5, 0.0, 0.05, format="%.2f",
            key="wf_minw")
        wf_maxw = st.slider(
            "Max weight per asset", 0.5, 1.0, 1.0, 0.05, format="%.2f",
            key="wf_maxw")

    wm1, wm2, wm3 = st.columns(3)
    with wm1:
        wf_m_mv = st.checkbox("Minimum Variance", value=True, key="wf_m_mv")
        wf_m_ms = st.checkbox("Maximum Sharpe", value=False, key="wf_m_ms")
    with wm2:
        wf_m_erc = st.checkbox("ERC", value=True, key="wf_m_erc")
        wf_m_md = st.checkbox("Max Diversification", value=True,
                              key="wf_m_md")
    with wm3:
        wf_m_bl = st.checkbox("Black-Litterman", value=False, key="wf_m_bl")
    wf_eq = st.checkbox(
        "Equal Weight (no optimisation)", value=True, key="wf_eq",
        help="Constant 1/N targets rebalanced on the same schedule — "
             "never optimised.")
    _wf_methods = ([m for m, on in
                    (("min_variance", wf_m_mv), ("max_sharpe", wf_m_ms),
                     ("erc", wf_m_erc), ("max_diversification", wf_m_md),
                     ("black_litterman", wf_m_bl)) if on])
    _wf_labels = [WALK_FORWARD_LABELS[m] for m in _wf_methods]
    if _wf_labels:
        wc1, wc2 = st.columns(2)
        with wc1:
            wf_static_method = st.selectbox(
                "Static counterpart method", _wf_labels, index=0,
                key="wf_static_method",
                help="Frozen initial targets of this method, rebalanced on "
                     "the same schedule — answers whether re-estimating "
                     "actually helped.")
        with wc2:
            wf_drag_method = st.selectbox(
                "Cost-drag twin (0 bps) for", ["None"] + _wf_labels, index=0,
                key="wf_drag_method",
                help="Re-runs one method's exact target schedule at zero "
                     "cost; drag = gross final − net final (compounding "
                     "included, not summed costs).")
    else:
        wf_static_method, wf_drag_method = None, "None"

    wf_bl_cfg, wf_bl_err = None, None
    wf_delta_bench = False
    if wf_m_bl:
        with st.expander("Black-Litterman inputs (walk-forward)",
                         expanded=True):
            wf_bl_cfg, wf_bl_err = _bl_views_editor(
                "wf_bl", tickers,
                ref_default="Equal Weights — proxy prior")
            wf_delta_bench = st.checkbox(
                "Estimate δ from benchmark (each training window)",
                value=False, key="wf_delta_bench",
                help="δ is recalculated per rebalance from that window's "
                     "benchmark observations only; falls back to the slider "
                     "on invalid estimates.")
        if wf_bl_err:
            st.error(f"Black-Litterman inputs: {wf_bl_err}")
        st.warning(BL_HINDSIGHT_WARNING)
        if wf_bl_cfg is not None and wf_bl_cfg["ref"].startswith("Current"):
            st.warning(
                "Current-Portfolio reference uses today's weights — labelled "
                "RETROSPECTIVE and excluded from strict out-of-sample claims. "
                "Prefer the Equal Weights prior for genuine walk-forward.")

    _wf_key = (tuple(tickers), str(wf_start), str(wf_end), int(wf_lookback),
               str(wf_window), str(wf_freq_label), float(wf_costs),
               float(wf_minw), float(wf_maxw), tuple(_wf_methods),
               str(wf_static_method), str(wf_drag_method),
               repr(wf_bl_cfg), bool(wf_delta_bench), str(base_ccy),
               float(bt_capital), float(rf_rate), str(bench_sym_bt))
    _wf_freq = {"Monthly": "monthly", "Quarterly": "quarterly",
                "Semi-annual": "semi-annual",
                "Annual": "annual"}[wf_freq_label]

    if st.button("Run Walk-Forward Analysis", key="wf_run"):
        if not _wf_methods:
            st.error("Select at least one walk-forward method.")
            return  # section-local return (was st.stop()): sibling tabs keep rendering
        if wf_m_bl and (wf_bl_err or wf_bl_cfg is None):
            st.error(f"Black-Litterman inputs: "
                     f"{wf_bl_err or 'unavailable'}")
            return  # section-local return (was st.stop()): sibling tabs keep rendering
        try:
            # Convert only the era the run can touch (estimation lookback
            # + test window, with a 1-year buffer for calendar edges): full
            # max-history conversion would fail on ancient FX gaps (e.g. a
            # 1988 listing vs FX history from 2003) that this window never
            # needs. Gaps INSIDE the needed era still raise honestly below.
            _wf_need_from = (pd.Timestamp(wf_start) - pd.DateOffset(
                years=int(wf_lookback) + 1)).date()
            _wf_native = bt_full.loc[
                pd.Timestamp(_wf_need_from):pd.Timestamp(wf_end)]
            if _wf_native.empty:
                raise ValueError("no market history covers the selected "
                                 "walk-forward era.")
            _wf_base_full = base_price_panel(
                _wf_native, tickers, instruments, ccy.get("fx_hist"),
                base_ccy)
        except ValueError as e:
            st.error(f"Base-currency panel: {e}")
            return  # section-local return (was st.stop()): sibling tabs keep rendering
        _wf_bench_full = None
        if bench_sym_bt is not None:
            try:
                _wf_bench_nat = load_benchmark_data(bench_sym_bt, "max")
                _wf_bench_nat = _wf_bench_nat.loc[
                    pd.Timestamp(_wf_need_from):pd.Timestamp(wf_end)]
                _wf_bench_full = base_benchmark_series(
                    _wf_bench_nat, bench_sym_bt,
                    base_ccy, ccy.get("fx_hist"))
            except ValueError as e:
                st.warning(f"Walk-forward benchmark unavailable: {e}")
            except Exception:
                st.warning("Could not load benchmark data "
                           "(network or API error).")
        _wf_results: dict = {}
        _wf_ok = True
        for _m in _wf_methods:
            _bcfg = WalkForwardConfig(
                method=_m, test_start=wf_start, test_end=wf_end,
                lookback_years=float(wf_lookback),
                window_type=str(wf_window).lower(),
                rebalance_frequency=_wf_freq,
                transaction_cost_bps=float(wf_costs),
                base_currency=base_ccy, min_weight=float(wf_minw),
                max_weight=float(wf_maxw), risk_free_annual=float(rf_rate))
            if _m == "black_litterman":
                _bcfg.bl_views = list(wf_bl_cfg["views"])
                _bcfg.bl_delta = float(wf_bl_cfg["delta"])
                _bcfg.bl_tau = float(wf_bl_cfg["tau"])
                _bcfg.bl_delta_from_benchmark = bool(wf_delta_bench)
                if wf_bl_cfg["ref"].startswith("Current"):
                    _bcfg.bl_ref = "retrospective_current"
                    _bcfg.bl_ref_weights = np.asarray(weights, dtype=float)
            try:
                _res = run_walk_forward(
                    _wf_base_full, tickers, _bcfg, float(bt_capital),
                    benchmark_base=_wf_bench_full)
            except ValueError as e:
                st.error(f"Walk-forward {WALK_FORWARD_LABELS[_m]} failed: {e}")
                _wf_ok = False
                continue
            if not _res.success:
                st.error(f"Walk-forward {WALK_FORWARD_LABELS[_m]} failed: "
                         f"{_res.message}")
                _wf_ok = False
                continue
            _wf_results[f"WF {WALK_FORWARD_LABELS[_m]}"] = _res
        if not _wf_results:
            st.error("No walk-forward strategy completed.")
            return  # section-local return (was st.stop()): sibling tabs keep rendering
        # Static counterpart: frozen INITIAL targets of the selected method
        # on the identical window/capital/costs/schedule (fair comparison).
        _wf_static: dict = {}
        try:
            _w0_name = f"WF {wf_static_method}"
            _w0 = _wf_results[_w0_name].target_weights.iloc[0].values
            _static_window, _ = prepare_backtest_data(
                _wf_base_full, tickers, pd.Timestamp(wf_start),
                pd.Timestamp(wf_end))
            _wf_static[f"Static {wf_static_method}"] = backtest_rebalanced(
                _static_window, _w0, float(bt_capital), _wf_freq,
                float(wf_costs), float(rf_rate),
                name=f"Static {wf_static_method}")
            if wf_eq:
                _wf_static["Equal Weight"] = backtest_rebalanced(
                    _static_window, np.ones(len(tickers)) / len(tickers),
                    float(bt_capital), _wf_freq, float(wf_costs),
                    float(rf_rate), name="Equal Weight")
        except (ValueError, KeyError) as e:
            st.warning(f"Static counterpart unavailable: {e}")
        if _wf_bench_full is not None and bench_sym_bt is not None:
            try:
                _wf_static[bench_sym_bt] = backtest_benchmark(
                    _wf_bench_full, _static_window.index, float(bt_capital),
                    float(rf_rate), name=bench_sym_bt)
            except (ValueError, NameError) as e:
                st.warning(f"Walk-forward benchmark skipped: {e}")
            except Exception:
                st.warning("Walk-forward benchmark skipped "
                           "(network or API error).")
        # Zero-cost twin for the drag diagnostic (same targets by
        # deterministic construction — verified in tests).
        _wf_drag: dict = {}
        if wf_drag_method and wf_drag_method != "None":
            _dm = [m for m in _wf_methods
                   if WALK_FORWARD_LABELS[m] == wf_drag_method]
            if _dm:
                _dcfg = WalkForwardConfig(
                    method=_dm[0], test_start=wf_start, test_end=wf_end,
                    lookback_years=float(wf_lookback),
                    window_type=str(wf_window).lower(),
                    rebalance_frequency=_wf_freq, transaction_cost_bps=0.0,
                    base_currency=base_ccy, min_weight=float(wf_minw),
                    max_weight=float(wf_maxw),
                    risk_free_annual=float(rf_rate))
                if _dm[0] == "black_litterman":
                    _dcfg.bl_views = list(wf_bl_cfg["views"])
                    _dcfg.bl_delta = float(wf_bl_cfg["delta"])
                    _dcfg.bl_tau = float(wf_bl_cfg["tau"])
                    _dcfg.bl_delta_from_benchmark = bool(wf_delta_bench)
                    if wf_bl_cfg["ref"].startswith("Current"):
                        _dcfg.bl_ref = "retrospective_current"
                        _dcfg.bl_ref_weights = np.asarray(weights,
                                                          dtype=float)
                try:
                    _dres = run_walk_forward(
                        _wf_base_full, tickers, _dcfg, float(bt_capital),
                        benchmark_base=_wf_bench_full)
                    if _dres.success:
                        _wf_drag[f"WF {wf_drag_method} (0 bps)"] = _dres
                except ValueError as e:
                    st.warning(f"Zero-cost twin unavailable: {e}")
        st.session_state["wf_results"] = {
            "key": _wf_key, "dynamic": _wf_results, "static": _wf_static,
            "drag": _wf_drag,
            "test_window": (str(wf_start), str(wf_end)),
        }
        if not _wf_ok:
            st.warning("Some walk-forward methods failed — completed "
                       "strategies are shown below; failures carry no "
                       "invented weights.")

    _wf_cached = st.session_state.get("wf_results")
    if _wf_cached and _wf_cached.get("key") != _wf_key:
        st.info("Walk-forward inputs changed — press Run Walk-Forward "
                "Analysis to refresh the results below.")
        _wf_cached = None
    if _wf_cached:
        _wf_dyn = _wf_cached["dynamic"]
        _wf_stat = _wf_cached.get("static", {})
        _wf_all = {**_wf_dyn, **_wf_stat, **_wf_cached.get("drag", {})}
        _wf_cap = float(bt_capital)
        st.markdown(f"### Walk-Forward Growth of {fmt_usd(_wf_cap)}")
        st.plotly_chart(plot_backtest_growth(_wf_all, _wf_cap,
                                             currency=base_ccy),
                        width="stretch")
        st.caption(
            f"Test window: {_wf_cached['test_window'][0]} to "
            f"{_wf_cached['test_window'][1]}. Static lines freeze the "
            "selected method's initial walk-forward targets; dynamic lines "
            "re-estimate at every event using only strictly-prior data. "
            "Same window, capital, currency, universe and costs throughout."
        )
        st.markdown("### Walk-Forward vs Static Comparison")
        _render_backtest_comparison(_wf_all)
        st.caption(
            "Benchmark Total Costs show as N/A (the passive leg is not "
            "cost-modelled)."
        )

        st.markdown("### Turnover Diagnostics")
        _to_rows = []
        for _nm, _r in _wf_dyn.items():
            _td = turnover_diagnostics(_r)
            _to_rows.append({
                "Strategy": _nm,
                "Scheduled": _td["n_scheduled"],
                "Successful": _td["n_successful"],
                "Failed/Skipped": _td["n_failed"],
                "Avg Turnover": fmt_pct(_td["avg_turnover"]),
                "Median Turnover": fmt_pct(_td["median_turnover"]),
                "Max Turnover": fmt_pct(_td["max_turnover"]),
                "Cumulative Turnover": fmt_pct(_td["cumulative_turnover"]),
                "Total Costs": fmt_usd(_td["total_costs"]),
            })
        st.dataframe(pd.DataFrame(_to_rows), width="stretch",
                     hide_index=True)
        _drag_res = _wf_cached.get("drag", {})
        if _drag_res:
            _dnm = next(iter(_drag_res))
            _gross = _drag_res[_dnm].metrics["final_value"]
            _base_nm = _dnm.replace(" (0 bps)", "")
            if _base_nm in _wf_dyn:
                _net = _wf_dyn[_base_nm].metrics["final_value"]
                d1, d2, d3 = st.columns(3)
                d1.metric("Gross Final Value (0 bps)", fmt_usd(_gross))
                d2.metric("Net Final Value", fmt_usd(_net))
                d3.metric("Transaction Cost Drag", fmt_usd(_gross - _net),
                          help="Zero-cost final wealth minus net final "
                               "wealth — lost compounding included, not "
                               "summed costs.")
        st.caption(RF_CONSTANT_NOTE)
        st.caption(
            "Currency conversion covers the estimation era onward (lookback "
            "plus test window); older history the run cannot touch is never "
            "converted. Per-event training windows are listed in the event "
            "table below.")

        _wf_names = list(_wf_dyn.keys())
        _wf_chart = st.selectbox(
            "Detail strategy", _wf_names, index=0, key="wf_chart_method",
            help="Drives the allocation, event and drawdown views below.")
        _wr = _wf_dyn[_wf_chart]
        st.markdown(f"### Allocation Through Time — {_wf_chart}")
        if _wr.target_weights is not None and not _wr.target_weights.empty:
            st.plotly_chart(plot_weight_drift(
                _wr.target_weights, title=f"Target Allocation — {_wf_chart}"),
                width="stretch")
            st.caption(
                "Estimated target weights at each successful rebalance "
                "(re-estimated from strictly-prior data); holdings drift "
                "between events.")
            with st.expander("Target weights by date (CSV)", expanded=False):
                st.dataframe(_wr.target_weights.style.format("{:.2%}"),
                             width="stretch")
                st.download_button(
                    "Download target weights (CSV)",
                    data=_wr.target_weights.to_csv().encode("utf-8"),
                    file_name="walkforward-target-weights.csv",
                    mime="text/csv", key="wf_dl_weights")
            _stab = _wr.metrics
            s1, s2 = st.columns(2)
            s1.metric("Mean |Δ target| (L1)",
                      f"{_stab.get('weight_stability_l1_mean', 0.0):.2%}",
                      help="Average absolute target-weight change between "
                           "consecutive events.")
            s2.metric("Max |Δ target| (L1)",
                      f"{_stab.get('weight_stability_l1_max', 0.0):.2%}")
        st.markdown(f"### Rebalance Events — {_wf_chart}")
        _ev = _wr.events.copy()
        if not _ev.empty:
            _ev_disp = pd.DataFrame({
                "Execution Date": pd.to_datetime(
                    _ev["Execution Date"]).dt.date,
                "Training Start": pd.to_datetime(
                    _ev["Training Start"]).dt.date,
                "Training End": pd.to_datetime(
                    _ev["Training End"]).dt.date,
                "Method": _ev["Method"],
                "Status": _ev["Status"],
                "Turnover": _ev["Turnover"].map(fmt_pct),
                "Transaction Cost": _ev["Transaction Cost"].map(fmt_usd),
                "Value Before": _ev["Portfolio Value Before"].map(fmt_usd),
                "Value After": _ev["Portfolio Value After"].map(fmt_usd),
                "Reason": _ev["Reason"],
            })
            st.dataframe(_ev_disp, width="stretch", hide_index=True)
            if not _wr.failures.empty:
                st.warning(
                    f"{len(_wr.failures)} rebalance(s) failed/skipped — "
                    "holdings were kept and the run continued. See Reason "
                    "above; no fallback weights were invented.")
            st.download_button(
                "Download rebalance events (CSV)",
                data=_wr.events.to_csv(index=False).encode("utf-8"),
                file_name="walkforward-events.csv",
                mime="text/csv", key="wf_dl_events")
            st.download_button(
                "Download wealth path (CSV)",
                data=_wr.values.rename("value").to_csv().encode("utf-8"),
                file_name="walkforward-wealth.csv",
                mime="text/csv", key="wf_dl_wealth")
            st.caption(
                f"Base Currency: {base_ccy}; method: {_wf_chart}; lookback: "
                f"{wf_lookback}y {str(wf_window).lower()}; frequency: "
                f"{wf_freq_label}; costs: {float(wf_costs):g} bps.")
        st.markdown(f"### Estimated vs Realised — {_wf_chart}")
        st.dataframe(ex_ante_realised_frame(_wr), width="stretch",
                     hide_index=True)
        st.caption(
            "Ex-ante estimates are labelled 'Estimated at rebalance' and are "
            "not predictions — realised performance below is historical fact, "
            "not a forecast test.")
        st.markdown(f"### Drawdown — {_wf_chart}")
        _wf_bench_name = (bench_sym_bt if bench_sym_bt in _wf_all else None)
        _wf_static_name = next(
            (n for n in _wf_all
             if n.startswith("Static ") and n in _wf_stat), None)
        _wf_dd_b = _wf_all[_wf_bench_name].drawdowns if _wf_bench_name else None
        st.plotly_chart(
            plot_drawdown(_wr.drawdowns, _wf_dd_b,
                          _wf_bench_name or "Benchmark",
                          _wr.metrics["max_drawdown"]),
            width="stretch",
        )
        _wfd = _wr.metrics
        wdd1, wdd2, wdd3 = st.columns(3)
        wdd1.metric("Current Drawdown", fmt_pct(_wfd["current_drawdown"]))
        wdd2.metric("Maximum Drawdown", fmt_pct(_wfd["max_drawdown"]))
        wdd3.metric("Total Costs", fmt_usd(_wfd["total_costs"]))
        if _wf_static_name is not None:
            st.caption(
                f"Static counterpart '{_wf_static_name}' final value: "
                f"{fmt_usd(_wf_all[_wf_static_name].metrics['final_value'])} "
                f"vs walk-forward {fmt_usd(_wfd['final_value'])} — the gap "
                f"answers whether re-estimating helped.")

    # ---- Prepare the common-date window (real data only) ----
    # Convert native history to base currency FIRST, so analytics run on
    # base-currency returns (FX movement included), never mixed natives.
    # Only the asked window (plus a 30-day edge buffer for the bounded FX
    # carry-forward) is converted: ancient history outside the window can
    # neither help nor fail the run. Results are identical to converting
    # everything first (pinned by test_era_scoped_conversion_matches_full_panel).
    try:
        _bt_need_from = pd.Timestamp(bt_start) - pd.Timedelta(days=30)
        bt_base_full = base_price_panel(
            bt_full.loc[_bt_need_from:pd.Timestamp(bt_end)],
            tickers, instruments, ccy.get("fx_hist"), base_ccy)
        bt_prices, bt_notes = prepare_backtest_data(
            bt_base_full, tickers, pd.Timestamp(bt_start), pd.Timestamp(bt_end))
    except ValueError as e:
        st.error(f"Backtest setup: {e}")
        return  # section-local return (was st.stop()): sibling tabs keep rendering
    for _note in bt_notes:
        st.warning(_note)
    if any(instruments[t].native_currency != base_ccy for t in tickers):
        st.caption(
            f"Holdings translated to {base_ccy} with date-matched historical "
            "FX before simulation, so currency movement is part of returns.")


    bench_full = None
    if bench_sym_bt is not None and show_bench:
        try:
            _bench_native = load_benchmark_data(bench_sym_bt, "max")
            _bench_native = _bench_native.loc[
                _bt_need_from:pd.Timestamp(bt_end)]
            bench_full = base_benchmark_series(
                _bench_native, bench_sym_bt, base_ccy, ccy.get("fx_hist"))
        except ValueError as e:
            st.warning(f"Benchmark unavailable: {e}")
        except Exception:
            st.warning("Could not load benchmark data (network or API error).")

    # ---- Run the selected strategies (pure historical engine) ----
    _freq = {"Monthly": "monthly", "Quarterly": "quarterly",
             "Semi-annual": "semi-annual", "Annual": "annual"}[freq_label]
    _eq_w = np.ones(len(tickers)) / len(tickers)
    _results: dict = {}
    try:
        if show_cwbh:
            _results["Current Weights — Retrospective (Buy & Hold)"] = backtest_buy_and_hold(
                bt_prices, weights, float(bt_capital), rf_rate,
                name="Current Weights — Retrospective (Buy & Hold)")
        if show_cwr:
            _name = f"Current Weights — Retrospective ({freq_label})"
            _results[_name] = backtest_rebalanced(
                bt_prices, weights, float(bt_capital), _freq, float(bt_costs),
                rf_rate, name=_name)
        if show_ewbh:
            _results["Equal Weight — Buy & Hold"] = backtest_buy_and_hold(
                bt_prices, _eq_w, float(bt_capital), rf_rate,
                name="Equal Weight — Buy & Hold")
        if show_ewr:
            _name = f"Equal Weight — {freq_label}"
            _results[_name] = backtest_rebalanced(
                bt_prices, _eq_w, float(bt_capital), _freq, float(bt_costs),
                rf_rate, name=_name)
        if bench_full is not None:
            try:
                _results[bench_sym_bt] = backtest_benchmark(
                    bench_full, bt_prices.index, float(bt_capital), rf_rate,
                    name=bench_sym_bt)
            except ValueError as e:
                st.warning(f"Benchmark skipped: {e}")
    except ValueError as e:
        st.error(f"Backtest failed: {e}")
        return  # section-local return (was st.stop()): sibling tabs keep rendering
    if not _results:
        st.info("Select at least one strategy to run the backtest.")
        return  # section-local return (was st.stop()): sibling tabs keep rendering

    # ---- Growth of Initial Capital (actual wealth paths) ----
    st.markdown("---")
    st.markdown(f"### Growth of {fmt_usd(float(bt_capital))}")
    st.plotly_chart(plot_backtest_growth(_results, float(bt_capital),
                                                currency=base_ccy),
                    width="stretch")
    st.caption(
        f"{len(bt_prices)} common trading days "
        f"({bt_prices.index[0].date()} to {bt_prices.index[-1].date()}); "
        "one line per selected strategy."
    )

    # ---- Strategy Comparison ----
    st.markdown("---")
    st.markdown("### Strategy Comparison")
    _table = compare_backtests(_results)
    _disp = pd.DataFrame(index=_table.index)
    for _col in _table.columns:
        _is_bench = _results[_col].is_benchmark
        _cells = []
        for _metric in _table.index:
            _v = _table.loc[_metric, _col]
            if _metric == "Total Costs" and _is_bench:
                _cells.append("N/A")
            elif _metric in ("Final Value", "Total Costs"):
                _cells.append(fmt_usd(_v))
            elif _metric in ("Total Return", "CAGR", "Annualised Return",
                             "Volatility", "Max Drawdown", "Best Day",
                             "Worst Day"):
                _cells.append(fmt_pct(_v))
            elif _metric in ("Sharpe", "Sortino", "Calmar"):
                _cells.append(f"{_v:.2f}")
            elif _metric == "Rebalances":
                _cells.append(f"{int(_v)}")
            else:
                _cells.append(str(_v))
        _disp[_col] = _cells
    st.dataframe(_disp, width="stretch")
    st.caption(
        "Benchmark Total Costs show as N/A (the passive leg is not "
        "cost-modelled). CAGR uses actual calendar elapsed time; "
        "Annualised Return is the log-space mean and is kept separate."
    )

    # ---- Drawdown (selected strategy vs benchmark) ----
    st.markdown("---")
    st.markdown("### Drawdown")
    _dd_default = next(
        (n for n, r in _results.items() if not r.is_benchmark),
        next(iter(_results)))
    _dd_choice = st.selectbox(
        "Drawdown strategy", list(_results.keys()),
        index=list(_results.keys()).index(_dd_default), key="bt_dd_choice")
    _bench_dd_name = (bench_sym_bt if bench_sym_bt in _results
                      and bench_sym_bt != _dd_choice else None)
    _sel_dd = _results[_dd_choice].drawdowns
    _b_dd = _results[_bench_dd_name].drawdowns if _bench_dd_name else None
    st.plotly_chart(
        plot_drawdown(_sel_dd, _b_dd, _bench_dd_name or "Benchmark",
                      _results[_dd_choice].metrics["max_drawdown"]),
        width="stretch",
    )
    _dm = _results[_dd_choice].metrics
    dd1, dd2, dd3 = st.columns(3)
    dd1.metric("Current Drawdown", fmt_pct(_dm["current_drawdown"]))
    dd2.metric("Maximum Drawdown", fmt_pct(_dm["max_drawdown"]))
    dd3.metric("Duration (days)", f"{_dm['duration_days']}")
    st.caption(
        f"Peak {_fmt_bt_date(_dm['peak_date'])} → "
        f"Trough {_fmt_bt_date(_dm['trough_date'])} → "
        f"Recovery {_fmt_bt_date(_dm['recovery_date'])}."
    )

    # ---- Rebalance Statistics (currency values stay 2-per-row) ----
    st.markdown("---")
    st.markdown("### Rebalance Statistics")
    _rb_names = [n for n, r in _results.items() if not r.is_benchmark]
    if not _rb_names:
        st.caption("Rebalance statistics apply to portfolio strategies — "
                   "select one above.")
        _rb_choice = None
    else:
        _rb_choice = st.selectbox(
            "Statistics for", _rb_names,
            index=0, key="bt_rb_choice") if len(_rb_names) > 1 else _rb_names[0]
        if len(_rb_names) == 1:
            st.caption(f"Showing statistics for {_rb_choice}.")
        _rm = _results[_rb_choice].metrics
        t1, t2 = st.columns(2)
        t1.metric("Average Turnover", fmt_pct(_rm["avg_turnover"]),
                  help="Mean per-event traded notional ÷ value before.")
        t2.metric("Total Turnover", fmt_pct(_rm["total_turnover"]),
                  help="Sum of per-event turnover across all rebalances.")
        t3, t4 = st.columns(2)
        t3.metric("Rebalances", f"{_rm['n_rebalances']}")
        t4.metric("Total Transaction Costs", fmt_usd(_rm["total_costs"]))

    # ---- Advanced Details ----
    st.markdown("---")
    st.markdown("### Advanced Details")
    if _rb_names:
        with st.expander("Rebalance Events"):
            _ev = _results[_rb_choice].rebalance_events
            if _ev is None or _ev.empty:
                st.caption("No rebalance events (buy-and-hold, or the window "
                           "covers a single schedule period).")
            else:
                _evd = _ev.copy()
                _evd["Date"] = pd.to_datetime(_evd["Date"]).dt.strftime("%Y-%m-%d")
                _evd["Portfolio Value Before"] = _evd["Portfolio Value Before"].map(fmt_usd)
                _evd["Turnover"] = _evd["Turnover"].map(fmt_pct)
                _evd["Transaction Cost"] = _evd["Transaction Cost"].map(fmt_usd)
                _evd["Portfolio Value After"] = _evd["Portfolio Value After"].map(fmt_usd)
                st.dataframe(_evd, width="stretch")
            st.caption("Every event reconciles: value before − costs = value after.")
        with st.expander("Trade Detail"):
            _tr = _results[_rb_choice].trades
            if _tr is None or _tr.empty:
                st.caption("No individual trades (buy-and-hold).")
            else:
                _trd = _tr.copy()
                _trd["Date"] = pd.to_datetime(_trd["Date"]).dt.strftime("%Y-%m-%d")
                _trd["Before Weight"] = _trd["Before Weight"].map(fmt_pct)
                _trd["Target Weight"] = _trd["Target Weight"].map(fmt_pct)
                _trd["Trade Value"] = _trd["Trade Value"].map(
                    lambda v: fmt_usd(v) if v >= 0 else f"-{fmt_usd(-v)}")
                _trd["Transaction Cost"] = _trd["Transaction Cost"].map(fmt_usd)
                st.dataframe(_trd, width="stretch")
        with st.expander("Rolling Metrics (12M)"):
            _lr = _results[_rb_choice].log_returns
            if len(_lr) < 252:
                st.warning(
                    f"Only {len(_lr)} daily returns — need at least 252 for "
                    "12-month rolling metrics. Widen the date range.")
            else:
                try:
                    _specs = [
                        ("Rolling 12M Return",
                         lambda s: rolling_annualised_return(s, 252),
                         "Return", ".0%"),
                        ("Rolling Volatility",
                         lambda s: rolling_volatility(s, 252),
                         "Volatility (σ)", ".0%"),
                        ("Rolling Sharpe",
                         lambda s: rolling_sharpe(s, rf_rate, 252),
                         "Sharpe", ".1f"),
                    ]
                    _broll = _results[bench_sym_bt].log_returns \
                        if bench_sym_bt in _results else None
                    _roll_series = [(_t, _fn(_lr),
                                     _fn(_broll) if _broll is not None else None,
                                     _y, _f)
                                    for _t, _fn, _y, _f in _specs]
                except ValueError as e:
                    st.warning(f"Rolling metrics unavailable: {e}")
                    _roll_series = []
                for _t, _ps, _bs, _y, _f in _roll_series:
                    _fig = go.Figure()
                    _fig.add_trace(go.Scatter(
                        x=_ps.index, y=_ps.values, mode="lines",
                        name=_rb_choice,
                        line=dict(color="#58a6ff", width=2)))
                    if _bs is not None:
                        _fig.add_trace(go.Scatter(
                            x=_bs.index, y=_bs.values, mode="lines",
                            name=bench_sym_bt,
                            line=dict(color="#f0883e", width=1.5)))
                    _fig.update_layout(**CHART_THEME, title=_t,
                                        xaxis_title="Date", yaxis_title=_y,
                                        yaxis_tickformat=_f)
                    st.plotly_chart(_fig, width="stretch")
        with st.expander("Weight Drift"):
            _wdf = _results[_rb_choice].weights
            if _wdf is None or _wdf.empty:
                st.caption("No weight history for this strategy.")
            else:
                st.plotly_chart(
                    plot_weight_drift(_wdf, f"Weight Drift — {_rb_choice}"),
                    width="stretch")
                st.caption("Buy-and-hold weights drift with prices; "
                           "rebalanced weights snap back to targets.")
    else:
        st.caption("Advanced details need a portfolio strategy selected above.")

    # ---- Static Out-of-Sample Allocation Comparison ----
    st.markdown("---")
    st.markdown("### Static Out-of-Sample Allocation Comparison")
    st.caption(
        "Portfolio weights are estimated using data **strictly before** the "
        "test period, then frozen throughout the historical test. No "
        "re-estimation occurs during the test; periodic rebalancing (if any) "
        "returns to those same frozen weights."
    )
    oo1, oo2, oo3, oo4 = st.columns(4)
    with oo1:
        _oos_default = max(
            bt_min_date,
            (pd.Timestamp(bt_max_date) - pd.DateOffset(years=3)).date())
        oos_start = st.date_input(
            "Test Start Date", value=_oos_default,
            min_value=bt_min_date, max_value=bt_max_date, key="oos_start",
            help="Estimation uses only data strictly before this date.")
    with oo2:
        oos_lookback = st.selectbox(
            "Estimation Lookback", [1, 2, 3, 5], index=2, key="oos_lookback",
            format_func=lambda y: f"{y} Year{'s' if y > 1 else ''} (~{y * 252} obs)",
            help="Calendar years of history ending strictly before test start.")
    with oo3:
        oos_freq_label = st.selectbox(
            "Test Rebalancing",
            ["Monthly", "Quarterly", "Semi-annual", "Annual"],
            index=1, key="oos_freq",
            help="Rebalance during the test back to the frozen weights.")
    with oo4:
        oos_costs = st.number_input(
            "Test Costs (bps)", min_value=0.0, max_value=1000.0,
            value=10.0, step=1.0, key="oos_costs")

    os1, os2, os3 = st.columns(3)
    with os1:
        oos_eq = st.checkbox("Equal Weight", value=True, key="oos_eq")
        oos_mv = st.checkbox("Minimum Variance", value=False, key="oos_mv")
    with os2:
        oos_ms = st.checkbox("Maximum Sharpe", value=False, key="oos_ms")
        oos_erc = st.checkbox("Risk Parity", value=True, key="oos_erc")
    with os3:
        oos_md = st.checkbox("Max Diversification", value=True, key="oos_md")
        oos_bl = st.checkbox("Black-Litterman", value=False, key="oos_bl")
    oos_cur = st.checkbox(
        "Current Weights — Retrospective", value=False, key="oos_cur",
        help="Today's mix evaluated over the test window — a retrospective "
             "hypothetical with hindsight.")
    if oos_cur:
        st.caption(
            "Current Weights — Retrospective uses today's live weights over "
            "the test window; treat it as a hindsight benchmark, not an "
            "achievable historical strategy.")

    oos_bl_cfg, oos_bl_err = None, None
    oos_delta_bench = False
    if oos_bl:
        with st.expander("Black-Litterman inputs (out-of-sample)", expanded=True):
            oos_bl_cfg, oos_bl_err = _bl_views_editor("oos_bl", tickers)
            oos_delta_bench = st.checkbox(
                "Estimate δ from benchmark (estimation window)", value=False,
                key="oos_delta_bench",
                help="δ = (benchmark excess return) / (benchmark variance), "
                     "annualised simple units over the estimation window. "
                     "Falls back to the slider on invalid estimates.")
        if oos_bl_err:
            st.error(f"Black-Litterman inputs: {oos_bl_err}")
        st.warning(
            "Black-Litterman views are treated as assumptions available at "
            "the test start date. Historical results are hypothetical and "
            "may contain human hindsight if the views were created using "
            "later knowledge.")

    try:
        oos_T = pd.Timestamp(oos_start)
        # Convert to base currency BEFORE estimation: estimation covariance
        # and all test wealth paths then live in base-currency space. Only
        # the estimation era onward is converted (same era-scoping as the
        # static setup above — identical numbers, no ancient-FX failures).
        _oos_need_from = (oos_T - pd.DateOffset(
            years=int(oos_lookback) + 1)).date()
        oos_base_full = base_price_panel(
            bt_full.loc[pd.Timestamp(_oos_need_from):pd.Timestamp(bt_end)],
            tickers, instruments, ccy.get("fx_hist"), base_ccy)
        oos_est, oos_info = estimation_window(
            oos_base_full, tickers, oos_T, float(oos_lookback))
    except ValueError as e:
        st.error(f"Estimation window: {e}")
        return  # section-local return (was st.stop()): sibling tabs keep rendering
    st.caption(
        f"Estimation period: {oos_info['est_start']} to {oos_info['est_end']} "
        f"({oos_info['n_obs']:,} trading days); test starts "
        f"{oos_info['test_start']}. No observation dated ≥ test start "
        "entered estimation."
    )
    try:
        oos_prices, oos_notes = prepare_backtest_data(
            oos_base_full, tickers, oos_T, pd.Timestamp(bt_end))
    except ValueError as e:
        st.error(f"Test window: {e}")
        return  # section-local return (was st.stop()): sibling tabs keep rendering
    for _note in oos_notes:
        st.warning(_note)

    oos_bench_full = None
    if bench_sym_bt is not None:
        try:
            _oos_bench_native = load_benchmark_data(bench_sym_bt, "max")
            _oos_bench_native = _oos_bench_native.loc[
                pd.Timestamp(_oos_need_from):pd.Timestamp(bt_end)]
            oos_bench_full = base_benchmark_series(
                _oos_bench_native, bench_sym_bt, base_ccy, ccy.get("fx_hist"))
        except ValueError as e:
            st.warning(f"Out-of-sample benchmark unavailable: {e}")
        except Exception:
            st.warning("Could not load benchmark data (network or API error).")

    try:
        oos_rets = simple_returns_from_prices(oos_est, tickers)
        oos_mu = annualised_mean_returns_simple(oos_rets)
        oos_cov = annualised_covariance_simple(oos_rets)
    except ValueError as e:
        st.error(f"Estimation statistics: {e}")
        return  # section-local return (was st.stop()): sibling tabs keep rendering

    _oos_freq = {"Monthly": "monthly", "Quarterly": "quarterly",
                 "Semi-annual": "semi-annual", "Annual": "annual"}[oos_freq_label]
    _oos_cap = float(bt_capital)
    _oos_cost = float(oos_costs)
    oos_results: dict = {}
    try:
        if oos_eq:
            oos_results["Equal Weight"] = backtest_rebalanced(
                oos_prices, np.ones(len(tickers)) / len(tickers),
                _oos_cap, _oos_freq, _oos_cost, rf_rate,
                name="Equal Weight (OOS)")
        if oos_mv:
            _r = min_variance(oos_mu.values, oos_cov.values, rf_rate, 0.0, 1.0)
            if _r.success and _r.weights is not None:
                oos_results["Minimum Variance"] = backtest_rebalanced(
                    oos_prices, _r.weights, _oos_cap, _oos_freq, _oos_cost,
                    rf_rate, name="Minimum Variance (OOS)")
            else:
                st.error(f"❌ Minimum Variance estimation failed: {_r.message}")
        if oos_ms:
            _r = max_sharpe(oos_mu.values, oos_cov.values, rf_rate, 0.0, 1.0)
            if _r.success and _r.weights is not None:
                oos_results["Maximum Sharpe"] = backtest_rebalanced(
                    oos_prices, _r.weights, _oos_cap, _oos_freq, _oos_cost,
                    rf_rate, name="Maximum Sharpe (OOS)")
            else:
                st.error(f"❌ Maximum Sharpe estimation failed: {_r.message}")
        if oos_erc:
            _r = equal_risk_contribution(
                tickers, oos_mu.values, oos_cov, rf_rate, 0.0, 1.0)
            if _r.success and _r.weights is not None:
                oos_results["Risk Parity"] = backtest_rebalanced(
                    oos_prices, _r.weights.values, _oos_cap, _oos_freq,
                    _oos_cost, rf_rate, name="Risk Parity (OOS)")
            else:
                st.error(f"❌ Risk Parity estimation failed: {_r.message}")
        if oos_md:
            _r = maximum_diversification(
                tickers, oos_mu.values, oos_cov, rf_rate, 0.0, 1.0)
            if _r.success and _r.weights is not None:
                oos_results["Max Diversification"] = backtest_rebalanced(
                    oos_prices, _r.weights.values, _oos_cap, _oos_freq,
                    _oos_cost, rf_rate, name="Max Diversification (OOS)")
            else:
                st.error(f"❌ Max Diversification estimation failed: {_r.message}")
        if oos_bl:
            if oos_bl_err or oos_bl_cfg is None:
                st.error(f"❌ Black-Litterman inputs: "
                         f"{oos_bl_err or 'unavailable'}")
            else:
                _oos_delta = oos_bl_cfg["delta"]
                if oos_delta_bench:
                    if oos_bench_full is None:
                        st.warning("Benchmark δ estimate unavailable — "
                                   "using the slider value.")
                    else:
                        try:
                            _bwin = oos_bench_full.loc[
                                (oos_bench_full.index >= pd.Timestamp(
                                    oos_info["est_start"]))
                                & (oos_bench_full.index < oos_T)].dropna()
                            if len(_bwin) < 2:
                                raise ValueError("too few benchmark points")
                            _br = _bwin.pct_change().dropna()
                            _oos_delta = estimate_risk_aversion_from_benchmark(
                                float(_br.mean() * 252 - rf_rate),
                                float(_br.var() * 252))
                            st.caption(f"Benchmark-implied δ = {_oos_delta:.2f} "
                                       f"(estimation window).")
                        except ValueError as e:
                            st.warning(f"Benchmark δ estimate rejected ({e}) — "
                                       "using the slider value.")
                _oos_post, _oos_alloc = _run_bl_pipeline(
                    tickers, oos_cov, oos_bl_cfg["ref"], weights,
                    _oos_delta, oos_bl_cfg["tau"], oos_bl_cfg["views"],
                    rf_rate, 0.0, 1.0)
                if _oos_alloc.success and _oos_alloc.weights is not None:
                    oos_results["Black-Litterman"] = backtest_rebalanced(
                        oos_prices, _oos_alloc.weights.values, _oos_cap,
                        _oos_freq, _oos_cost, rf_rate,
                        name="Black-Litterman (OOS)")
                else:
                    st.error(f"❌ Black-Litterman estimation failed: "
                             f"{_oos_alloc.message}")
        if oos_cur:
            oos_results["Current — Retrospective"] = backtest_rebalanced(
                oos_prices, np.asarray(weights, dtype=float),
                _oos_cap, _oos_freq, _oos_cost, rf_rate,
                name="Current — Retrospective (OOS)")
        if oos_bench_full is not None and bench_sym_bt is not None:
            try:
                oos_results[bench_sym_bt] = backtest_benchmark(
                    oos_bench_full, oos_prices.index, _oos_cap, rf_rate,
                    name=f"{bench_sym_bt} (OOS)")
            except ValueError as e:
                st.warning(f"Out-of-sample benchmark skipped: {e}")
    except ValueError as e:
        st.error(f"Out-of-sample backtest failed: {e}")
        return  # section-local return (was st.stop()): sibling tabs keep rendering
    if not oos_results:
        st.info("Select at least one out-of-sample strategy.")
        return  # section-local return (was st.stop()): sibling tabs keep rendering

    st.markdown(f"### Out-of-Sample Growth of {fmt_usd(_oos_cap)}")
    st.plotly_chart(plot_backtest_growth(oos_results, _oos_cap,
                                                currency=base_ccy),
                    width="stretch")
    st.caption(
        f"Test window: {oos_prices.index[0].date()} to "
        f"{oos_prices.index[-1].date()} ({len(oos_prices):,} trading days); "
        "all lines start at the same initial capital."
    )

    st.markdown("### Out-of-Sample Comparison")
    _render_backtest_comparison(oos_results)
    st.caption(
        "Benchmark Total Costs show as N/A (the passive leg is not "
        "cost-modelled). Frozen estimation weights were never re-estimated "
        "during the test."
    )


# ============================================================

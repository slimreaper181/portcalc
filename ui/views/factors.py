"""views.factors — Factor Analysis tab (rendered by the app shell).

Extracted from the application shell; analytics live in
``analytics/``, fetching in ``data/``. Receives the shared
context dict and renders into the active tab.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from analytics.backtest import (
    backtest_benchmark,
    backtest_buy_and_hold,
    backtest_rebalanced,
    prepare_backtest_data,
)
from analytics.factors import (
    FACTOR_LABELS,
    MODEL_DESCRIPTIONS,
    MODEL_FACTORS,
    MODEL_LABELS,
    align_factor_returns,
    check_min_observations,
    describe_factor_loading,
    describe_market_beta,
    factor_attribution,
    format_p_value,
    plot_factor_attribution,
    plot_rolling_exposure,
    rolling_factor_regression,
    run_model,
    simple_returns_from_wealth,
    to_decimal_returns,
    us_scope_note,
)
from analytics.validation import validate_ticker_symbol
from data.pipeline import (
    base_benchmark_series,
    base_price_panel,
    load_backtest_data,
    load_benchmark_data,
    load_factor_data,
)
from ui.formatting import fmt_pct


def render(ctx):
    """Render the Factor Analysis tab from the shared context."""
    base_ccy = ctx["base_ccy"]
    ccy = ctx["ccy"]
    instruments = ctx["instruments"]
    rf_rate = ctx["rf_rate"]
    tickers = ctx["tickers"]
    weights = ctx["weights"]

    st.markdown("## Factor Analysis")
    st.caption(
        "Why did this portfolio perform the way it did? Simple daily "
        "portfolio returns (never log returns) regressed on simple "
        "Fama-French factor returns, with excess returns measured against "
        "the **French daily RF** — not the app-level risk-free input. "
        "Historical and model-dependent throughout; never a forecast."
    )

    # ---- Long-history prices through the same cached yfinance pipeline ----
    try:
        fa_full = load_backtest_data(tickers)
    except ValueError as e:
        st.error(f"Factor data error: {e}")
        return  # section-local return (was st.stop()): sibling tabs keep rendering
    except Exception:
        st.error("Could not load price history (network or API error). "
                 "Check your connection and try again.")
        return  # section-local return (was st.stop()): sibling tabs keep rendering
    # Range for the window widgets comes from the native common history (no
    # FX involved, so ancient listings never block widget setup). Currency
    # conversion below covers only the selected era — identical numbers to
    # converting everything first (see era-equivalence test), while FX gaps
    # outside the asked window can no longer fail the run.
    fa_common = fa_full.dropna(how="any")
    if len(fa_common) < 2:
        st.error("No overlapping history available for factor analysis.")
        return  # section-local return (was st.stop()): sibling tabs keep rendering
    fa_min_date = fa_common.index[0].date()
    fa_max_date = fa_common.index[-1].date()

    _scope = us_scope_note(tickers)
    if _scope:
        st.warning(
            "This factor model uses US Fama-French factors. Results for "
            "portfolios containing non-US assets may be economically less "
            f"meaningful. ({_scope})"
        )
    if base_ccy != "USD":
        st.warning(
            "Factor analysis uses US Fama-French factors expressed in USD "
            "market-return space. Portfolio returns have been translated "
            f"into {base_ccy}, so factor loadings may reflect both security "
            "and FX effects and should be interpreted cautiously."
        )

    # ---- Controls ----
    st.markdown("### Controls")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        fa_source = st.selectbox(
            "Return Source",
            ["Current Weights — Retrospective",
             "Equal Weight",
             "Selected Backtest Strategy"],
            index=0, key="fa_source",
            help="Current Weights freezes today's mix as the fixed target "
                 "(retrospective hypothetical). Selected Backtest Strategy "
                 "rebuilds the first strategy ticked in the Backtest tab.")
    with fc2:
        fa_model_key = st.selectbox(
            "Model",
            ["capm", "ff3", "ff5", "ff5_mom"],
            index=3, key="fa_model",
            format_func=lambda k: MODEL_LABELS[k],
            help="CAPM < FF3 < FF5 < FF5+Mom in explanatory power; "
                 "more factors need more data.")
    with fc3:
        fa_roll_win = st.selectbox(
            "Rolling Window (trading days)", [63, 126, 252], index=2,
            key="fa_rollwin",
            help="Trailing window for rolling exposures (past data only).")
    fd1, fd2, fd3 = st.columns(3)
    with fd1:
        fa_start = st.date_input(
            "Start Date", value=fa_min_date,
            min_value=fa_min_date, max_value=fa_max_date, key="fa_start")
    with fd2:
        fa_end = st.date_input(
            "End Date", value=fa_max_date,
            min_value=fa_min_date, max_value=fa_max_date, key="fa_end")
    with fd3:
        if "fa_benchmark" not in st.session_state:
            st.session_state.fa_benchmark = st.session_state.get(
                "bt_benchmark", "SPY")
        st.text_input(
            "Benchmark (comparison only)", key="fa_benchmark",
            help="Any Yahoo Finance ticker, e.g. SPY, QQQ, ^GSPC.")
    st.caption(MODEL_DESCRIPTIONS[fa_model_key])

    # Translate the selected era into base currency up front: factor
    # regressions then describe the investor's base-currency experience.
    # The French factors themselves are never converted (US datasets).
    # "Selected Backtest Strategy" may analyse the Backtest tab's window,
    # so the era covers both windows when that source is chosen.
    try:
        _fa_era_start = pd.Timestamp(fa_start)
        if fa_source == "Selected Backtest Strategy":
            _fa_era_start = min(
                _fa_era_start,
                pd.Timestamp(st.session_state.get("bt_start", fa_start)))
        _fa_need_from = _fa_era_start - pd.Timedelta(days=30)
        _fa_need_to = max(pd.Timestamp(fa_end),
                          pd.Timestamp(st.session_state.get("bt_end", fa_end)))
        fa_full = base_price_panel(
            fa_full.loc[_fa_need_from:_fa_need_to],
            tickers, instruments, ccy.get("fx_hist"), base_ccy)
    except ValueError as e:
        st.error(f"Factor data error: {e}")
        return  # section-local return (was st.stop()): sibling tabs keep rendering

    # ---- Build the analysed wealth series (backtest engine, no Alpaca) ----
    _FA_CAPITAL = 10_000.0
    _fa_freq_map = {"Monthly": "monthly", "Quarterly": "quarterly",
                    "Semi-annual": "semi-annual", "Annual": "annual"}
    fa_values = None
    fa_source_label = ""
    try:
        if fa_source == "Selected Backtest Strategy":
            _bt_specs = []
            if st.session_state.get("bt_s_cwbh"):
                _bt_specs.append(("Current Weights — Retrospective (Buy & Hold)",
                                  "bh", weights))
            if st.session_state.get("bt_s_cwr"):
                _fl = st.session_state.get("bt_freq", "Quarterly")
                _bt_specs.append(
                    (f"Current Weights — Retrospective ({_fl})",
                     "rebal", weights))
            if st.session_state.get("bt_s_ewbh"):
                _bt_specs.append(("Equal Weight — Buy & Hold", "bh",
                                  np.ones(len(tickers)) / len(tickers)))
            if st.session_state.get("bt_s_ewr"):
                _fl = st.session_state.get("bt_freq", "Quarterly")
                _bt_specs.append((f"Equal Weight — {_fl}", "rebal",
                                  np.ones(len(tickers)) / len(tickers)))
            if not _bt_specs:
                st.info("Tick a portfolio strategy in the Backtest tab to "
                        "analyse it here — or pick another return source.")
                return  # section-local return (was st.stop()): sibling tabs keep rendering
            _sel_name, _sel_kind, _sel_w = _bt_specs[0]
            _bt_start = st.session_state.get("bt_start", fa_start)
            _bt_end = st.session_state.get("bt_end", fa_end)
            _bt_cap = float(st.session_state.get("bt_capital", _FA_CAPITAL))
            _bt_cost = float(st.session_state.get("bt_costs", 10.0))
            _bt_freq = _fa_freq_map.get(
                st.session_state.get("bt_freq", "Quarterly"), "quarterly")
            _bt_prices, _bt_notes = prepare_backtest_data(
                fa_full, tickers, pd.Timestamp(_bt_start), pd.Timestamp(_bt_end))
            for _n in _bt_notes:
                st.warning(_n)
            if _sel_kind == "bh":
                _bt_res = backtest_buy_and_hold(
                    _bt_prices, _sel_w, _bt_cap, rf_rate, name=_sel_name)
            else:
                _bt_res = backtest_rebalanced(
                    _bt_prices, _sel_w, _bt_cap, _bt_freq, _bt_cost,
                    rf_rate, name=_sel_name)
            fa_values = _bt_res.values
            fa_source_label = _sel_name
        else:
            _fa_prices, _fa_notes = prepare_backtest_data(
                fa_full, tickers, pd.Timestamp(fa_start), pd.Timestamp(fa_end))
            for _n in _fa_notes:
                st.warning(_n)
            if fa_source == "Current Weights — Retrospective":
                _fa_w = np.asarray(weights, dtype=float)
                fa_source_label = ("Current Weights — Retrospective "
                                   "(Quarterly, frictionless)")
            else:
                _fa_w = np.ones(len(tickers)) / len(tickers)
                fa_source_label = "Equal Weight (Quarterly, frictionless)"
            _fa_res = backtest_rebalanced(
                _fa_prices, _fa_w, _FA_CAPITAL, "quarterly", 0.0,
                rf_rate, name=fa_source_label)
            fa_values = _fa_res.values
    except ValueError as e:
        st.error(f"Return-source setup: {e}")
        return  # section-local return (was st.stop()): sibling tabs keep rendering
    st.caption(f"Analysing **{fa_source_label}** "
               f"({len(fa_values)} trading days).")

    # ---- French factor data (cached; percent → decimal explicitly) ----
    try:
        with st.spinner("Loading French factor data…"):
            _french = load_factor_data()
    except ValueError as e:
        st.error(f"Factor data unavailable: {e}")
        return  # section-local return (was st.stop()): sibling tabs keep rendering
    except Exception:
        st.error("Could not load factor data (network or API error). "
                 "Other tabs are unaffected.")
        return  # section-local return (was st.stop()): sibling tabs keep rendering
    try:
        _factors = to_decimal_returns(
            pd.concat([_french["ff5"], _french["mom"]], axis=1,
                      sort=False).sort_index())
    except ValueError as e:
        st.error(f"Factor data malformed: {e}")
        return  # section-local return (was st.stop()): sibling tabs keep rendering

    # ---- Simple returns + inner-join alignment + excess vs French RF ----
    _needed = MODEL_FACTORS[fa_model_key]
    try:
        _port_simple = simple_returns_from_wealth(fa_values)
        _y_al, _X_al = align_factor_returns(
            _port_simple, _factors[[*_needed, "RF"]])
    except ValueError as e:
        st.error(f"Factor alignment: {e}")
        return  # section-local return (was st.stop()): sibling tabs keep rendering
    _rf_al = _X_al["RF"]
    _XA = _X_al.drop(columns=["RF"])
    _y_excess = _y_al - _rf_al

    try:
        _warn_small = check_min_observations(len(_y_excess), len(_needed))
    except ValueError as e:
        st.error(f"Not enough data: {e}")
        return  # section-local return (was st.stop()): sibling tabs keep rendering
    if _warn_small:
        st.warning(
            f"Only {len(_y_excess)} overlapping observations (< 252) — "
            "treat loadings as tentative.")
    try:
        _res = run_model(fa_model_key, _y_excess, _XA)
    except ValueError as e:
        st.error(f"Regression failed: {e}")
        return  # section-local return (was st.stop()): sibling tabs keep rendering

    st.caption(
        f"Factor sample: {_y_al.index[0].date()} – {_y_al.index[-1].date()} "
        f"({_res.observations:,} observations). Portfolio observations after "
        f"{_y_al.index[-1].date()} are excluded because factor observations "
        "are not yet available."
    )

    # ---- Factor Exposures (2 rows × 3, short values) ----
    st.markdown("---")
    st.markdown("### Factor Exposures")
    _flist = _res.factor_names
    for _i in range(0, len(_flist), 3):
        _cols = st.columns(3)
        for _col, _f in zip(_cols, _flist[_i:_i + 3]):
            with _col:
                st.metric(FACTOR_LABELS[_f], f"{_res.betas[_f]:.2f}",
                          help=f"Excess-return loading on {FACTOR_LABELS[_f]} "
                               f"({_f}).")
                if _f == "Mkt-RF":
                    st.caption(describe_market_beta(_res.betas[_f]))
                else:
                    st.caption(describe_factor_loading(_f, _res.betas[_f]))
    st.caption(
        "Loadings are historical associations from this sample — "
        "descriptive, never prescriptive, and a positive loading is not "
        "inherently “good”.")

    # ---- Model Summary ----
    st.markdown("---")
    st.markdown("### Model Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Historical Alpha (ann.)", fmt_pct(_res.alpha_annualised),
              help="Exact compounding (1+αd)^252−1. Model-implied and "
                   "historical — not expected future alpha.")
    s2.metric("R²", f"{_res.r_squared:.3f}")
    s3.metric("Adjusted R²", f"{_res.adj_r_squared:.3f}")
    s4.metric("Residual Volatility", fmt_pct(_res.residual_volatility),
              help="Annualised idiosyncratic variation vs this model.")
    st.caption(
        f"{_res.observations:,} daily observations · HAC/Newey-West robust "
        f"standard errors (lag {_res.hac_lags}) · Durbin-Watson "
        f"{_res.durbin_watson:.2f} · design condition number "
        f"{_res.condition_number:.1f}. R² estimates how much historical "
        "variation in excess returns the model explains — not a share of "
        "return “earned from” factors.")

    # ---- Regression Detail ----
    st.markdown("---")
    st.markdown("### Regression Detail")
    _rows = [{
        "Factor": "Alpha (daily)",
        "Loading": f"{_res.alpha:.5f}",
        "Robust SE": f"{_res.alpha_se:.5f}",
        "t-stat": f"{_res.alpha_t:.2f}",
        "p-value": format_p_value(_res.alpha_p),
    }]
    for _f in _res.factor_names:
        _p = _res.p_values[_f]
        _rows.append({
            "Factor": FACTOR_LABELS[_f],
            "Loading": f"{_res.betas[_f]:.2f}" + (" *" if _p < 0.05 else ""),
            "Robust SE": f"{_res.std_errors[_f]:.3f}",
            "t-stat": f"{_res.t_stats[_f]:.2f}",
            "p-value": format_p_value(_p),
        })
    st.dataframe(pd.DataFrame(_rows), width="stretch",
                 hide_index=True)
    st.caption("* p < 0.05 under HAC robust errors — statistical significance "
               "only, never proof of a permanent economic exposure. Only "
               "factors in the selected model are shown.")

    # ---- Historical Attribution (arithmetic, model-implied) ----
    st.markdown("---")
    st.markdown("### Historical Attribution")
    try:
        _contrib = factor_attribution(_res, _y_excess, _XA)
    except ValueError as e:
        st.warning(f"Attribution unavailable: {e}")
        _contrib = None
    if _contrib is not None:
        st.plotly_chart(plot_factor_attribution(_contrib),
                        width="stretch")
        st.caption(
            f"Model-implied historical attribution (annualised arithmetic: "
            f"fitted {_contrib['Fitted (linear, ann.)']:.2%} vs realised mean "
            f"{_contrib['Realised mean (ann.)']:.2%}). Additive by construction "
            f"— it decomposes the fitted mean, not compounded wealth, and is "
            f"not an exact causal decomposition.")

    # ---- Rolling Exposure (trailing windows only) ----
    st.markdown("---")
    st.markdown("### Rolling Exposure")
    _roll_factor = st.selectbox(
        "Rolling factor", _res.factor_names,
        index=0, key="fa_rollfactor",
        help="Trailing-window OLS loadings; each point uses only past data.")
    _show_roll_alpha = st.checkbox(
        "Show rolling historical alpha (annualised)", value=False,
        key="fa_rollalpha")
    try:
        _roll = rolling_factor_regression(_y_excess, _XA, int(fa_roll_win))
    except ValueError as e:
        st.warning(f"Rolling exposure unavailable: {e}")
        _roll = None
    if _roll is not None:
        st.plotly_chart(
            plot_rolling_exposure(
                _roll.index, _roll[_roll_factor],
                f"{FACTOR_LABELS[_roll_factor]} ({int(fa_roll_win)}d)"),
            width="stretch")
        if _show_roll_alpha:
            _alpha_ann = (_roll["alpha"] + 1.0) ** 252 - 1.0
            st.plotly_chart(
                plot_rolling_exposure(
                    _roll.index, _alpha_ann,
                    f"Historical Alpha, ann. ({int(fa_roll_win)}d)"),
                width="stretch")
            st.caption("Rolling alpha is annualised by exact compounding; "
                       "short windows make it noisy — interpret with care.")

    # ---- Benchmark comparison (optional, side-by-side) ----
    with st.expander("Portfolio vs Benchmark loadings"):
        try:
            _fa_bench_sym = validate_ticker_symbol(
                st.session_state.fa_benchmark)
        except ValueError as e:
            st.error(f"Invalid benchmark ticker: {e}")
            _fa_bench_sym = None
        if _fa_bench_sym is not None:
            try:
                _bench_native = load_benchmark_data(_fa_bench_sym, "max")
                _bench_native = _bench_native.loc[
                    _fa_need_from:fa_values.index[-1]]
                _bench_full = base_benchmark_series(
                    _bench_native, _fa_bench_sym, base_ccy, ccy.get("fx_hist"))
            except ValueError as e:
                st.warning(f"Benchmark unavailable: {e}")
                _bench_full = None
            except Exception:
                st.warning("Could not load benchmark data.")
                _bench_full = None
            if _bench_full is not None:
                try:
                    _bench_res = backtest_benchmark(
                        _bench_full, fa_values.index, _FA_CAPITAL, rf_rate,
                        name=_fa_bench_sym)
                    _bench_simple = simple_returns_from_wealth(
                        _bench_res.values)
                    _by, _bX = align_factor_returns(
                        _bench_simple, _factors[[*_needed, "RF"]])
                    _b_excess = _by - _bX["RF"]
                    _bX = _bX.drop(columns=["RF"])
                    check_min_observations(len(_b_excess), len(_needed))
                    _bres = run_model(fa_model_key, _b_excess, _bX)
                except ValueError as e:
                    st.warning(f"Benchmark regression unavailable: {e}")
                    _bres = None
                if _bres is not None:
                    _comp = pd.DataFrame({
                        "Factor": [FACTOR_LABELS[_f] for _f in _needed],
                        "Portfolio": [f"{_res.betas[_f]:.2f}"
                                      for _f in _needed],
                        _fa_bench_sym: [f"{_bres.betas[_f]:.2f}"
                                        for _f in _needed],
                    })
                    st.dataframe(_comp, width="stretch",
                                 hide_index=True)
                    st.caption(
                        "Same model, same alignment rules — useful for "
                        "asking whether the portfolio leans more "
                        "growth/momentum/small-cap than "
                        f"{_fa_bench_sym}. Descriptive only.")

    # ---- Methodology ----
    with st.expander("Methodology"):
        _md = "\n".join(
            f"**{MODEL_LABELS[k]}** — {MODEL_DESCRIPTIONS[k]}"
            for k in ["capm", "ff3", "ff5", "ff5_mom"])
        st.markdown(
            f"""
            {_md}

            **Alpha** — intercept: mean excess return left unexplained by the
            factors. Historical and model-implied; annualised by exact
            compounding. Never a forecast.
            **Beta / loadings** — sensitivity to each factor; never annualised.
            **R² / Adjusted R²** — share of historical excess-return variation
            explained by the model.
            **t-stat / p-value** — HAC-robust significance; p < 0.05 is
            statistical evidence in-sample, not proof of permanence.
            **Risk-free rate** — the French daily RF series aligned to the
            factor dates (the app-level rate is used everywhere else).
            **HAC/Newey-West** — robust covariance, lag 5 trading days.
            **Attribution** — arithmetic (β × mean factor × 252); sums to the
            fitted mean, not to compounded wealth.
            **Scope** — US Fama-French factors; US-listed foreign firms
            without a suffix are undetectable and not specially handled.
            **Data** — French library percent units converted to decimals;
            no Alpaca, TradingView or live data enters factor analysis.
            """
        )

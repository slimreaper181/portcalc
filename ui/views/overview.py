"""views.overview — Overview tab (rendered by the app shell).

Extracted from the application shell; analytics live in
``analytics/``, fetching in ``data/``. Receives the shared
context dict and renders into the active tab.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from analytics.currency import (
    currency_symbol,
    freshness_report,
    fx_rate_between,
    fx_shock_portfolio,
    quote_age_days,
)
from analytics.positions import build_analysis_snapshot, format_money, summarize_portfolio_csv
from analytics.returns import cumulative_growth_from_log_returns
from data.pipeline import (
    build_core_analytics,
    build_currency_layer,
    build_position_valuations,
    load_alpaca_trades,
    load_current_fx,
    load_market_data,
    resolve_valuation_prices,
)
from ui.formatting import fmt_pct, fmt_usd, style_matrix
from ui.theme import CHART_THEME


def render(ctx):
    """Render the Overview tab from the shared context."""
    alpaca_status = ctx["alpaca_status"]
    base_ccy = ctx["base_ccy"]
    ccy = ctx["ccy"]
    core = ctx["core"]
    corr = ctx["corr"]
    cov = ctx["cov"]
    current_prices = ctx["current_prices"]
    instruments = ctx["instruments"]
    port_return = ctx["port_return"]
    port_sharpe = ctx["port_sharpe"]
    port_vol = ctx["port_vol"]
    portfolio_value = ctx["portfolio_value"]
    pos_by_ticker = ctx["pos_by_ticker"]
    positions = ctx["positions"]
    price_view = ctx["price_view"]
    prices = ctx["prices"]
    rc = ctx["rc"]
    rets = ctx["rets"]
    rf_rate = ctx["rf_rate"]
    tickers = ctx["tickers"]
    val_agg = ctx["val_agg"]
    val_warnings = ctx["val_warnings"]
    valuation_prices = ctx["valuation_prices"]
    valuations = ctx["valuations"]
    weights = ctx["weights"]
    mu = ctx["mu"]
    port_daily_returns = ctx["port_daily_returns"]
    val_by_ticker = ctx["val_by_ticker"]
    native_panel = ctx["native_panel"]

    st.markdown("## Portfolio Overview")

    # History period selector lives here
    hist_col, _ = st.columns([2, 8])
    with hist_col:
        hist_period = st.selectbox(
            "History period",
            ["6mo", "1y", "2y", "5y"],
            index=2,
            key="hist_period_tab1",
        )

    if hist_period != "2y":
        with st.spinner("Fetching data…"):
            try:
                prices, current_prices, rf_rate = load_market_data(tickers, hist_period)
                valuation_prices, price_view, alpaca_status = resolve_valuation_prices(
                    tickers, current_prices)
                ccy = build_currency_layer(
                    tickers, prices, valuation_prices, base_ccy)
                native_panel = ccy["native_panel"]
                instruments = ccy["instruments"]
                core = build_core_analytics(tickers, ccy["base_panel"],
                                            ccy["base_current"], rf_rate)
                _native_current_norm = {
                    t: float(valuation_prices.get(t, float("nan")))
                    * ccy["instruments"][t].quote_scale for t in tickers}
                positions, valuations, val_agg, val_warnings = \
                    build_position_valuations(
                        tickers, ccy["instruments"], _native_current_norm,
                        price_view, ccy["fx_hist"], ccy["fx_now"], base_ccy)
                val_by_ticker = {v.ticker: v for v in valuations}
                pos_by_ticker = positions
            except ValueError as e:
                st.error(f"Data error: {e}")
                return  # section-local return (was st.stop()): sibling tabs keep rendering
            except Exception:
                st.error("Could not load market data for the selected period.")
                return  # section-local return (was st.stop()): sibling tabs keep rendering
            prices = core["prices"]
            current_prices = core["current_prices"]
            rf_rate = core["rf_rate"]
            weights = core["weights"]
            portfolio_value = core["portfolio_value"]
            rets = core["rets"]
            mu = core["mu"]
            cov = core["cov"]
            corr = core["corr"]
            port_return = core["port_return"]
            port_vol = core["port_vol"]
            port_sharpe = core["port_sharpe"]
            rc = core["rc"]
            port_daily_returns = core["port_daily_returns"]

    for _w in val_warnings:
        st.warning(_w)

    # Headline portfolio value on its own row: full width, never truncated.
    st.metric(f"Portfolio Value ({base_ccy})", fmt_usd(portfolio_value),
              help="Total current market value across all positions, "
              f"in {base_ccy}.")
    _fx_ts_txt = "-"
    try:
        if ccy.get("fx_timestamp") is not None:
            _fx_ts_txt = ccy["fx_timestamp"].strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass
    st.caption(
        f"Historical prices through {prices.index[-1].date()} · "
        f"FX through {_fx_ts_txt} · FX source: Yahoo Finance.")

    # Small metrics (percentages / ratios): up to 4-5 per row is fine.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected Return", fmt_pct(port_return))
    c2.metric("Volatility (σ)", fmt_pct(port_vol))
    c3.metric("Sharpe Ratio", f"{port_sharpe:.2f}")
    c4.metric("Risk-Free Rate", fmt_pct(rf_rate))
    _pnl_tot = val_agg.get("total_pnl")
    _dd_stat = None
    try:
        import math as _math

        from analytics.performance import drawdown_stats as _dd_stats
        _wealth = port_daily_returns.copy()
        _wealth = _wealth[np.isfinite(_wealth.values)]
        if len(_wealth) >= 2:
            _dd_stat = _dd_stats((_wealth.cumsum().apply(_math.exp)))
    except Exception:
        _dd_stat = None
    k1, k2 = st.columns(2)
    k1.metric(
        f"Unrealised P&L ({base_ccy})",
        (f"{_pnl_tot:+,.2f}" if _pnl_tot is not None
         and np.isfinite(_pnl_tot) else "—"),
        help="Current market value minus purchase cost (price/FX P&L).")
    k2.metric(
        "Max Drawdown",
        fmt_pct(_dd_stat["max_drawdown"]) if _dd_stat else "—",
        help="Peak-to-trough decline on the current history window.")

    # Deterministic portfolio insights (analytics, never advice).
    try:
        from ui.insights import build_insights
        _wts = val_agg.get("weights", {})
        _wvals = np.array([float(_wts.get(t, 0.0)) for t in tickers])
        _rc_arr = np.asarray(rc, dtype=float)
        _rc_top = int(np.argmax(_rc_arr)) if len(_rc_arr) else 0
        _facts = {
            "n_assets": len(tickers),
            "largest_weight": float(_wvals.max()) if len(_wvals) else None,
            "largest_ticker": tickers[int(np.argmax(_wvals))] if _wvals.size else None,
            "top3_weight": float(np.sort(_wvals)[-3:].sum())
            if len(_wvals) >= 3 else None,
            "max_rc_ticker": tickers[_rc_top] if tickers else None,
            "max_rc_share": float(_rc_arr[_rc_top]) if len(_rc_arr) else None,
            "max_rc_weight": float(_wvals[_rc_top]) if len(_wvals) else None,
            "max_drawdown": (float(_dd_stat["max_drawdown"])
                             if _dd_stat else None),
        }
        try:
            from analytics.allocation import (
                effective_number_of_holdings as _eff_n)
            _facts["effective_holdings"] = float(_eff_n(_wvals))
        except Exception:
            pass
        if len(tickers) > 1:
            try:
                _cm = corr.values.copy()
                np.fill_diagonal(_cm, -np.inf)
                _ij = np.unravel_index(np.argmax(_cm), _cm.shape)
                _facts["max_corr_pair"] = (
                    f"{tickers[_ij[0]]}/{tickers[_ij[1]]}")
                _facts["max_corr_value"] = float(_cm[_ij])
            except Exception:
                pass
        _insights = build_insights(_facts)
    except Exception:
        _insights = []
    if _insights:
        st.markdown("### Portfolio Insights")
        for _ins in _insights:
            _icon = {"notable": "🔶", "watch": "🔸"}.get(_ins.severity, "ℹ️")
            st.markdown(f"{_icon} {_ins.text}")
        st.caption("Deterministic observations from current analytics — "
                   "not investment advice.")

    st.markdown("---")

    # Positions table from the single central valuation (base currency).
    _sym = currency_symbol(base_ccy)
    pos_head, pos_refresh = st.columns([4, 1])
    with pos_head:
        st.markdown("### Positions")
    with pos_refresh:
        if st.button("↻ Refresh live prices", key="refresh_live",
                     help="Clear the short-TTL live-price/FX caches and fetch "
                          "fresh Alpaca trades and FX."):
            load_alpaca_trades.clear()
            try:
                load_current_fx.clear()
            except Exception:
                pass
            st.rerun()
    rows = []
    for _v in valuations:
        _p = pos_by_ticker.get(_v.ticker, {})
        _buy = _p.purchase_date.isoformat() if getattr(
            _p, "purchase_date", None) else "—"
        _buy_px = (format_money(_p.purchase_price_native,
                                _v.native_currency)
                   if getattr(_p, "purchase_price_native", None) else "—")
        _pnl = _v.unrealised_pnl if _v.unrealised_pnl is not None else float("nan")
        _pnlp = (_v.unrealised_pnl_pct * 100 if _v.unrealised_pnl_pct is not None
                 else float("nan"))
        _cost = (_v.base_cost_basis
                 if _v.base_cost_basis is not None else float("nan"))
        rows.append({
            "Ticker": _v.ticker,
            "Shares": _v.shares,
            "Buy Date": _buy,
            "Buy Px": _buy_px,
            "Now Px": format_money(_v.native_current_price,
                                   _v.native_currency),
            "Source": _v.price_source,
            "Ccy": _v.native_currency,
            f"Cost ({base_ccy})": _cost,
            f"Value ({base_ccy})": _v.base_market_value,
            f"P&L ({base_ccy})": _pnl,
            "P&L (%)": _pnlp,
            "Weight": val_agg["weights"].get(_v.ticker, 0.0),
        })
    pos_df = pd.DataFrame(rows)
    st.dataframe(
        pos_df.style
        .format({
            f"Cost ({base_ccy})": _sym + "{:,.2f}",
            f"Value ({base_ccy})": _sym + "{:,.2f}",
            f"P&L ({base_ccy})": _sym + "{:+,.2f}",
            "P&L (%)": "{:+.2f}%",
            "Weight": "{:.1%}",
        }, na_rep="—")
        .map(lambda v: "color: #3fb950" if isinstance(v, (int, float)) and v > 0
                  else ("color: #f85149" if isinstance(v, (int, float)) and v < 0 else ""),
                  subset=[f"P&L ({base_ccy})", "P&L (%)"]),
        width="stretch",
    )
    st.caption(
        "Unrealised P&L = current market value − purchase cost (Price/FX "
        "P&L). Cash dividends received are excluded — see Estimated "
        "Dividend Income in Security Detail.")
    if alpaca_status is None:
        st.caption("Live US equity prices: Alpaca IEX. Historical portfolio "
                   "analytics: Yahoo Finance. TradingView: charting only.")
    elif alpaca_status == "disabled":
        st.caption("Alpaca credentials not configured — current prices use "
                   "Yahoo Finance. Historical analytics: Yahoo Finance.")
    else:
        st.warning(f"Live prices unavailable ({alpaca_status}) — current "
                   "prices use Yahoo Finance. Historical analytics unaffected.")

    st.markdown("---")

    # ---- Quote-currency exposure (current base-currency market values) ----
    st.markdown("### Quote Currency Exposure")
    _expo: dict[str, float] = {}
    for _v in valuations:
        _expo[_v.native_currency] = _expo.get(_v.native_currency, 0.0) \
            + _v.base_market_value
    _expo_tot = sum(_expo.values())
    _e1, _e2 = st.columns([1, 1])
    with _e1:
        fig_fx = go.Figure(go.Pie(
            labels=list(_expo),
            values=[_expo[k] / _expo_tot for k in _expo],
            hole=0.45,
            marker=dict(colors=px.colors.qualitative.Plotly),
            textinfo="label+percent",
            textfont=dict(family="IBM Plex Mono", size=13),
        ))
        fig_fx.update_layout(**CHART_THEME, title="Exposure by Quotation Currency")
        st.plotly_chart(fig_fx, width="stretch")
    with _e2:
        for _ccy, _val in sorted(_expo.items(), key=lambda kv: -kv[1]):
            st.metric(f"{_ccy} exposure",
                      f"{_val / _expo_tot:.1%} ({fmt_usd(_val)})")
    st.caption("Quote Currency Exposure: share of current portfolio value "
               "economically exposed to each quotation currency — not "
               "corporate revenue exposure.")

    with st.expander("FX Shock — prices held constant"):
        st.caption(
            "First-order static sensitivity: local security prices held "
            "constant while each non-base currency moves. Illustration, "
            "not a forecast.")
        _shocks = {}
        _scols = st.columns(max(1, len([c for c in _expo if c != base_ccy]) or 1))
        _i = 0
        for _ccy in sorted(_expo):
            if _ccy == base_ccy:
                continue
            with _scols[_i % len(_scols)]:
                _shocks[_ccy] = st.number_input(
                    f"{_ccy} shock (%)", value=0.0, step=1.0,
                    key=f"fx_shock_{_ccy}") / 100.0
            _i += 1
        if any(_shocks.values()):
            try:
                _sr = fx_shock_portfolio(
                    {k: v for k, v in _expo.items()}, _shocks, base_ccy)
                s1, s2 = st.columns(2)
                s1.metric("Shocked Value", fmt_usd(_sr["shocked_total"]))
                s2.metric("Change", f"{_sr['pct_change']:+.2%}")
            except ValueError as e:
                st.error(str(e))

    # ---- Data quality, freshness, methodology ----
    with st.expander("Data Quality & Freshness"):
        _now = datetime.now(timezone.utc)
        _q_ages, _items = {}, []
        for _v in valuations:
            _age = quote_age_days(_v.current_price_timestamp, _now)
            _q_ages[_v.ticker] = _age
            _ts = (_v.current_price_timestamp.strftime("%Y-%m-%d %H:%M UTC")
                   if _v.current_price_timestamp else "unknown")
            _items.append({
                "Item": f"{_v.ticker} price ({_v.price_source})",
                "Status": "ok" if _age is not None and _age <= 5 else
                          ("stale" if _age is not None else "unknown"),
                "As of": _ts,
            })
        _fx_age = quote_age_days(ccy.get("fx_timestamp"), _now)
        _items.append({
            "Item": "Current FX (Yahoo Finance)",
            "Status": "ok" if _fx_age is not None and _fx_age <= 5 else
                      ("stale" if _fx_age is not None else "unknown"),
            "As of": (ccy["fx_timestamp"].strftime("%Y-%m-%d %H:%M UTC")
                      if ccy.get("fx_timestamp") else "unknown"),
        })
        _n_purch = sum(1 for t in tickers
                       if pos_by_ticker.get(t) is not None
                       and getattr(pos_by_ticker[t], "purchase_price_native",
                                   None) is not None)
        _items.append({"Item": "Purchase prices",
                       "Status": f"{_n_purch} / {len(tickers)}",
                       "As of": "position entry"})
        _items.append({"Item": "Native currencies",
                       "Status": f"{len(instruments)} / {len(tickers)}",
                       "As of": "metadata lookup"})
        st.dataframe(pd.DataFrame(_items), width="stretch",
                     hide_index=True)
        for _w in freshness_report(_q_ages, _fx_age):
            st.warning(_w)
        st.caption(f"Last refreshed: {_now.strftime('%H:%M:%S UTC')} "
                   "(current quotes/FX; historical datasets have their own "
                   "as-of dates shown per section).")

    with st.expander("FX Conversion Details"):
        _fxrows = []
        for _v in valuations:
            try:
                _r = fx_rate_between(_v.native_currency, base_ccy,
                                     ccy["fx_now"])
                _rl = f"{_r:.4f}"
            except ValueError:
                _rl = "n/a"
            _fxrows.append({
                "Ticker": _v.ticker,
                "Pair": f"{_v.native_currency} → {base_ccy}",
                "Rate": _rl,
                "Basis": "Current" if _v.native_currency != base_ccy
                         else "Identity (no FX)",
            })
        st.dataframe(pd.DataFrame(_fxrows), width="stretch",
                     hide_index=True)

    with st.expander("Methodology & Data Sources"):
        st.markdown(
            f"""
            **Current US prices** — Alpaca IEX (fallback: Yahoo Finance).
            **Fallback prices** — Yahoo Finance.
            **Historical prices** — Yahoo Finance (adjusted closes).
            **Current FX** — Yahoo Finance (short TTL).
            **Historical FX** — Yahoo Finance (long TTL).
            **Factor data** — Kenneth R. French Data Library.
            **Charts** — TradingView (native quotation, display only).

            Prices remain in each security's native market currency.
            Current portfolio values are translated using current FX rates.
            Historical analytics and backtests use date-matched historical
            FX, so currency movements are included in base-currency
            investment returns.
            """
        )

    # ---- Portfolio data export / snapshot ----
    st.markdown("### Portfolio Data")
    _d1, _d2 = st.columns(2)
    with _d1:
        _csv_df = summarize_portfolio_csv(
            valuations, {t: pos_by_ticker[t] for t in tickers
                         if t in pos_by_ticker})
        st.download_button(
            "Download Portfolio Summary (CSV)",
            data=_csv_df.to_csv(index=False), file_name="portcalc-summary.csv",
            mime="text/csv", key="dl_csv")
    with _d2:
        _snap = build_analysis_snapshot(
            tickers, {t: pos_by_ticker[t].shares for t in tickers
                      if t in pos_by_ticker},
            {t: (pos_by_ticker[t].purchase_date.isoformat()
                 if pos_by_ticker[t].purchase_date else None)
             for t in tickers if t in pos_by_ticker},
            base_ccy, "2y",
            st.session_state.get("bt_benchmark", "SPY"),
            float(rf_rate), None, 0.0, 1.0, {}, None, "Yahoo Finance",
            prices.index[-1].date().isoformat())
        st.download_button(
            "Download Analysis Snapshot (JSON)",
            data=json.dumps(_snap, indent=2),
            file_name=f"{_snap['run_id']}.json", mime="application/json",
            key="dl_snap")
        st.caption(f"Run ID {_snap['run_id']} · inputs only, no secrets.")

    st.markdown("---")

    col_weights, col_rc = st.columns(2)

    with col_weights:
        st.markdown("### Weight Allocation")
        fig_pie = go.Figure(go.Pie(
            labels=tickers,
            values=weights,
            hole=0.45,
            marker=dict(colors=px.colors.qualitative.Plotly),
            textinfo="label+percent",
            textfont=dict(family="IBM Plex Mono", size=13),
        ))
        fig_pie.update_layout(**CHART_THEME, title="Portfolio Weights")
        st.plotly_chart(fig_pie, width="stretch")
        _wv = np.asarray(weights, dtype=float)
        _order = np.argsort(_wv)[::-1]
        _top3 = float(_wv[_order[:3]].sum()) if len(_wv) >= 3 else float(_wv.sum())
        try:
            from analytics.allocation import (
                effective_number_of_holdings as _eff_n2)
            _eff_txt = f"Effective holdings: {_eff_n2(_wv):.1f}. "
        except Exception:
            _eff_txt = ""
        st.caption(
            f"Largest: {tickers[_order[0]]} {float(_wv[_order[0]]):.1%}. "
            f"Top 3: {_top3:.1%}. {_eff_txt}"
            "Concentration raises single-name risk.")

    with col_rc:
        st.markdown("### Risk Contributions")
        fig_rc = go.Figure(go.Bar(
            x=tickers, y=rc,
            marker_color="#58a6ff",
            text=[f"{v:.1%}" for v in rc],
            textposition="outside",
        ))
        fig_rc.update_layout(
            **CHART_THEME,
            title="Risk Contribution per Asset",
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig_rc, width="stretch")

    st.markdown("---")

    col_cov, col_corr = st.columns(2)

    with col_cov:
        st.markdown("### Covariance Matrix")
        st.dataframe(style_matrix(cov, ".5f"), width="stretch")

    with col_corr:
        st.markdown("### Correlation Matrix")
        st.dataframe(style_matrix(corr, ".4f"), width="stretch")

    # Cumulative returns chart (log returns -> exp(cumsum), NOT (1+r).cumprod())
    st.markdown("---")
    st.markdown("### Cumulative Returns")

    cum_ret = cumulative_growth_from_log_returns(rets)
    fig_cum = go.Figure()
    for col in tickers:
        fig_cum.add_trace(go.Scatter(
            x=cum_ret.index, y=cum_ret[col],
            mode="lines", name=col,
        ))
    fig_cum.update_layout(
        **CHART_THEME,
        title="Cumulative Return (base = 1, from log returns)",
        xaxis_title="Date", yaxis_title=f"Growth of {currency_symbol(base_ccy)}1",
    )
    st.plotly_chart(fig_cum, width="stretch")

    # Propagate (possibly rebuilt) pipeline state to later tabs.
    return {
        "base_ccy": base_ccy,
        "ccy": ccy,
        "prices": prices,
        "current_prices": current_prices,
        "rf_rate": rf_rate,
        "weights": weights,
        "portfolio_value": portfolio_value,
        "rets": rets,
        "mu": mu,
        "cov": cov,
        "corr": corr,
        "port_return": port_return,
        "port_vol": port_vol,
        "port_sharpe": port_sharpe,
        "port_daily_returns": port_daily_returns,
        "rc": rc,
        "positions": positions,
        "valuations": valuations,
        "val_agg": val_agg,
        "val_warnings": val_warnings,
        "val_by_ticker": val_by_ticker,
        "pos_by_ticker": pos_by_ticker,
        "valuation_prices": valuation_prices,
        "price_view": price_view,
        "alpaca_status": alpaca_status,
        "instruments": instruments,
        "native_panel": native_panel,
        "core": core,
        "core": core,
    }

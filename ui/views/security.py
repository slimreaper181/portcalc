"""views.security — Security Detail tab (rendered by the app shell).

Extracted from the application shell; analytics live in
``analytics/``, fetching in ``data/``. Receives the shared
context dict and renders into the active tab.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from analytics.currency import decompose_pnl, fx_rate_between, rate_on_date
from analytics.performance import log_returns_from_prices
from analytics.positions import estimate_dividend_income, format_money
from analytics.security import period_returns, security_summary
from analytics.validation import validate_ticker_symbol
from components.tradingview import render_tradingview_chart, resolve_tradingview_symbol
from data.pipeline import (
    base_benchmark_series,
    load_benchmark_data,
    load_dividends_history,
    load_exchange_hint,
)
from ui.formatting import fmt_pct, fmt_usd, format_trade_time
from ui.theme import CHART_THEME


def render(ctx):
    """Render the Security Detail tab from the shared context."""
    base_ccy = ctx["base_ccy"]
    ccy = ctx["ccy"]
    cov = ctx["cov"]
    native_panel = ctx["native_panel"]
    port_daily_returns = ctx["port_daily_returns"]
    pos_by_ticker = ctx["pos_by_ticker"]
    prices = ctx["prices"]
    rets = ctx["rets"]
    rf_rate = ctx["rf_rate"]
    tickers = ctx["tickers"]
    val_agg = ctx["val_agg"]
    val_by_ticker = ctx["val_by_ticker"]
    weights = ctx["weights"]

    st.markdown("## Security Detail")

    if st.session_state.get("security_ticker") not in tickers:
        st.session_state.security_ticker = tickers[0]
    sel = st.selectbox(
        "Security", tickers, key="security_ticker",
        help="Holdings come from your current portfolio. Analytics below "
             "follow this selection.",
    )
    sel_idx = tickers.index(sel)

    # ---- Top security summary from the single central valuation ----
    _v = val_by_ticker.get(sel)
    _pos = pos_by_ticker.get(sel)
    if _v is None or _pos is None:
        st.error(f"Valuation unavailable for {sel}.")
        return  # section-local return (was st.stop()): sibling tabs keep rendering
    _per_share_base = _v.base_market_value / _v.shares if _v.shares else float("nan")

    st.markdown(f"### {sel}")
    # Native quotation vs base-currency value stay visually distinct.
    n1, n2 = st.columns(2)
    n1.metric(f"Latest Native Price ({_v.native_currency})",
              format_money(_v.native_current_price, _v.native_currency),
              help=f"Source: {_v.price_source}. "
                   f"{_v.native_currency} quotation (×{_pos.quote_scale:g} "
                   f"{_pos.quote_unit or _v.native_currency}).")
    n2.metric(f"Price in {base_ccy}", fmt_usd(_per_share_base),
              help="Same native quote translated at current FX.")
    h1, h2 = st.columns(2)
    h1.metric("Position Value", fmt_usd(_v.base_market_value))
    h2.metric("Portfolio Weight", fmt_pct(val_agg["weights"].get(sel, 0.0)),
              help="Share of total portfolio market value.")
    st.caption(
        f"{_v.shares:g} shares · {_v.price_source}"
        + (f" · last trade {format_trade_time(_v.current_price_timestamp)}"
           if _v.current_price_timestamp else " · last close") + ".")
    # Purchase / cost basis block (never invented).
    if _v.native_cost_basis is not None and _v.base_cost_basis is not None:
        _src_label = {"estimate": "Historical Close Estimate",
                      "manual": "Manual Execution Price",
                      "legacy": "Legacy Cost Basis"}.get(
                          _v.cost_source or "", _v.cost_source or "Unknown")
        p1, p2 = st.columns(2)
        p1.metric("Purchase Date",
                  _pos.purchase_date.isoformat() if _pos.purchase_date
                  else "unknown")
        p2.metric("Estimated Purchase Price",
                  format_money(_pos.purchase_price_native,
                               _v.native_currency))
        st.caption(
            f"Price Source: {_src_label} · Cost Basis: "
            f"{format_money(_v.native_cost_basis, _v.native_currency)} → "
            f"{fmt_usd(_v.base_cost_basis)} at purchase-date FX.")
        st.metric("Unrealised Price/FX P&L",
                  fmt_usd(_v.unrealised_pnl)
                  if _v.unrealised_pnl is not None else "n/a",
                  help="Current value minus purchase cost; cash dividends "
                       "excluded.")
        if _v.unrealised_pnl_pct is not None:
            st.caption(f"({_v.unrealised_pnl_pct:+.2%} on cost; cash dividends "
                       f"excluded — see below.)")
        # Exact multiplicative decomposition + estimated dividends.
        try:
            _local_ret = (_v.native_current_price
                          - _pos.purchase_price_native) \
                / _pos.purchase_price_native
            _purch_fx = rate_on_date(
                ccy.get("fx_hist"), _v.native_currency, base_ccy,
                _pos.purchase_date) if _pos.purchase_date else None
            _cur_fx = fx_rate_between(
                _v.native_currency, base_ccy, ccy["fx_now"]) \
                if ccy.get("fx_now") else None
            if (_purch_fx is not None and _cur_fx
                    and _v.base_cost_basis):
                _fx_ret = (_cur_fx - _purch_fx) / _purch_fx
                _dec = decompose_pnl(_v.base_cost_basis, _local_ret, _fx_ret)
                st.caption(
                    f"Local price effect {fmt_usd(_dec['local_effect'])} · "
                    f"FX effect {fmt_usd(_dec['fx_effect'])} · interaction "
                    f"{fmt_usd(_dec['interaction'])} (exact).")
        except (ValueError, ZeroDivisionError, TypeError):
            pass
        if _pos.purchase_date is not None:
            try:
                _divs = load_dividends_history(sel)
                _div_tot, _div_ev, _div_skip = estimate_dividend_income(
                    _divs, _pos.purchase_date, _v.shares,
                    lambda d: rate_on_date(
                        ccy.get("fx_hist"), _v.native_currency, base_ccy, d))
                if _div_tot > 0 or _div_ev:
                    st.metric("Estimated Dividend Income", fmt_usd(_div_tot),
                              help="Cash dividends with ex-date after the "
                                   "recorded purchase date, FX-converted "
                                   "per payment date. Estimate only.")
                    if _v.unrealised_pnl is not None:
                        st.metric("Estimated Total P&L",
                                  fmt_usd(_v.unrealised_pnl + _div_tot),
                                  help="Unrealised Price/FX P&L plus "
                                       "estimated dividends received.")
                if _div_skip:
                    st.caption(f"{_div_skip} dividend payment(s) skipped "
                               f"(no FX on payment date).")
            except ValueError as e:
                st.caption(f"Dividend estimate unavailable: {e}")
    else:
        st.caption("No cost basis recorded — add a purchase date (or legacy "
                   "cost) to enable P&L.")

    st.markdown("---")

    # ---- Holding analytics (project's own data pipeline only) ----
    st.markdown("### Holding Analytics")
    sec_rets = rets[sel].dropna()

    # Benchmark leg for beta: reuse the Performance tab benchmark (SPY
    # default) through the same cached loader — no extra request.
    bench_period = st.session_state.get("hist_period_tab1", "2y")
    bench_rets_sec = None
    sec_bench_sym = None
    try:
        sec_bench_sym = validate_ticker_symbol(
            st.session_state.get("bench_ticker", "SPY"))
    except ValueError:
        sec_bench_sym = None
    if sec_bench_sym is not None and sec_bench_sym != sel:
        try:
            _bp = load_benchmark_data(sec_bench_sym, bench_period)
            _bp = base_benchmark_series(
                _bp, sec_bench_sym, base_ccy, ccy.get("fx_hist"))
            bench_rets_sec = log_returns_from_prices(_bp)
        except Exception:
            bench_rets_sec = None  # beta shows as unavailable, tab continues

    try:
        sec = security_summary(
            sec_rets, weights, cov.values, sel_idx, rf_rate,
            portfolio_log_returns=port_daily_returns,
            benchmark_log_returns=bench_rets_sec,
        )
    except ValueError as e:
        st.error(f"Holding analytics unavailable: {e}")
        sec = None

    if sec is not None:
        a1, a2, a3 = st.columns(3)
        a1.metric("Annualised Return", fmt_pct(sec["annualised_return"]),
                  help="Mean daily log return × 252, this holding only.")
        a2.metric("Annualised Volatility", fmt_pct(sec["annualised_volatility"]),
                  help="Daily std × √252, this holding only.")
        a3.metric("Sharpe Ratio", f"{sec['sharpe']:.2f}",
                  help="Holding excess return per unit of its own volatility.")
        b1, b2, b3 = st.columns(3)
        with b1:
            if sec["beta_vs_benchmark"] is not None:
                st.metric(
                    f"Beta vs {sec_bench_sym}", f"{sec['beta_vs_benchmark']:.2f}",
                    help="Historical sensitivity to the benchmark. "
                         "Association, not causality.")
            else:
                st.metric("Beta", "n/a")
                st.caption(sec.get("beta_error")
                           or "Benchmark unavailable for beta.")
        with b2:
            st.metric("Contribution to Portfolio Risk",
                      fmt_pct(sec["risk_contribution"]),
                      help="This holding's share of portfolio volatility "
                           "(same model as the Overview risk breakdown).")
        with b3:
            if sec["correlation_to_portfolio"] is not None:
                st.metric("Correlation to Portfolio",
                          f"{sec['correlation_to_portfolio']:.2f}",
                          help="Co-movement with the portfolio return series. "
                               "Association, not causality.")
                st.caption(sec["correlation_note"])
            else:
                st.metric("Correlation to Portfolio", "n/a")
                st.caption(sec.get("correlation_error")
                           or "Correlation unavailable.")

    st.markdown("---")

    # ---- TradingView advanced chart (visual analysis only) ----
    # Full-width terminal chart: metrics above, performance below.
    st.markdown("### Interactive Chart")
    size_col, _ = st.columns([1, 3])
    with size_col:
        chart_size = st.selectbox(
            "Chart Size", ["Standard", "Large"], index=1, key="tv_chart_size",
            help="Standard renders a ~650px chart, Large ~900px.",
        )
    # Actual rendered chart heights (attribution line added on top).
    tv_height = 900 if chart_size == "Large" else 650
    hints = {}
    for t in tickers:
        try:
            hints[t] = load_exchange_hint(t)
        except Exception:
            hints[t] = None
    watchlist = []
    for t in tickers:
        try:
            watchlist.append(resolve_tradingview_symbol(t, hints.get(t)).symbol)
        except ValueError:
            continue

    override_key = f"tv_override_{sel}"
    with st.expander("TradingView symbol override (advanced)"):
        st.text_input(
            "TradingView symbol", key=override_key, placeholder="e.g. NASDAQ:AAPL",
            help="Chart display only. Never affects portfolio data or analytics.",
        )
        st.caption(f"Yahoo ticker: {sel}")
    override_val = (st.session_state.get(override_key) or "").strip() or None

    tv_failed = False
    try:
        resolved = resolve_tradingview_symbol(sel, hints.get(sel), override_val)
    except ValueError as e:
        tv_failed = True
        st.warning(f"TradingView chart unavailable for {sel}: {e} "
                   "Portfolio Calc analytics above are unaffected; you can "
                   "change the symbol inside the widget once it loads.")
        resolved = None

    if not tv_failed and resolved is not None:
        if not resolved.resolved:
            st.caption(f"ℹ️ {resolved.note}")
        try:
            render_tradingview_chart(resolved.symbol, watchlist, height=tv_height)
        except Exception:
            st.warning(
                "The TradingView widget could not display this symbol. "
                "Portfolio Calc analytics above are unaffected — try the "
                "widget's symbol search or the override above.")
        st.caption(
            "Streamlit selector drives Portfolio Calc analytics; the widget "
            "watchlist switches only the embedded chart. Chart data is "
            "TradingView's own and is never used in calculations. "
            "TradingView chart shown in the security's native market "
            "quotation.")

    # ---- Reproducible internal price chart (native market quotation) ----
    with st.expander("Portfolio data chart (reproducible)"):
        st.caption("Plotted from Portfolio Calc's own yfinance history in "
                   "the security's native market quotation — "
                   "the reproducible application data, unlike the widget above.")
        _nat_ccy = val_by_ticker[sel].native_currency
        fig_sec = go.Figure()
        fig_sec.add_trace(go.Scatter(
            x=native_panel.index, y=native_panel[sel],
            mode="lines", name=sel, line=dict(color="#58a6ff", width=2),
        ))
        fig_sec.update_layout(**CHART_THEME, title=f"{sel} Price History",
                              xaxis_title="Date",
                              yaxis_title=f"Price ({_nat_ccy})")
        st.plotly_chart(fig_sec, width="stretch")

    # ---- Performance snapshot (project's own prices) ----
    st.markdown("---")
    st.markdown("### Performance Snapshot")
    try:
        snap = period_returns(prices[sel])
    except ValueError as e:
        st.warning(f"Performance snapshot unavailable: {e}")
        snap = None
    if snap is not None:
        cols = st.columns(5)
        for col, label in zip(cols, ["1M", "3M", "6M", "YTD", "1Y"]):
            entry = snap[label]
            if entry["return"] is None:
                col.metric(f"{label} Return", "n/a",
                           help="Not enough history for this window.")
            else:
                col.metric(
                    f"{label} Return", f"{entry['return']:+.2%}",
                    help=f"{entry['start']} → {entry['end']}, from actual prices.")
        st.caption("Window returns use date-based ranges on Portfolio Calc "
                   "price history — never scraped from TradingView.")


# ============================================================

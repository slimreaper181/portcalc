"""views.security — Security Detail tab (rendered by the app shell).

Extracted from the application shell; analytics live in
``analytics/``, fetching in ``data/``. Receives the shared
context dict and renders into the active tab.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    load_intraday,
    load_market_data,
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
    live_snapshot = ctx.get("live_snapshot", {})
    ledger_state = ctx.get("ledger_state")
    ledger_state = ctx.get("ledger_state")
    watchlist = ctx.get("watchlist", [])

    from ui.theme import page_header as _header
    _header("SECURITY TERMINAL", "Security Detail",
            "Quote, position, lots, risk relationship and history.")

    _choices = list(tickers) + [w for w in (watchlist or [])
                                if w not in tickers]
    if st.session_state.get("security_ticker") not in _choices:
        st.session_state.security_ticker = tickers[0]
    sel = st.selectbox(
        "Security", _choices, key="security_ticker",
        help="Holdings come from your current portfolio; watchlist symbols "
             "open market analytics with position panels marked Not held.",
    )
    _held = sel in tickers
    if not _held:
        st.info(f"{sel} is on the watchlist — not held. Market analytics "
                "below use its own history; position panels show Not held.")
    sel_idx = tickers.index(sel) if _held else 0

    # ---- Top security summary from the single central valuation ----
    _v = val_by_ticker.get(sel)
    _pos = pos_by_ticker.get(sel)
    if _held and (_v is None or _pos is None):
        st.error(f"Valuation unavailable for {sel}.")
        return  # section-local return (was st.stop()): sibling tabs keep rendering
    _have_portfolio = bool(_held and _v is not None and _pos is not None)
    _per_share_base = (_v.base_market_value / _v.shares
                       if (_have_portfolio and _v.shares)
                       else float("nan"))

    st.markdown(f"### {sel}")

    # ---- Live quote header (Programme 17): mark, day move, book, feed ----
    _mk = (live_snapshot.get(sel, {}) or {})
    _mark, _dcp = _mk.get("mark"), _mk.get("day_change_pct")
    m1, m2, m3 = st.columns(3)
    m1.metric("Mark Price",
              format_money(_mark, (_v.native_currency if _v is not None
                                   else base_ccy)) if _mark is not None
              else "—",
              delta=(f"{_dcp:+.2%}" if _dcp is not None else None),
              help=f"Valuation mark ({_mk.get('method', 'unknown')}).")
    m2.metric("Bid / Ask",
              (f"{_mk.get('bid')} / {_mk.get('ask')}"
               if _mk.get("bid") is not None
               and _mk.get("ask") is not None else "—"),
              help="Live order-book legs when the quote carries them.")
    m3.metric("Feed / Status",
              f"{_mk.get('source', '?')} · {_mk.get('status', '?')}")
    with st.expander("Quote detail (why the mark may differ)", expanded=False):
        def _qstr(v):
            if v is None:
                return "—"
            try:
                return f"{float(v):,.4f}"
            except (TypeError, ValueError):
                return str(v)
        _qr = [("Last trade", _qstr(_mk.get("last"))),
               ("Bid", _qstr(_mk.get("bid"))),
               ("Ask", _qstr(_mk.get("ask"))),
               ("Spread (bps)", _qstr(_mk.get("spread_bps"))),
               ("Previous close", _qstr(_mk.get("prev_close"))),
               ("Quote age (s)", _qstr(_mk.get("age_seconds"))),
               ("Feed", str(_mk.get("feed") or "—")),
               ("Method", str(_mk.get("method") or "—"))]
        st.dataframe(pd.DataFrame(
            [{"Field": k, "Value": v} for k, v in _qr]),
            width="stretch", hide_index=True)
        st.caption(
            "Portfolio Calc values positions at the mark above. "
            "TradingView draws its own feed — small differences reflect "
            "source/feed/timing, never calculation input (TradingView stays "
            "display-only).")

    if not _held:
        st.info("Not held — no position, lots or P&L for this symbol.")
    if _have_portfolio:
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

    # ---- Open lots + ticker transactions (Programme 39) ----
    if _have_portfolio and ledger_state is not None:
        _lots = [l for l in ledger_state.open_lots if l.ticker == sel]
        with st.expander(f"Open Lots — {sel} ({len(_lots)})",
                         expanded=False):
            if not _lots:
                st.caption("No open lots (fully sold or cash-exempt "
                           "aggregate without ledger lots).")
            else:
                for _l in _lots:
                    _lv = (_l.shares_remaining
                           * (_mk.get("mark") or float("nan")))
                    st.markdown(
                        f"**{_l.purchase_date}** — "
                        f"{_l.shares_remaining:g} shares @ "
                        f"{_l.execution_price_native:,.2f} "
                        f"{_l.native_currency} "
                        f"(cost {format_money(_l.cost_basis_native, _l.native_currency)}"
                        f"{'' if _l.purchase_fx else '; FX unknown'}) · "
                        f"{_l.source}")
            _tx_all = [t for t in (st.session_state.get("ledger_txns") or [])
                       if isinstance(t, dict) and t.get("ticker") == sel]
            if _tx_all:
                st.markdown("#### Transactions")
                st.dataframe(pd.DataFrame([{
                    "Date": str(t.get("date", ""))[:10], "Type": t.get("type"),
                    "Qty": t.get("quantity"), "Price": t.get("price_native"),
                    "Ccy": t.get("currency"), "Fee": t.get("fee"),
                } for t in sorted(
                    _tx_all, key=lambda d: (str(d.get("date", "")),
                                            str(d.get("transaction_id",
                                                      ""))))]),
                    width="stretch", hide_index=True)

    st.markdown("---")

    # ---- Holding analytics (project's own data pipeline only) ----
    st.markdown("### Holding Analytics")
    if not _have_portfolio:
        # Watchlist symbol: own history only (cached single-ticker fetch),
        # market metrics without portfolio relationship.
        try:
            _solo = load_market_data([sel], "2y")[0]
            _lr = np.log(_solo[sel] / _solo[sel].shift(1)).dropna()
            _ann_ret = float(_lr.mean() * 252)
            _ann_vol = float(_lr.std() * np.sqrt(252))
            _sh = ((_ann_ret - rf_rate) / _ann_vol) if _ann_vol > 0 else 0.0
            s1, s2, s3 = st.columns(3)
            s1.metric("Annualised Return", fmt_pct(_ann_ret))
            s2.metric("Annualised Volatility", fmt_pct(_ann_vol))
            s3.metric("Sharpe Ratio", f"{_sh:.2f}")
            st.caption("Market-only analytics from its own history "
                       "(not held — no portfolio relationship).")
            sec, sec_rets, bench_rets_sec = None, _lr, None
            _px_series, _nat_series = _solo[sel], _solo[sel]
            try:
                from data.fx import get_instrument as _gi
                _nat_ccy = _gi(sel).native_currency
            except ValueError:
                _nat_ccy = base_ccy
        except ValueError as e:
            st.warning(f"Market history unavailable for {sel}: {e}")
            sec, sec_rets, bench_rets_sec = None, None, None
            _px_series, _nat_series, _nat_ccy = None, None, None
    else:
        sec_rets = rets[sel].dropna()
        _px_series, _nat_series = prices[sel], native_panel[sel]
        _nat_ccy = val_by_ticker[sel].native_currency

    # Benchmark leg for beta: reuse the Performance tab benchmark (SPY
    # default) through the same cached loader — no extra request.
    # Portfolio relationship needs holdings; watchlist symbols skip it.
    bench_period = st.session_state.get("hist_period_tab1", "2y")
    bench_rets_sec = None
    sec_bench_sym = None
    sec = None
    if not _have_portfolio:
        pass
    else:
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
        if _px_series is None or _nat_series is None:
            st.caption("Price history unavailable for this symbol.")
        else:
            st.caption("Plotted from Portfolio Calc's own yfinance history in "
                       "the security's native market quotation — "
                       "the reproducible application data, unlike the widget above.")
            fig_sec = go.Figure()
            fig_sec.add_trace(go.Scatter(
                x=_nat_series.index, y=_nat_series,
                mode="lines", name=sel, line=dict(color="#58a6ff", width=2),
            ))
            fig_sec.update_layout(**CHART_THEME, title=f"{sel} Price History",
                                  xaxis_title="Date",
                                  yaxis_title=f"Price ({_nat_ccy})")
            st.plotly_chart(fig_sec, width="stretch")

    # ---- Intraday (display-only; never into daily analytics) ----
    with st.expander("Intraday (display only)", expanded=False):
        _intra = st.selectbox("Window", ["1D", "5D"], index=0,
                              key="sec_intraday",
                              help="Intraday bars illustrate the session; "
                                   "daily-return analytics never use them.")
        try:
            _iser = load_intraday(sel, 1 if _intra == "1D" else 5)
            _ifig = go.Figure(go.Scatter(
                x=_iser.index, y=_iser.values, mode="lines", name=sel,
                line=dict(color="#58a6ff", width=1.5)))
            _ifig.update_layout(**CHART_THEME, title=f"{sel} Intraday",
                                xaxis_title="", yaxis_title="Price")
            st.plotly_chart(_ifig, width="stretch")
        except ValueError as e:
            st.caption(f"Intraday unavailable: {e}")

    # ---- Performance snapshot (project's own prices) ----
    st.markdown("---")
    st.markdown("### Performance Snapshot")
    try:
        if _px_series is None:
            raise ValueError("no price history for this symbol")
        snap = period_returns(_px_series)
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

"""
app.py
------
Portfolio Analytics & Risk Dashboard — Streamlit entry point.

Tabs:
  1. 📈 Overview      — positions, returns, correlation/covariance
  2. ⚠️  Risk          — VaR (parametric, historical, Monte Carlo) + CVaR
  3. 🎯 Optimise      — efficient frontier, max Sharpe, min variance, target constraints
  4. 📊 Scenario      — Monte Carlo forward simulation with stress scenarios
  5. 📉 Performance   — benchmark comparison, drawdowns, downside risk, rolling stats
  6. 🔍 Security Detail — single-holding analytics + TradingView chart

Run locally:
    streamlit run app.py

The app is designed to be modular so the analytics modules can be imported
and used independently of Streamlit (e.g. in a desktop GUI or CLI).

Numerical conventions:
  * Market-data returns are daily LOG returns (see analytics.returns).
  * Weight vectors, mu vectors and covariance matrices always share the
    canonical (sorted) ticker order defined in analytics.validation.
"""

import os
import json
from datetime import date, timedelta

import numpy as np
import streamlit as st

from analytics.validation import (
    canonical_tickers,
    validate_shares,
    validate_ticker_symbol,
)
from analytics.currency import validate_base_currency
from analytics.positions import (
    export_portfolio,
    format_money,
    normalise_quote,
    position_from_dict,
    resolve_purchase_price,
    validate_positions_import,
)
from data.fx import SUPPORTED_BASE_CURRENCIES, get_instrument
from data.pipeline import (
    build_core_analytics,
    build_currency_layer,
    build_position_valuations,
    load_alpaca_trades,
    load_backtest_data,
    load_benchmark_data,
    load_current_fx,
    load_dividends_history,
    load_exchange_hint,
    load_factor_data,
    load_fx_history,
    load_instruments,
    load_market_data,
    load_raw_history,
    load_splits,
    resolve_valuation_prices,
)
from ui.errors import render_section
from ui.feedback import flash, render_flash
from ui.views import backtest as views_backtest
from ui.views import factors as views_factors
from ui.views import performance as views_performance
from ui.views import security as views_security
from ui.views import optimise as views_optimise
from ui.views import scenario as views_scenario
from ui.views import overview as views_overview
from ui.views import risk as views_risk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "portfolio.json")
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")

DEFAULT_PORTFOLIO = {
    "AAPL":  {"shares": 10, "avg_cost": 150.0},
    "MSFT":  {"shares": 5,  "avg_cost": 300.0},
    "GOOGL": {"shares": 3,  "avg_cost": 130.0},
    "AMZN":  {"shares": 4,  "avg_cost": 180.0},
    "NVDA":  {"shares": 8,  "avg_cost": 500.0},
}

# ---------------------------------------------------------------------------
# Page configuration + visual theme (single design system in ui/theme.py)
# ---------------------------------------------------------------------------

from ui.theme import inject_theme
inject_theme()


# ---------------------------------------------------------------------------
# Portfolio persistence helpers
# ---------------------------------------------------------------------------

def load_portfolio() -> dict:
    """Load portfolio from JSON file, falling back to defaults."""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                data = json.load(f)
            # Normalise keys on load (upper-case, stripped).
            cleaned = {}
            for k, v in data.items():
                try:
                    sym = validate_ticker_symbol(k)
                except ValueError:
                    continue
                cleaned[sym] = v
            if cleaned:
                return cleaned
        except (json.JSONDecodeError, IOError):
            pass
    return DEFAULT_PORTFOLIO.copy()


def save_portfolio(portfolio: dict) -> None:
    """Persist portfolio to JSON file."""
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(portfolio, f, indent=2)
    except IOError as e:
        st.warning(f"Could not save portfolio: {e}")


def load_settings() -> dict:
    """Load UI settings (base currency); defaults when missing/invalid."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                try:
                    base = validate_base_currency(data.get("base_currency", "USD"))
                except ValueError:
                    base = "USD"
                return {"base_currency": base}
        except (json.JSONDecodeError, IOError, ValueError):
            pass
    return {"base_currency": "USD"}


def save_settings(settings: dict) -> None:
    """Persist UI settings (never secrets — base currency only)."""
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump({"schema_version": 1,
                       "base_currency": settings.get("base_currency", "USD")},
                      f, indent=2)
    except IOError as e:
        st.warning(f"Could not save settings: {e}")


def _complete_add_position(sym, shares, purchase_date, price_native,
                           source, manual, inst):
    """Store one aggregate position (single purchase date, no tax lots).

    ``price_native`` is per current split-adjusted share; ``source`` is one
    of ``estimate``/``manual``/``legacy``. An existing ticker is overwritten
    (with a warning) — multiple purchase lots are not supported.
    """
    if sym in st.session_state.portfolio:
        st.warning(f"{sym} already held — updating it in place. Multiple "
                   "purchase lots are not supported yet.")
    st.session_state.portfolio[sym] = {
        "shares": float(shares),
        "purchase_date": (purchase_date.isoformat()
                          if hasattr(purchase_date, "isoformat")
                          and purchase_date is not None else None),
        "purchase_price_native": float(price_native),
        "purchase_price_source": source,
        "manual_purchase_price": bool(manual),
        "native_currency": inst.get("native_currency")
        if isinstance(inst, dict) else getattr(inst, "native_currency", None),
        "quote_unit": inst.get("quote_unit")
        if isinstance(inst, dict) else getattr(inst, "quote_unit", None),
        "quote_scale": float(inst.get("quote_scale", 1.0))
        if isinstance(inst, dict) else float(
            getattr(inst, "quote_scale", 1.0)),
    }
    save_portfolio(st.session_state.portfolio)
    st.session_state.data_loaded = False
    _label = {"estimate": "estimated close", "manual": "actual execution",
              "legacy": "legacy cost"}.get(source, source)
    # Flash (not st.success): per-run deltas are dropped by st.rerun().
    flash(f"Added {sym} ({_label}).", "success")
    st.rerun()


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_portfolio()

if "base_currency" not in st.session_state:
    st.session_state.base_currency = load_settings().get("base_currency", "USD")

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# Task 9 ledger/watchlist session state (JSON-safe shapes only).
if "ledger_txns" not in st.session_state:
    st.session_state.ledger_txns = []
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "ledger_cleared" not in st.session_state:
    st.session_state.ledger_cleared = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sidebar — Portfolio Editor
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📈 Portfolio Editor")
    render_flash()  # rerun-surviving confirmations render once, here

    st.markdown('<div class="pc-kicker">PORTFOLIO</div>',
                unsafe_allow_html=True)
    # --- Base currency (global valuation currency) ---
    if st.session_state.get("base_currency") not in SUPPORTED_BASE_CURRENCIES:
        st.session_state.base_currency = "USD"
    _prev_base = st.session_state.base_currency
    st.selectbox(
        "Base Currency", list(SUPPORTED_BASE_CURRENCIES),
        key="base_currency",
        help="Currency used to value and aggregate the portfolio.")
    if st.session_state.base_currency != _prev_base:
        save_settings({"base_currency": st.session_state.base_currency})

    st.markdown("---")

    # --- Add position: Ticker / Shares / Purchase Date ---
    with st.expander("➕ Add / Update Position", expanded=False):
        new_ticker = st.text_input("Ticker", placeholder="e.g. TSLA").upper().strip()
        new_shares = st.number_input("Shares", min_value=0.0, step=0.5, value=1.0)
        _today = date.today()
        buy_date = st.date_input(
            "Purchase Date",
            value=st.session_state.get(
                "new_buy_date", date.today() - timedelta(days=365)),
            max_value=date.today(), key="new_buy_date",
            help="Estimated using the market closing price on the selected "
                 "date. Your actual execution price may have differed.")
        use_override = st.checkbox(
            "Use actual execution price", value=False, key="new_use_override",
            help="Advanced: override the estimate with your broker fill "
                 "(in the security's native currency).")
        new_override = st.number_input(
            "Actual execution price (native)", min_value=0.0, step=0.5,
            value=0.0, key="new_override") if use_override else 0.0

        if st.button("Add Position"):
            try:
                sym = validate_ticker_symbol(new_ticker)
                sh = validate_shares(float(new_shares))
                if use_override and (
                        not np.isfinite(new_override) or new_override <= 0):
                    raise ValueError(
                        "Override price must be positive when enabled.")
            except ValueError as e:
                st.error(str(e))
            else:
                try:
                    inst = get_instrument(sym)
                except ValueError as e:
                    st.error(str(e))
                    inst = None
                if inst is not None:
                    if use_override:
                        _complete_add_position(
                            sym, sh, None, float(new_override), "manual",
                            True, inst)
                    else:
                        try:
                            raw = load_raw_history(sym)
                            splits = load_splits(sym)
                        except ValueError as e:
                            st.error(str(e))
                        else:
                            res = resolve_purchase_price(
                                raw, splits, buy_date, _today)
                            if res["status"] == "ok":
                                # res["price"] is in raw quote units (e.g. GBp)
                                # — normalise to native currency before storing.
                                _complete_add_position(
                                    sym, sh, res["used_date"],
                                    normalise_quote(res["price"],
                                                    inst.quote_scale),
                                    "estimate", False, inst)
                            elif res["status"] == "missing":
                                st.session_state.pending_buy = {
                                    "ticker": sym, "shares": sh,
                                    "inst": {"native_currency": inst.native_currency,
                                             "quote_unit": inst.quote_unit,
                                             "quote_scale": inst.quote_scale},
                                    "prev": [res["prev"][0].isoformat(),
                                             normalise_quote(
                                                 res["prev"][1],
                                                 inst.quote_scale)],
                                    "next": [res["next"][0].isoformat(),
                                             normalise_quote(
                                                 res["next"][1],
                                                 inst.quote_scale)],
                                }
                                st.rerun()
                            else:
                                st.error(res["message"])

        # Pending non-trading-date choice (explicit, never silent).
        pending = st.session_state.get("pending_buy")
        if pending:
            st.info(
                f"No market price exists on the selected date for "
                f"{pending['ticker']}. Choose explicitly:")
            _pdate, _pprice = pending["prev"]
            _ndate, _nprice = pending["next"]
            _nccy = pending["inst"]["native_currency"]
            _opts = {
                f"{_pdate} — prev close {format_money(_pprice, _nccy)}":
                    (_pdate, _pprice),
                f"{_ndate} — next close {format_money(_nprice, _nccy)}":
                    (_ndate, _nprice),
            }
            _choice = st.radio("Trading date", list(_opts),
                               key="pending_buy_choice")
            _c1, _c2 = st.columns(2)
            with _c1:
                if st.button("Use date", key="pending_buy_ok"):
                    _d, _px = _opts[_choice]
                    _complete_add_position(
                        pending["ticker"], pending["shares"],
                        date.fromisoformat(_d), float(_px), "estimate",
                        False, pending["inst"])
                    st.session_state.pending_buy = None
                    st.rerun()
            with _c2:
                if st.button("Cancel", key="pending_buy_cancel"):
                    st.session_state.pending_buy = None
                    st.rerun()

    st.markdown('<div class="pc-kicker">HOLDINGS</div>',
                unsafe_allow_html=True)
    # --- Current positions ---
    to_remove = []
    for ticker, pos in list(st.session_state.portfolio.items()):
        col1, col2 = st.columns([3, 1])
        try:
            _p = position_from_dict(ticker, pos)
            _buy = _p.purchase_date.isoformat() if _p.purchase_date else "no date"
            _src = {"estimate": "est", "manual": "actual",
                    "legacy": "legacy"}.get(
                        _p.purchase_price_source or "", "?")
            col1.markdown(
                f"**{ticker}** — {_p.shares:g} sh · buy {_buy} ({_src})")
        except ValueError:
            col1.markdown(f"**{ticker}** — (invalid entry, remove me)")
        if col2.button("✕", key=f"rm_{ticker}"):
            to_remove.append(ticker)

    for t in to_remove:
        del st.session_state.portfolio[t]
        save_portfolio(st.session_state.portfolio)
        st.session_state.data_loaded = False
        flash(f"Removed {t}.", "info")
        st.rerun()

    st.markdown('<div class="pc-kicker">DATA</div>',
                unsafe_allow_html=True)
    with st.expander("Market data", expanded=False):
        if st.button("🔄 Refresh Market Data"):
            st.session_state.data_loaded = False
            for _cache_fn in (load_market_data, load_benchmark_data,
                              load_backtest_data, load_factor_data,
                              load_exchange_hint, load_alpaca_trades,
                              load_instruments, load_fx_history, load_current_fx,
                              load_raw_history, load_splits, load_dividends_history):
                try:
                    _cache_fn.clear()
                except Exception:
                    pass
            flash("Market data refreshed.", "success")
            st.rerun()
        st.caption(
            "Current US prices: Alpaca IEX (Yahoo fallback). Historical "
            "prices, FX: Yahoo Finance. Factors: Kenneth French library. "
            "Charts: TradingView (display only).")

    st.markdown('<div class="pc-kicker">FILES</div>',
                unsafe_allow_html=True)
    # --- Export / Import (positions + settings only, never secrets) ---
    with st.expander("Import / export / reset", expanded=False):
        st.markdown("### Portfolio Data")
        try:
            _export_doc = export_portfolio(
                st.session_state.portfolio,
                st.session_state.get("base_currency", "USD"))
            st.download_button(
                "Export Portfolio (JSON)", data=json.dumps(_export_doc, indent=2),
                file_name="portcalc-portfolio.json", mime="application/json",
                key="export_json")
        except ValueError as e:
            st.warning(f"Export unavailable: {e}")
        _uploaded = st.file_uploader("Import Portfolio (JSON)", type=["json"],
                                     key="import_json")
        if _uploaded is not None:
            if st.button("Replace portfolio with import", key="import_apply"):
                try:
                    _doc = json.load(_uploaded)
                    _imp_positions, _imp_base = validate_positions_import(_doc)
                except (ValueError, json.JSONDecodeError) as e:
                    st.error(f"Import rejected: {e}")
                except Exception:
                    st.error("Import rejected: unreadable file.")
                else:
                    st.session_state.portfolio = _imp_positions
                    if _imp_base in SUPPORTED_BASE_CURRENCIES:
                        st.session_state.base_currency = _imp_base
                        save_settings({"base_currency": _imp_base})
                    save_portfolio(st.session_state.portfolio)
                    st.session_state.data_loaded = False
                    flash(f"Imported {len(_imp_positions)} positions "
                          f"(base {_imp_base}).", "success")
                    st.rerun()

        # --- Reset with intentional confirmation ---
        if st.button("Reset Portfolio", key="reset_start"):
            st.session_state.confirm_reset = True
            st.rerun()
        if st.session_state.get("confirm_reset"):
            st.warning("Delete ALL positions? This cannot be undone.")
            _r1, _r2 = st.columns(2)
            with _r1:
                if st.button("Confirm reset", key="reset_confirm"):
                    st.session_state.portfolio = {}
                    save_portfolio(st.session_state.portfolio)
                    st.session_state.confirm_reset = False
                    st.session_state.data_loaded = False
                    flash("Portfolio reset — all positions removed.", "warning")
                    st.rerun()
            with _r2:
                if st.button("Cancel", key="reset_cancel"):
                    st.session_state.confirm_reset = False
                    st.rerun()

    st.markdown('<div class="pc-kicker">ADVANCED</div>',
                unsafe_allow_html=True)
    with st.expander("Assumptions", expanded=False):
        st.caption(
            "Adjusted closes (splits/dividends per Yahoo). End-of-day "
            "execution approximation; fractional shares; no market impact "
            "beyond explicit bps costs. Risk-free rate fixed at 5% across "
            "historical windows. French factors are US datasets. "
            "TradingView charts are display-only.")


try:
    tickers = canonical_tickers(list(st.session_state.portfolio.keys()))
except ValueError as e:
    st.warning(f"Portfolio problem: {e} Add at least one valid position in the sidebar.")
    st.stop()

if not tickers:
    st.warning("Add at least one position in the sidebar to get started.")
    st.stop()

# De-duplicated tickers are canonicalised; warn if the raw keys differed.
if len(tickers) != len(st.session_state.portfolio):
    st.info("Duplicate tickers were merged (symbols are case-insensitive).")

with st.spinner("Loading market data…"):
    # Default history period — can be overridden per-tab
    try:
        prices, current_prices, rf_rate = load_market_data(tickers, "2y")
    except ValueError as e:
        st.error(f"Market data error: {e}")
        st.stop()
    except Exception:
        st.error(
            "Could not load market data (network or API error). "
            "Check your connection and click Refresh Market Data to retry."
        )
        st.stop()

# Current valuation prices: Alpaca latest trade for supported US equities,
# Yahoo fallback otherwise. Historical `prices` series are untouched.
valuation_prices, price_view, alpaca_status = resolve_valuation_prices(
    tickers, current_prices)

# Multi-currency layer: quote-normalise native data, convert history and
# current prices into the base currency. All downstream analytics operate
# on the base-currency panel — never on mixed native currencies.
base_ccy = st.session_state.get("base_currency", "USD")
try:
    ccy = build_currency_layer(tickers, prices, valuation_prices, base_ccy)
except ValueError as e:
    st.error(f"Currency setup: {e}")
    st.stop()
native_panel = ccy["native_panel"]
instruments = ccy["instruments"]

try:
    core = build_core_analytics(tickers, ccy["base_panel"],
                                ccy["base_current"], rf_rate)
except ValueError as e:
    st.error(f"Data error: {e}")
    st.stop()

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
port_daily_returns = core["port_daily_returns"]
rc = core["rc"]

# Central position valuations (single path reused by every tab).
# Native current prices, quote-normalised; base values come from the engine.
_native_current_norm = {
    t: float(valuation_prices.get(t, float("nan")))
    * ccy["instruments"][t].quote_scale for t in tickers}
try:
    positions, valuations, val_agg, val_warnings = build_position_valuations(
        tickers, ccy["instruments"], _native_current_norm, price_view,
        ccy["fx_hist"], ccy["fx_now"], base_ccy)
except ValueError as e:
    st.error(f"Valuation error: {e}")
    st.stop()
val_by_ticker = {v.ticker: v for v in valuations}
pos_by_ticker = positions

# ---------------------------------------------------------------------------
# Task 9 — ledger build (migration + FX resolution + derived state)
# ---------------------------------------------------------------------------

from analytics.ledger import (
    apply_ledger,
    average_cost_per_share,
    open_shares_by_ticker,
    unrealised_by_ticker,
)
from analytics.persistence import (
    migrate_v1_to_ledger,
    rebase_ledger_fx,
    transaction_from_dict,
    transaction_to_dict,
)

ledger_notes: list = []
ledger_error: str | None = None
_ledger_raw = [t for t in st.session_state.ledger_txns
               if isinstance(t, dict)]
if not _ledger_raw and not st.session_state.ledger_cleared \
        and st.session_state.portfolio:
    # Deterministic auto-migration: aggregate positions → opening BUYs.
    _mig, _skipped = migrate_v1_to_ledger(st.session_state.portfolio)
    _ledger_raw = [transaction_to_dict(t) for t in _mig]
    st.session_state.ledger_txns = _ledger_raw
    if _skipped:
        ledger_notes.append(
            "Migration skipped undated entries (reported, never invented): "
            + ", ".join(_skipped) + ".")
_ledger_txns = []
for _d in _ledger_raw:
    try:
        _ledger_txns.append(transaction_from_dict(_d))
    except ValueError as e:
        ledger_error = f"Ledger entry rejected: {e}"
        _ledger_txns = []
        break
if ledger_error is None and _ledger_txns:
    # Fill unknown FX legs from date-matched history (base-aware); user
    # legs are preserved. Base switches recompute presentation only.
    _adj, _unresolved = rebase_ledger_fx(
        _ledger_txns, base_ccy, ccy.get("fx_hist"), only_missing=True)
    _ledger_txns = _adj
    if _unresolved:
        ledger_notes.append(
            "Purchase-date FX unavailable for "
            f"{len(_unresolved)} ledger leg(s); base cost stays unknown "
            "there (never assumed).")
try:
    ledger_state = apply_ledger(_ledger_txns) if (
        ledger_error is None and _ledger_txns) else None
except ValueError as e:
    ledger_error = f"Ledger inconsistent: {e}"
    ledger_state = None
ledger_open_shares = open_shares_by_ticker(ledger_state) \
    if ledger_state else {}
ledger_cash = dict(ledger_state.cash) if ledger_state else {}

# ---------------------------------------------------------------------------
# Task 9 — live mark snapshot (one consistent read for all current cards)
# ---------------------------------------------------------------------------

from data.pipeline import build_live_snapshot, load_previous_closes

try:
    _prev_closes = load_previous_closes(tuple(tickers))
except ValueError:
    _prev_closes = {}
try:
    _watch_syms = [w for w in st.session_state.watchlist
                   if w not in tickers]
    live_snapshot = build_live_snapshot(
        tickers + _watch_syms, current_prices, _prev_closes)
except Exception:
    live_snapshot = {"_status": {"feed": None,
                                 "feed_note": "snapshot unavailable",
                                 "n_symbols": 0, "at": ""}}

# Live securities value from marks (native marks × current FX) + cash = NAV.
# Legacy Yahoo-based valuations above stay untouched for cost/P&L analytics.
live_holdings: dict = {}
live_day_pnl_native: dict = {}
for _t in tickers:
    _m = live_snapshot.get(_t, {})
    _mark = _m.get("mark")
    _sh = float(ledger_open_shares.get(
        _t, st.session_state.portfolio.get(_t, {}).get("shares", 0.0)) or 0.0)
    _inst = ccy["instruments"][_t]
    _fx_now = ccy["fx_now"].get(_inst.native_currency)
    if _mark is None or not np.isfinite(float(_mark)) or _sh <= 0 \
            or _fx_now is None or not np.isfinite(float(_fx_now)) \
            or float(_fx_now) <= 0:
        live_holdings[_t] = {"native_value": None, "base_value": None,
                             "day_pnl_base": None, "shares": _sh}
        continue
    _native_value = _sh * float(_mark) * float(_inst.quote_scale)
    live_holdings[_t] = {
        "native_value": _native_value,
        "base_value": _native_value * float(_fx_now),
        "shares": _sh,
        "day_pnl_base": None,
    }
    _prev = _m.get("prev_close")
    if _prev is not None and float(_prev) > 0:
        _day_native = (float(_mark) - float(_prev)) * _sh \
            * float(_inst.quote_scale)
        live_holdings[_t]["day_pnl_base"] = _day_native * float(_fx_now)
_cash_base = 0.0
for _ccy, _bal in ledger_cash.items():
    _r = ccy["fx_now"].get(_ccy)
    if _r is not None and np.isfinite(float(_r)) and float(_r) > 0:
        _cash_base += float(_bal) * float(_r)
_live_sec_base = sum(v["base_value"] for v in live_holdings.values()
                    if v["base_value"] is not None)
portfolio_nav = _live_sec_base + _cash_base
day_pnl_base = sum(v["day_pnl_base"] for v in live_holdings.values()
                   if v["day_pnl_base"] is not None)
live_incomplete = [t for t, v in live_holdings.items()
                   if v["base_value"] is None]

# Shared render context for ui/views tabs (single source of shell state).
# Overview may return an updated dict (history-period rebuild) merged back.
ctx = {
    "tickers": tickers,
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
    "live_snapshot": live_snapshot,
    "live_holdings": live_holdings,
    "ledger_state": ledger_state,
    "ledger_error": ledger_error,
    "ledger_notes": ledger_notes,
    "ledger_cash": ledger_cash,
    "portfolio_nav": portfolio_nav,
    "day_pnl_base": day_pnl_base,
    "live_incomplete": live_incomplete,
    "watchlist": list(st.session_state.watchlist),
}

# ---------------------------------------------------------------------------
# Application shell header (compact context — details live in the tabs)
# ---------------------------------------------------------------------------

from ui.formatting import format_date, format_money, format_signed_money

_header_bench = st.session_state.get("bt_benchmark", "SPY")
try:
    _header_through = format_date(prices.index[-1])
except Exception:
    _header_through = "—"
st.markdown("## Portfolio Calc")
st.caption("Portfolio Research & Risk")
_h1, _h2, _h3, _h4 = st.columns(4)
_h1.metric("NAV", format_money(portfolio_nav, base_ccy),
           help="Live securities value (mark prices) + cash.")
_h2.metric("Day P&L", format_signed_money(day_pnl_base, base_ccy)
           if live_holdings else "—",
           help="Mark vs previous official close (pre/post-market marks "
                "compared against the last close; see methodology).")
_h3.metric("Cash", format_money(_cash_base, base_ccy),
           help="Currency-keyed cash from the transaction ledger.")
_h4.metric("Base Currency", base_ccy)
_live_note = live_snapshot.get("_status", {}).get("feed_note", "")
st.caption(f"Benchmark {str(_header_bench)} · Data through {_header_through} "
           f"· Live: {_live_note}"
           + (" · Partial marks: " + ", ".join(live_incomplete)
              if live_incomplete else ""))
st.markdown("---")

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_overview, tab_risk, tab_optimise, tab_scenario, tab_perf, tab_security, tab_backtest, tab_factors = st.tabs([
    "📈 Overview",
    "⚠️  Risk",
    "🎯 Optimise",
    "📊 Scenario Analysis",
    "📉 Performance",
    "🔍 Security Detail",
    "🧪 Backtest",
    "📐 Factor Analysis",
])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tab_overview:
    _ov_out = render_section("Overview", views_overview.render, ctx)
    if isinstance(_ov_out, dict):
        ctx.update(_ov_out)
# TAB 2 — RISK
# ============================================================
with tab_risk:
    render_section("Risk", views_risk.render, ctx)
# TAB 3 — OPTIMISE
# ============================================================
with tab_optimise:
    render_section("Optimise", views_optimise.render, ctx)
# TAB 4 — SCENARIO ANALYSIS
# ============================================================
with tab_scenario:
    render_section("Scenario Analysis", views_scenario.render, ctx)
# TAB 5 — PERFORMANCE vs BENCHMARK
# ============================================================
with tab_perf:
    render_section("Performance", views_performance.render, ctx)
# TAB 6 — SECURITY DETAIL
# ============================================================
with tab_security:
    render_section("Security Detail", views_security.render, ctx)
# TAB 7 — BACKTEST
# ============================================================
with tab_backtest:
    render_section("Backtest", views_backtest.render, ctx)
# TAB 8 — FACTOR ANALYSIS
# ============================================================
with tab_factors:
    render_section("Factor Analysis", views_factors.render, ctx)

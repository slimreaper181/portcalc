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
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from data.market_data import (
    fetch_price_history,
    fetch_benchmark_history,
    fetch_current_prices,
    fetch_dividends,
    fetch_exchange_code,
    fetch_raw_closes,
    fetch_risk_free_rate,
    fetch_splits,
)
from analytics.returns import (
    daily_returns,
    annualised_mean_returns,
    annualised_cov_matrix,
    correlation_matrix,
    cumulative_growth_from_log_returns,
)
from analytics.portfolio import (
    portfolio_expected_return,
    portfolio_std,
    sharpe_ratio,
    risk_contributions,
    weights_from_shares,
)
from analytics.var import (
    parametric_var,
    historical_var,
    monte_carlo_var,
    to_daily,
    mean_to_daily,
)
from analytics.optimisation import (
    min_variance,
    max_sharpe,
    target_return,
    target_volatility,
    efficient_frontier,
    feasible_return_range,
    feasible_volatility_range,
    rebalance_trades,
)
from analytics.scenario import (
    simulate_portfolio_paths,
    run_predefined_scenario,
    summarise_future_metrics,
    explain_scenario_results,
    plot_simulation_paths,
    plot_final_distribution,
    plot_confidence_bands,
    SCENARIO_DEFINITIONS,
)
from analytics.performance import (
    align_return_series,
    best_worst_periods,
    drawdown_series,
    growth_of_capital,
    historical_cvar,
    log_returns_from_prices,
    monthly_returns,
    plot_drawdown,
    plot_growth_comparison,
    rolling_annualised_return,
    rolling_sharpe,
    rolling_volatility,
    sharpe_from_log,
    summarise_performance,
    underwater_episodes,
)
from analytics.security import (
    period_returns,
    security_summary,
)
from components.tradingview import (
    render_tradingview_chart,
    resolve_tradingview_symbol,
)
from data.alpaca import (
    AlpacaError,
    build_price_view,
    fetch_latest_trades,
    get_alpaca_credentials,
    is_alpaca_supported_symbol,
)
from data.fx import (
    SUPPORTED_BASE_CURRENCIES,
    FX_PAIRS,
    fetch_current_fx,
    fetch_fx_history,
    get_instrument,
)
from analytics.currency import (
    convert_price_series,
    convert_value,
    currency_symbol,
    decompose_pnl,
    freshness_report,
    fx_rate_between,
    fx_shock_portfolio,
    quote_age_days,
    rate_on_date,
    validate_base_currency,
)
from analytics.positions import (
    Position,
    build_analysis_snapshot,
    estimate_dividend_income,
    export_portfolio,
    format_money,
    position_from_dict,
    resolve_purchase_price,
    summarize_portfolio_csv,
    validate_positions_import,
    value_portfolio,
    value_position,
)
from data.factors import fetch_french_factors
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
from analytics.backtest import (
    backtest_benchmark,
    backtest_buy_and_hold,
    backtest_rebalanced,
    compare_backtests,
    plot_backtest_growth,
    plot_weight_drift,
    prepare_backtest_data,
)
from analytics.allocation import (
    AllocationResult,
    allocation_summary,
    annualised_covariance_simple,
    annualised_mean_returns_simple,
    black_litterman_allocation,
    black_litterman_posterior,
    equal_risk_contribution,
    estimate_risk_aversion_from_benchmark,
    estimation_window,
    maximum_diversification,
    simple_returns_from_prices,
)
from analytics.validation import (
    align_market_data,
    aligned_portfolio_returns,
    canonical_tickers,
    sanitize_prices,
    validate_horizon_years,
    validate_monthly_contrib,
    validate_risk_free_rate,
    validate_shares,
    validate_simulation_count,
    validate_ticker_symbol,
    validate_weight_bounds,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "portfolio.json")
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")

CHART_THEME = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Mono, monospace", color="#c9d1d9", size=12),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    margin=dict(l=60, r=30, t=60, b=60),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d"),
)

DEFAULT_PORTFOLIO = {
    "AAPL":  {"shares": 10, "avg_cost": 150.0},
    "MSFT":  {"shares": 5,  "avg_cost": 300.0},
    "GOOGL": {"shares": 3,  "avg_cost": 130.0},
    "AMZN":  {"shares": 4,  "avg_cost": 180.0},
    "NVDA":  {"shares": 8,  "avg_cost": 500.0},
}

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Portfolio Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
  }
  h1, h2, h3, h4 { font-family: 'IBM Plex Mono', monospace; color: #e6edf3; }
  .stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 14px; }
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
  }
  .metric-card .value { font-size: 26px; font-weight: 600; color: #58a6ff; }
  .metric-card .label { font-size: 12px; color: #8b949e; margin-top: 4px; }
  div[data-testid="stSidebarContent"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
  }
  .stButton>button {
    background-color: #238636;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
  }
  .stButton>button:hover { background-color: #2ea043; }
  div[data-testid="stMetricValue"] {
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
    /* Never ellipsise financial values: allow wrapping instead of clipping.
       Layouts below keep large currency metrics to 2-3 cards per row so
       wrapping is a last resort, not the norm. */
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
  }
  div[data-testid="stMetricValue"] > div {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
  }
</style>
""", unsafe_allow_html=True)


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
    st.success(f"Added {sym} ({_label}).")
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def style_matrix(df: pd.DataFrame, fmt: str = ".4f") -> Styler:
    """Apply background gradient styling to a numeric DataFrame."""
    return (
        df.style
        .format("{:" + fmt + "}")
        .background_gradient(cmap="Blues", axis=None)
    )


def fmt_usd(value: float, currency: str | None = None) -> str:
    """Format money in the active base currency (explicit override allowed).

    Existing call sites keep working unchanged; the base currency is read
    from session state. Use an explicit currency for native-currency values.
    """
    try:
        ccy = currency or st.session_state.get("base_currency", "USD")
    except Exception:
        ccy = currency or "USD"
    return format_money(value, ccy)


def fmt_pct(value: float) -> str:
    return f"{value:.2%}"


def build_core_analytics(
    tickers: list[str],
    prices: pd.DataFrame,
    current_prices: dict,
    rf_rate: float,
) -> dict:
    """Sanitise market data, align to canonical ticker order, compute core stats.

    Raises:
        ValueError: with a user-facing message if data is unusable.
    """
    tickers = canonical_tickers(tickers)
    clean_prices = sanitize_prices(prices, tickers)
    if len(clean_prices) < 30:
        st.warning(
            f"Only {len(clean_prices)} trading days of overlapping history — "
            "statistics and VaR estimates will be noisy. "
            "Consider a longer history period or fewer tickers."
        )

    # Current prices must be present and positive for every ticker.
    bad = [t for t in tickers
           if not np.isfinite(float(current_prices.get(t, np.nan)))
           or float(current_prices.get(t, 0.0)) <= 0]
    if bad:
        raise ValueError(
            f"No usable current price for: {', '.join(bad)}. "
            "Check the ticker symbols."
        )

    shares_dict = {t: st.session_state.portfolio[t]["shares"] for t in tickers}
    for t, sh in shares_dict.items():
        if not np.isfinite(sh) or sh <= 0:
            raise ValueError(
                f"Position {t} has invalid shares ({sh!r}); "
                "shares must be positive."
            )
    weights, w_tickers = weights_from_shares(shares_dict, current_prices)
    if list(w_tickers) != list(tickers) or len(weights) != len(tickers):
        raise ValueError("Internal error: weight/ticker ordering mismatch.")

    portfolio_value = float(sum(
        st.session_state.portfolio[t]["shares"] * float(current_prices[t])
        for t in tickers
    ))
    if not np.isfinite(portfolio_value) or portfolio_value <= 0:
        raise ValueError(
            "Portfolio value is zero — check share counts and prices."
        )

    try:
        rf_rate = validate_risk_free_rate(rf_rate)
    except ValueError:
        rf_rate = 0.05

    rets = daily_returns(clean_prices)
    rets = rets.dropna(how="any")
    if rets.empty or len(rets) < 2:
        raise ValueError("Not enough return observations to compute statistics.")
    aligned = align_market_data(tickers, returns=rets)
    rets = aligned["returns"]

    mu = annualised_mean_returns(rets)
    cov = annualised_cov_matrix(rets)
    corr = correlation_matrix(rets)
    # Defensive reindex: mu/cov/corr rows/cols follow the canonical order.
    mu = align_market_data(tickers, mu=mu)["mu"]
    cov = align_market_data(tickers, cov=cov)["cov"]
    corr = corr.reindex(index=tickers, columns=tickers)

    if not np.all(np.isfinite(mu.values)):
        raise ValueError("Expected returns contain non-finite values.")
    if not np.all(np.isfinite(cov.values)):
        raise ValueError("Covariance matrix contains non-finite values.")

    port_return = portfolio_expected_return(weights, mu.values)
    port_vol = portfolio_std(weights, cov.values)
    port_sharpe = sharpe_ratio(port_return, port_vol, rf_rate)
    port_daily_returns = aligned_portfolio_returns(rets, weights, tickers)
    rc = risk_contributions(weights, cov.values)

    for name, val in (("expected return", port_return), ("volatility", port_vol)):
        if not np.isfinite(val):
            raise ValueError(f"Portfolio {name} is not finite — check market data.")

    return {
        "tickers": tickers,
        "prices": clean_prices,
        "current_prices": {t: float(current_prices[t]) for t in tickers},
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
    }


# ---------------------------------------------------------------------------
# Sidebar — Portfolio Editor
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📈 Portfolio Editor")
    st.markdown("---")

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
                                _complete_add_position(
                                    sym, sh, res["used_date"],
                                    res["price"], "estimate", False, inst)
                            elif res["status"] == "missing":
                                st.session_state.pending_buy = {
                                    "ticker": sym, "shares": sh,
                                    "inst": {"native_currency": inst.native_currency,
                                             "quote_unit": inst.quote_unit,
                                             "quote_scale": inst.quote_scale},
                                    "prev": [res["prev"][0].isoformat(),
                                             res["prev"][1]],
                                    "next": [res["next"][0].isoformat(),
                                             res["next"][1]],
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

    # --- Current positions ---
    st.markdown("### Current Positions")
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
        st.rerun()

    st.markdown("---")

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
        st.rerun()

    # --- Export / Import (positions + settings only, never secrets) ---
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
                st.success(f"Imported {len(_imp_positions)} positions "
                           f"(base {_imp_base}).")
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
                st.rerun()
        with _r2:
            if st.button("Cancel", key="reset_cancel"):
                st.session_state.confirm_reset = False
                st.rerun()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def load_market_data(tickers: list[str], period: str):
    prices  = fetch_price_history(tickers, period=period)
    current = fetch_current_prices(tickers)
    rf      = fetch_risk_free_rate()
    return prices, current, rf


@st.cache_data(ttl=300, show_spinner=False)
def load_benchmark_data(benchmark_ticker: str, period: str):
    """Cached benchmark history for the same period as the portfolio."""
    return fetch_benchmark_history(benchmark_ticker, period=period)


@st.cache_data(ttl=3600, show_spinner=False)
def load_backtest_data(tickers: list[str]):
    """Cached full-length history for backtesting (same yfinance pipeline)."""
    return fetch_price_history(list(tickers), period="max")


@st.cache_data(ttl=86400, show_spinner=False)
def load_factor_data():
    """Cached French factor datasets (change only daily)."""
    return fetch_french_factors()


@st.cache_data(ttl=3600, show_spinner=False)
def load_exchange_hint(ticker: str):
    """Cached yfinance exchange code (TradingView resolution hint only)."""
    try:
        return fetch_exchange_code(ticker)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_instruments(tickers):
    """Cached canonical instrument metadata (currency, quote scale, tz).

    Returns ``(instruments, problems)`` where problems lists tickers whose
    metadata could not be established; callers must fail clearly rather
    than assume USD.
    """
    instruments, problems = {}, []
    for t in list(tickers):
        try:
            instruments[t] = get_instrument(t)
        except ValueError as e:
            problems.append(f"{t}: {e}")
        except Exception as e:
            problems.append(f"{t}: metadata lookup failed ({e}).")
    return instruments, problems


@st.cache_data(ttl=86400, show_spinner=False)
def load_fx_history():
    """Cached daily USD-per-unit FX history (long TTL — history is static)."""
    return fetch_fx_history()


@st.cache_data(ttl=60, show_spinner=False)
def load_current_fx():
    """Cached current FX snapshot: ``{pair: (rate, timestamp_utc)}``.

    Short TTL; lightweight 5-day fetch. Derived from the same Yahoo
    series family as historical FX so sources never disagree.
    """
    out = {}
    for pair in ("GBPUSD=X", "EURUSD=X"):
        try:
            out[pair] = fetch_current_fx(pair)
        except ValueError as e:
            out[pair] = (None, str(e))
        except Exception as e:
            out[pair] = (None, f"FX request failed ({e}).")
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def load_raw_history(ticker: str):
    """Cached raw closes for purchase-price lookup (long TTL)."""
    return fetch_raw_closes(ticker)


@st.cache_data(ttl=86400, show_spinner=False)
def load_splits(ticker: str):
    """Cached split history for cost-basis adjustment (long TTL)."""
    return fetch_splits(ticker)


@st.cache_data(ttl=86400, show_spinner=False)
def load_dividends_history(ticker: str):
    """Cached dividend history for income estimates (long TTL)."""
    return fetch_dividends(ticker)


@st.cache_data(ttl=10, show_spinner=False)
def load_alpaca_trades(symbols):
    """Cached Alpaca latest trades (short 10s TTL — live data).

    Never raises: returns ``(trades, status)`` where ``status`` is ``None``
    on success, ``"disabled"`` when credentials are absent, or a short
    error message otherwise. Callers fall back per symbol to Yahoo.
    """
    try:
        creds = get_alpaca_credentials()
    except Exception:
        return {}, "disabled"
    try:
        trades = fetch_latest_trades(list(symbols), credentials=creds)
    except AlpacaError as e:
        return {}, str(e)
    except Exception as e:
        return {}, f"Live-price request failed ({e})."
    return trades, None


def resolve_valuation_prices(tickers, yahoo_current):
    """Blend Alpaca live trades over Yahoo current prices (valuation only).

    Returns ``(valuation_dict, price_view, alpaca_status)``. Historical
    ``prices`` series are never touched — live data is used solely for
    current prices, market value and P&L display.
    """
    supported = sorted(t for t in tickers if is_alpaca_supported_symbol(t))
    trades, status = load_alpaca_trades(tuple(supported)) if supported else ({}, None)
    view = build_price_view(tickers, yahoo_current, trades)
    return {t: view[t].price for t in tickers}, view, status


def build_currency_layer(tickers, native_prices, native_current, base_ccy):
    """Quote-normalise native data and convert everything to base currency.

    Pipeline per holding: quoted price → × quote_scale → native currency
    → × date-matched historical FX → base currency. Returns a dict with
    ``instruments``, ``native_panel``, ``base_panel``, ``fx_hist`` (None
    when every holding is already in base currency — no FX fetch at all),
    ``base_current``, ``fx_now`` (USD-per-unit map), ``fx_timestamp`` and
    ``native_ccy``. Raises ``ValueError`` with user-facing messages instead
    of guessing (never assumes USD, never invents FX).
    """
    base = validate_base_currency(base_ccy)
    instruments, problems = load_instruments(tuple(tickers))
    if problems:
        raise ValueError(
            "Currency metadata unavailable — " + "; ".join(problems))
    nat_ccy = {t: instruments[t].native_currency for t in tickers}
    scales = {t: instruments[t].quote_scale for t in tickers}
    native_panel = native_prices.reindex(columns=tickers).copy()
    for t in tickers:
        native_panel[t] = native_panel[t] * scales[t]
    need_fx = any(c != base for c in nat_ccy.values())
    fx_hist = None
    if need_fx:
        try:
            fx_hist = load_fx_history()
        except ValueError as e:
            raise ValueError(f"FX history unavailable: {e}")
    base_panel = base_price_panel(native_prices, tickers, instruments,
                                  fx_hist, base)
    usd_now = {"USD": 1.0}
    fx_ts = None
    if need_fx:
        snap = load_current_fx()
        for pair, (rate, ts) in snap.items():
            ccy = FX_PAIRS[pair][0]
            if rate is not None and np.isfinite(rate) and rate > 0:
                usd_now[ccy] = float(rate)
                if ts is not None and (fx_ts is None or ts > fx_ts):
                    fx_ts = ts
        # Graceful fallback: historical tail when the 5-day snapshot
        # misses a needed pair (labelled by its own timestamp downstream).
        if fx_hist is not None:
            for ccy in {c for c in nat_ccy.values() if c != "USD"}:
                if ccy not in usd_now:
                    tail = fx_hist[ccy].dropna()
                    if not tail.empty and tail.iloc[-1] > 0:
                        usd_now[ccy] = float(tail.iloc[-1])
                        fx_ts = tail.index[-1].to_pydatetime().replace(
                            tzinfo=timezone.utc)
    base_current = {}
    for t in tickers:
        px = float(native_current.get(t, float("nan"))) * scales[t]
        try:
            base_current[t] = convert_value(px, nat_ccy[t], base, usd_now)
        except ValueError as e:
            raise ValueError(f"{t}: current FX unavailable ({e}).")
    return {
        "base_currency": base,
        "instruments": instruments,
        "native_ccy": nat_ccy,
        "native_panel": native_panel,
        "base_panel": base_panel,
        "fx_hist": fx_hist,
        "base_current": base_current,
        "fx_now": usd_now,
        "fx_timestamp": fx_ts,
        "need_fx": need_fx,
    }


def format_trade_time(ts):
    """Format an Alpaca trade timestamp as ET (UTC fallback). Never raises."""
    if ts is None:
        return "time unknown"
    try:
        from zoneinfo import ZoneInfo

        return ts.astimezone(ZoneInfo("America/New_York")).strftime("%H:%M:%S ET")
    except Exception:
        try:
            return ts.strftime("%H:%M:%S UTC")
        except Exception:
            return "time unknown"


def base_price_panel(native_quoted_panel, tickers, instruments, fx_hist,
                     base_ccy):
    """Quote-normalise a native history panel and convert it to base currency.

    Same pipeline as the central currency layer, reusable by Backtest /
    Factor tabs on their own history windows: quoted → × quote_scale →
    native → × date-matched historical FX → base. Raises ``ValueError``
    naming the offending ticker instead of guessing.
    """
    panel = native_quoted_panel.reindex(columns=tickers).copy()
    out = pd.DataFrame(index=panel.index)
    for t in tickers:
        col = panel[t] * instruments[t].quote_scale
        try:
            if fx_hist is None:
                if instruments[t].native_currency != base_ccy:
                    raise ValueError(
                        f"No FX history to convert {t} "
                        f"({instruments[t].native_currency} → {base_ccy}).")
                out[t] = col.copy()
            else:
                out[t] = convert_price_series(
                    col, instruments[t].native_currency, base_ccy, fx_hist)
        except ValueError as e:
            raise ValueError(f"{t}: {e}")
    return out


def base_benchmark_series(bench_native, bench_sym, base_ccy, fx_hist):
    """Convert a benchmark history to base currency (native → base).

    Resolves the benchmark's own instrument metadata (caret tickers are
    never pence-scaled). Raises ``ValueError`` with context on failure.
    """
    try:
        inst = get_instrument(bench_sym)
    except ValueError as e:
        raise ValueError(f"Benchmark currency unknown: {e}")
    native = pd.Series(bench_native, dtype=float) * inst.quote_scale
    if inst.native_currency == base_ccy or fx_hist is None:
        if inst.native_currency != base_ccy:
            raise ValueError(
                f"No FX history to convert benchmark {bench_sym} "
                f"({inst.native_currency} → {base_ccy}).")
        return native.rename(bench_sym)
    try:
        return convert_price_series(
            native, inst.native_currency, base_ccy,
            fx_hist).rename(bench_sym)
    except ValueError as e:
        raise ValueError(f"Benchmark {bench_sym}: {e}")


def build_position_valuations(tickers, instruments, native_current_norm,
                              price_view, fx_hist, usd_now, base_ccy):
    """Parse every position and value it in native + base currency.

    Single valuation path reused by every tab (Overview, Security Detail,
    …). Returns ``(positions, valuations, agg, warnings)``. Invalid entries
    are skipped with a warning each — never silently dropped, never fatal.
    """
    positions, valuations, warnings = {}, [], []
    for t in tickers:
        try:
            pos = position_from_dict(t, st.session_state.portfolio[t])
        except ValueError as e:
            warnings.append(f"{t} skipped: {e}")
            continue
        # Enrich missing currency metadata from the live instrument record.
        inst = instruments.get(t)
        if pos.native_currency is None and inst is not None:
            pos = Position(
                ticker=pos.ticker, shares=pos.shares,
                purchase_date=pos.purchase_date,
                purchase_price_native=pos.purchase_price_native,
                purchase_price_source=pos.purchase_price_source,
                manual_purchase_price=pos.manual_purchase_price,
                native_currency=inst.native_currency,
                quote_unit=pos.quote_unit or inst.quote_unit,
                quote_scale=pos.quote_scale
                if pos.quote_scale != 1.0 else inst.quote_scale,
                legacy_avg_cost=pos.legacy_avg_cost)
        if pos.native_currency is None:
            warnings.append(
                f"{t} skipped: native currency unknown (not assumed USD).")
            continue
        positions[t] = pos
        purchase_fx = None
        if (pos.purchase_date is not None
                and pos.purchase_price_native is not None):
            purchase_fx = rate_on_date(
                fx_hist, pos.native_currency, base_ccy, pos.purchase_date) \
                if fx_hist is not None or pos.native_currency != base_ccy \
                else 1.0
            if purchase_fx is None:
                warnings.append(
                    f"{t}: no FX on purchase date — P&L omitted, value kept.")
        quote = price_view.get(t)
        try:
            current_fx = fx_rate_between(
                pos.native_currency, base_ccy, usd_now)
        except ValueError as e:
            warnings.append(f"{t} skipped: {e}")
            del positions[t]
            continue
        try:
            valuations.append(value_position(
                pos, float(native_current_norm[t]),
                quote.source if quote else "Yahoo",
                quote.timestamp if quote else None,
                purchase_fx, current_fx, base_ccy))
        except ValueError as e:
            warnings.append(f"{t} skipped: {e}")
            del positions[t]
    if not valuations:
        detail = "; ".join(warnings) if warnings else "no usable data"
        raise ValueError(f"No positions could be valued — {detail}.")
    agg = value_portfolio(valuations)  # hard reconciliation inside
    return positions, valuations, agg, warnings


# ---------------------------------------------------------------------------
# Institutional allocation helpers (Optimise + Backtest tabs share these)
# ---------------------------------------------------------------------------

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


def _bl_views_editor(prefix: str, tickers: list[str]):
    """Render Black-Litterman reference/delta/tau/views inputs.

    Returns ``(config, error)`` where config is None when inputs are
    invalid. Config: ``{"ref", "delta", "tau", "views"}``.
    """
    ref_choice = st.selectbox(
        "Reference weights (prior)",
        ["Current Portfolio — proxy prior", "Equal Weights — proxy prior"],
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
                     views, rf_rate, min_w, max_w):
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
        tickers, post.posterior, cov_df, delta, rf_rate, min_w, max_w)
    return post, alloc


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
                st.stop()
            except Exception:
                st.error("Could not load market data for the selected period.")
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
        rows.append({
            "Ticker": _v.ticker,
            "Shares": _v.shares,
            "Buy Date": _buy,
            "Buy Px": _buy_px,
            "Now Px": format_money(_v.native_current_price,
                                   _v.native_currency),
            "Source": _v.price_source,
            f"Value ({base_ccy})": _v.base_market_value,
            f"P&L ({base_ccy})": _pnl,
            "P&L (%)": _pnlp,
            "Weight": val_agg["weights"].get(_v.ticker, 0.0),
        })
    pos_df = pd.DataFrame(rows)
    st.dataframe(
        pos_df.style
        .format({
            f"Value ({base_ccy})": _sym + "{:,.2f}",
            f"P&L ({base_ccy})": _sym + "{:+,.2f}",
            "P&L (%)": "{:+.2f}%",
            "Weight": "{:.1%}",
        }, na_rep="—")
        .map(lambda v: "color: #3fb950" if isinstance(v, (int, float)) and v > 0
                  else ("color: #f85149" if isinstance(v, (int, float)) and v < 0 else ""),
                  subset=[f"P&L ({base_ccy})", "P&L (%)"]),
        use_container_width=True,
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
        st.plotly_chart(fig_fx, use_container_width=True)
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
        st.dataframe(pd.DataFrame(_items), use_container_width=True,
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
        st.dataframe(pd.DataFrame(_fxrows), use_container_width=True,
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
        st.plotly_chart(fig_pie, use_container_width=True)

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
        st.plotly_chart(fig_rc, use_container_width=True)

    st.markdown("---")

    col_cov, col_corr = st.columns(2)

    with col_cov:
        st.markdown("### Covariance Matrix")
        st.dataframe(style_matrix(cov, ".5f"), use_container_width=True)

    with col_corr:
        st.markdown("### Correlation Matrix")
        st.dataframe(style_matrix(corr, ".4f"), use_container_width=True)

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
    st.plotly_chart(fig_cum, use_container_width=True)


# ============================================================
# TAB 2 — RISK
# ============================================================
with tab_risk:
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
    st.plotly_chart(fig_dist, use_container_width=True)

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
        st.plotly_chart(fig_mc, use_container_width=True)


# ============================================================
# TAB 3 — OPTIMISE
# ============================================================
with tab_optimise:
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
            target_r_val = st.slider(
                "Target annual return",
                r_min_bound, r_max_bound,
                st.session_state.target_r_val,
                0.005,
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
                v_min_bound, v_max_bound,
                st.session_state.target_v_val,
                0.005,
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
            st.plotly_chart(fig_ef, use_container_width=True)

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
                    use_container_width=True,
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
                             use_container_width=True)

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
                st.dataframe(mtab, use_container_width=True)
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
                st.plotly_chart(fig_w, use_container_width=True)

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
                    st.plotly_chart(fig_rc, use_container_width=True)
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
                st.dataframe(post_df, use_container_width=True)
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
# TAB 4 — SCENARIO ANALYSIS
# ============================================================
with tab_scenario:
    st.markdown("## Scenario & Monte Carlo Simulation")

    sc1, sc2 = st.columns([1, 2])

    with sc1:
        st.markdown("### Simulation Settings")

        horizon_years = st.slider(
            "Time horizon (years)", 1, 10, 5, 1, key="sc_horizon"
        )
        initial_investment = st.number_input(
            f"Initial investment ({currency_symbol(base_ccy).strip() or base_ccy})",
            min_value=1_000.0,
            value=float(max(portfolio_value, 1_000.0)),
            step=1_000.0,
            key="sc_initial",
        )
        monthly_contrib = st.number_input(
            f"Monthly contribution ({currency_symbol(base_ccy).strip() or base_ccy})",
            min_value=0.0, value=0.0, step=100.0,
            key="sc_monthly",
        )
        n_simulations = st.slider(
            "Number of simulations", 1_000, 20_000, 5_000, 1_000, key="sc_nsims"
        )

        scenario_choice = st.selectbox(
            "Scenario",
            ["Normal", "Market Crash", "Bull Market", "High Volatility", "Custom"],
            key="sc_scenario",
        )

        if scenario_choice == "Custom":
            custom_shock = st.slider(
                "Return shock", -0.50, 0.50, 0.0, 0.01,
                format="%+.0f%%", key="sc_custom_shock"
            )
            custom_vol = st.slider(
                "Volatility multiplier", 0.5, 3.0, 1.0, 0.1,
                format="%.1fx", key="sc_custom_vol"
            )
        else:
            custom_shock = 0.0
            custom_vol   = 1.0

        run_btn = st.button("▶  Run Simulation", use_container_width=True)

    with sc2:
        if run_btn or st.session_state.get("sc_ran"):
            st.session_state.sc_ran = True

            # Validate simulation inputs before running.
            try:
                _years = validate_horizon_years(float(horizon_years))
                _n_sims = validate_simulation_count(int(n_simulations))
                _monthly = validate_monthly_contrib(float(monthly_contrib))
                if not np.isfinite(initial_investment) or initial_investment <= 0:
                    raise ValueError("Initial investment must be positive.")
                if scenario_choice == "Custom" and custom_vol <= 0:
                    raise ValueError("Volatility multiplier must be positive.")
            except ValueError as e:
                st.error(f"Invalid simulation settings: {e}")
                st.stop()

            with st.spinner(f"Running {_n_sims:,} simulations…"):
                try:
                    mu_s, cov_s, sc_info = run_predefined_scenario(
                        mu.values, cov.values,
                        scenario=scenario_choice,
                        return_shock=custom_shock,
                        vol_multiplier=custom_vol,
                    )

                    paths = simulate_portfolio_paths(
                        weights=weights,
                        mu_annual=mu_s,
                        cov_annual=cov_s,
                        initial_value=float(initial_investment),
                        years=_years,
                        n_sims=_n_sims,
                        monthly_contrib=_monthly,
                    )

                    metrics, pct_df = summarise_future_metrics(
                        paths, float(initial_investment), _years,
                        monthly_contrib=_monthly,
                    )
                except ValueError as e:
                    st.error(f"Simulation error: {e}")
                    st.stop()
                except Exception:
                    st.error(
                        "The simulation failed unexpectedly. Try fewer "
                        "simulations or a shorter horizon."
                    )
                    st.stop()

            # Key metrics: large currency values get 2 columns x 2 rows
            # so full "$6,589.31"-style values always fit.
            r1a, r1b = st.columns(2)
            r1a.metric("Expected Final Value", fmt_usd(metrics["expected_value"]))
            r1b.metric("Median Outcome",        fmt_usd(metrics["median_value"]))
            r2a, r2b = st.columns(2)
            r2a.metric("Worst 5% Outcome",     fmt_usd(metrics["worst_5pct"]))
            r2b.metric("Probability of Loss",  fmt_pct(metrics["prob_loss"]))
            if metrics.get("contributions", 0) > 0:
                _csym = currency_symbol(base_ccy)
                st.caption(
                    f"Total invested: {fmt_usd(metrics['total_contributed'])} "
                    f"({_csym}{metrics['initial_capital']:,.0f} initial + "
                    f"{_csym}{metrics['contributions']:,.0f} contributions). "
                    f"Expected P&L vs invested: "
                    f"{_csym}{metrics['expected_profit_loss']:+,.0f} "
                    f"({metrics['return_on_invested']:+.1%})."
                )

            st.markdown("---")

            # Explainability
            explanation = explain_scenario_results(
                metrics, sc_info, float(initial_investment),
                monthly_contrib=_monthly, years=_years,
            )
            st.info(explanation)

            # Charts
            st.plotly_chart(
                plot_simulation_paths(
                    paths, pct_df, float(initial_investment), sc_info["label"],
                    total_contributed=metrics["total_contributed"],
                ),
                use_container_width=True,
            )
            st.plotly_chart(
                plot_confidence_bands(pct_df, float(initial_investment), sc_info["label"]),
                use_container_width=True,
            )
            st.plotly_chart(
                plot_final_distribution(paths, metrics, float(initial_investment), sc_info["label"]),
                use_container_width=True,
            )

        else:
            st.markdown(
                """
                ### How to use this tab
                1. Set your **time horizon** and **initial investment** on the left.
                2. Optionally add a **monthly contribution**.
                3. Choose a **scenario** (Normal, Market Crash, Bull Market, High Volatility, or Custom).
                4. Click **▶ Run Simulation**.

                The engine simulates thousands of forward paths using Geometric Brownian Motion
                calibrated to your portfolio's historical returns and covariance.
                """
            )

    # Scenario comparison (always shown)
    st.markdown("---")
    st.markdown("### Scenario Comparison")

    if st.button("Compare All Scenarios", key="sc_compare"):
        try:
            _years_c = validate_horizon_years(float(horizon_years))
            _monthly_c = validate_monthly_contrib(float(monthly_contrib))
            if not np.isfinite(initial_investment) or initial_investment <= 0:
                raise ValueError("Initial investment must be positive.")
        except ValueError as e:
            st.error(f"Invalid comparison settings: {e}")
            st.stop()

        comp_rows = []
        compare_sims = 2_000  # lighter weight for comparison

        try:
            for sc_name in ["Normal", "Market Crash", "Bull Market", "High Volatility"]:
                mu_c, cov_c, info_c = run_predefined_scenario(mu.values, cov.values, sc_name)
                p = simulate_portfolio_paths(
                    weights, mu_c, cov_c, float(initial_investment),
                    years=_years_c, n_sims=compare_sims,
                    monthly_contrib=_monthly_c,
                )
                m, _ = summarise_future_metrics(
                    p, float(initial_investment), _years_c,
                    monthly_contrib=_monthly_c,
                )
                comp_rows.append({
                    "Scenario": info_c["label"],
                    "Expected Value": m["expected_value"],
                    "Median": m["median_value"],
                    "Worst 5%": m["worst_5pct"],
                    "Best 5%": m["best_5pct"],
                    "P(Loss)": m["prob_loss"],
                    "Avg Max Drawdown": m["mean_max_drawdown"],
                })
        except ValueError as e:
            st.error(f"Comparison failed: {e}")
            st.stop()
        except Exception:
            st.error("Scenario comparison failed unexpectedly.")
            st.stop()

        comp_df = pd.DataFrame(comp_rows).set_index("Scenario")
        if _monthly_c > 0:
            st.caption(
                f"Comparison includes monthly contributions of "
                f"{format_money(float(_monthly_c), base_ccy)} "
                f"(loss probabilities vs total invested capital)."
            )
        _csym = currency_symbol(base_ccy)
        st.dataframe(
            comp_df.style.format({
                "Expected Value": f"{_csym}{{:,.0f}}",
                "Median": f"{_csym}{{:,.0f}}",
                "Worst 5%": f"{_csym}{{:,.0f}}",
                "Best 5%": f"{_csym}{{:,.0f}}",
                "P(Loss)": "{:.1%}",
                "Avg Max Drawdown": "{:.1%}",
            }),
            use_container_width=True,
        )


# ============================================================
# TAB 5 — PERFORMANCE vs BENCHMARK
# ============================================================
with tab_perf:
    st.markdown("## Performance vs Benchmark")

    if "bench_ticker" not in st.session_state:
        st.session_state.bench_ticker = "SPY"

    set1, set2, set3 = st.columns([2, 2, 3])
    with set1:
        st.text_input("Benchmark ticker", key="bench_ticker",
                      help="Any Yahoo Finance ticker, e.g. SPY, QQQ, ^GSPC, ^FTSE.")
    with set2:
        roll_win = st.selectbox(
            "Rolling window (trading days)",
            [21, 63, 126, 252], index=1, key="perf_rollwin",
            help="21d ≈ 1 month, 63d ≈ 1 quarter, 126d ≈ half year, 252d ≈ 1 year.",
        )
    with set3:
        bench_period = st.session_state.get("hist_period_tab1", "2y")
        st.caption(
            f"Benchmark is loaded over the same **{bench_period}** history period "
            "as the portfolio and all comparisons use only common trading dates."
        )

    try:
        bench_sym = validate_ticker_symbol(st.session_state.bench_ticker)
    except ValueError as e:
        st.error(f"Invalid benchmark ticker: {e}")
        bench_sym = None

    if bench_sym is not None:
        with st.spinner(f"Loading benchmark {bench_sym}…"):
            try:
                bench_prices = load_benchmark_data(bench_sym, bench_period)
            except ValueError as e:
                st.error(f"Benchmark error: {e}")
                bench_prices = None
            except Exception:
                st.error(
                    "Could not load benchmark data (network or API error). "
                    "Check the ticker and your connection."
                )
                bench_prices = None

        if bench_prices is not None:
            try:
                # Translate the benchmark into base currency first, so all
                # relative metrics live in the investor's return space.
                bench_prices = base_benchmark_series(
                    bench_prices, bench_sym, base_ccy, ccy.get("fx_hist"))
                bench_rets_full = log_returns_from_prices(bench_prices)
                p_al, b_al = align_return_series(port_daily_returns, bench_rets_full)
            except ValueError as e:
                st.error(f"Benchmark comparison unavailable: {e}")
                p_al = None
            else:
                try:
                    perf = summarise_performance(
                        port_daily_returns, bench_rets_full, rf_rate)
                except ValueError as e:
                    st.warning(f"Relative metrics unavailable: {e}")
                    perf = summarise_performance(
                        port_daily_returns, None, rf_rate)
                perf_p = perf["portfolio"]
                perf_b = perf["benchmark"]
                perf_r = perf["relative"]

                # ---- Growth of $10,000 ----
                st.markdown("---")
                _g_sym = currency_symbol(base_ccy)
                st.markdown(f"### Growth of {_g_sym}10,000")
                g_p = growth_of_capital(p_al, 10_000.0)
                g_b = growth_of_capital(b_al, 10_000.0)
                gh1, gh2 = st.columns(2)
                gh1.metric("Portfolio", fmt_usd(float(g_p.iloc[-1])),
                           help=f"Final value of {_g_sym}10,000 in the portfolio.")
                gh2.metric(bench_sym, fmt_usd(float(g_b.iloc[-1])),
                           help=f"Final value of {_g_sym}10,000 in {bench_sym}.")
                st.plotly_chart(
                    plot_growth_comparison(g_p, g_b, 10_000.0, bench_sym,
                                           currency=base_ccy),
                    use_container_width=True,
                )
                st.caption(
                    f"Both series start at {_g_sym}10,000 on {g_p.index[0].date()} "
                    f"({len(p_al):,} common trading dates)."
                )

                # ---- Performance summary panel ----
                st.markdown("---")
                st.markdown("### Performance Summary")
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.markdown("#### PORTFOLIO PERFORMANCE")
                    st.metric("Annualised Return", fmt_pct(perf_p["ann_return"]),
                              help="Mean daily log return × 252.")
                    st.metric("Annualised Volatility", fmt_pct(perf_p["ann_vol"]),
                              help="Daily std × √252.")
                    st.metric("Sharpe", f"{perf_p['sharpe']:.2f}",
                              help="Excess return over cash per unit of total volatility.")
                    st.metric("Sortino", f"{perf_p['sortino']:.2f}",
                              help="Excess return per unit of downside volatility (losses only).")
                    st.metric("Calmar", f"{perf_p['calmar']:.2f}",
                              help="Annualised return per unit of maximum drawdown.")
                    st.metric("Maximum Drawdown", fmt_pct(perf_p["max_drawdown"]),
                              help="Worst peak-to-trough fall on the aligned history.")
                with s2:
                    st.markdown("#### BENCHMARK")
                    if perf_b is not None:
                        st.metric("Annualised Return", fmt_pct(perf_b["ann_return"]))
                        st.metric("Annualised Volatility", fmt_pct(perf_b["ann_vol"]))
                        st.metric("Sharpe", f"{perf_b['sharpe']:.2f}")
                        st.metric("Maximum Drawdown", fmt_pct(perf_b["max_drawdown"]))
                    else:
                        st.caption("Benchmark metrics unavailable.")
                with s3:
                    st.markdown("#### RELATIVE PERFORMANCE")
                    if perf_r["beta"] is not None:
                        beta = perf_r["beta"]
                        st.metric("Beta", f"{beta:.2f}",
                                  help="Portfolio sensitivity to benchmark moves.")
                        gap = (beta - 1.0) * 100
                        direction = "more" if gap >= 0 else "less"
                        st.caption(
                            f"Beta {beta:.2f}: the portfolio has historically moved "
                            f"≈{abs(gap):.0f}% {direction} than {bench_sym}. "
                            "Association, not causality."
                        )
                        st.metric(
                            "Historical Alpha", f"{perf_r['historical_alpha']:+.2%}",
                            help="Past CAPM-style excess return. Historical and "
                                 "model-dependent — not expected future alpha.",
                        )
                        st.metric("Tracking Error", fmt_pct(perf_r["tracking_error"]),
                                  help="Annualised std of portfolio-minus-benchmark returns.")
                        st.metric("Information Ratio", f"{perf_r['information_ratio']:.2f}",
                                  help="Annualised active return per unit of tracking error.")
                    else:
                        st.caption("Relative metrics unavailable for this benchmark.")

                # ---- Portfolio vs benchmark table ----
                st.markdown("---")
                st.markdown("### Portfolio vs Benchmark")
                p_simple = np.exp(p_al) - 1
                b_simple = np.exp(b_al) - 1

                def _fmt_cell(metric: str, v: float) -> str:
                    if v is None or not np.isfinite(v):
                        return "n/a"
                    if metric == "Sharpe":
                        return f"{v:.2f}"
                    if "Day" in metric:
                        return f"{v:+.2%}"
                    return f"{v:.2%}"

                _metrics = ["Annualised Return", "Volatility", "Sharpe",
                            "Max Drawdown", "Best Day", "Worst Day"]
                _p_vals = [perf_p["ann_return"], perf_p["ann_vol"],
                           perf_p["sharpe"], perf_p["max_drawdown"],
                           p_simple.max(), p_simple.min()]
                _b_vals = ([perf_b["ann_return"], perf_b["ann_vol"],
                            perf_b["sharpe"], perf_b["max_drawdown"],
                            b_simple.max(), b_simple.min()]
                           if perf_b is not None else [np.nan] * 6)
                perf_table = pd.DataFrame(
                    {
                        "Portfolio": [_fmt_cell(m, v) for m, v in zip(_metrics, _p_vals)],
                        bench_sym: [_fmt_cell(m, v) for m, v in zip(_metrics, _b_vals)],
                    },
                    index=_metrics,
                )
                st.dataframe(perf_table, use_container_width=True)
                st.caption(
                    f"Best day: portfolio {p_simple.idxmax().date()} "
                    f"({p_simple.max():.2%}) vs {bench_sym} {b_simple.idxmax().date()} "
                    f"({b_simple.max():.2%}). Worst day: portfolio "
                    f"{p_simple.idxmin().date()} ({p_simple.min():.2%}) vs "
                    f"{bench_sym} {b_simple.idxmin().date()} ({b_simple.min():.2%})."
                )

                # ---- Drawdown chart + underwater episodes ----
                st.markdown("---")
                st.markdown("### Drawdowns")
                dd_p = drawdown_series(g_p)
                dd_b = drawdown_series(g_b)
                st.plotly_chart(
                    plot_drawdown(dd_p, dd_b, bench_sym, perf_p["max_drawdown"]),
                    use_container_width=True,
                )
                st.markdown("#### Five Largest Drawdowns (Portfolio, Full History)")
                wealth_full = growth_of_capital(port_daily_returns, 1.0)
                try:
                    uw = underwater_episodes(wealth_full, top_n=5)
                except ValueError as e:
                    st.warning(f"Underwater analysis unavailable: {e}")
                    uw = None
                if uw is not None and not uw.empty:
                    disp = uw.copy()
                    for col in ("Peak", "Trough"):
                        disp[col] = pd.to_datetime(disp[col]).dt.strftime("%Y-%m-%d")
                    disp["Recovery"] = disp["Recovery"].apply(
                        lambda d: "Not yet recovered"
                        if pd.isna(d) else pd.Timestamp(d).strftime("%Y-%m-%d")
                    )
                    st.dataframe(
                        disp.style.format({
                            "Drawdown %": "{:.2%}",
                            "Duration (days)": "{:.0f}",
                        }),
                        use_container_width=True,
                    )
                else:
                    st.caption("No drawdown episodes found.")

                # ---- Rolling analytics ----
                st.markdown("---")
                st.markdown(f"### Rolling Analytics ({int(roll_win)}-Day Window)")
                if len(p_al) < int(roll_win):
                    st.warning(
                        f"Only {len(p_al)} common observations — need at least "
                        f"{int(roll_win)} for the selected rolling window. "
                        "Choose a shorter window or load a longer history."
                    )
                else:
                    try:
                        rr_p = rolling_annualised_return(p_al, int(roll_win))
                        rr_b = rolling_annualised_return(b_al, int(roll_win))
                        rv_p = rolling_volatility(p_al, int(roll_win))
                        rv_b = rolling_volatility(b_al, int(roll_win))
                        rs_p = rolling_sharpe(p_al, rf_rate, int(roll_win))
                        rs_b = rolling_sharpe(b_al, rf_rate, int(roll_win))
                    except ValueError as e:
                        st.warning(f"Rolling analytics unavailable: {e}")
                    else:
                        def _roll_fig(p_s, b_s, title, ytitle, fmt):
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=p_s.index, y=p_s.values, mode="lines",
                                name="Portfolio", line=dict(color="#58a6ff", width=2)))
                            fig.add_trace(go.Scatter(
                                x=b_s.index, y=b_s.values, mode="lines",
                                name=bench_sym, line=dict(color="#f0883e", width=1.5)))
                            fig.update_layout(**CHART_THEME, title=title,
                                              xaxis_title="Date", yaxis_title=ytitle,
                                              yaxis_tickformat=fmt)
                            return fig

                        st.plotly_chart(_roll_fig(
                            rr_p, rr_b, "Rolling Annualised Return",
                            "Return", ".0%"), use_container_width=True)
                        st.plotly_chart(_roll_fig(
                            rv_p, rv_b, "Rolling Annualised Volatility",
                            "Volatility (σ)", ".0%"), use_container_width=True)
                        st.plotly_chart(_roll_fig(
                            rs_p, rs_b, "Rolling Sharpe Ratio",
                            "Sharpe", ".1f"), use_container_width=True)

                # ---- Best / worst periods ----
                st.markdown("---")
                st.markdown("### Best / Worst Periods (Portfolio)")
                try:
                    bw = best_worst_periods(port_daily_returns)
                except ValueError as e:
                    st.warning(f"Best/worst periods unavailable: {e}")
                else:
                    # Day/month values carry a date suffix ("% — date"), so
                    # use 2 columns x 2 rows instead of one tight row of 4.
                    bw1, bw2 = st.columns(2)
                    bw1.metric("Best Day",
                               f"{bw['best_day'][1]:+.2%} — {bw['best_day'][0].date()}")
                    bw2.metric("Worst Day",
                               f"{bw['worst_day'][1]:+.2%} — {bw['worst_day'][0].date()}")
                    bw3, bw4 = st.columns(2)
                    bw3.metric("Best Month", f"{bw['best_month'][1]:+.2%} — {bw['best_month'][0]}")
                    bw4.metric("Worst Month",
                               f"{bw['worst_month'][1]:+.2%} — {bw['worst_month'][0]}")
                    st.caption(
                        "Monthly returns compound daily log returns "
                        "(exp of summed logs − 1), never simple averages."
                    )

    # ---- Metric glossary ----
    with st.expander("Metric glossary"):
        st.markdown(
            """
            **Sharpe** — excess return over cash per unit of total volatility. Higher is better.
            **Sortino** — like Sharpe but only downside volatility counts; upside swings are not penalised.
            **Calmar** — annualised return divided by the worst peak-to-trough drawdown.
            **Beta** — how strongly the portfolio has historically moved with the benchmark (1.0 = in step).
            **Alpha (historical)** — past return left over after accounting for beta; model-dependent, not a forecast.
            **Tracking Error** — annualised volatility of portfolio-minus-benchmark returns.
            **Information Ratio** — active return per unit of tracking error.
            **VaR** — loss threshold crossed only in the worst 5% of periods.
            **CVaR / Expected Shortfall** — average loss once the VaR threshold is crossed.
            **Maximum Drawdown** — largest observed peak-to-trough fall; 0 means a prior peak.
            """
        )


# ============================================================
# TAB 6 — SECURITY DETAIL
# ============================================================
with tab_security:
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
        st.stop()
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
        st.plotly_chart(fig_sec, use_container_width=True)

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
# TAB 7 — BACKTEST
# ============================================================
def _fmt_bt_date(d) -> str:
    """Format a drawdown date for display (NaT → 'not yet recovered')."""
    try:
        ts = pd.Timestamp(d)
        if pd.isna(ts):
            return "not yet recovered"
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return "n/a"


with tab_backtest:
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
        st.stop()
    except Exception:
        st.error("Could not load backtest history (network or API error). "
                 "Check your connection and try again.")
        st.stop()

    bt_common = bt_full.dropna(how="any")
    if len(bt_common) < 2:
        st.error("No overlapping history available for backtesting.")
        st.stop()
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

    # ---- Prepare the common-date window (real data only) ----
    # Convert native history to base currency FIRST, so analytics run on
    # base-currency returns (FX movement included), never mixed natives.
    try:
        bt_base_full = base_price_panel(
            bt_full, tickers, instruments, ccy.get("fx_hist"), base_ccy)
        bt_prices, bt_notes = prepare_backtest_data(
            bt_base_full, tickers, pd.Timestamp(bt_start), pd.Timestamp(bt_end))
    except ValueError as e:
        st.error(f"Backtest setup: {e}")
        st.stop()
    for _note in bt_notes:
        st.warning(_note)
    if any(instruments[t].native_currency != base_ccy for t in tickers):
        st.caption(
            f"Holdings translated to {base_ccy} with date-matched historical "
            "FX before simulation, so currency movement is part of returns.")

    try:
        bench_sym_bt = validate_ticker_symbol(st.session_state.bt_benchmark)
    except ValueError as e:
        st.error(f"Invalid benchmark ticker: {e}")
        bench_sym_bt = None

    bench_full = None
    if bench_sym_bt is not None and show_bench:
        try:
            _bench_native = load_benchmark_data(bench_sym_bt, "max")
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
        st.stop()
    if not _results:
        st.info("Select at least one strategy to run the backtest.")
        st.stop()

    # ---- Growth of Initial Capital (actual wealth paths) ----
    st.markdown("---")
    st.markdown(f"### Growth of {fmt_usd(float(bt_capital))}")
    st.plotly_chart(plot_backtest_growth(_results, float(bt_capital),
                                                currency=base_ccy),
                    use_container_width=True)
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
    st.dataframe(_disp, use_container_width=True)
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
        use_container_width=True,
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
                st.dataframe(_evd, use_container_width=True)
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
                st.dataframe(_trd, use_container_width=True)
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
                    st.plotly_chart(_fig, use_container_width=True)
        with st.expander("Weight Drift"):
            _wdf = _results[_rb_choice].weights
            if _wdf is None or _wdf.empty:
                st.caption("No weight history for this strategy.")
            else:
                st.plotly_chart(
                    plot_weight_drift(_wdf, f"Weight Drift — {_rb_choice}"),
                    use_container_width=True)
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
        # and all test wealth paths then live in base-currency space.
        oos_base_full = base_price_panel(
            bt_full, tickers, instruments, ccy.get("fx_hist"), base_ccy)
        oos_est, oos_info = estimation_window(
            oos_base_full, tickers, oos_T, float(oos_lookback))
    except ValueError as e:
        st.error(f"Estimation window: {e}")
        st.stop()
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
        st.stop()
    for _note in oos_notes:
        st.warning(_note)

    oos_bench_full = None
    if bench_sym_bt is not None:
        try:
            _oos_bench_native = load_benchmark_data(bench_sym_bt, "max")
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
        st.stop()

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
        st.stop()
    if not oos_results:
        st.info("Select at least one out-of-sample strategy.")
        st.stop()

    st.markdown(f"### Out-of-Sample Growth of {fmt_usd(_oos_cap)}")
    st.plotly_chart(plot_backtest_growth(oos_results, _oos_cap,
                                                currency=base_ccy),
                    use_container_width=True)
    st.caption(
        f"Test window: {oos_prices.index[0].date()} to "
        f"{oos_prices.index[-1].date()} ({len(oos_prices):,} trading days); "
        "all lines start at the same initial capital."
    )

    st.markdown("### Out-of-Sample Comparison")
    _oos_table = compare_backtests(oos_results)
    _oos_disp = pd.DataFrame(index=_oos_table.index)
    for _col in _oos_table.columns:
        _is_bench = oos_results[_col].is_benchmark
        _cells = []
        for _metric in _oos_table.index:
            _v = _oos_table.loc[_metric, _col]
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
        _oos_disp[_col] = _cells
    st.dataframe(_oos_disp, use_container_width=True)
    st.caption(
        "Benchmark Total Costs show as N/A (the passive leg is not "
        "cost-modelled). Frozen estimation weights were never re-estimated "
        "during the test."
    )


# ============================================================
# TAB 8 — FACTOR ANALYSIS
# ============================================================
with tab_factors:
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
        st.stop()
    except Exception:
        st.error("Could not load price history (network or API error). "
                 "Check your connection and try again.")
        st.stop()
    # Translate history into base currency up front: factor regressions
    # then describe the investor's base-currency experience. The French
    # factors themselves are never converted (US factor datasets).
    try:
        fa_full = base_price_panel(
            fa_full, tickers, instruments, ccy.get("fx_hist"), base_ccy)
    except ValueError as e:
        st.error(f"Factor data error: {e}")
        st.stop()

    fa_common = fa_full.dropna(how="any")
    if len(fa_common) < 2:
        st.error("No overlapping history available for factor analysis.")
        st.stop()
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
                st.stop()
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
        st.stop()
    st.caption(f"Analysing **{fa_source_label}** "
               f"({len(fa_values)} trading days).")

    # ---- French factor data (cached; percent → decimal explicitly) ----
    try:
        with st.spinner("Loading French factor data…"):
            _french = load_factor_data()
    except ValueError as e:
        st.error(f"Factor data unavailable: {e}")
        st.stop()
    except Exception:
        st.error("Could not load factor data (network or API error). "
                 "Other tabs are unaffected.")
        st.stop()
    try:
        _factors = to_decimal_returns(
            pd.concat([_french["ff5"], _french["mom"]], axis=1,
                      sort=False).sort_index())
    except ValueError as e:
        st.error(f"Factor data malformed: {e}")
        st.stop()

    # ---- Simple returns + inner-join alignment + excess vs French RF ----
    _needed = MODEL_FACTORS[fa_model_key]
    try:
        _port_simple = simple_returns_from_wealth(fa_values)
        _y_al, _X_al = align_factor_returns(
            _port_simple, _factors[[*_needed, "RF"]])
    except ValueError as e:
        st.error(f"Factor alignment: {e}")
        st.stop()
    _rf_al = _X_al["RF"]
    _XA = _X_al.drop(columns=["RF"])
    _y_excess = _y_al - _rf_al

    try:
        _warn_small = check_min_observations(len(_y_excess), len(_needed))
    except ValueError as e:
        st.error(f"Not enough data: {e}")
        st.stop()
    if _warn_small:
        st.warning(
            f"Only {len(_y_excess)} overlapping observations (< 252) — "
            "treat loadings as tentative.")
    try:
        _res = run_model(fa_model_key, _y_excess, _XA)
    except ValueError as e:
        st.error(f"Regression failed: {e}")
        st.stop()

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
    st.dataframe(pd.DataFrame(_rows), use_container_width=True,
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
                        use_container_width=True)
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
            use_container_width=True)
        if _show_roll_alpha:
            _alpha_ann = (_roll["alpha"] + 1.0) ** 252 - 1.0
            st.plotly_chart(
                plot_rolling_exposure(
                    _roll.index, _alpha_ann,
                    f"Historical Alpha, ann. ({int(fa_roll_win)}d)"),
                use_container_width=True)
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
                    st.dataframe(_comp, use_container_width=True,
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

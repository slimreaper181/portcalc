"""pipeline — cached data loaders + composed portfolio data plumbing.

Streamlit-aware composition layer between raw ``data.*`` fetchers and the
pure ``analytics.*`` engines:

* ``load_*``: ``st.cache_data`` fetch wrappers (single network source —
  no fetch logic is duplicated anywhere else).
* ``build_*`` / ``base_*``: multi-source assembly (prices + instruments +
  FX + valuations) reused by every tab.

Analytics here stay numerical; widgets/session-state reads are limited to
cache scope and the portfolio-backed core builder (documented below).
"""

from __future__ import annotations

from datetime import timezone

import numpy as np
import pandas as pd
import streamlit as st

from analytics.currency import (
    convert_price_series,
    convert_value,
    fx_rate_between,
    rate_on_date,
    validate_base_currency,
)
from analytics.portfolio import (
    portfolio_expected_return,
    portfolio_std,
    risk_contributions,
    sharpe_ratio,
    weights_from_shares,
)
from analytics.positions import (
    Position,
    position_from_dict,
    value_portfolio,
    value_position,
)
from analytics.returns import (
    annualised_cov_matrix,
    annualised_mean_returns,
    correlation_matrix,
    daily_returns,
)
from analytics.validation import (
    align_market_data,
    aligned_portfolio_returns,
    canonical_tickers,
    sanitize_prices,
    validate_risk_free_rate,
)
from data.alpaca import (
    AlpacaError,
    build_price_view,
    fetch_latest_trades,
    get_alpaca_credentials,
    is_alpaca_supported_symbol,
)
from data.factors import fetch_french_factors
from data.fx import (
    FX_PAIRS,
    fetch_current_fx,
    fetch_fx_history,
    get_instrument,
)
from data.market_data import (
    fetch_benchmark_history,
    fetch_current_prices,
    fetch_dividends,
    fetch_exchange_code,
    fetch_price_history,
    fetch_raw_closes,
    fetch_risk_free_rate,
    fetch_splits,
)

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

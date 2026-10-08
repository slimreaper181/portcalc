"""
market_data.py
--------------
Fetches live and historical stock price data via yfinance.

All functions validate ticker symbols up front and raise ``ValueError``
with actionable messages for empty results, unknown tickers, or unusable
price data — the Streamlit UI converts these into friendly warnings.
"""

import yfinance as yf
import pandas as pd
import numpy as np

from analytics.validation import canonical_tickers, validate_ticker_symbol


def _normalise_requested(tickers: list[str]) -> list[str]:
    """Validate and canonicalise a requested ticker list."""
    return canonical_tickers(list(tickers))


def fetch_price_history(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.

    Args:
        tickers: List of stock ticker symbols (e.g. ['AAPL', 'MSFT'])
        period:  yfinance period string ('1y', '2y', '6mo', etc.)

    Returns:
        DataFrame of daily adjusted close prices with columns in canonical
        (sorted) ticker order and only complete (all-ticker) trading days.

    Raises:
        ValueError: if tickers are invalid, data cannot be downloaded, a
            ticker has no history at all, or no overlapping history remains.
    """
    wanted = _normalise_requested(tickers)

    try:
        raw = yf.download(wanted, period=period, auto_adjust=True, progress=False)
    except Exception as e:
        raise ValueError(
            f"Could not download market data (network/API error: {e}). "
            "Check your connection and try again."
        )
    if raw is None or raw.empty:
        raise ValueError(
            f"No market data returned for {', '.join(wanted)}. "
            "Check ticker symbols and network connectivity."
        )

    # yfinance returns MultiIndex when >1 ticker
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]]
            prices.columns = wanted
    except KeyError as e:
        raise ValueError(f"Unexpected price response from market data API: {e}")

    # Normalise column labels (yfinance may return different casing/order)
    prices.columns = [str(c).strip().upper() for c in prices.columns]
    # Reindex to canonical order; missing tickers -> absent columns.
    prices = prices.reindex(columns=wanted)

    missing = [t for t in wanted if t not in prices.columns or prices[t].dropna().empty]
    if missing:
        raise ValueError(
            f"No price history available for: {', '.join(missing)}. "
            "Check the ticker symbols (delisted or invalid symbols have no data)."
        )

    # Keep only days where every ticker traded (aligned panel).
    prices = prices.dropna(how="any")
    if prices.empty:
        raise ValueError(
            "No overlapping price history across the selected tickers "
            "(their trading calendars do not intersect)."
        )
    if ((prices <= 0) | (~np.isfinite(prices))).any().any():
        bad = prices.columns[
            ((prices <= 0) | (~np.isfinite(prices))).any()
        ].tolist()
        raise ValueError(f"Invalid (zero/negative/NaN) prices for: {', '.join(bad)}.")
    return prices


def fetch_benchmark_history(ticker: str, period: str = "2y") -> pd.Series:
    """
    Download adjusted-close history for a single benchmark ticker.

    Args:
        ticker: Benchmark symbol (e.g. 'SPY', 'QQQ', '^GSPC').
        period:  yfinance period string — callers must pass the exact same
            period used for the portfolio so date ranges never diverge.

    Returns:
        Series of daily adjusted close prices named after the ticker.

    Raises:
        ValueError: if the ticker is invalid or no history is available.
    """
    sym = validate_ticker_symbol(ticker)
    df = fetch_price_history([sym], period=period)
    return df[sym].copy()


def fetch_current_prices(tickers: list[str]) -> dict[str, float]:
    """
    Fetch the most recent closing price for each ticker.

    Returns:
        Dict mapping ticker -> latest price (USD). Prices that cannot be
        retrieved are NaN; callers must handle NaN/zero prices explicitly.

    Raises:
        ValueError: if the ticker list itself is invalid.
    """
    wanted = _normalise_requested(tickers)
    prices: dict[str, float] = {}
    for t in wanted:
        price = np.nan
        try:
            info = yf.Ticker(t).fast_info
            price = float(info.last_price)
        except Exception:
            price = np.nan
        if not np.isfinite(price):
            try:
                hist = yf.Ticker(t).history(period="5d")
                if hist is not None and not hist.empty:
                    price = float(hist["Close"].iloc[-1])
            except Exception:
                price = np.nan
        prices[t] = float(price) if np.isfinite(price) else np.nan
    failed = [t for t, p in prices.items() if not np.isfinite(p) or p <= 0]
    if failed and len(failed) == len(wanted):
        raise ValueError(
            f"Could not retrieve current prices for any ticker "
            f"({', '.join(failed)}). Check symbols and connectivity."
        )
    return prices


def fetch_exchange_code(ticker: str) -> str | None:
    """
    Best-effort yfinance exchange code for a ticker (e.g. ``NMS``, ``NYQ``).

    Used only as a hint for TradingView symbol resolution. Returns ``None``
    when the lookup fails or yields nothing usable — callers must fall back
    gracefully. Never raises for network/API problems (only for invalid
    ticker strings).
    """
    sym = validate_ticker_symbol(ticker)
    try:
        code = yf.Ticker(sym).fast_info.exchange
    except Exception:
        return None
    if not code or not isinstance(code, str):
        return None
    code = code.strip().upper()
    return code or None


def _naive_utc_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalise an index to tz-naive UTC for cross-source alignment."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def fetch_raw_closes(ticker: str) -> pd.Series:
    """
    Raw (unadjusted) daily closes — actual traded prices including splits.

    Unlike adjusted closes, these reflect what a broker fill would have
    been (up to intraday timing). Index is tz-naive UTC, ascending, unique.
    """
    sym = validate_ticker_symbol(ticker)
    try:
        hist = yf.Ticker(sym).history(period="max", auto_adjust=False)
    except Exception as e:
        raise ValueError(
            f"Could not download raw history for {sym} "
            f"(network/API error: {e})."
        )
    if hist is None or hist.empty or "Close" not in hist.columns:
        raise ValueError(f"No raw price history available for {sym}.")
    closes = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    if closes.empty:
        raise ValueError(f"No raw price history available for {sym}.")
    closes.index = _naive_utc_index(closes.index)
    closes = closes.sort_index()
    closes = closes[~closes.index.duplicated(keep="last")]
    return closes.rename(sym)


def fetch_splits(ticker: str) -> pd.Series:
    """Stock-split series (e.g. 4.0 = 4-for-1); empty when none occurred."""
    sym = validate_ticker_symbol(ticker)
    try:
        splits = yf.Ticker(sym).splits
    except Exception as e:
        raise ValueError(f"Could not download split history for {sym} ({e}).")
    if splits is None or len(splits) == 0:
        return pd.Series(dtype=float)
    s = pd.to_numeric(splits, errors="coerce").dropna()
    s = s[s > 0]
    try:
        s.index = _naive_utc_index(s.index)
    except Exception:
        pass
    return s.sort_index()


def fetch_dividends(ticker: str) -> pd.Series:
    """Cash dividends per share by date; empty when none paid."""
    sym = validate_ticker_symbol(ticker)
    try:
        divs = yf.Ticker(sym).dividends
    except Exception as e:
        raise ValueError(
            f"Could not download dividend history for {sym} ({e})."
        )
    if divs is None or len(divs) == 0:
        return pd.Series(dtype=float)
    d = pd.to_numeric(divs, errors="coerce").dropna()
    d = d[d > 0]
    try:
        d.index = _naive_utc_index(d.index)
    except Exception:
        pass
    return d.sort_index()


def fetch_risk_free_rate() -> float:
    """
    Approximate the annualised risk-free rate using the 13-week US T-Bill (^IRX).
    Returns a decimal (e.g. 0.053 for 5.3%).

    Falls back to 5% if the quote is unavailable or implausible.
    """
    try:
        tbill = yf.Ticker("^IRX").history(period="5d")
        if tbill is None or tbill.empty:
            return 0.05
        rate = float(tbill["Close"].iloc[-1]) / 100
    except Exception:
        return 0.05  # sensible default
    if not np.isfinite(rate) or rate < -0.05 or rate > 0.25:
        return 0.05
    return rate

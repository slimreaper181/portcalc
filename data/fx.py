"""
fx.py
-----
Foreign-exchange data, instrument metadata and raw corporate-action history.

This module is the canonical source for everything currency-related that
requires network I/O or exchange metadata:

* ``Instrument`` — one canonical record per ticker (exchange, native
  currency, quote unit/scale, timezone, Alpaca support). Nothing else in
  the codebase should maintain competing ticker assumptions for these
  fields.
* FX retrieval — current + historical Yahoo FX series, pair routing with
  USD triangulation, timestamps in UTC.
* Raw (unadjusted) closes, splits and dividends for purchase-price and
  dividend estimation.

Pure conversion math lives in ``analytics.currency``; this module only
fetches and structures data. All functions validate ticker symbols up
front and raise ``ValueError`` with actionable messages — the Streamlit
UI converts these into friendly warnings.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from analytics.validation import canonical_tickers, validate_ticker_symbol

#: Officially supported portfolio base currencies.
SUPPORTED_BASE_CURRENCIES = ("USD", "GBP", "EUR")

#: Yahoo FX pairs quoted as USD-per-unit, e.g. GBPUSD=X ≈ 1.27 means
#: one GBP buys 1.27 USD. All conversions route through USD legs.
FX_PAIRS = {
    "GBPUSD=X": ("GBP", "USD"),
    "EURUSD=X": ("EUR", "USD"),
}

#: Yahoo exchange code → IANA timezone for session-date handling.
EXCHANGE_TIMEZONES = {
    "NMS": "America/New_York",
    "NYQ": "America/New_York",
    "NYS": "America/New_York",
    "NASDAQ": "America/New_York",
    "NYSE": "America/New_York",
    "ASE": "America/New_York",
    "PCX": "America/New_York",
    "LSE": "Europe/London",
    "GER": "Europe/Berlin",
    "XETRA": "Europe/Berlin",
    "XFRA": "Europe/Berlin",
    "HAM": "Europe/Berlin",
    "MUN": "Europe/Berlin",
    "EPA": "Europe/Paris",
    "XPAR": "Europe/Paris",
}


@dataclass(frozen=True)
class Instrument:
    """Canonical instrument metadata for one ticker.

    * ``native_currency`` — ISO code of actual money (GBP, never GBp).
    * ``quote_unit`` — quotation label as traded (``GBp`` for LSE pence).
    * ``quote_scale`` — multiply a quoted price to get native currency
      (0.01 for GBp pence → pounds, 1.0 otherwise).
    """

    ticker: str
    exchange: str | None
    native_currency: str
    quote_unit: str
    quote_scale: float
    timezone: str
    alpaca_supported: bool
    yahoo_symbol: str


def _normalise_currency(raw: str | None) -> tuple[str, str, float]:
    """Map a Yahoo currency code to (native_currency, quote_unit, scale).

    Raises:
        ValueError: if no usable currency was reported (never assume USD).
    """
    if not raw or not isinstance(raw, str):
        raise ValueError(
            "No currency metadata reported for this ticker — refusing to "
            "assume USD. Check the symbol."
        )
    code = raw.strip()
    upper = code.upper()
    if upper in ("GBP", "GBX"):
        return "GBP", "GBp", 0.01
    if upper == "ZAC":
        return "ZAR", "ZAc", 0.01
    if len(code) == 3 and code.isalpha():
        return code.upper(), code.upper(), 1.0
    raise ValueError(
        f"Unrecognised currency code {raw!r} — refusing to guess. "
        "Check the symbol."
    )


def get_instrument(ticker: str) -> Instrument:
    """Resolve canonical metadata for a ticker via yfinance.

    Index tickers (``^...``) are treated as pure index points: quote scale
    is forced to 1.0 even if the vendor labels the currency in pence.

    Raises:
        ValueError: on invalid tickers or when currency metadata is missing.
    """
    from data.alpaca import is_alpaca_supported_symbol

    sym = validate_ticker_symbol(ticker)
    try:
        info = yf.Ticker(sym).fast_info
        raw_ccy = info.currency
        exchange = info.exchange
    except Exception as e:
        raise ValueError(
            f"Could not retrieve metadata for {sym} "
            f"(network/API error: {e}). Check your connection."
        )
    native, unit, scale = _normalise_currency(raw_ccy)
    if sym.startswith("^"):
        # Index points are not tradeable pence — never apply a pence scale.
        scale = 1.0
        unit = native
    tz = EXCHANGE_TIMEZONES.get(str(exchange).strip().upper(), "UTC")
    return Instrument(
        ticker=sym,
        exchange=str(exchange) if exchange else None,
        native_currency=native,
        quote_unit=unit,
        quote_scale=float(scale),
        timezone=tz,
        alpaca_supported=is_alpaca_supported_symbol(sym),
        yahoo_symbol=sym,
    )


def _as_utc_naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalise an index to tz-naive UTC for cross-source alignment."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def _extract_close_frame(raw: pd.DataFrame, label: str) -> pd.DataFrame:
    """Pull Close columns out of a yfinance download (single/multi ticker)."""
    if raw is None or raw.empty:
        raise ValueError(f"No market data returned for {label}.")
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            out = raw["Close"].copy()
        except KeyError as e:
            raise ValueError(f"Unexpected price response for {label}: {e}")
    else:
        col = "Close" if "Close" in raw.columns else None
        if col is None:
            raise ValueError(f"Unexpected price response for {label}.")
        out = raw[[col]].copy()
    return out


def fetch_fx_history(
    pairs: list[str] | None = None, period: str = "max"
) -> pd.DataFrame:
    """Download daily USD-per-unit FX history.

    Args:
        pairs: Yahoo FX symbols (subset of ``FX_PAIRS``). ``None`` fetches all.
        period: yfinance period (``"max"`` for history, ``"5d"`` for a
            lightweight recent snapshot).

    Returns:
        DataFrame indexed by tz-naive UTC date with one column per
        foreign currency (``GBP``, ``EUR``) holding USD-per-unit rates.

    Raises:
        ValueError: on download failure or missing pairs.
    """
    wanted = list(pairs) if pairs else list(FX_PAIRS)
    unknown = [p for p in wanted if p not in FX_PAIRS]
    if unknown:
        raise ValueError(f"Unsupported FX pair(s): {', '.join(unknown)}.")
    try:
        raw = yf.download(wanted, period=period, auto_adjust=True,
                          progress=False)
    except Exception as e:
        raise ValueError(
            f"Could not download FX data (network/API error: {e}). "
            "Check your connection and try again."
        )
    closes = _extract_close_frame(raw, ", ".join(wanted))
    if isinstance(closes, pd.Series):
        closes = closes.to_frame()
    # Map Yahoo columns back to single-ticker labels. A single-pair
    # download yields a lone "Close" column — assign it directly.
    if len(wanted) == 1 and (
            len(closes.columns) == 1 and closes.columns[0] not in wanted):
        closes.columns = wanted
    colmap: dict[str, str] = {}
    for c in closes.columns:
        name = str(c).strip().upper()
        if name in wanted:
            colmap[c] = name
        elif name in [w.replace("=X", "") for w in wanted]:
            colmap[c] = name + "=X"
    closes = closes.rename(columns=colmap)
    missing = [p for p in wanted if p not in closes.columns
               or closes[p].dropna().empty]
    if missing:
        raise ValueError(
            f"No FX history available for: {', '.join(missing)}. "
            "Check connectivity."
        )
    out = pd.DataFrame(index=_as_utc_naive(closes.index))
    for pair in wanted:
        base_ccy = FX_PAIRS[pair][0]
        out[base_ccy] = pd.to_numeric(closes[pair], errors="coerce").values
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    # Drop fully-empty bars (e.g. today's incomplete quote); scattered
    # interior NaNs survive for the bounded alignment layer downstream.
    out = out.dropna(how="all")
    if out.empty:
        raise ValueError("No FX history available. Check connectivity.")
    present = out.dropna()
    bad = [c for c in out.columns
           if ((present[c] <= 0) | (~np.isfinite(present[c]))).any()]
    if bad:
        raise ValueError(f"Invalid FX rates for: {', '.join(bad)}.")
    return out


def fetch_current_fx(pair: str, period: str = "5d") -> tuple[float, datetime | None]:
    """Latest FX rate + UTC timestamp for one pair (USD-per-unit).

    Uses a lightweight recent window (``period="5d"``) rather than full
    history. Returns ``(rate, timestamp_utc)``; timestamp ``None`` if
    unavailable.
    """
    if pair not in FX_PAIRS:
        raise ValueError(f"Unsupported FX pair: {pair}.")
    hist = fetch_fx_history([pair], period=period)
    ccy = FX_PAIRS[pair][0]
    series = hist[ccy].dropna()
    if series.empty:
        raise ValueError(f"No FX history available for {pair}.")
    ts = series.index[-1]
    ts_utc = ts if ts.tzinfo else ts.tz_localize("UTC")
    try:
        ts_utc = ts_utc.tz_convert("UTC")
    except Exception:
        pass
    return float(series.iloc[-1]), ts_utc.to_pydatetime()


def fetch_raw_closes(ticker: str) -> pd.Series:
    """Raw (unadjusted) daily closes — actual traded prices, splits included.

    Unlike adjusted closes, these reflect what a broker fill would have
    been (up to intraday timing). Index is tz-naive UTC.
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
    closes.index = _as_utc_naive(closes.index)
    closes = closes.sort_index()
    closes = closes[~closes.index.duplicated(keep="last")]
    return closes.rename(sym)


def fetch_splits(ticker: str) -> pd.Series:
    """Stock-split series (e.g. 4.0 = 4-for-1); empty when none occurred."""
    sym = validate_ticker_symbol(ticker)
    try:
        splits = yf.Ticker(sym).splits
    except Exception as e:
        raise ValueError(
            f"Could not download split history for {sym} ({e})."
        )
    if splits is None or len(splits) == 0:
        return pd.Series(dtype=float)
    s = pd.to_numeric(splits, errors="coerce").dropna()
    s = s[s > 0]
    try:
        s.index = _as_utc_naive(s.index)
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
        d.index = _as_utc_naive(d.index)
    except Exception:
        pass
    return d.sort_index()

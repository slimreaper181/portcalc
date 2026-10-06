"""
alpaca.py
---------
Alpaca live-price layer for **current US equity valuation only**.

Architecture:
    Historical analytics (returns, covariance, VaR, optimisation,
    scenarios, benchmarks) → yfinance pipeline in ``market_data.py``.
    Current portfolio valuation (prices, market value, P&L) → Alpaca
    latest trade where available, with per-symbol fallback to Yahoo.
    TradingView → chart/display only, never used in calculations.

Live prices are never appended to historical series and never feed any
historical computation. Credentials come from environment variables
(``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY``) with an ``st.secrets``
fallback; nothing secret is logged, committed, or exposed in messages.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

from analytics.validation import validate_ticker_symbol

BASE_URL = "https://data.alpaca.markets"
DEFAULT_FEED = "iex"
DEFAULT_TIMEOUT = 10

# Conservative US-equity shape: 1–5 plain letters only. Anything with a
# suffix (".L", ".B"), caret ("^FTSE"), dash or slash is treated as
# non-US / unsupported and stays on the Yahoo source.
_ALPACA_SYMBOL_RE = re.compile(r"^[A-Z]{1,5}$")


class AlpacaError(Exception):
    """Alpaca request failed (network, auth, entitlement, rate limit, …).

    Attributes:
        status: HTTP status code when known, else ``None`` (e.g. timeouts).
    """

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class AlpacaConfigError(AlpacaError):
    """Alpaca credentials are missing (not an API failure)."""


@dataclass(frozen=True)
class Credentials:
    """Alpaca API credentials. ``repr`` is redacted — secrets never leak."""

    key_id: str
    secret: str

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "Credentials(key_id='***', secret='***')"


@dataclass(frozen=True)
class LatestTrade:
    """One latest-trade snapshot from Alpaca."""

    symbol: str
    price: float
    timestamp: datetime | None
    exchange: str | None
    feed: str


@dataclass(frozen=True)
class PriceQuote:
    """Resolved current price for one ticker (single source of truth)."""

    price: float
    source: str  # "Alpaca IEX" or "Yahoo"
    timestamp: datetime | None = None
    live: bool = False  # True only for a fresh Alpaca trade


def get_alpaca_credentials() -> Credentials:
    """Load Alpaca credentials: environment variables → ``st.secrets`` fallback.

    Raises:
        AlpacaConfigError: if either variable is missing/blank.
    """
    key = os.environ.get("APCA_API_KEY_ID", "").strip()
    secret = os.environ.get("APCA_API_SECRET_KEY", "").strip()
    if not key or not secret:
        try:
            import streamlit as st

            key = str(st.secrets.get("APCA_API_KEY_ID", "") or "").strip()
            secret = str(st.secrets.get("APCA_API_SECRET_KEY", "") or "").strip()
        except Exception:
            key, secret = "", ""
    if not key or not secret:
        raise AlpacaConfigError(
            "Alpaca credentials not configured. Set APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY (Yahoo fallback is used meanwhile)."
        )
    return Credentials(key_id=key, secret=secret)


def is_alpaca_supported_symbol(symbol: str) -> bool:
    """Conservative US-equity check. Never raises — unknown → ``False``."""
    try:
        sym = str(symbol).strip().upper()
    except Exception:
        return False
    return bool(_ALPACA_SYMBOL_RE.match(sym))


def _headers(creds: Credentials) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": creds.key_id,
        "APCA-API-SECRET-KEY": creds.secret,
        "Accept": "application/json",
    }


def _request_json(
    path: str, creds: Credentials, params: dict, timeout: int
) -> dict:
    """GET Alpaca Market Data, mapping failures to :class:`AlpacaError`."""
    url = f"{BASE_URL}{path}"
    try:
        resp = requests.get(url, headers=_headers(creds), params=params,
                            timeout=timeout)
    except requests.Timeout as e:
        raise AlpacaError(f"Alpaca request timed out ({e}).") from e
    except requests.RequestException as e:
        raise AlpacaError(f"Alpaca network error ({e}).") from e
    if resp.status_code == 401:
        raise AlpacaError("Alpaca authentication failed (401): check API keys.",
                          status=401)
    if resp.status_code == 403:
        raise AlpacaError("Alpaca entitlement error (403): feed not authorised.",
                          status=403)
    if resp.status_code == 429:
        raise AlpacaError("Alpaca rate limit hit (429): retry shortly.",
                          status=429)
    if resp.status_code >= 400:
        raise AlpacaError(f"Alpaca API error ({resp.status_code}).",
                          status=resp.status_code)
    try:
        data = resp.json()
    except ValueError as e:
        raise AlpacaError("Alpaca returned a malformed response.") from e
    if not isinstance(data, dict):
        raise AlpacaError("Alpaca returned a malformed response.")
    return data


def _parse_timestamp(raw) -> datetime | None:
    """Parse Alpaca's ISO-8601 trade time (nanoseconds tolerated)."""
    if not raw or not isinstance(raw, str):
        return None
    try:
        text = raw.strip().replace("Z", "+00:00")
        tz = ""
        m = re.match(r"^(.*?)([+-]\d{2}:?\d{2})$", text)
        if m:
            text, tz = m.group(1), m.group(2)
            if ":" not in tz:
                tz = f"{tz[:3]}:{tz[3:]}"
        if "." in text:
            head, frac = text.split(".", 1)
            digits = "".join(ch for ch in frac if ch.isdigit())[:6]
            text = f"{head}.{digits}" if digits else head
        dt = datetime.fromisoformat(text + tz)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, OverflowError):
        return None


def _parse_trade(symbol: str, entry: dict, feed: str) -> LatestTrade | None:
    """Parse one trade object; ``None`` when price is unusable."""
    if not isinstance(entry, dict):
        return None
    try:
        price = float(entry.get("p"))
    except (TypeError, ValueError):
        return None
    import math

    if not math.isfinite(price) or price <= 0:
        return None
    exch = entry.get("x")
    return LatestTrade(
        symbol=symbol,
        price=price,
        timestamp=_parse_timestamp(entry.get("t")),
        exchange=str(exch) if exch else None,
        feed=feed,
    )


def fetch_latest_trades(
    symbols: list[str],
    credentials: Credentials | None = None,
    feed: str = DEFAULT_FEED,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, LatestTrade]:
    """Batch latest trades via ``GET /v2/stocks/trades/latest``.

    Only Alpaca-supported US symbols are requested (others are skipped for
    the caller to fall back to Yahoo). Symbols are validated before the URL
    is built, so malicious strings raise ``ValueError`` and never reach HTTP.

    Returns:
        Mapping of requested symbol → :class:`LatestTrade` for every symbol
        with a usable trade. Symbols with missing/malformed trades are
        simply absent (caller falls back per symbol).

    Raises:
        ValueError: on invalid ticker input.
        AlpacaConfigError: when no credentials are supplied or found.
        AlpacaError: on transport/auth/entitlement/rate-limit/API failures.
    """
    creds = credentials or get_alpaca_credentials()
    wanted: list[str] = []
    for s in symbols or []:
        sym = validate_ticker_symbol(s)  # rejects injection garbage
        if is_alpaca_supported_symbol(sym) and sym not in wanted:
            wanted.append(sym)
    if not wanted:
        return {}
    data = _request_json(
        "/v2/stocks/trades/latest", creds,
        {"symbols": ",".join(wanted), "feed": feed}, timeout,
    )
    trades = data.get("trades")
    if not isinstance(trades, dict):
        raise AlpacaError("Alpaca response missing 'trades'.")
    out: dict[str, LatestTrade] = {}
    for sym in wanted:
        trade = _parse_trade(sym, trades.get(sym), feed)
        if trade is not None:
            out[sym] = trade
    return out


def fetch_latest_trade(
    symbol: str,
    credentials: Credentials | None = None,
    feed: str = DEFAULT_FEED,
    timeout: int = DEFAULT_TIMEOUT,
) -> LatestTrade | None:
    """Single-symbol latest trade; ``None`` when unsupported or missing."""
    creds = credentials or get_alpaca_credentials()
    sym = validate_ticker_symbol(symbol)
    if not is_alpaca_supported_symbol(sym):
        return None
    data = _request_json(
        f"/v2/stocks/{sym}/trades/latest", creds, {"feed": feed}, timeout,
    )
    entry = data.get("trade", data)
    return _parse_trade(sym, entry, feed)


def is_fresh(timestamp: datetime | None, max_age_minutes: int = 60) -> bool:
    """True only for a recent trade timestamp (stale values aren't 'live')."""
    if timestamp is None:
        return False
    now = datetime.now(timezone.utc)
    ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
    return timedelta(0) <= (now - ts) <= timedelta(minutes=max_age_minutes)


def build_price_view(
    tickers: list[str],
    yahoo_prices: dict[str, float],
    alpaca_trades: dict[str, LatestTrade] | None,
) -> dict[str, PriceQuote]:
    """Central price resolution: Alpaca trade where usable, else Yahoo.

    This is the single source of truth for *display/valuation* prices
    (Positions table + Security Detail). Historical analytics never see it.
    """
    view: dict[str, PriceQuote] = {}
    for t in tickers:
        yahoo = float(yahoo_prices.get(t, float("nan")))
        trade = (alpaca_trades or {}).get(t)
        if (trade is not None and trade.price is not None
                and trade.price > 0):
            import math

            if math.isfinite(trade.price):
                view[t] = PriceQuote(
                    price=trade.price,
                    source=f"Alpaca {trade.feed.upper()}",
                    timestamp=trade.timestamp,
                    live=is_fresh(trade.timestamp),
                )
                continue
        view[t] = PriceQuote(price=yahoo, source="Yahoo",
                             timestamp=None, live=False)
    return view

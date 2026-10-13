"""
tradingview.py
--------------
Official TradingView Advanced Chart (free widget) integration.

* :func:`resolve_tradingview_symbol` maps Yahoo Finance tickers to
  TradingView symbols (curated US listings, LSE ``.L`` suffix, known
  indices, optional yfinance exchange hint). Anything ambiguous falls back
  to the bare ticker with symbol-change left enabled — never a crash.
* :func:`tradingview_widget_html` builds the embed HTML with JSON
  serialization (plus ``<``/``>``/``&`` escaping) so tickers and overrides
  cannot break out of the widget configuration.
* :func:`render_tradingview_chart` displays it via
  ``streamlit.components.v1.html``.

The chart is a visual market-analysis tool only. TradingView data is never
used in any portfolio calculation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from analytics.validation import validate_ticker_symbol

# Room reserved for the required TradingView attribution line.
_ATTRIBUTION_PX = 32

# TradingView symbol charset: exchange prefix, separators, no whitespace or
# HTML/JS metacharacters.
_TV_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.:/_-]{0,63}$")

# Known Yahoo index tickers → TradingView symbols.
_INDEX_MAP: dict[str, str] = {
    "^GSPC": "SP:SPX",
    "^IXIC": "NASDAQ:IXIC",
    "^DJI": "DJ:DJI",
    "^RUT": "TVC:RUT",
    "^FTSE": "TVC:UKX",
    "^FTMC": "TVC:MCX",
}

# Curated exchange prefixes for well-known US listings only. Anything not
# listed here (and without a reliable exchange hint) falls back to the bare
# ticker rather than a guessed exchange.
_US_LISTINGS: dict[str, str] = {
    # NASDAQ
    "AAPL": "NASDAQ", "MSFT": "NASDAQ", "NVDA": "NASDAQ",
    "GOOGL": "NASDAQ", "GOOG": "NASDAQ", "AMZN": "NASDAQ",
    "META": "NASDAQ", "TSLA": "NASDAQ", "AVGO": "NASDAQ",
    "COST": "NASDAQ", "NFLX": "NASDAQ", "AMD": "NASDAQ",
    "AMAT": "NASDAQ", "INTC": "NASDAQ", "CSCO": "NASDAQ",
    "ADBE": "NASDAQ", "PEP": "NASDAQ",
    # NYSE
    "JPM": "NYSE", "JNJ": "NYSE", "XOM": "NYSE", "BAC": "NYSE",
    "WMT": "NYSE", "PG": "NYSE", "UNH": "NYSE", "HD": "NYSE",
    "MA": "NYSE", "V": "NYSE", "CVX": "NYSE",
}

# yfinance exchange codes → TradingView prefixes (used only when the
# metadata lookup succeeds; unknown codes are ignored, never guessed).
_EXCHANGE_CODE_MAP: dict[str, str] = {
    "NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ",
    "NYQ": "NYSE", "NYS": "NYSE", "ASE": "AMEX",
    "LSE": "LSE",
}


@dataclass(frozen=True)
class ResolvedSymbol:
    """Result of Yahoo → TradingView symbol resolution."""
    symbol: str      # Symbol to hand to the widget, e.g. "NASDAQ:AAPL".
    resolved: bool   # True when an exchange-qualified mapping was found.
    note: str        # Human-readable explanation for the UI.


def validate_tradingview_symbol(symbol: str) -> str:
    """Validate/normalise a TradingView symbol (or user override).

    Raises ``ValueError`` on anything outside the strict symbol charset,
    which blocks HTML/JS injection through ticker or override fields.
    """
    sym = str(symbol).strip().upper()
    if not _TV_SYMBOL_RE.match(sym):
        raise ValueError(
            f"Invalid TradingView symbol {symbol!r}: use 1–64 characters "
            "from A–Z, 0–9 and . : / _ - (e.g. NASDAQ:AAPL, TVC:UKX)."
        )
    return sym


def _safe_json(config: dict) -> str:
    """JSON-serialize widget config, escaping HTML-significant characters.

    ``json.dumps`` alone does not escape ``<``; a payload containing
    ``</script>`` would otherwise break out of the embed block. The
    ``\\uXXXX`` escapes are valid JSON and decode to identical values.
    """
    return (
        json.dumps(config, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def resolve_tradingview_symbol(
    yahoo_ticker: str,
    exchange_hint: str | None = None,
    override: str | None = None,
) -> ResolvedSymbol:
    """Resolve a Yahoo Finance ticker to a TradingView symbol.

    Args:
        yahoo_ticker: Portfolio ticker, e.g. ``AAPL``, ``BARC.L``, ``^GSPC``.
        exchange_hint: Optional yfinance exchange code (``NMS``, ``NYQ``,
            ``LSE``, …). Unknown codes are ignored.
        override: Optional explicit TradingView symbol; validated strictly
            and used verbatim when valid. Affects only the displayed chart.

    Returns:
        :class:`ResolvedSymbol` with the widget symbol, whether an
        exchange-qualified mapping was found, and an explanatory note.

    Raises:
        ValueError: on invalid Yahoo ticker or invalid override input.
    """
    if override is not None and str(override).strip():
        sym = validate_tradingview_symbol(override)
        return ResolvedSymbol(
            symbol=sym, resolved=True,
            note=f"Using override symbol {sym} for the chart only.",
        )

    yahoo = validate_ticker_symbol(yahoo_ticker)  # rejects garbage early

    if yahoo in _INDEX_MAP:
        sym = _INDEX_MAP[yahoo]
        return ResolvedSymbol(
            symbol=sym, resolved=True,
            note=f"Index {yahoo} maps to {sym}.",
        )

    if yahoo.endswith(".L"):
        base = yahoo[:-2]
        if base and re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]*", base):
            sym = f"LSE:{base}"
            return ResolvedSymbol(
                symbol=sym, resolved=True,
                note=f"London listing {yahoo} maps to {sym}.",
            )

    if "." not in yahoo and not yahoo.startswith("^") and "=" not in yahoo:
        if yahoo in _US_LISTINGS:
            sym = f"{_US_LISTINGS[yahoo]}:{yahoo}"
            return ResolvedSymbol(
                symbol=sym, resolved=True,
                note=f"US listing {yahoo} maps to {sym}.",
            )
        if exchange_hint:
            prefix = _EXCHANGE_CODE_MAP.get(str(exchange_hint).strip().upper())
            if prefix:
                sym = f"{prefix}:{yahoo}"
                return ResolvedSymbol(
                    symbol=sym, resolved=True,
                    note=f"Exchange metadata maps {yahoo} to {sym}.",
                )

    # Graceful fallback: bare ticker, symbol search stays enabled.
    return ResolvedSymbol(
        symbol=yahoo, resolved=False,
        note=(
            f"No exchange-qualified mapping for {yahoo}; showing it as-is. "
            "Use the widget's symbol search or the override below if needed."
        ),
    )


def tradingview_widget_html(
    symbol: str,
    watchlist: list[str] | tuple[str, ...] = (),
    interval: str = "D",
    theme: str = "dark",
    height: int = 900,
) -> str:
    """Build the official Advanced Chart embed HTML for a symbol.

    Height chain (every level gets an explicit height so the rendered
    candlestick area — not just the iframe — hits the requested size)::

        Streamlit iframe (height + attribution allowance)
        └─ html/body (100%, margin/padding reset)
           └─ .tradingview-widget-container (explicit ``height`` px)
              ├─ .tradingview-widget-container__widget (explicit px —
              │  the exact class the official embed script targets; a
              │  random ``id`` here leaves the widget at its ~200px default)
              └─ attribution line (required, never removed)

    Args:
        symbol: Resolved TradingView symbol (strictly validated).
        watchlist: Extra symbols for the widget watchlist (each validated).
        interval: Chart interval (``D`` daily default).
        theme: ``dark`` (matches the terminal aesthetic) or ``light``.
        height: Actual chart height in px (Standard ≈ 650, Large ≈ 900).
            The attribution line is added on top of this value.

    Returns:
        Self-contained HTML string for ``streamlit.components.v1.html``.
        The container is fluid (``width:100%``) with ``autosize`` enabled so
        the widget always spans the full Streamlit content width.
    """
    sym = validate_tradingview_symbol(symbol)
    clean_watchlist: list[str] = []
    for entry in watchlist:
        validated = validate_tradingview_symbol(entry)
        if validated != sym and validated not in clean_watchlist:
            clean_watchlist.append(validated)

    if interval not in ("1", "5", "15", "60", "D", "W", "M"):
        raise ValueError(f"Unsupported chart interval {interval!r}.")
    if theme not in ("dark", "light"):
        raise ValueError(f"Unsupported chart theme {theme!r}.")
    if not isinstance(height, int) or not 300 <= height <= 1200:
        raise ValueError(f"Chart height must be 300–1200px, got {height!r}.")

    config = {
        "autosize": True,
        "width": "100%",
        "height": height,
        "symbol": sym,
        "interval": interval,
        "timezone": "exchange",
        "theme": theme,
        "style": "1",  # candlesticks
        "locale": "en",
        "withdateranges": True,
        "hide_side_toolbar": False,
        "allow_symbol_change": True,
        "watchlist": [sym, *clean_watchlist],
        "support_host": "https://www.tradingview.com",
    }
    payload = _safe_json(config)
    return (
        "<style>"
        "html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;"
        "background:#0d1117;}"
        "</style>"
        f'<div class="tradingview-widget-container" '
        f'style="width:100%;height:{height + _ATTRIBUTION_PX}px;">'
        f'<div class="tradingview-widget-container__widget" '
        f'style="width:100%;height:{height}px;"></div>'
        '<div class="tradingview-widget-copyright" '
        f'style="height:{_ATTRIBUTION_PX}px;line-height:{_ATTRIBUTION_PX}px;'
        'text-align:center;font-size:12px;font-family:monospace;">'
        '<a href="https://www.tradingview.com/" rel="noopener nofollow" '
        'target="_blank" style="color:#58a6ff;text-decoration:none;">'
        '<span>Track all markets on TradingView</span></a>'
        "</div>"
        '<script type="text/javascript" '
        'src="https://s3.tradingview.com/external-embedding/'
        'embed-widget-advanced-chart.js" async>'
        f"{payload}"
        "</script>"
        "</div>"
    )


def render_tradingview_chart(
    symbol: str,
    watchlist: list[str] | tuple[str, ...] = (),
    interval: str = "D",
    theme: str = "dark",
    height: int = 900,
) -> None:
    """Render the Advanced Chart widget inside Streamlit (full width).

    The iframe is sized to the actual chart height plus the attribution
    line and a small margin, so no large blank region appears beneath a
    short chart (or vice versa).
    """
    from streamlit.components.v1 import html as _html

    _html(tradingview_widget_html(symbol, watchlist, interval, theme, height),
          height=height + _ATTRIBUTION_PX + 8, scrolling=False)

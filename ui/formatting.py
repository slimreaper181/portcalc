"""formatting — one money/percent/number/date vocabulary (pure functions).

Wraps the audited :func:`analytics.positions.format_money` (NaN/inf-safe)
and extends it to signed values, ratios, counts and dates. Missing or
non-finite values render as an em dash — never ``nan``/``inf``/clipped
decimals.
"""

from __future__ import annotations

import math
from datetime import date, datetime

from analytics.positions import format_money as _engine_money

MISSING = "—"


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def format_money(amount, currency: str = "USD") -> str:
    """Base-currency money; ``None``/NaN/inf → em dash."""
    if amount is None or not _finite(amount):
        return MISSING
    try:
        return _engine_money(float(amount), currency)
    except (ValueError, TypeError):
        return MISSING


def format_signed_money(amount, currency: str = "USD") -> str:
    """Money with explicit sign (+$1.20 / −$0.40 / —)."""
    if amount is None or not _finite(amount):
        return MISSING
    v = float(amount)
    if v == 0.0:
        return _engine_money(0.0, currency)
    sign = "+" if v > 0 else "−"
    return sign + _engine_money(abs(v), currency)


def format_percent(x, digits: int = 2) -> str:
    """Fraction → percent (0.1234 → 12.34%)."""
    if x is None or not _finite(x):
        return MISSING
    return f"{float(x) * 100:.{int(digits)}f}%"


def format_signed_percent(x, digits: int = 2) -> str:
    """Signed percent (+3.21% / −1.05% / —)."""
    if x is None or not _finite(x):
        return MISSING
    v = float(x)
    if v == 0.0:
        return f"{0.0:.{int(digits)}f}%"
    sign = "+" if v > 0 else "−"
    return f"{sign}{abs(v) * 100:.{int(digits)}f}%"


def format_ratio(x, digits: int = 2) -> str:
    """Plain ratio (Sharpe/Sortino/Calmar/DR)."""
    if x is None or not _finite(x):
        return MISSING
    return f"{float(x):.{int(digits)}f}"


def format_number(x, digits: int = 2) -> str:
    """Thousands-separated number."""
    if x is None or not _finite(x):
        return MISSING
    return f"{float(x):,.{int(digits)}f}"


def format_int(x) -> str:
    """Integer count."""
    if x is None or not _finite(x):
        return MISSING
    return f"{int(float(x)):,d}"


def format_date(d) -> str:
    """ISO date (accepts date/datetime/Timestamp/str)."""
    if d is None:
        return MISSING
    if isinstance(d, str):
        return d[:10] if len(d) >= 10 else d
    if isinstance(d, datetime):
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    try:
        import pandas as pd
        ts = pd.Timestamp(d)
        if ts is not pd.NaT and pd.notna(ts):
            return ts.date().isoformat()
    except (ValueError, TypeError):
        pass
    return MISSING


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


def fmt_usd(value, currency=None):
    """Money in the active base currency (explicit override allowed).

    Session-aware shell default: reads the base currency from Streamlit
    session state unless ``currency`` is given. Safe outside a session
    (falls back to USD).
    """
    if currency is None:
        try:
            import streamlit as st
            currency = st.session_state.get("base_currency", "USD")
        except Exception:
            currency = "USD"
    return format_money(value, currency)


def fmt_pct(value):
    """Fraction to percent (session-independent)."""
    return format_percent(value)


def style_matrix(df, fmt=".4f"):
    """Blue-gradient styled numeric matrix (correlation/covariance views)."""
    return (
        df.style
        .format("{:" + fmt + "}")
        .background_gradient(cmap="Blues", axis=None)
    )

"""
currency.py
-----------
Pure foreign-exchange and multi-currency mathematics (no network, no UI).

Pipeline (never just relabelling symbols)::

    native price (quote units)
      → quote-unit normalisation (÷100 for GBp, …)
      → × date-matched historical FX
      → base-currency price
      → each module's existing return methodology

All FX routing goes through USD legs: with ``u(X)`` = USD-per-unit of X,
``rate(A → B) = u(A) / u(B)``. Identity conversions need no data.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

#: Officially supported portfolio base currencies (extensible mapping).
BASE_CURRENCIES = ("USD", "GBP", "EUR")

#: Display symbols. Unknown codes fall back to the code itself + space.
CURRENCY_SYMBOLS = {"USD": "$", "GBP": "£", "EUR": "€"}

#: Maximum forward-fill when aligning FX to equity dates (trading days).
FX_ALIGN_FFILL_LIMIT = 5


def validate_base_currency(code: str) -> str:
    """Validate a base-currency code (upper-cased, extensible set)."""
    ccy = str(code or "").strip().upper()
    if ccy not in BASE_CURRENCIES:
        raise ValueError(
            f"Unsupported base currency {code!r}; "
            f"expected one of {list(BASE_CURRENCIES)}."
        )
    return ccy


def currency_symbol(code: str) -> str:
    """Display symbol for a currency (``USD`` → ``$``)."""
    return CURRENCY_SYMBOLS.get(str(code or "").strip().upper(), f"{code} ")


def fx_rate_between(
    from_ccy: str, to_ccy: str, usd_per_unit: dict[str, float]
) -> float:
    """Convert 1 unit of ``from_ccy`` into ``to_ccy`` via USD legs.

    ``rate = u(from) / u(to)`` where ``u`` maps currency → USD-per-unit
    (with ``u("USD") == 1``). Identical currencies return exactly 1.0 with
    no data required. Raises on missing/non-positive rates — a missing FX
    rate is never silently treated as 1.
    """
    src = str(from_ccy).strip().upper()
    dst = str(to_ccy).strip().upper()
    if src == dst:
        return 1.0
    try:
        u_src = float(usd_per_unit[src])
        u_dst = float(usd_per_unit[dst])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(
            f"No FX rate available to convert {src} → {dst}."
        ) from e
    if not (math.isfinite(u_src) and math.isfinite(u_dst)):
        raise ValueError(f"Non-finite FX rate for {src} → {dst}.")
    if u_src <= 0 or u_dst <= 0:
        raise ValueError(f"Non-positive FX rate for {src} → {dst}.")
    return u_src / u_dst


def convert_value(
    amount: float, from_ccy: str, to_ccy: str, usd_per_unit: dict[str, float]
) -> float:
    """Convert a monetary amount between currencies (validated, unrounded).

    Internal precision is retained — rounding happens only for display.
    """
    if not math.isfinite(amount):
        raise ValueError(f"Amount must be finite, got {amount!r}.")
    return float(amount) * fx_rate_between(from_ccy, to_ccy, usd_per_unit)


def align_fx_to_index(
    fx: pd.Series,
    index: pd.DatetimeIndex,
    limit: int = FX_ALIGN_FFILL_LIMIT,
    label: str = "FX",
) -> pd.Series:
    """Align an FX series to equity trading dates with bounded carry-forward.

    FX markets trade on more days than equities, so the series is reindexed
    and forward-filled (never back-filled — that would use future data).
    Any remaining gap raises with the first gap date instead of silently
    dropping long outages.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Target index must be a DatetimeIndex.")
    s = pd.to_numeric(pd.Series(fx, dtype=float), errors="coerce")
    if s.empty:
        raise ValueError(f"No {label} observations supplied.")
    aligned = s.reindex(index).ffill(limit=int(limit))
    if aligned.isna().any():
        first_gap = aligned[aligned.isna()].index[0]
        raise ValueError(
            f"{label} has an unfillable gap at/before {first_gap.date()}: "
            f"no observation within {limit} prior trading days. "
            "Refusing to invent FX data."
        )
    if ((aligned <= 0) | (~np.isfinite(aligned))).any():
        raise ValueError(f"{label} contains zero/negative/non-finite rates.")
    return aligned


def rate_on_date(
    fx_history: pd.DataFrame,
    from_ccy: str,
    to_ccy: str,
    when,
    limit: int = 30,
) -> float | None:
    """Native→base rate prevailing on a past date (or ``None`` if unknown).

    Uses only FX observations on/before ``when`` (forward-fill, bounded —
    never future data, never an invented rate). ``None`` means the caller
    must treat the dependent value as unknown, not as 1.0.
    """
    src = str(from_ccy).strip().upper()
    dst = str(to_ccy).strip().upper()
    if src == dst:
        return 1.0
    if not isinstance(fx_history, pd.DataFrame) or fx_history.empty:
        return None
    try:
        day = pd.Timestamp(when).date()
    except Exception:
        return None
    frame = fx_history.copy()
    if "USD" not in frame.columns:
        frame["USD"] = 1.0
    if src not in frame.columns or dst not in frame.columns:
        return None
    past = frame.loc[frame.index.date <= day]
    if past.empty:
        return None
    tail = past.tail(limit + 1)
    u_src = tail[src].dropna()
    u_dst = tail[dst].dropna()
    if u_src.empty or u_dst.empty:
        return None
    # Last common observation on/before the date (bounded look-back).
    common = u_src.index.intersection(u_dst.index)
    if common.empty:
        return None
    use = common.max()
    if (tail.index.max() - use).days > limit:
        return None
    r_src, r_dst = float(u_src.loc[use]), float(u_dst.loc[use])
    if not (math.isfinite(r_src) and math.isfinite(r_dst)):
        return None
    if r_src <= 0 or r_dst <= 0:
        return None
    return r_src / r_dst


def convert_price_series(
    native: pd.Series,
    native_ccy: str,
    base_ccy: str,
    fx_usd_per_unit: pd.DataFrame,
    limit: int = FX_ALIGN_FFILL_LIMIT,
) -> pd.Series:
    """Convert a native-unit price series to base currency, date-matched.

    ``base_price[t] = native[t] × rate(native_ccy → base_ccy at t)`` using
    only FX observations on/before each date. Same-currency input is
    returned (copied) untouched.
    """
    src = str(native_ccy).strip().upper()
    dst = str(base_ccy).strip().upper()
    s = pd.to_numeric(pd.Series(native, dtype=float), errors="coerce")
    if s.empty:
        raise ValueError("No native prices supplied.")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("Price series needs a DatetimeIndex.")
    if src == dst:
        return s.copy()
    if not isinstance(fx_usd_per_unit, pd.DataFrame) or fx_usd_per_unit.empty:
        raise ValueError("No FX history supplied for conversion.")
    if src not in fx_usd_per_unit.columns or dst not in fx_usd_per_unit.columns:
        # USD leg is the constant 1.0 and needs no series.
        frame = fx_usd_per_unit.copy()
        if "USD" not in frame.columns:
            frame["USD"] = 1.0
        missing = [c for c in (src, dst) if c not in frame.columns]
        if missing:
            raise ValueError(
                f"No FX history for currency: {', '.join(missing)}."
            )
        fx_usd_per_unit = frame
    u_src = align_fx_to_index(fx_usd_per_unit[src], s.index, limit,
                              label=f"{src} FX")
    u_dst = align_fx_to_index(fx_usd_per_unit[dst], s.index, limit,
                              label=f"{dst} FX")
    rate = (u_src / u_dst).replace([np.inf, -np.inf], np.nan)
    if rate.isna().any() or (rate <= 0).any():
        raise ValueError(
            f"Invalid {src} → {dst} conversion rate on some dates."
        )
    return (s * rate).rename(s.name)


def validate_return_series(
    returns,
    name: str = "returns",
    min_obs: int = 2,
) -> pd.Series:
    """Validate a return series before any analytics pipeline consumes it.

    Checks: finite-or-NaN values with no ±infinity, a sorted unique
    DatetimeIndex, and at least ``min_obs`` non-NaN observations. Returns a
    cleaned copy (NaNs dropped). Raises ``ValueError`` otherwise.
    """
    s = pd.Series(returns, dtype=float)
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError(f"{name} needs a DatetimeIndex.")
    if s.index.duplicated().any():
        raise ValueError(f"{name} contains duplicate dates.")
    if not s.index.is_monotonic_increasing:
        raise ValueError(f"{name} dates are not in order.")
    if np.isinf(s.values).any():
        raise ValueError(f"{name} contains infinite returns (impossible).")
    clean = s.dropna()
    if len(clean) < min_obs:
        raise ValueError(
            f"{name} has only {len(clean)} usable observations; "
            f"at least {min_obs} are required."
        )
    if not np.all(np.isfinite(clean.values)):
        raise ValueError(f"{name} contains non-finite returns.")
    return clean


def check_weights_sum(weights: dict[str, float], tol: float = 1e-6) -> float:
    """Hard validation that weights sum to ~1.0; returns the total."""
    total = 0.0
    for t, w in weights.items():
        if not math.isfinite(w):
            raise ValueError(f"Weight for {t} is not finite.")
        total += float(w)
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"Portfolio weights sum to {total:.6f}, not 1.0 — refusing to "
            "display inconsistent currency-converted weights."
        )
    return total


def decompose_pnl(
    base_cost: float, local_ret: float, fx_ret: float
) -> dict[str, float]:
    """Exact multiplicative P&L decomposition (base currency).

    ``R_base = (1 + R_local)(1 + R_fx) − 1``, so with base cost ``C``::

        local_effect  = C × R_local
        fx_effect     = C × R_fx
        interaction   = C × R_local × R_fx

    which sums exactly to ``C × R_base``. Presented as exact, never as an
    approximation. All inputs must be finite.
    """
    for name, v in (("base_cost", base_cost), ("local_ret", local_ret),
                    ("fx_ret", fx_ret)):
        if not math.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v!r}.")
    local_eff = base_cost * local_ret
    fx_eff = base_cost * fx_ret
    inter = base_cost * local_ret * fx_ret
    return {
        "local_effect": float(local_eff),
        "fx_effect": float(fx_eff),
        "interaction": float(inter),
        "total": float(local_eff + fx_eff + inter),
    }


def fx_shock_portfolio(
    exposures: dict[str, float],
    shocks: dict[str, float],
    base_ccy: str,
) -> dict:
    """Static first-order FX shock with local security prices held constant.

    ``exposures`` maps quote currency → current base-currency market value.
    ``shocks`` maps non-base currency → fractional FX move
    (e.g. ``+0.05`` = that currency strengthens 5% vs base). Returns the
    shocked total, absolute/percentage change and per-currency deltas.
    This is a sensitivity illustration, not a forecast.
    """
    base = str(base_ccy).strip().upper()
    total = 0.0
    for ccy, val in exposures.items():
        if not math.isfinite(val):
            raise ValueError(f"Exposure for {ccy} is not finite.")
        total += float(val)
    deltas: dict[str, float] = {}
    for ccy, shock in shocks.items():
        c = str(ccy).strip().upper()
        if c == base and shock:
            raise ValueError(
                f"Cannot shock base currency {base} against itself."
            )
        if not math.isfinite(shock):
            raise ValueError(f"Shock for {ccy} is not finite.")
        if c not in exposures:
            raise ValueError(f"No {c} exposure in the portfolio to shock.")
        deltas[c] = float(exposures[c] * shock)
    moved = total + sum(deltas.values())
    return {
        "base_total": total,
        "shocked_total": moved,
        "absolute_change": moved - total,
        "pct_change": (moved - total) / total if total else 0.0,
        "per_currency": deltas,
    }


def quote_age_days(ts_utc: datetime | None, now_utc: datetime) -> float | None:
    """Age of a quote timestamp in days (``None`` when timestamp missing)."""
    if ts_utc is None:
        return None
    try:
        ts = ts_utc if ts_utc.tzinfo else ts_utc.replace(tzinfo=timezone.utc)
        now = now_utc if now_utc.tzinfo else now_utc.replace(tzinfo=timezone.utc)
        return max(0.0, (now - ts).total_seconds() / 86400.0)
    except Exception:
        return None


def freshness_report(
    quote_ages_days: dict[str, float | None],
    fx_age_days: float | None,
    stale_after_days: float = 5.0,
) -> list[str]:
    """Conservative mixed-freshness warnings (never alarmist).

    Warns only when data is missing or materially stale (> ``stale_after``
    days, covering long weekends/holidays), or when sources disagree by more
    than that window (mixed freshness must never look uniformly fresh).
    """
    warnings: list[str] = []
    missing = sorted(t for t, age in quote_ages_days.items() if age is None)
    if missing:
        warnings.append(
            "No timestamp for current quote: "
            + ", ".join(missing)
            + " — treat as last close, not live."
        )
    stale = sorted(t for t, age in quote_ages_days.items()
                   if age is not None and age > stale_after_days)
    if stale:
        warnings.append(
            "Stale current quote (> "
            f"{stale_after_days:g} days): "
            + ", ".join(stale) + "."
        )
    if fx_age_days is None:
        warnings.append("No timestamp for current FX — treat as last close.")
    elif fx_age_days > stale_after_days:
        warnings.append(
            f"Stale current FX (> {stale_after_days:g} days old)."
        )
    known = [a for a in list(quote_ages_days.values()) + [fx_age_days]
             if a is not None]
    if len(known) >= 2 and (max(known) - min(known)) > stale_after_days:
        warnings.append(
            "Mixed data freshness across quotes/FX — figures combine "
            "observations of different ages."
        )
    return warnings

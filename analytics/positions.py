"""
positions.py
------------
Typed position model, derived valuation engine and cost-basis accounting.

Separation (deliberate)::

    Position input/state (shares, purchase date, manual price, …)
      → value_position / value_portfolio (pure derivation)
      → PositionValuation (calculated result)

Volatile live values are never stored inside ``Position`` — quotes and FX
are passed in per calculation. All money math keeps full float precision;
rounding happens only in ``format_money`` for display.

Cost-basis convention (documented choice — "Option B")
-------------------------------------------------------
The user enters **current split-adjusted share counts**. The estimated
purchase price is the raw (dividend-excluded) historical close divided by
the cumulative split factor since purchase, i.e. a per-current-share
transaction price. Hence ``cost = shares_now × price_split_adjusted``.
Cash dividends are NOT included in unrealised P&L (labelled accordingly);
see :func:`estimate_dividend_income` for separately-reported income.
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from analytics.currency import (
    BASE_CURRENCIES,
    convert_value,
    currency_symbol,
    validate_base_currency,
)
from analytics.validation import validate_ticker_symbol

SCHEMA_VERSION = 1

#: Purchase-price provenance labels.
SOURCE_ESTIMATE = "estimate"    # raw historical close, split-adjusted
SOURCE_MANUAL = "manual"        # user-supplied actual execution price
SOURCE_LEGACY = "legacy"        # migrated pre-Task-6 avg_cost


@dataclass
class Position:
    """One aggregate acquisition (single purchase date; no tax lots)."""

    ticker: str
    shares: float
    purchase_date: date | None = None
    purchase_price_native: float | None = None
    purchase_price_source: str | None = None
    manual_purchase_price: bool = False
    native_currency: str | None = None
    quote_unit: str | None = None
    quote_scale: float = 1.0
    legacy_avg_cost: float | None = None


@dataclass
class PositionValuation:
    """Derived valuation for one position (recalculate, never persist)."""

    ticker: str
    shares: float
    native_currency: str
    native_current_price: float
    price_source: str
    current_price_timestamp: datetime | None
    base_currency: str
    current_fx_rate: float
    current_fx_timestamp: datetime | None
    native_market_value: float
    base_market_value: float
    native_cost_basis: float | None = None
    base_cost_basis: float | None = None
    cost_source: str | None = None
    unrealised_pnl: float | None = None
    unrealised_pnl_pct: float | None = None
    dividend_income_base: float = 0.0
    warnings: tuple = ()


# ---------------------------------------------------------------------------
# Money formatting (single formatter for the whole UI)
# ---------------------------------------------------------------------------

def format_money(value: float | None, currency: str) -> str:
    """Format money: ``$1,234.56`` / ``-£312.42`` / ``€0.00`` / ``n/a``.

    ``None``/NaN → ``"n/a"`` (unavailable, never zero). Full precision is
    kept internally; this rounds only the displayed string.
    """
    ccy = str(currency or "").strip().upper()
    sym = currency_symbol(ccy)
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(v):
        return "n/a"
    sign = "-" if v < 0 else ""
    return f"{sign}{sym}{abs(v):,.2f}"


# ---------------------------------------------------------------------------
# Quote normalisation
# ---------------------------------------------------------------------------

def normalise_quote(price: float, quote_scale: float) -> float:
    """Convert a quoted price to native currency (÷100 for GBp, …)."""
    if not math.isfinite(price):
        raise ValueError(f"Quoted price must be finite, got {price!r}.")
    if not math.isfinite(quote_scale) or quote_scale <= 0:
        raise ValueError(f"Quote scale must be positive, got {quote_scale!r}.")
    return float(price) * float(quote_scale)


# ---------------------------------------------------------------------------
# Splits + purchase-price resolution (pure; data injected for testability)
# ---------------------------------------------------------------------------

def cumulative_split_factor(splits: pd.Series, purchase_date: date) -> float:
    """Product of split ratios strictly after the purchase date.

    yfinance encodes e.g. a 4-for-1 split as 4.0 (new shares per old share),
    so the per-current-share price is ``raw_close / factor``.
    """
    if splits is None or len(splits) == 0:
        return 1.0
    s = pd.to_numeric(pd.Series(splits), errors="coerce").dropna()
    s = s[s > 0]
    mask = pd.DatetimeIndex(s.index).date > purchase_date
    selected = s[mask.values] if hasattr(mask, "values") else s[list(mask)]
    factor = 1.0
    for v in selected:
        factor *= float(v)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Invalid split history around the purchase date.")
    return factor


def resolve_purchase_price(
    raw_closes: pd.Series,
    splits: pd.Series | None,
    purchase_date: date,
    today: date,
) -> dict:
    """Resolve an estimated per-current-share purchase price.

    Uses the raw (dividend-excluded) close on the purchase date divided by
    subsequent splits — coherent with current split-adjusted share counts.

    Returns a dict with ``status``:
      - ``"ok"`` → ``price``, ``raw_close``, ``split_factor``, ``used_date``
      - ``"missing"`` → ``prev``/``next`` ``(date, price)`` candidates for
        the user to choose explicitly (never silent substitution)
      - ``"invalid"`` → ``message`` (future date, pre-history, bad data)
    """
    if not isinstance(purchase_date, date) or isinstance(purchase_date, datetime):
        if isinstance(purchase_date, datetime):
            purchase_date = purchase_date.date()
        else:
            return {"status": "invalid",
                    "message": f"Malformed purchase date: {purchase_date!r}."}
    if purchase_date > today:
        return {"status": "invalid",
                "message": f"Purchase date {purchase_date} is in the future."}
    closes = pd.to_numeric(pd.Series(raw_closes), errors="coerce").dropna()
    if closes.empty:
        return {"status": "invalid",
                "message": "No price history available for lookup."}
    try:
        trading_dates = sorted({pd.Timestamp(d).date() for d in closes.index})
    except Exception:
        return {"status": "invalid",
                "message": "Price history has unparseable dates."}
    if purchase_date in trading_dates:
        raw_close = float(closes[closes.index.date == purchase_date].iloc[0])
        factor = cumulative_split_factor(splits, purchase_date)
        return {"status": "ok",
                "price": raw_close / factor,
                "raw_close": raw_close,
                "split_factor": factor,
                "used_date": purchase_date}
    if purchase_date < trading_dates[0]:
        return {"status": "invalid",
                "message": f"Purchase date {purchase_date} predates available "
                           f"history (starts {trading_dates[0]})."}
    prev = max(d for d in trading_dates if d < purchase_date)
    nxt = min(d for d in trading_dates if d > purchase_date)
    prev_px = float(closes[closes.index.date == prev].iloc[0])
    next_px = float(closes[closes.index.date == nxt].iloc[0])
    return {"status": "missing",
            "message": f"No market price exists on {purchase_date} "
                       f"({purchase_date:%A}). Choose explicitly:",
            "prev": (prev, prev_px / cumulative_split_factor(splits, prev)),
            "next": (nxt, next_px / cumulative_split_factor(splits, nxt))}


def estimate_dividend_income(
    dividends: pd.Series,
    purchase_date: date,
    shares: float,
    fx_for_date,
) -> tuple[float, list, int]:
    """Estimated cash dividend income in base currency.

    Only dividends dated strictly after the purchase date count (documented
    entitlement approximation). ``fx_for_date(date)`` returns the native→base
    rate or ``None`` (skipped + counted, never assumed 1).

    Returns ``(total_base, events, skipped)``; inputs are never mutated.
    """
    if dividends is None or len(dividends) == 0:
        return 0.0, [], 0
    if not math.isfinite(shares) or shares < 0:
        raise ValueError(f"Shares must be finite and >= 0, got {shares!r}.")
    divs = pd.to_numeric(pd.Series(dividends), errors="coerce").dropna()
    divs = divs[divs > 0]
    total, events, skipped = 0.0, [], 0
    for ts, per_share in divs.items():
        try:
            d = pd.Timestamp(ts).date()
        except Exception:
            skipped += 1
            continue
        if d <= purchase_date:
            continue
        rate = fx_for_date(d)
        if rate is None or not math.isfinite(rate) or rate <= 0:
            skipped += 1
            continue
        amount = float(per_share) * float(shares) * float(rate)
        total += amount
        events.append({"date": d, "per_share_native": float(per_share),
                       "fx": float(rate), "base_amount": amount})
    return total, events, skipped


# ---------------------------------------------------------------------------
# Position parsing / serialisation (backward compatible)
# ---------------------------------------------------------------------------

def _parse_date(value) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return pd.Timestamp(str(value)).date()
    except (ValueError, TypeError):
        raise ValueError(f"Malformed purchase date: {value!r}.")


def position_from_dict(ticker: str, data: dict) -> Position:
    """Parse a stored position dict (new schema + legacy ``avg_cost``).

    Legacy ``avg_cost`` without a purchase date is preserved as a manual /
    legacy cost basis (never discarded); the UI should prompt for a
    purchase date later. Missing optional fields default to ``None``.
    Zero/negative shares and malformed tickers raise.
    """
    sym = validate_ticker_symbol(ticker)
    if not isinstance(data, dict):
        raise ValueError(f"Position {sym} must be a mapping.")
    try:
        shares = float(data.get("shares", 0.0))
    except (TypeError, ValueError):
        raise ValueError(f"Shares for {sym} must be a number.")
    if not math.isfinite(shares) or shares <= 0:
        raise ValueError(
            f"Shares for {sym} must be a positive number (got "
            f"{data.get('shares')!r}); zero-share rows are rejected.")
    purchase_date = _parse_date(data.get("purchase_date"))
    raw_flag = data.get("manual_purchase_price", False)
    if isinstance(raw_flag, bool):
        manual_flag = raw_flag
    elif raw_flag in (0, 1):
        manual_flag = bool(raw_flag)
    elif raw_flag in (None, ""):
        manual_flag = False
    else:
        raise ValueError(
            f"Manual-price flag for {sym} must be true/false.")
    legacy = data.get("avg_cost", data.get("legacy_avg_cost"))
    try:
        legacy = float(legacy) if legacy not in (None, "") else None
    except (TypeError, ValueError):
        raise ValueError(f"Legacy cost for {sym} must be a number.")
    if legacy is not None and (
            not math.isfinite(legacy) or legacy < 0):
        raise ValueError(f"Legacy cost for {sym} must be >= 0.")
    est_price = data.get("purchase_price_native")
    try:
        est_price = float(est_price) if est_price not in (None, "") else None
    except (TypeError, ValueError):
        raise ValueError(f"Stored purchase price for {sym} must be a number.")
    if est_price is not None and (
            not math.isfinite(est_price) or est_price <= 0):
        raise ValueError(f"Stored purchase price for {sym} must be positive.")
    source = data.get("purchase_price_source")
    if source is not None and source not in (
            "estimate", "manual", "legacy"):
        raise ValueError(f"Unknown price source for {sym}: {source!r}.")
    native_ccy = data.get("native_currency")
    if native_ccy is not None:
        native_ccy = str(native_ccy).strip().upper()
    quote_unit = data.get("quote_unit")
    try:
        scale = float(data.get("quote_scale", 1.0))
    except (TypeError, ValueError):
        raise ValueError(f"Quote scale for {sym} must be a number.")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Quote scale for {sym} must be positive.")
    # Effective cost basis: an explicit manual override wins, then a stored
    # estimate, then legacy avg_cost preserved as manual/legacy basis.
    # The manual override price itself lives in ``purchase_price_native``
    # with source ``"manual"``; ``manual_purchase_price`` is its bool flag.
    if est_price is not None and (source == "manual" or
                                  (source is None and manual_flag)):
        eff_price, eff_source, eff_manual = est_price, "manual", True
    elif est_price is not None and source in (None, "estimate"):
        eff_price, eff_source, eff_manual = est_price, "estimate", False
    elif est_price is not None and source == "legacy":
        eff_price, eff_source, eff_manual = est_price, "legacy", True
    elif legacy is not None and legacy > 0:
        eff_price, eff_source, eff_manual = legacy, "legacy", True
    else:
        eff_price, eff_source, eff_manual = None, None, False
    return Position(
        ticker=sym, shares=shares, purchase_date=purchase_date,
        purchase_price_native=eff_price, purchase_price_source=eff_source,
        manual_purchase_price=eff_manual, native_currency=native_ccy,
        quote_unit=quote_unit, quote_scale=scale, legacy_avg_cost=legacy)


def position_to_dict(pos: Position) -> dict:
    """Serialise a position (whitelisted fields only — never secrets)."""
    return {
        "shares": pos.shares,
        "purchase_date": pos.purchase_date.isoformat()
        if pos.purchase_date else None,
        "purchase_price_native": pos.purchase_price_native,
        "purchase_price_source": pos.purchase_price_source,
        "manual_purchase_price": bool(pos.manual_purchase_price),
        "native_currency": pos.native_currency,
        "quote_unit": pos.quote_unit,
        "quote_scale": pos.quote_scale,
        "avg_cost": pos.legacy_avg_cost,
    }


# ---------------------------------------------------------------------------
# Valuation engine (single path for every tab)
# ---------------------------------------------------------------------------

def value_position(
    pos: Position,
    current_native_price: float,
    price_source: str,
    price_timestamp,
    purchase_fx_rate: float | None,
    current_fx_rate: float,
    base_currency: str,
) -> PositionValuation:
    """Value one position in native + base currency (full float precision).

    * ``purchase_fx_rate``: native→base on the purchase date (``None`` when
      the cost basis is unknown — P&L stays ``None``, never invented).
    * P&L is **unrealised price/FX P&L** (dividends excluded by design).
    """
    base = validate_base_currency(base_currency)
    if not math.isfinite(current_native_price) or current_native_price <= 0:
        raise ValueError(
            f"Current native price for {pos.ticker} must be positive.")
    if not math.isfinite(current_fx_rate) or current_fx_rate <= 0:
        raise ValueError(
            f"Current FX rate for {pos.ticker} must be positive.")
    native_ccy = (pos.native_currency or "").strip().upper() or None
    if native_ccy is None:
        raise ValueError(
            f"Native currency for {pos.ticker} is unknown — refusing to "
            "assume USD.")
    native_value = pos.shares * current_native_price
    base_value = native_value * current_fx_rate
    native_cost = base_cost = pnl = pnl_pct = None
    cost_source = pos.purchase_price_source
    if (pos.purchase_price_native is not None
            and math.isfinite(pos.purchase_price_native)
            and pos.purchase_price_native > 0
            and purchase_fx_rate is not None
            and math.isfinite(purchase_fx_rate)
            and purchase_fx_rate > 0):
        native_cost = pos.shares * pos.purchase_price_native
        base_cost = native_cost * purchase_fx_rate
        pnl = base_value - base_cost
        pnl_pct = pnl / base_cost if base_cost else None
    return PositionValuation(
        ticker=pos.ticker, shares=pos.shares,
        native_currency=native_ccy,
        native_current_price=current_native_price,
        price_source=price_source or "Unknown",
        current_price_timestamp=price_timestamp,
        base_currency=base, current_fx_rate=current_fx_rate,
        current_fx_timestamp=None,
        native_market_value=native_value, base_market_value=base_value,
        native_cost_basis=native_cost, base_cost_basis=base_cost,
        cost_source=cost_source, unrealised_pnl=pnl,
        unrealised_pnl_pct=pnl_pct)


def value_portfolio(valuations: list[PositionValuation]) -> dict:
    """Aggregate valuations with hard reconciliation checks.

    * weights sum to 1 within 1e-6 (never silent 99.3%/101.7%);
    * total P&L == Σ position P&L; total cost == Σ position costs.
    """
    if not valuations:
        raise ValueError("No position valuations to aggregate.")
    base_ccys = {v.base_currency for v in valuations}
    if len(base_ccys) != 1:
        raise ValueError(
            f"Mixed base currencies in one valuation: {sorted(base_ccys)}.")
    total_value = sum(v.base_market_value for v in valuations)
    if not math.isfinite(total_value) or total_value <= 0:
        raise ValueError("Total portfolio value must be positive.")
    weights = {v.ticker: v.base_market_value / total_value
               for v in valuations}
    wsum = sum(weights.values())
    if abs(wsum - 1.0) > 1e-6:
        raise ValueError(
            f"Portfolio weights sum to {wsum:.6f}, not 1.0.")
    total_cost, total_pnl = 0.0, 0.0
    cost_known, pnl_known = True, True
    for v in valuations:
        if v.base_cost_basis is None:
            cost_known = False
        else:
            total_cost += v.base_cost_basis
        if v.unrealised_pnl is None:
            pnl_known = False
        else:
            total_pnl += v.unrealised_pnl
    return {
        "base_currency": valuations[0].base_currency,
        "total_base_value": total_value,
        "weights": weights,
        "total_base_cost": total_cost if cost_known else None,
        "total_pnl": total_pnl if pnl_known else None,
        "total_pnl_pct": (total_pnl / total_cost
                          if pnl_known and cost_known and total_cost else None),
    }


# ---------------------------------------------------------------------------
# Portfolio persistence schema (versioned JSON, never secrets)
# ---------------------------------------------------------------------------

EXPORT_SCHEMA_VERSION = 1
_SECRET_LIKE = ("key", "secret", "token", "password", "credential")


def export_portfolio(positions: dict[str, dict], base_currency: str,
                     run_id: str | None = None) -> dict:
    """Build the versioned export structure (positions + settings only).

    Only whitelisted position fields are serialised — API keys, env vars
    and Streamlit secrets can never be included (see test).
    """
    base = validate_base_currency(base_currency)
    out_positions = []
    for ticker in sorted(positions):
        data = positions[ticker]
        if not isinstance(data, dict):
            raise ValueError(f"Position {ticker} must be a mapping.")
        pos = position_from_dict(ticker, data)  # validates everything
        entry = position_to_dict(pos)
        entry["ticker"] = pos.ticker
        out_positions.append(entry)
    doc = {"schema_version": EXPORT_SCHEMA_VERSION,
           "base_currency": base, "positions": out_positions}
    if run_id is not None:
        doc["run_id"] = str(run_id)
    return doc


def validate_positions_import(data: dict) -> tuple[dict, str]:
    """Validate an imported portfolio document (untrusted input).

    Returns ``(positions_dict, base_currency)`` in app-native shape
    (``{ticker: {...}}`` mirroring ``position_to_dict`` fields).
    Rejects unknown schema versions, malformed tickers/shares/dates/prices
    and unknown currency codes. Never ``eval``/deserialise callables.
    """
    if not isinstance(data, dict):
        raise ValueError("Imported portfolio must be a JSON object.")
    version = data.get("schema_version")
    if version != EXPORT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported portfolio schema version: {version!r} "
            f"(expected {EXPORT_SCHEMA_VERSION}).")
    base = validate_base_currency(data.get("base_currency", ""))
    raw_positions = data.get("positions")
    if not isinstance(raw_positions, list) or not raw_positions:
        raise ValueError("Imported portfolio has no positions list.")
    positions: dict[str, dict] = {}
    for i, entry in enumerate(raw_positions):
        if not isinstance(entry, dict):
            raise ValueError(f"Position #{i} must be a mapping.")
        ticker = entry.get("ticker", "")
        sym = validate_ticker_symbol(ticker)
        if sym in positions:
            raise ValueError(f"Duplicate ticker in import: {sym}.")
        # position_from_dict performs full field validation.
        pos = position_from_dict(sym, entry)
        stored = position_to_dict(pos)
        if pos.native_currency is not None and not (
                len(pos.native_currency) == 3
                and pos.native_currency.isalpha()):
            raise ValueError(f"Unknown currency for {sym}.")
        positions[sym] = stored
    return positions, base


def build_analysis_snapshot(
    tickers: list[str],
    shares: dict[str, float],
    purchase_dates: dict[str, str | None],
    base_currency: str,
    period: str,
    benchmark: str,
    risk_free: float,
    optimisation_method: str | None,
    min_weight: float,
    max_weight: float,
    scenario_settings: dict | None,
    factor_model: str | None,
    fx_source: str,
    history_end: str,
) -> dict:
    """Versioned reproducible-analysis snapshot (inputs only, no secrets)."""
    base = validate_base_currency(base_currency)
    snap = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "run_id": ("PC-" + datetime.now(timezone.utc).strftime("%Y%m%d")
                   + "-" + secrets.token_hex(2).upper()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tickers": list(tickers),
        "shares": {t: float(shares[t]) for t in tickers},
        "purchase_dates": {t: purchase_dates.get(t) for t in tickers},
        "base_currency": base,
        "historical_period": str(period),
        "benchmark": str(benchmark),
        "risk_free": float(risk_free),
        "optimisation_method": optimisation_method,
        "min_weight": float(min_weight),
        "max_weight": float(max_weight),
        "scenario_settings": dict(scenario_settings or {}),
        "factor_model": factor_model,
        "fx_source": str(fx_source),
        "history_end": str(history_end),
    }
    return snap


def summarize_portfolio_csv(
    valuations: list[PositionValuation],
    positions: dict[str, Position],
) -> pd.DataFrame:
    """Human-readable portfolio summary frame for CSV export."""
    rows = []
    for v in valuations:
        pos = positions.get(v.ticker)
        purchase_dt = pos.purchase_date.isoformat() if pos and pos.purchase_date else ""
        pp = pos.purchase_price_native if pos else None
        rows.append({
            "Ticker": v.ticker,
            "Shares": v.shares,
            "Purchase Date": purchase_dt,
            "Purchase Price": round(pp, 4) if pp is not None else "",
            "Native Currency": v.native_currency,
            "Current Price": round(v.native_current_price, 4),
            "Market Value": round(v.base_market_value, 2),
            "Weight": "",
            "P&L": round(v.unrealised_pnl, 2)
            if v.unrealised_pnl is not None else "",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        total = df["Market Value"].sum()
        if total > 0:
            df["Weight"] = (df["Market Value"] / total).map(
                lambda w: f"{w:.1%}")
    return df

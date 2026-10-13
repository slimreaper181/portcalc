"""
execution.py
------------
Research execution simulator (Task 10, Milestone 7) — hypothetical fills only.

This module NEVER submits broker orders: there is no network code, no Alpaca
order endpoint, no credential use. It simulates fills for MARKET and LIMIT
orders against supplied quotes/bars with explicit, conservative rules:

* MARKET BUY fills near the ask, MARKET SELL near the bid, plus a
  configurable slippage (bps). Missing touch reference → REJECTED, never
  invented.
* LIMIT orders fill only when the supplied market touches the limit
  (BUY: ask ≤ limit; SELL: bid ≥ limit) at the better of limit/touch, or —
  given a bar/quote sequence — at the first touch, filled at the limit
  (conservative: no price improvement claimed, no queue priority modelled).
  Unfilled by the horizon end → CANCELLED with a reason.
* Partial fills come only from an explicit deterministic ``available_size``
  fixture (shares at the touch). Nothing invents market depth.
* Commission is an optional flat and/or bps charge, reported separately.
* Simulation never mutates the real ledger: :func:`apply_fills_sandbox`
  replays fills onto a holdings/cash copy for before/after study only.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .validation import canonical_tickers, validate_ticker_symbol

#: Order types and sides.
ORDER_TYPES = ("MARKET", "LIMIT")
SIDES = ("BUY", "SELL")

#: Simulated lifecycle states.
STATUSES = ("NEW", "PARTIALLY_FILLED", "FILLED", "CANCELLED", "REJECTED")

_counter = itertools.count(1)


def _new_order_id() -> str:
    return f"SIM-{next(_counter):06d}"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SimQuote:
    """One market observation for simulation (partial legs allowed)."""

    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    available_size: float | None = None  # deterministic depth fixture


@dataclass
class SimOrder:
    """One hypothetical order and its simulated outcome."""

    order_id: str = ""
    ticker: str = ""
    side: str = ""          # BUY | SELL
    order_type: str = ""    # MARKET | LIMIT
    quantity_requested: float = 0.0
    limit_price: float | None = None
    status: str = "NEW"
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    reference_price: float | None = None
    slippage_cost: float = 0.0
    commission: float = 0.0
    reason: str = ""


@dataclass
class SimConfig:
    """Simulation assumptions (all explicit, all reported)."""

    slippage_bps: float = 5.0
    commission_bps: float = 0.0
    commission_flat: float = 0.0

    def validated(self) -> "SimConfig":
        for label, v, lo in (("slippage_bps", self.slippage_bps, 0.0),
                             ("commission_bps", self.commission_bps, 0.0),
                             ("commission_flat", self.commission_flat, 0.0)):
            try:
                f = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"{label} must be a number.")
            if not np.isfinite(f) or f < lo:
                raise ValueError(f"{label} must be finite and ≥ {lo}.")
        self.slippage_bps = float(self.slippage_bps)
        self.commission_bps = float(self.commission_bps)
        self.commission_flat = float(self.commission_flat)
        return self


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check_order(side: str, order_type: str, qty: float,
                 limit: float | None) -> tuple[str, str, float, float | None]:
    side = str(side or "").strip().upper()
    if side not in SIDES:
        raise ValueError(f"Side must be BUY/SELL, got {side!r}.")
    otype = str(order_type or "").strip().upper()
    if otype not in ORDER_TYPES:
        raise ValueError(f"Order type must be MARKET/LIMIT, got {otype!r}.")
    try:
        q = float(qty)
    except (TypeError, ValueError):
        raise ValueError("Quantity must be a number.")
    if not np.isfinite(q) or q <= 0:
        raise ValueError("Quantity must be positive and finite.")
    lim = None
    if otype == "LIMIT":
        if limit is None:
            raise ValueError("LIMIT orders need a limit price.")
        try:
            lim = float(limit)
        except (TypeError, ValueError):
            raise ValueError("Limit price must be a number.")
        if not np.isfinite(lim) or lim <= 0:
            raise ValueError("Limit price must be positive and finite.")
    return side, otype, q, lim


def _pos(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) and f > 0 else None


def _touch(quote: SimQuote, side: str) -> float | None:
    """Best reference: BUY→ask, SELL→bid, falling back to last (stated)."""
    if side == "BUY":
        return _pos(quote.ask) or _pos(quote.last) or _pos(quote.bid)
    return _pos(quote.bid) or _pos(quote.last) or _pos(quote.ask)


def _commission_for(notional: float, cfg: SimConfig) -> float:
    return abs(notional) * cfg.commission_bps / 1e4 + cfg.commission_flat


# ---------------------------------------------------------------------------
# Single-order simulation
# ---------------------------------------------------------------------------

def simulate_market_order(ticker: str, side: str, quantity: float,
                          quote: SimQuote, config: SimConfig,
                          order_id: str | None = None) -> SimOrder:
    """Hypothetical MARKET fill against one quote (pure)."""
    cfg = config.validated()
    sym = validate_ticker_symbol(ticker)
    side, otype, qty, _ = _check_order(side, "MARKET", quantity, None)
    order = SimOrder(order_id=order_id or _new_order_id(), ticker=sym,
                     side=side, order_type=otype, quantity_requested=qty)
    ref = _touch(quote, side)
    if ref is None:
        order.status, order.reason = "REJECTED", "no market reference price"
        return order
    order.reference_price = ref
    slip = cfg.slippage_bps / 1e4
    fill = ref * (1.0 + slip) if side == "BUY" else ref * (1.0 - slip)
    depth = _pos(quote.available_size)
    filled = min(qty, depth) if depth is not None else qty
    order.filled_qty = float(filled)
    order.avg_fill_price = float(fill)
    order.slippage_cost = float(abs(fill - ref) * filled)
    order.commission = float(_commission_for(filled * fill, cfg))
    if filled <= 0:
        order.status, order.reason = "REJECTED", "no size available"
    elif filled < qty - 1e-12:
        order.status = "PARTIALLY_FILLED"
        order.reason = (f"depth fixture capped fill at {filled:g} of {qty:g} "
                        "shares (remainder assumed cancelled)")
    else:
        order.status, order.reason = "FILLED", "market fill at touch + slippage"
    return order


def simulate_limit_order(ticker: str, side: str, quantity: float,
                         limit_price: float, quote: SimQuote,
                         config: SimConfig,
                         order_id: str | None = None) -> SimOrder:
    """Hypothetical LIMIT fill against one quote (pure, conservative)."""
    cfg = config.validated()
    sym = validate_ticker_symbol(ticker)
    side, otype, qty, lim = _check_order(side, "LIMIT", quantity, limit_price)
    order = SimOrder(order_id=order_id or _new_order_id(), ticker=sym,
                     side=side, order_type=otype, quantity_requested=qty,
                     limit_price=lim)
    touched, touch_px = False, None
    if side == "BUY" and _pos(quote.ask) is not None and quote.ask <= lim:
        touched, touch_px = True, float(quote.ask)
    elif side == "SELL" and _pos(quote.bid) is not None and quote.bid >= lim:
        touched, touch_px = True, float(quote.bid)
    if not touched:
        order.status = "CANCELLED"
        order.reason = ("limit away from market at horizon end "
                        "(no fill, no queue priority modelled)")
        return order
    # Marketable: fill at the touch (never better than the limit allows —
    # BUY pays min(limit, ask)=ask here; SELL receives bid), plus slippage.
    slip = cfg.slippage_bps / 1e4
    if side == "BUY":
        fill = min(lim, touch_px) * (1.0 + slip)
        fill = min(fill, lim * (1.0 + slip))
    else:
        fill = max(lim, touch_px) * (1.0 - slip)
        fill = max(fill, lim * (1.0 - slip))
    order.reference_price = touch_px
    depth = _pos(quote.available_size)
    filled = min(qty, depth) if depth is not None else qty
    order.filled_qty = float(filled)
    order.avg_fill_price = float(fill)
    order.slippage_cost = float(abs(fill - touch_px) * filled)
    order.commission = float(_commission_for(filled * fill, cfg))
    if filled < qty - 1e-12:
        order.status = "PARTIALLY_FILLED"
        order.reason = (f"depth fixture capped fill at {filled:g} of {qty:g} "
                        "shares (remainder assumed cancelled)")
    else:
        order.status, order.reason = "FILLED", "limit touched (filled at touch)"
    return order


def simulate_limit_over_bars(ticker: str, side: str, quantity: float,
                             limit_price: float, bars: pd.DataFrame,
                             config: SimConfig,
                             sizes: list | None = None,
                             order_id: str | None = None) -> SimOrder:
    """LIMIT order walked over a deterministic bar/quote sequence.

    ``bars`` needs ``high``/``low`` columns (or a single ``price`` column
    treated as both). Optional ``sizes`` gives per-bar available size
    (default: full size at first touch). Fills accumulate at the limit
    (conservative — no price improvement); unfilled remainder → CANCELLED
    (or PARTIALLY_FILLED when something filled).
    """
    cfg = config.validated()
    sym = validate_ticker_symbol(ticker)
    side, otype, qty, lim = _check_order(side, "LIMIT", quantity, limit_price)
    order = SimOrder(order_id=order_id or _new_order_id(), ticker=sym,
                     side=side, order_type=otype, quantity_requested=qty,
                     limit_price=lim)
    if bars is None or bars.empty:
        order.status, order.reason = "REJECTED", "no market sequence supplied"
        return order
    if "price" in bars.columns and ("high" not in bars.columns
                                    or "low" not in bars.columns):
        highs = lows = pd.to_numeric(bars["price"], errors="coerce")
    else:
        try:
            highs = pd.to_numeric(bars["high"], errors="coerce")
            lows = pd.to_numeric(bars["low"], errors="coerce")
        except KeyError:
            order.status = "REJECTED"
            order.reason = "bars need high/low (or price) columns"
            return order
    remaining, filled, notional = qty, 0.0, 0.0
    touch_px = None
    for i in range(len(bars)):
        if remaining <= 1e-12:
            break
        try:
            hi, lo = float(highs.iloc[i]), float(lows.iloc[i])
        except (IndexError, TypeError, ValueError):
            continue
        if not np.isfinite(hi) or not np.isfinite(lo):
            continue
        hit = (lo <= lim) if side == "BUY" else (hi >= lim)
        if not hit:
            continue
        touch_px = lim if touch_px is None else touch_px
        cap = None
        if sizes is not None and i < len(sizes):
            cap = _pos(sizes[i])
        take = min(remaining, cap) if cap is not None else remaining
        filled += take
        notional += take * lim
        remaining -= take
    if filled <= 1e-12:
        order.status = "CANCELLED"
        order.reason = ("limit never touched in sequence "
                        "(no queue priority modelled)")
        return order
    order.filled_qty = float(filled)
    order.avg_fill_price = float(notional / filled)
    order.reference_price = float(touch_px)
    order.slippage_cost = 0.0  # filled at the limit by rule — stated
    order.commission = float(_commission_for(notional, cfg))
    if remaining > 1e-12:
        order.status = "PARTIALLY_FILLED"
        order.reason = (f"sequence filled {filled:g} of {qty:g} shares "
                        "(remainder assumed cancelled)")
    else:
        order.status, order.reason = "FILLED", "filled at limit on touch"
    return order


# ---------------------------------------------------------------------------
# Blotter + rebalance integration + cost comparison + sandbox
# ---------------------------------------------------------------------------

def blotter_frame(orders: list[SimOrder]) -> pd.DataFrame:
    """Execution blotter (§7.8 columns)."""
    rows = []
    for o in orders or []:
        rows.append({
            "Order ID": o.order_id, "Ticker": o.ticker, "Side": o.side,
            "Type": o.order_type, "Requested": o.quantity_requested,
            "Filled": o.filled_qty,
            "Average Fill": o.avg_fill_price,
            "Reference": o.reference_price,
            "Slippage": o.slippage_cost, "Commission": o.commission,
            "Status": o.status, "Note": o.reason,
        })
    cols = ["Order ID", "Ticker", "Side", "Type", "Requested", "Filled",
            "Average Fill", "Reference", "Slippage", "Commission",
            "Status", "Note"]
    return pd.DataFrame(rows, columns=cols)


def orders_from_intents(intents, order_type: str = "MARKET",
                        limits: dict | None = None) -> list[dict]:
    """Rebalancing intents → hypothetical order specs (HOLD/zero skipped)."""
    otype = str(order_type or "MARKET").strip().upper()
    if otype not in ORDER_TYPES:
        raise ValueError(f"Order type must be MARKET/LIMIT, got {otype!r}.")
    specs = []
    for intent in intents or []:
        side = str(getattr(intent, "side", "")).upper()
        if side not in SIDES:
            continue
        try:
            qty = abs(float(getattr(intent, "quantity_shares", 0.0)))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(qty) or qty <= 0:
            continue
        ticker = str(getattr(intent, "ticker", ""))
        lim = None
        if otype == "LIMIT":
            if not limits or ticker not in limits:
                raise ValueError(
                    f"LIMIT simulation needs a limit price for {ticker}.")
            lim = float(limits[ticker])
        specs.append({"ticker": ticker, "side": side, "quantity": qty,
                      "limit_price": lim})
    return specs


def simulate_specs(specs: list[dict], quotes: dict,
                   config: SimConfig) -> list[SimOrder]:
    """Simulate order specs against a ticker → SimQuote map (pure)."""
    orders = []
    for s in specs or []:
        q = quotes.get(s["ticker"], SimQuote())
        if s.get("limit_price") is not None:
            orders.append(simulate_limit_order(
                s["ticker"], s["side"], s["quantity"], s["limit_price"],
                q, config))
        else:
            orders.append(simulate_market_order(
                s["ticker"], s["side"], s["quantity"], q, config))
    return orders


def simulation_summary(orders: list[SimOrder]) -> dict:
    """Totals across simulated orders (descriptive only)."""
    orders = list(orders or [])
    gross = sum((o.avg_fill_price or 0.0) * o.filled_qty for o in orders)
    return {
        "n_orders": len(orders),
        "n_filled": sum(1 for o in orders if o.status == "FILLED"),
        "n_partial": sum(1 for o in orders
                         if o.status == "PARTIALLY_FILLED"),
        "n_cancelled": sum(1 for o in orders if o.status == "CANCELLED"),
        "n_rejected": sum(1 for o in orders if o.status == "REJECTED"),
        "filled_notional": float(gross),
        "slippage": float(sum(o.slippage_cost for o in orders)),
        "commissions": float(sum(o.commission for o in orders)),
        "total_friction": float(sum(o.slippage_cost + o.commission
                                    for o in orders)),
    }


def cost_comparison(simulated_friction: float, modelled_bps: float,
                    notional: float) -> dict:
    """Simulated spread/slippage/commission vs the modelled assumption (§7.13).

    ``modelled`` = bps/10⁴ × notional (the planner/backtest assumption);
    both legs are estimates — neither is presented as realised truth.
    """
    try:
        sim, bps, nom = (float(simulated_friction), float(modelled_bps),
                         float(notional))
    except (TypeError, ValueError):
        raise ValueError("Cost comparison inputs must be numbers.")
    if not all(np.isfinite([sim, bps, nom])) or nom <= 0 or bps < 0:
        raise ValueError("Invalid cost comparison inputs.")
    modelled = bps / 1e4 * nom
    return {"simulated": sim, "modelled": float(modelled),
            "difference": float(sim - modelled), "notional": float(nom)}


def apply_fills_sandbox(holdings: dict, cash: float,
                        orders: list[SimOrder],
                        prices: dict) -> tuple[dict, float]:
    """Replay fills onto a holdings/cash COPY (never the ledger).

    BUY adds shares (pays fill notional + commission from cash); SELL
    removes shares (blocked past available — remainder skipped, stated).
    Returns ``(new_holdings, new_cash)``; inputs are never mutated.
    """
    hold = {str(t): float(q) for t, q in (holdings or {}).items()}
    new_cash = float(cash)
    if not np.isfinite(new_cash):
        raise ValueError("Sandbox cash must be finite.")
    for o in orders or []:
        if o.status not in ("FILLED", "PARTIALLY_FILLED"):
            continue
        if o.avg_fill_price is None or o.filled_qty <= 0:
            continue
        px = prices.get(o.ticker)
        if o.side == "BUY":
            hold[o.ticker] = hold.get(o.ticker, 0.0) + o.filled_qty
            new_cash -= o.filled_qty * o.avg_fill_price + o.commission
        else:
            avail = hold.get(o.ticker, 0.0)
            take = min(avail, o.filled_qty)
            hold[o.ticker] = avail - take
            new_cash += take * o.avg_fill_price - o.commission
    return hold, new_cash

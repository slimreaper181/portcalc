"""
performance_cashflows.py
------------------------
Time-weighted and money-weighted returns around external cash flows.

* TWR links sub-period returns between external flows: deposits and
  withdrawals change capital but never count as performance.
* XIRR solves the annualised rate equating dated flows to zero (terminal
  portfolio value included as a final inflow). Structured failure when no
  valid solution exists (e.g. all flows one-signed).

Conventions: amounts are signed from the INVESTOR's perspective only inside
XIRR (outflows negative, terminal value positive). TWR takes dated
valuations plus dated external-flow amounts (positive = deposit). Full
float precision; rounding is presentation-only.
"""

from __future__ import annotations

import math
from datetime import date, datetime


def _to_ordinal(d) -> int:
    if isinstance(d, date):
        return d.toordinal() if not isinstance(d, datetime) \
            else d.date().toordinal()
    try:
        import pandas as pd
        return pd.Timestamp(d).date().toordinal()
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid date {d!r}.") from e


def time_weighted_return(valuations: list, external_flows: list) -> dict:
    """TWR over dated valuations with external cash flows.

    Args:
        valuations: ``[(date, portfolio_value)]`` sorted ascending, ≥2.
        external_flows: ``[(date, amount)]`` (amount > 0 deposit). Flows
            dated exactly on a valuation date apply BEFORE that date's
            return accrues (start-of-day convention, documented).

    Returns dict with ``twr`` (fraction), ``n_periods``. Raises
    ``ValueError`` on bad inputs (non-positive values break linking).
    """
    if valuations is None or len(valuations) < 2:
        raise ValueError("TWR needs at least 2 valuations.")
    pts = []
    for d, v in valuations:
        if v is None or not math.isfinite(float(v)) or float(v) <= 0:
            raise ValueError("Valuations must be positive and finite.")
        pts.append((_to_ordinal(d), float(v)))
    pts.sort(key=lambda p: p[0])
    flows = {}
    for d, a in (external_flows or []):
        if a is None or not math.isfinite(float(a)):
            raise ValueError("External flows must be finite.")
        day = _to_ordinal(d)
        flows[day] = flows.get(day, 0.0) + float(a)
    # Link sub-periods split at flow dates (flows adjust the base, never
    # the return).
    boundaries = sorted({p[0] for p in pts} | set(flows))
    values = {day: val for day, val in pts}
    # Carry last known value forward to flow-only dates for clean splitting.
    ordered_days = sorted(set(values) | set(flows))
    last = None
    filled = {}
    for day in ordered_days:
        if day in values:
            last = values[day]
        if last is None:
            raise ValueError("External flow predates first valuation.")
        filled[day] = last
    linked = 1.0
    days = sorted(filled)
    prev = days[0]
    for day in days[1:]:
        base = filled[prev] + flows.get(prev, 0.0)
        if base <= 0:
            raise ValueError("Non-positive capital base in TWR linking.")
        linked *= filled[day] / base
        prev = day
    # A deposit ON the final valuation date must not create performance:
    # the loop above already treats it as base for a zero-length stub only
    # when a later valuation exists; with none, strip its effect by construction.
    return {"twr": linked - 1.0, "n_periods": len(days) - 1}


def xirr(cashflows: list, guess: float = 0.1) -> dict:
    """Annualised money-weighted return (XIRR) via guarded secant/Newton.

    Args:
        cashflows: ``[(date, amount)]`` investor-signed (deposits/outflows
            negative, withdrawals/terminal value positive), ≥2 entries with
            both signs present.

    Returns dict with ``xirr`` (annual fraction). Raises ``ValueError``
    when no valid solution exists (single-signed flows) or convergence
    fails (structured failure — never a fabricated rate).
    """
    if cashflows is None or len(cashflows) < 2:
        raise ValueError("XIRR needs at least 2 cash flows.")
    flows = []
    for d, a in cashflows:
        if a is None or not math.isfinite(float(a)) or float(a) == 0:
            raise ValueError("Cash flows must be finite and non-zero.")
        flows.append((_to_ordinal(d), float(a)))
    flows.sort(key=lambda p: p[0])
    amounts = [a for _, a in flows]
    if not (any(a < 0 for a in amounts) and any(a > 0 for a in amounts)):
        raise ValueError(
            "XIRR needs both negative and positive flows (no valid solution).")
    t0 = flows[0][0]

    def _npv(r: float) -> float:
        if r <= -1.0:
            return math.inf
        total = 0.0
        for day, amount in flows:
            years = (day - t0) / 365.25
            total += amount / ((1.0 + r) ** years)
        return total

    def _d_npv(r: float) -> float:
        total = 0.0
        for day, amount in flows:
            years = (day - t0) / 365.25
            if years == 0:
                continue
            total += -years * amount / ((1.0 + r) ** (years + 1.0))
        return total

    r = float(guess)
    if not math.isfinite(r) or r <= -1.0:
        r = 0.1
    for _ in range(200):
        f = _npv(r)
        if abs(f) < 1e-9 * max(1.0, abs(sum(abs(a) for _, a in flows))):
            return {"xirr": r}
        d = _d_npv(r)
        step = f / d if d != 0 else 0.01
        # Guarded step: halve until NPV improves and rate stays valid.
        trial, improved = r, False
        for _ in range(60):
            trial = r - step
            if trial <= -0.9999:
                step /= 2.0
                continue
            try:
                f_trial = _npv(trial)
            except (OverflowError, ZeroDivisionError):
                step /= 2.0
                continue
            if abs(f_trial) < abs(f):
                improved = True
                break
            step /= 2.0
        if not improved:
            break
        if abs(trial - r) < 1e-12:
            r = trial
            break
        r = trial
    if abs(_npv(r)) < 1e-7 * max(1.0, abs(sum(abs(a) for _, a in flows))):
        return {"xirr": r}
    raise ValueError(
        "XIRR failed to converge for these cash flows (no valid solution).")


def benchmark_with_flows(benchmark_prices, flows: list,
                         initial: float) -> list:
    """Fair benchmark leg: same external flows buy benchmark units.

    Args:
        benchmark_prices: date → price mapping (base currency).
        flows: ``[(date, amount)]`` deposits positive (same dates as the
            portfolio's external flows where practical).
        initial: starting capital invested at the first price.

    Returns dated ``[(date, value)]`` valuations of the flow-matched
    benchmark portfolio. A deposit buys units at that date's price; value
    is units × price. Never compares a flow-heavy portfolio against a
    single initial benchmark investment.
    """
    if initial is None or not math.isfinite(float(initial)) \
            or float(initial) <= 0:
        raise ValueError("Initial capital must be positive.")
    px = {}
    for d, p in (benchmark_prices.items() if hasattr(benchmark_prices, "items")
                 else benchmark_prices):
        if p is None or not math.isfinite(float(p)) or float(p) <= 0:
            raise ValueError("Benchmark prices must be positive and finite.")
        try:
            import pandas as pd
            day = pd.Timestamp(d).date()
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid benchmark date {d!r}.") from e
        px[day] = float(p)
    if not px:
        raise ValueError("No benchmark prices supplied.")
    events = sorted([(d, float(a)) for d, a in (flows or [])])
    days = sorted(px)
    units = float(initial) / px[days[0]]
    applied: set = set()
    out = []
    for day in days:
        for k, (fday, famt) in enumerate(events):
            if k not in applied and famt > 0 and days[0] <= fday <= day:
                trade_days = [d for d in days if d >= fday]
                units += famt / px[trade_days[0]]
                applied.add(k)
        out.append((day, units * px[day]))
    return out

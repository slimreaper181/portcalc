"""
history.py
----------
Transaction-aware historical portfolio reconstruction (pure).

Rebuilds, for each date in a base-currency price panel, the holdings that
existed then (transactions with date ≤ D — never future lots), the cash
position then (all ledger flows up to D), and the resulting NAV series.
External cash flows are tracked separately so TWR/XIRR never mistake a
deposit for performance (Programmes 44/48/49/50).

Cash in base currency needs date-matched FX: pass the historical
USD-per-unit frame used for the base panel (same source — no second FX
truth). Dates whose cash cannot be converted are reported with unknown
cash (never invented).

All inputs explicit; insufficient data raises ``ValueError``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from analytics.ledger import Transaction


def _usd_per_unit(fx_hist, ccy: str, day) -> float | None:
    ccy = str(ccy).strip().upper()
    if ccy == "USD":
        return 1.0
    if fx_hist is None:
        return None
    try:
        frame = fx_hist.copy()
        past = frame.loc[frame.index.date <= day]
        if past.empty or ccy not in past.columns:
            return None
        tail = past[ccy].dropna()
        if tail.empty:
            return None
        v = float(tail.iloc[-1])
        return v if math.isfinite(v) and v > 0 else None
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def reconstruct_history(base_prices: pd.DataFrame, tickers: list[str],
                        transactions: list[Transaction],
                        fx_hist=None, base_currency: str = "USD",
                        initial_cash: dict | None = None) -> pd.DataFrame:
    """Daily holdings/cash/NAV history from the transaction record.

    Args:
        base_prices: base-currency price panel (any window; common dates).
        tickers: universe (fixed; assets simply hold zero before purchase).
        transactions: ledger entries (any order; internally date-ordered).
        fx_hist: historical USD-per-unit FX frame for cash conversion
            (None only safe when all cash is in ``base_currency``).
        base_currency: reporting currency.
        initial_cash: optional starting balances by currency (date of the
            first panel bar; use for pre-history cash, disclosed as such).

    Returns DataFrame indexed by date with ``nav``, ``securities_value``,
    ``cash_value`` (None when unconvertible), ``external_flow`` (dated
    deposits positive / withdrawals negative, base-converted when known)
    and per-ticker ``shares_<T>`` columns.

    Raises ``ValueError`` on empty panels or non-positive prices.
    """
    from analytics.validation import canonical_tickers
    tickers = canonical_tickers(list(tickers))
    base_currency = str(base_currency).strip().upper()
    if base_prices is None or base_prices.empty:
        raise ValueError("No price history supplied.")
    if not isinstance(base_prices.index, pd.DatetimeIndex):
        raise ValueError("Price history needs a DatetimeIndex.")
    panel = base_prices.reindex(columns=tickers).sort_index()
    if ((panel <= 0) | (~np.isfinite(panel))).any().any():
        bad = panel.columns[((panel <= 0)
                             | (~np.isfinite(panel))).any()].tolist()
        raise ValueError(f"Invalid prices for: {', '.join(bad)}.")
    txns = sorted([t for t in (transactions or [])
                   if isinstance(t, Transaction)],
                  key=lambda t: (t.date, t.transaction_id))
    first_day = panel.index[0].date()
    cash = dict(initial_cash or {})
    for ccy in list(cash):
        if not math.isfinite(float(cash[ccy])):
            raise ValueError(f"Invalid initial cash for {ccy}.")
    shares = {t: 0.0 for t in tickers}
    ti = 0
    rows = []
    for ts in panel.index:
        day = ts.date() if hasattr(ts, "date") else ts
        # Apply every transaction dated ≤ D (never future lots).
        while ti < len(txns) and txns[ti].date <= day:
            tx = txns[ti]
            if tx.ticker is not None and tx.ticker in shares:
                if tx.type == "BUY":
                    shares[tx.ticker] += tx.quantity
                elif tx.type == "SELL":
                    shares[tx.ticker] = max(
                        shares[tx.ticker] - tx.quantity, 0.0)
            ccy = tx.currency
            amt = 0.0
            if tx.type == "BUY":
                amt = -(tx.quantity * tx.price_native + tx.fee)
            elif tx.type == "SELL":
                amt = tx.quantity * tx.price_native - tx.fee
            elif tx.type in ("DIVIDEND", "INTEREST"):
                amt = tx.quantity * tx.price_native
            elif tx.type == "FEE":
                amt = -tx.fee
            elif tx.type == "DEPOSIT":
                amt = tx.quantity * tx.price_native
            elif tx.type == "WITHDRAWAL":
                amt = -(tx.quantity * tx.price_native)
            if amt:
                cash[ccy] = cash.get(ccy, 0.0) + amt
            ti += 1
        sec_value = float(sum(shares[t] * float(panel[t].loc[ts])
                              for t in tickers))
        cash_value, cash_known, flow = 0.0, True, 0.0
        for ccy, bal in cash.items():
            u = _usd_per_unit(fx_hist, ccy, day)
            ub = _usd_per_unit(fx_hist, base_currency, day)
            if u is None or ub is None or ub <= 0:
                cash_known = False
                continue
            cash_value += float(bal) * u / ub
        row = {"nav": (sec_value + cash_value if cash_known else None),
               "securities_value": sec_value,
               "cash_value": (cash_value if cash_known else None),
               "external_flow": 0.0}
        for t in tickers:
            row[f"shares_{t}"] = shares[t]
        rows.append((ts, row))
    # Build explicitly to keep column order stable.
    data: dict = {"nav": [], "securities_value": [], "cash_value": [],
                  "external_flow": []}
    share_cols = {t: [] for t in tickers}
    for _, row in rows:
        for k in data:
            data[k].append(row[k])
        for t in tickers:
            share_cols[t].append(row[f"shares_{t}"])
    for t in tickers:
        data[f"shares_{t}"] = share_cols[t]
    frame = pd.DataFrame(data, index=panel.index)
    # External flows: deposits/withdrawals dated in the window (base FX).
    flows = {}
    for tx in txns:
        if tx.type not in ("DEPOSIT", "WITHDRAWAL"):
            continue
        if tx.date < first_day or tx.date > panel.index[-1].date():
            continue
        amt = tx.quantity * tx.price_native
        if tx.type == "WITHDRAWAL":
            amt = -amt
        u = _usd_per_unit(fx_hist, tx.currency, tx.date)
        ub = _usd_per_unit(fx_hist, base_currency, tx.date)
        if u is None or ub is None or ub <= 0:
            continue
        flows[tx.date] = flows.get(tx.date, 0.0) + amt * u / ub
    for ts in frame.index:
        day = ts.date() if hasattr(ts, "date") else ts
        if day in flows:
            frame.loc[ts, "external_flow"] = flows[day]
    return frame


def twr_from_history(frame: pd.DataFrame) -> dict:
    """TWR from a reconstructed history frame (external_flow-aware).

    The frame's NAV is INCLUSIVE of same-day flows (deposit cash lands in
    NAV that day), so each flow-day leg uses ``(V(D) − F(D)) / V(D−1)``
    (end-of-day flow convention): flows never count as performance.
    Leading unfunded dates (NAV ≤ 0) start measurement at first funding;
    a later zero base (full withdrawal) ends measurement honestly.
    """
    nav = [(ts.date() if hasattr(ts, "date") else ts, float(v))
           for ts, v in frame["nav"].items() if v is not None
           and math.isfinite(float(v))]
    flows = {}
    for ts, f in frame["external_flow"].items():
        day = ts.date() if hasattr(ts, "date") else ts
        if float(f) != 0.0:
            flows[day] = flows.get(day, 0.0) + float(f)
    # Start at first funded valuation (earlier zeros carry no performance).
    start = next((i for i, (_, v) in enumerate(nav) if v > 0), None)
    if start is None or len(nav) - start < 2:
        raise ValueError("Insufficient funded NAV history for TWR.")
    linked, n = 1.0, 0
    for i in range(start + 1, len(nav)):
        day, value = nav[i]
        prev_day, prev_value = nav[i - 1]
        if prev_value <= 0:
            raise ValueError("Capital fully withdrawn mid-window: TWR ends.")
        linked *= (value - flows.get(day, 0.0)) / prev_value
        n += 1
    if not math.isfinite(linked) or linked < 0:
        raise ValueError("TWR linking failed on this history.")
    return {"twr": linked - 1.0, "n_periods": n}


def xirr_from_ledger(transactions: list[Transaction],
                     terminal_value: float, terminal_date,
                     fx_hist=None, base_currency: str = "USD") -> dict:
    """XIRR from ledger external flows + terminal portfolio value."""
    from analytics.performance_cashflows import xirr
    base_currency = str(base_currency).strip().upper()
    flows = []
    for tx in transactions or []:
        if not isinstance(tx, Transaction):
            continue
        if tx.type == "DEPOSIT":
            sign = -1.0
        elif tx.type == "WITHDRAWAL":
            sign = 1.0
        else:
            continue
        amt = tx.quantity * tx.price_native
        u = _usd_per_unit(fx_hist, tx.currency, tx.date)
        ub = _usd_per_unit(fx_hist, base_currency, tx.date)
        if u is None or ub is None or ub <= 0:
            raise ValueError(
                f"Cannot convert {tx.type} on {tx.date} to {base_currency}.")
        flows.append((tx.date, sign * amt * u / ub))
    if terminal_value is None or not math.isfinite(float(terminal_value)) \
            or float(terminal_value) <= 0:
        raise ValueError("Terminal value must be positive.")
    try:
        import pandas as pd
        tday = pd.Timestamp(terminal_date).date()
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid terminal date: {e}")
    flows.append((tday, float(terminal_value)))
    return xirr(flows)

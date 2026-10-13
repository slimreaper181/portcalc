"""Tests for analytics.history (reconstruction, TWR/XIRR from ledger)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from analytics.history import (
    reconstruct_history,
    twr_from_history,
    xirr_from_ledger,
)
from analytics.ledger import Transaction


def _tx(tid, day, type, ticker=None, qty=0.0, px=1.0, ccy="USD", fx=1.0):
    return Transaction(transaction_id=tid, date=date(2026, 1, day),
                       type=type, ticker=ticker, quantity=qty,
                       price_native=px, currency=ccy, purchase_fx=fx)


def _px():
    idx = pd.bdate_range("2026-01-01", periods=60)
    return pd.DataFrame({"AAPL": np.linspace(100.0, 120.0, 60)},
                        index=idx)


def test_holdings_reflect_only_executed_lots():
    txns = [_tx("d", 2, "DEPOSIT", qty=100000, px=1.0),
            _tx("b", 5, "BUY", "AAPL", 10, 100.0)]
    frame = reconstruct_history(_px(), ["AAPL"], txns)
    assert (frame["shares_AAPL"].iloc[:2] == 0.0).all()  # no future lots
    assert (frame["shares_AAPL"].iloc[2:] == 10.0).all()
    assert frame["nav"].iloc[0] == pytest.approx(0.0)  # unfunded era: 0
    assert frame["nav"].iloc[-1] == pytest.approx(
        100000 - 1000 + 10 * 120.0)
    assert (frame["external_flow"] != 0).sum() == 1


def test_twr_ignores_deposits_in_history():
    # Idle mid-window deposit: TWR links clean sub-periods (deposit is
    # capital, never performance) — idle cash then correctly dilutes the
    # second leg, unlike a naive end/start return.
    txns = [_tx("d", 2, "DEPOSIT", qty=10000, px=1.0),
            _tx("b", 2, "BUY", "AAPL", 100, 100.0),
            _tx("d2", 30, "DEPOSIT", qty=10000, px=1.0)]
    frame = reconstruct_history(_px(), ["AAPL"], txns)
    out = twr_from_history(frame)
    v = frame["nav"]
    fday = date(2026, 1, 30)
    i = v.index[v.index.date == fday][0]
    pos = v.index.get_loc(i)
    # Exact linked expectation: pre-flow value over the FIRST FUNDED bar
    # (position 1 — position 0 is the unfunded era), times the post-flow run.
    leg1 = (v.iloc[pos] - 10000.0) / v.iloc[1]
    leg2 = v.iloc[-1] / v.iloc[pos]
    assert out["twr"] == pytest.approx(leg1 * leg2 - 1.0, rel=1e-9)
    naive = v.iloc[-1] / v.iloc[2] - 1.0  # treats deposit as performance
    assert out["twr"] != pytest.approx(naive)
    assert out["twr"] < naive  # idle cash drags, not boosts


def test_twr_deployed_deposit_creates_no_performance():
    # Deposit immediately and FULLY deployed at that day's market price:
    # TWR equals the pure buy-and-hold price path (flows neither create
    # nor dilute). Amounts derive from the panel so execution is exact.
    px = _px()["AAPL"]
    buy_px = round(float(px.iloc[21]), 2)
    dep = round(90 * buy_px, 2)
    txns = [_tx("d", 2, "DEPOSIT", qty=10000, px=1.0),
            _tx("b", 2, "BUY", "AAPL", 100, 100.0),
            _tx("d2", 30, "DEPOSIT", qty=dep, px=1.0),
            _tx("b2", 30, "BUY", "AAPL", 90, buy_px)]
    frame = reconstruct_history(_px(), ["AAPL"], txns)
    out = twr_from_history(frame)
    # Cent-rounded execution leaves sub-cent dust; tolerance reflects that
    # (full-precision execution would telescope exactly).
    assert out["twr"] == pytest.approx(px.iloc[-1] / px.iloc[1] - 1.0,
                                       rel=1e-3)


def test_xirr_from_ledger_flows():
    txns = [_tx("d", 2, "DEPOSIT", qty=10000, px=1.0),
            _tx("d2", 30, "DEPOSIT", qty=5000, px=1.0)]
    out = xirr_from_ledger(txns, 18000.0, date(2026, 3, 2))
    assert -0.5 < out["xirr"] < 5.0
    with pytest.raises(ValueError):
        xirr_from_ledger(txns, -10.0, date(2026, 3, 2))


def test_insufficient_data_rejected():
    with pytest.raises(ValueError):
        reconstruct_history(pd.DataFrame(), ["AAPL"], [])
    with pytest.raises(ValueError):
        twr_from_history(pd.DataFrame(
            {"nav": [1.0], "external_flow": [0.0]}))

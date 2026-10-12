"""Tests for currency-keyed cash accounting (Programme 114 + precision)."""

from datetime import date

import pytest

from analytics.ledger import Transaction, apply_ledger


def _t(tid, day, type, ccy="USD", qty=0.0, px=1.0, fee=0.0, fx=1.0,
       ticker=None):
    return Transaction(transaction_id=tid, date=date(2026, 1, day),
                       type=type, ticker=ticker, quantity=qty,
                       price_native=px, currency=ccy, fee=fee,
                       purchase_fx=fx)


def test_cash_start_buy_fee_sell_reconcile():
    # Programme 114: $10,000 → buy 10 @ $100 + fee → sell later → reconcile.
    st = apply_ledger([
        _t("d", 1, "DEPOSIT", qty=10000.0),
        _t("b", 2, "BUY", qty=10.0, px=100.0, ticker="AAPL", fee=5.0),
        _t("s", 3, "SELL", qty=10.0, px=110.0, ticker="AAPL", fee=5.0),
    ])
    assert st.cash == {"USD": pytest.approx(10000 - 1005 + 1095)}
    assert st.fees_paid_base == pytest.approx(10.0)
    assert st.realised[0].fee_native == pytest.approx(5.0)


def test_cash_precision_not_rounded_early():
    st = apply_ledger([
        _t("d", 1, "DEPOSIT", qty=10000.123456),
        _t("b", 2, "BUY", qty=3.333333, px=111.111111, ticker="AAPL"),
    ])
    expect = 10000.123456 - 3.333333 * 111.111111
    assert st.cash["USD"] == pytest.approx(expect, rel=1e-9)


def test_multi_currency_cash_independent():
    st = apply_ledger([
        _t("d1", 1, "DEPOSIT", qty=5000.0),
        _t("d2", 1, "DEPOSIT", "GBP", qty=3000.0),
        _t("w", 2, "WITHDRAWAL", qty=1000.0),
    ])
    assert st.cash == {"USD": pytest.approx(4000.0),
                       "GBP": pytest.approx(3000.0)}
    with pytest.raises(ValueError, match="negative"):
        apply_ledger([_t("w", 2, "WITHDRAWAL", qty=1.0)])

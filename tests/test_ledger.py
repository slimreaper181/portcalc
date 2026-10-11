"""Tests for analytics.ledger (transactions, FIFO, cash, realised P&L)."""

from datetime import date

import pytest

from analytics.ledger import (
    LedgerState,
    Transaction,
    apply_ledger,
    average_cost_per_share,
    cost_by_ticker,
    new_transaction_id,
    open_shares_by_ticker,
    unrealised_by_ticker,
    validate_transaction,
)


def _tx(day, type, ticker=None, qty=0.0, px=0.0, ccy="USD", fee=0.0,
        fx=1.0, tid=None, source="user"):
    return Transaction(
        transaction_id=tid or f"t-{day}-{type}-{ticker or ccy}",
        date=date(2026, 1, day), type=type, ticker=ticker, quantity=qty,
        price_native=px, currency=ccy, fee=fee, purchase_fx=fx,
        source=source)


def _buy(day, ticker, qty, px, ccy="USD", fee=0.0, fx=1.0, tid=None):
    return _tx(day, "BUY", ticker, qty, px, ccy, fee, fx, tid)


# ---------------------------------------------------------------------------
# Validation + single buy
# ---------------------------------------------------------------------------

def test_transaction_validation():
    with pytest.raises(ValueError):
        validate_transaction(_tx(1, "HOLD", "AAPL", 1, 1))
    with pytest.raises(ValueError):
        validate_transaction(_tx(1, "BUY", "AAPL", -1, 100))
    with pytest.raises(ValueError):
        validate_transaction(_tx(1, "BUY", "AAPL", 1, 0))
    with pytest.raises(ValueError):
        validate_transaction(_tx(1, "BUY", None, 1, 100))
    with pytest.raises(ValueError):
        validate_transaction(_tx(1, "FEE", "AAPL", 5, 1))
    t = validate_transaction(_tx(1, "BUY", "aapl", 10, 100.0))
    assert t.ticker == "AAPL" and t.type == "BUY"
    assert new_transaction_id() != new_transaction_id()


def test_single_buy_creates_lot_and_moves_cash():
    st = apply_ledger([_tx(1, "DEPOSIT", qty=10000, px=1.0),
                       _buy(2, "AAPL", 10, 100.0, fee=5.0)])
    assert st.cash == {"USD": pytest.approx(10000 - 1000 - 5)}
    assert open_shares_by_ticker(st) == {"AAPL": pytest.approx(10.0)}
    lot = st.open_lots[0]
    assert lot.shares_remaining == pytest.approx(10.0)
    assert lot.cost_basis_native == pytest.approx(1005.0)  # fee in basis
    assert lot.cost_basis_base == pytest.approx(1005.0)
    assert st.realised == []


def test_two_buys_partial_sell_fifo():
    # Programme 113: BUY 10 @100, BUY 5 @120, SELL 12 @130.
    txns = [_tx(1, "DEPOSIT", qty=10000, px=1.0),
            _buy(2, "AAPL", 10, 100.0, tid="b1"),
            _buy(3, "AAPL", 5, 120.0, tid="b2"),
            _tx(4, "SELL", "AAPL", 12, 130.0, tid="s1")]
    st = apply_ledger(txns)
    # FIFO: 10 from b1 + 2 from b2; 3 remain at second-lot basis.
    assert open_shares_by_ticker(st) == {"AAPL": pytest.approx(3.0)}
    assert st.open_lots[0].transaction_id == "b2"
    assert st.open_lots[0].cost_basis_native == pytest.approx(3 * 120.0)
    (ev,) = st.realised
    assert ev.shares_sold == pytest.approx(12.0)
    assert ev.cost_native == pytest.approx(10 * 100.0 + 2 * 120.0)
    assert ev.proceeds_native == pytest.approx(12 * 130.0)
    assert ev.realised_pnl_native == pytest.approx(1560 - 1240)
    assert ev.realised_pnl_base == pytest.approx(320.0)
    assert ev.lots_consumed == [("lot-b1", pytest.approx(10.0)),
                                ("lot-b2", pytest.approx(2.0))]
    assert st.cash == {"USD": pytest.approx(10000 - 1600 + 1560)}


def test_full_sell_then_buy_after_sell():
    txns = [_tx(1, "DEPOSIT", qty=5000, px=1.0),
            _buy(2, "AAPL", 10, 100.0),
            _tx(3, "SELL", "AAPL", 10, 110.0),
            _buy(4, "AAPL", 4, 90.0)]
    st = apply_ledger(txns)
    assert open_shares_by_ticker(st) == {"AAPL": pytest.approx(4.0)}
    assert len(st.realised) == 1
    assert st.realised[0].realised_pnl_native == pytest.approx(100.0)
    assert st.cash == {"USD": pytest.approx(5000 - 1000 + 1100 - 360)}


def test_oversell_rejected_no_shorting():
    txns = [_tx(1, "DEPOSIT", qty=5000, px=1.0),
            _buy(2, "AAPL", 3, 100.0),
            _tx(3, "SELL", "AAPL", 4, 110.0)]
    with pytest.raises(ValueError, match="only 3"):
        apply_ledger(txns)


def test_duplicate_ids_rejected_and_order_deterministic():
    a = _buy(5, "AAPL", 1, 100.0, tid="same")
    b = _buy(2, "AAPL", 1, 100.0, tid="same")
    with pytest.raises(ValueError, match="Duplicate"):
        apply_ledger([a, b])
    # Same-date ordering by id is deterministic.
    x = _buy(5, "AAPL", 1, 100.0, tid="x")
    y = _buy(5, "AAPL", 2, 100.0, tid="y")
    st1 = apply_ledger([_tx(1, "DEPOSIT", qty=99999, px=1.0), x, y])
    st2 = apply_ledger([_tx(1, "DEPOSIT", qty=99999, px=1.0), y, x])
    assert [l.transaction_id for l in st1.open_lots] == \
        [l.transaction_id for l in st2.open_lots]


# ---------------------------------------------------------------------------
# Fees / dividends / deposits / withdrawals / interest
# ---------------------------------------------------------------------------

def test_buy_fee_in_basis_sell_fee_in_proceeds():
    txns = [_tx(1, "DEPOSIT", qty=10000, px=1.0),
            _buy(2, "AAPL", 10, 100.0, fee=10.0),
            _tx(3, "SELL", "AAPL", 10, 110.0, fee=5.0)]
    st = apply_ledger(txns)
    (ev,) = st.realised
    assert ev.cost_native == pytest.approx(1010.0)
    assert ev.proceeds_native == pytest.approx(1100.0 - 5.0)
    assert ev.realised_pnl_native == pytest.approx(85.0)


def test_standalone_fee_moves_cash_only():
    st = apply_ledger([_tx(1, "DEPOSIT", qty=1000, px=1.0),
                       Transaction(transaction_id="f1", date=date(2026, 1, 2),
                                   type="FEE", quantity=7.0, price_native=1.0,
                                   currency="USD", fee=7.0, purchase_fx=1.0)])
    assert st.cash == {"USD": pytest.approx(993.0)}
    assert st.fees_paid_base == pytest.approx(7.0)
    assert open_shares_by_ticker(st) == {}


def test_dividend_adds_cash_not_shares():
    txns = [_tx(1, "DEPOSIT", qty=10000, px=1.0),
            _buy(2, "AAPL", 100, 50.0),
            _tx(10, "DIVIDEND", "AAPL", 100, 0.5, tid="d1")]
    st = apply_ledger(txns)
    assert open_shares_by_ticker(st) == {"AAPL": pytest.approx(100.0)}
    assert st.cash == {"USD": pytest.approx(10000 - 5000 + 50)}
    assert st.dividend_income_native == {"AAPL": pytest.approx(50.0)}
    assert st.dividend_income_base == pytest.approx(50.0)


def test_deposit_withdrawal_reconcile():
    st = apply_ledger([_tx(1, "DEPOSIT", None, 10000, 1.0, "USD"),
                       _tx(2, "DEPOSIT", None, 500, 1.0, "EUR"),
                       _tx(3, "WITHDRAWAL", None, 2000, 1.0, "USD")])
    assert st.cash["USD"] == pytest.approx(8000.0)
    assert st.cash["EUR"] == pytest.approx(500.0)


def test_negative_cash_rejected_unless_allowed():
    with pytest.raises(ValueError, match="negative"):
        apply_ledger([_buy(2, "AAPL", 10, 100.0)])
    st = apply_ledger([_buy(2, "AAPL", 10, 100.0)],
                      allow_negative_cash=True)
    assert st.cash == {"USD": pytest.approx(-1000.0)}


# ---------------------------------------------------------------------------
# Multicurrency lots (Programme 117)
# ---------------------------------------------------------------------------

def test_gbp_lot_records_native_and_base():
    txns = [_tx(1, "DEPOSIT", None, 10000, 1.0, "GBP"),
            _buy(2, "BARC.L", 100, 5.00, "GBP", fx=1.25, tid="g1"),
            _tx(3, "SELL", "BARC.L", 40, 5.50, "GBP", fx=1.30, tid="g2")]
    st = apply_ledger(txns)
    assert open_shares_by_ticker(st) == {"BARC.L": pytest.approx(60.0)}
    assert st.open_lots[0].cost_basis_native == pytest.approx(60 * 5.00)
    assert st.open_lots[0].cost_basis_base == pytest.approx(60 * 5.00 * 1.25)
    (ev,) = st.realised
    assert ev.proceeds_native == pytest.approx(40 * 5.50)
    assert ev.cost_native == pytest.approx(40 * 5.00)
    assert ev.proceeds_base == pytest.approx(220.0 * 1.30)
    assert ev.cost_base == pytest.approx(200.0 * 1.25)
    assert ev.realised_pnl_base == pytest.approx(286.0 - 250.0)
    # FX identity: realised_base − native realised at sale FX.
    assert ev.fx_effect_base == pytest.approx(
        ev.realised_pnl_base - ev.realised_pnl_native * 1.30)
    assert st.cash == {"GBP": pytest.approx(10000 - 500 + 220)}


def test_unknown_fx_leaves_base_unknown_not_invented():
    txns = [_tx(1, "DEPOSIT", qty=10000, px=1.0, ccy="GBP"),
            _buy(2, "BARC.L", 100, 5.00, "GBP", fx=None, tid="u1")]
    st = apply_ledger(txns)
    assert st.open_lots[0].cost_basis_base is None
    assert st.open_lots[0].cost_basis_native == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Derived views: averages, unrealised, reconciliation
# ---------------------------------------------------------------------------

def test_average_cost_display_not_used_for_realised():
    st = apply_ledger([_tx(1, "DEPOSIT", qty=99999, px=1.0),
                       _buy(2, "AAPL", 10, 100.0),
                       _buy(3, "AAPL", 10, 200.0)])
    avg, ccy = average_cost_per_share(st)["AAPL"]
    assert avg == pytest.approx(150.0) and ccy == "USD"


def test_unrealised_reconciles_to_lots():
    st = apply_ledger([_tx(1, "DEPOSIT", qty=99999, px=1.0),
                       _buy(2, "AAPL", 10, 100.0),
                       _buy(3, "AAPL", 5, 120.0)])
    u = unrealised_by_ticker(st, {"AAPL": 130.0}, {"USD": 1.0}, "USD")
    row = u["AAPL"]
    assert row["market_base"] == pytest.approx(15 * 130.0)
    assert row["cost_base"] == pytest.approx(1000.0 + 600.0)
    assert row["unrealised_base"] == pytest.approx(1950.0 - 1600.0)
    with pytest.raises(ValueError):
        unrealised_by_ticker(st, {}, {"USD": 1.0}, "USD")


def test_full_flow_cash_lots_realised_unrealised_income_nav():
    # Programme 207: deposit → buy → buy → dividend → partial sell → fee.
    txns = [_tx(1, "DEPOSIT", qty=10000, px=1.0),
            _buy(2, "AAPL", 10, 100.0, tid="a1"),
            _buy(3, "AAPL", 10, 110.0, tid="a2"),
            _tx(4, "DIVIDEND", "AAPL", 20, 0.5, tid="d1"),
            _tx(5, "SELL", "AAPL", 5, 120.0, tid="s1"),
            Transaction(transaction_id="f1", date=date(2026, 1, 6),
                        type="FEE", quantity=2.0, price_native=1.0,
                        currency="USD", fee=2.0, purchase_fx=1.0)]
    st = apply_ledger(txns)
    assert st.cash == {"USD": pytest.approx(10000 - 2100 + 10 + 600 - 2)}
    assert open_shares_by_ticker(st) == {"AAPL": pytest.approx(15.0)}
    (ev,) = st.realised
    assert ev.cost_native == pytest.approx(5 * 100.0)
    assert ev.realised_pnl_native == pytest.approx(600 - 500)
    u = unrealised_by_ticker(st, {"AAPL": 125.0}, {"USD": 1.0}, "USD")
    # remaining: 5@100 + 10@110 = 1600 cost; 15×125 = 1875 value.
    assert u["AAPL"]["cost_base"] == pytest.approx(1600.0)
    assert u["AAPL"]["unrealised_base"] == pytest.approx(275.0)
    nav = u["AAPL"]["market_base"] + st.cash["USD"]
    assert nav == pytest.approx(1875.0 + 8508.0)
    assert cost_by_ticker(st)["AAPL"][0] == pytest.approx(1600.0)

"""Tests for analytics.persistence (v2 schema, migration, roundtrips)."""

import json
from datetime import date

import pytest

from analytics.ledger import Transaction, apply_ledger, open_shares_by_ticker
from analytics.persistence import (
    PORTFOLIO_SCHEMA_VERSION,
    migrate_v1_to_ledger,
    rebase_ledger_fx,
    transaction_from_dict,
    transaction_to_dict,
    validate_portfolio_import,
    export_portfolio_v2,
)


def _pos(shares=10.0, pdate="2026-08-13", px=305.26, ccy="USD",
         unit="USD", scale=1.0, source="estimate", manual=False):
    return {"shares": shares, "purchase_date": pdate,
            "purchase_price_native": px, "purchase_price_source": source,
            "manual_purchase_price": manual, "native_currency": ccy,
            "quote_unit": unit, "quote_scale": scale}


def test_v1_import_still_accepted():
    doc = {"schema_version": 1, "base_currency": "USD",
           "positions": [dict({"ticker": "AAPL"}, **_pos())]}
    out = validate_portfolio_import(doc)
    assert out.schema_version == 1 and out.base_currency == "USD"
    assert out.positions["AAPL"]["shares"] == pytest.approx(10.0)
    assert out.transactions == [] and out.watchlist == []


def test_v2_roundtrip_with_ledger_and_watchlist():
    txns = [Transaction(transaction_id="t1", date=date(2026, 1, 5),
                        type="DEPOSIT", quantity=10000.0, price_native=1.0,
                        currency="USD", purchase_fx=1.0),
            Transaction(transaction_id="t2", date=date(2026, 8, 13),
                        type="BUY", ticker="AAPL", quantity=10.0,
                        price_native=305.26, currency="USD", purchase_fx=1.0)]
    doc = export_portfolio_v2({"AAPL": _pos()}, "USD", txns, ["MSFT"],
                              {"theme": "dark"})
    blob = json.dumps(doc)
    assert "APCA" not in blob and "secret" not in blob.lower()
    assert doc["schema_version"] == PORTFOLIO_SCHEMA_VERSION
    back = validate_portfolio_import(json.loads(blob))
    assert len(back.transactions) == 2
    assert back.watchlist == ["MSFT"]
    assert back.settings == {"theme": "dark"}
    assert back.transactions[1].price_native == pytest.approx(305.26)


def test_import_rejects_garbage():
    with pytest.raises(ValueError):
        validate_portfolio_import({"schema_version": 99})
    with pytest.raises(ValueError):
        validate_portfolio_import(
            {"schema_version": 2, "base_currency": "USD", "positions": [],
             "transactions": [{"transaction_id": "x"}]})
    dup = {"schema_version": 2, "base_currency": "USD", "positions": [],
           "transactions": [
               {"transaction_id": "d", "date": "2026-01-01", "type": "DEPOSIT",
                "quantity": 1, "price_native": 1.0, "currency": "USD"},
               {"transaction_id": "d", "date": "2026-01-02", "type": "DEPOSIT",
                "quantity": 1, "price_native": 1.0, "currency": "USD"}]}
    with pytest.raises(ValueError, match="Duplicate transaction id"):
        validate_portfolio_import(dup)
    with pytest.raises(ValueError):
        validate_portfolio_import(
            {"schema_version": 2, "base_currency": "XX", "positions": []})


def test_migration_equivalence_and_skips():
    positions = {"AAPL": _pos(), "BARC.L": _pos(100.0, "2026-08-13", 5.205,
                                                "GBP", "GBp", 0.01),
                 "LEGACY": {"shares": 5.0, "legacy_avg_cost": 50.0,
                            "native_currency": "USD", "quote_unit": "USD",
                            "quote_scale": 1.0}}
    txns, skipped = migrate_v1_to_ledger(positions)
    assert skipped == ["LEGACY"]  # undated average: reported, not invented
    assert [t.transaction_id for t in txns] == ["mig-v1-AAPL-0",
                                                "mig-v1-BARC.L-1"]
    # Deterministic: same input → same ids.
    txns2, _ = migrate_v1_to_ledger(positions)
    assert [t.transaction_id for t in txns2] == \
        [t.transaction_id for t in txns]
    # Migrated lots price economics into the ledger (FX resolved later).
    dep = Transaction(transaction_id="dep", date=date(2026, 1, 2),
                      type="DEPOSIT", quantity=10**9, price_native=1.0,
                      currency="USD", purchase_fx=1.0)
    st = apply_ledger([dep] + txns, allow_negative_cash=True)
    assert open_shares_by_ticker(st)["AAPL"] == pytest.approx(10.0)
    assert open_shares_by_ticker(st)["BARC.L"] == pytest.approx(100.0)


def test_transaction_dict_roundtrip_precision():
    from datetime import datetime, timezone
    tx = Transaction(transaction_id="p1", date=date(2026, 8, 13), type="BUY",
                     ticker="BARC.L", quantity=123.456789,
                     price_native=5.205123, currency="GBP", fee=1.234567,
                     purchase_fx=1.23456789, notes="n", source="user")
    tx.created_at = datetime(2026, 8, 13, 10, 5, tzinfo=timezone.utc)
    back = transaction_from_dict(transaction_to_dict(tx))
    assert back.quantity == pytest.approx(123.456789, rel=1e-12)
    assert back.purchase_fx == pytest.approx(1.23456789, rel=1e-12)
    assert back.date == date(2026, 8, 13)
    assert back.move_cash is True
    assert back.created_at == tx.created_at
    # Absent stamp round-trips as absent (no invented timestamps).
    tx.created_at = None
    assert transaction_from_dict(
        transaction_to_dict(tx)).created_at is None


def test_rebase_preserves_native_recomputes_base():
    import pandas as pd
    from analytics.ledger import Transaction
    fx = pd.DataFrame({"GBP": [1.25, 1.30], "EUR": [1.10, 1.12]},
                      index=pd.bdate_range("2026-08-01", periods=2))
    tx = Transaction(transaction_id="r1", date=date(2026, 8, 3), type="BUY",
                     ticker="BARC.L", quantity=100.0, price_native=5.0,
                     currency="GBP", purchase_fx=None)
    user = Transaction(transaction_id="r2", date=date(2026, 8, 3),
                       type="BUY", ticker="BARC.L", quantity=10.0,
                       price_native=5.0, currency="GBP", purchase_fx=9.99)
    out, unresolved = rebase_ledger_fx([tx, user], "EUR", fx)
    # Native economics untouched; the missing leg resolves via date-matched
    # FX (1.25/1.10 on 2026-08-03); the user-recorded leg is preserved
    # (only_missing default).
    assert out[0].quantity == 100.0 and out[0].price_native == 5.0
    assert out[0].purchase_fx == pytest.approx(1.25 / 1.10)
    assert out[1].purchase_fx == pytest.approx(9.99)
    assert unresolved == []
    # Unknown FX resolves to None (never invented).
    out2, unresolved2 = rebase_ledger_fx([tx], "EUR",
                                         pd.DataFrame({"GBP": [1.25]},
                                                      index=pd.bdate_range(
                                                          "2020-01-01",
                                                          periods=1)))
    assert out2[0].purchase_fx is None
    assert unresolved2 == ["r1"]

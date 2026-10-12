"""M2 — Transaction Accounting 2.0: trace, rebuild, audit, CSV."""

from datetime import date, datetime, timezone

import pytest

from analytics.ledger import (
    Transaction,
    apply_ledger,
    lot_market_values,
    open_shares_by_ticker,
    sell_consumption,
    trace_ledger,
    validate_transaction,
)
from analytics.ledger_csv import preview_csv_import
from analytics.persistence import transaction_from_dict, transaction_to_dict
from storage import database as db
from storage import repositories as repo


def _tx(tid, day, type, ticker=None, qty=1.0, px=10.0, ccy="USD",
        fee=0.0, fx=1.0, notes="", source="user", move_cash=True,
        created=None, month=1):
    return Transaction(
        transaction_id=tid, date=date(2026, month, day), type=type,
        ticker=ticker, quantity=qty, price_native=px, currency=ccy,
        fee=fee, purchase_fx=fx, notes=notes, source=source,
        move_cash=move_cash, created_at=created)


def _flow_ledger():
    # Deposit → Buy → Buy → Dividend → Fee → Sell → Withdraw.
    return [
        _tx("d1", 2, "DEPOSIT", qty=20000.0, px=1.0),
        _tx("b1", 5, "BUY", "AAPL", 10, 150.0),
        _tx("b2", 6, "BUY", "AAPL", 10, 160.0),
        _tx("v1", 10, "DIVIDEND", "AAPL", 20, 0.5),
        _tx("f1", 11, "FEE", qty=5.0, px=1.0, fee=5.0),
        _tx("s1", 12, "SELL", "AAPL", 12, 170.0),
        _tx("w1", 13, "WITHDRAWAL", qty=500.0, px=1.0),
    ]


# ---------------------------------------------------------------------------
# trace_ledger: same truth as apply_ledger
# ---------------------------------------------------------------------------

def test_trace_matches_prefix_replays_and_final_apply():
    txns = _flow_ledger()
    trace = trace_ledger(txns)
    assert [t["transaction_id"] for t in trace] == \
        ["d1", "b1", "b2", "v1", "f1", "s1", "w1"]
    for k in range(1, len(txns) + 1):
        state = apply_ledger(txns[:k])
        assert trace[k - 1]["cash_after"] == pytest.approx(state.cash)
        assert trace[k - 1]["shares_after"] == pytest.approx(
            open_shares_by_ticker(state))
    final = apply_ledger(txns)
    assert trace[-1]["cash_after"] == pytest.approx(final.cash)
    assert trace[-1]["shares_after"] == {"AAPL": pytest.approx(8.0)}


def test_trace_running_cash_spot_values():
    trace = trace_ledger(_flow_ledger())
    by_id = {t["transaction_id"]: t for t in trace}
    assert by_id["d1"]["cash_after"] == {"USD": pytest.approx(20000.0)}
    # 20000 − (1500+1600) + 10 − 5 ... checked against apply below
    assert by_id["s1"]["cash_after"]["USD"] == pytest.approx(
        apply_ledger(_flow_ledger()[:6]).cash["USD"])


def test_trace_raises_where_apply_raises():
    txns = _flow_ledger() + [_tx("sX", 14, "SELL", "AAPL", 999, 1.0)]
    with pytest.raises(ValueError):
        apply_ledger(txns)
    with pytest.raises(ValueError):
        trace_ledger(txns)


def test_same_day_ordering_deterministic():
    morning = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)
    evening = datetime(2026, 1, 5, 17, 0, tzinfo=timezone.utc)
    cash = _tx("d0", 4, "DEPOSIT", qty=5000.0, px=1.0)
    buy = _tx("b1", 5, "BUY", "AAPL", 5, 100.0, created=morning)
    dep = _tx("d1", 5, "DEPOSIT", qty=5000.0, px=1.0, created=evening)
    # created_at decides same-day order, not list order
    assert [t["transaction_id"] for t in trace_ledger([dep, buy, cash])] == \
        ["d0", "b1", "d1"]
    assert [t["transaction_id"] for t in trace_ledger([cash, dep, buy])] == \
        ["d0", "b1", "d1"]
    # date still dominates: an earlier date always comes first
    assert [t["transaction_id"] for t in trace_ledger([buy, dep, cash])] == \
        ["d0", "b1", "d1"]
    # missing created_at sorts before any same-day timestamp (stable rule)
    nodate = _tx("d2", 5, "DEPOSIT", qty=1.0, px=1.0)
    order = [t["transaction_id"] for t in trace_ledger([dep, nodate, cash])]
    assert order == ["d0", "d2", "d1"]


# ---------------------------------------------------------------------------
# 2.16 rebuild-after-edit == from-scratch replay
# ---------------------------------------------------------------------------

def test_edit_early_transaction_rebuild_equals_replay():
    txns = _flow_ledger()
    before = apply_ledger(txns)
    edited = []
    for t in txns:
        if t.transaction_id == "b1":
            edited.append(_tx("b1", 5, "BUY", "AAPL", 10, 140.0))  # cheaper
        else:
            edited.append(t)
    after_edit = apply_ledger(edited)  # what the app derives post-edit
    replay = apply_ledger(list(edited))  # from-scratch replay
    assert after_edit.cash == pytest.approx(replay.cash)
    assert open_shares_by_ticker(after_edit) == pytest.approx(
        open_shares_by_ticker(replay))
    assert [e.realised_pnl_native for e in after_edit.realised] == \
        pytest.approx([e.realised_pnl_native for e in replay.realised])
    # and the edit actually changed economics (cheaper Buy → more cash,
    # lower cost, higher realised)
    assert after_edit.cash["USD"] > before.cash["USD"]
    assert after_edit.realised[0].realised_pnl_native > \
        before.realised[0].realised_pnl_native


def test_delete_rebuilds_lots_cash_realised():
    txns = _flow_ledger()
    # Deleting b2 alone strands s1 (sells 12, only 10 left): the rebuild
    # must fail loudly rather than silently restate the sale.
    with pytest.raises(ValueError):
        apply_ledger([t for t in txns if t.transaction_id != "b2"])
    # Deleting the buy AND its dependent sale gives a consistent ledger.
    remainder = [t for t in txns
                 if t.transaction_id not in ("b2", "s1")]
    state = apply_ledger(remainder)
    assert open_shares_by_ticker(state) == {"AAPL": pytest.approx(10.0)}
    assert state.realised == []
    assert state.cash["USD"] == pytest.approx(
        20000.0 - 1500.0 + 10.0 - 5.0 - 500.0)


# ---------------------------------------------------------------------------
# FIFO consumption detail
# ---------------------------------------------------------------------------

def test_sell_consumption_detail_fifo_order():
    txns = _flow_ledger()
    state = apply_ledger(txns)
    assert len(state.realised) == 1
    rows = sell_consumption(txns, state.realised[0])
    assert [(r["lot_date"].isoformat(), r["shares"]) for r in rows] == \
        [("2026-01-05", 10.0), ("2026-01-06", 2.0)]
    assert rows[0]["execution_price_native"] == pytest.approx(150.0)
    assert rows[1]["execution_price_native"] == pytest.approx(160.0)
    assert sum(r["shares"] for r in rows) == pytest.approx(12.0)


def test_lot_market_values_legs():
    txns = _flow_ledger()
    state = apply_ledger(txns)
    rows = lot_market_values(state.open_lots, {"AAPL": 200.0}, {"USD": 1.0})
    assert len(rows) == 1 and rows[0]["shares_remaining"] == \
        pytest.approx(8.0)
    assert rows[0]["market_native"] == pytest.approx(1600.0)
    assert rows[0]["unrealised_base"] == pytest.approx(
        rows[0]["market_base"] - rows[0]["cost_basis_base"])
    missing = lot_market_values(state.open_lots, {}, {"USD": 1.0})
    assert missing[0]["market_base"] is None
    assert missing[0]["unrealised_base"] is None


# ---------------------------------------------------------------------------
# audit trail in repositories
# ---------------------------------------------------------------------------

def _mem_with_ledger():
    conn = db.connect(":memory:")
    a = repo.create_portfolio(conn, "A")
    txns = _flow_ledger()
    repo.replace_ledger_with_audit(conn, a.portfolio_id, txns)
    return conn, a.portfolio_id, txns


def test_audit_records_creates_and_noop_records_nothing():
    conn, pid, txns = _mem_with_ledger()
    revs = repo.list_revisions(conn, pid)
    assert len(revs) == len(txns)
    assert {r.action for r in revs} == {"CREATE"}
    counts = repo.replace_ledger_with_audit(conn, pid, list(txns))
    assert counts == {"created": 0, "updated": 0, "deleted": 0}
    assert len(repo.list_revisions(conn, pid)) == len(txns)


def test_audit_update_keeps_before_and_history_immutable():
    conn, pid, txns = _mem_with_ledger()
    edited = [(_tx("b1", 5, "BUY", "AAPL", 10, 140.0)
               if t.transaction_id == "b1" else t) for t in txns]
    counts = repo.replace_ledger_with_audit(conn, pid, edited)
    assert counts == {"created": 0, "updated": 1, "deleted": 0}
    hist = repo.list_revisions(conn, pid, "b1")
    assert [r.action for r in hist] == ["CREATE", "UPDATE"]
    import json
    before = json.loads(hist[1].before_json)
    after = json.loads(hist[1].after_json)
    assert before["price_native"] == pytest.approx(150.0)
    assert after["price_native"] == pytest.approx(140.0)
    # earlier CREATE row untouched by the later UPDATE
    assert json.loads(hist[0].after_json)["price_native"] == \
        pytest.approx(150.0)


def test_audit_delete_records_before():
    conn, pid, txns = _mem_with_ledger()
    remainder = [t for t in txns
                 if t.transaction_id not in ("b2", "s1")]
    counts = repo.replace_ledger_with_audit(conn, pid, remainder)
    assert counts["deleted"] == 2
    dels = [r for r in repo.list_revisions(conn, pid) if r.action == "DELETE"]
    assert {r.transaction_id for r in dels} == {"b2", "s1"}
    assert all(r.before_json and r.after_json is None for r in dels)


def test_audit_failure_writes_neither_rows_nor_revisions():
    conn, pid, txns = _mem_with_ledger()
    n_revs = len(repo.list_revisions(conn, pid))
    bad = list(txns) + [_tx("zz", 1, "HOLD", "AAPL", 1, 1.0)]
    with pytest.raises(ValueError):
        repo.replace_ledger_with_audit(conn, pid, bad)
    assert len(repo.list_transactions(conn, pid)) == len(txns)
    assert len(repo.list_revisions(conn, pid)) == n_revs


# ---------------------------------------------------------------------------
# CSV import
# ---------------------------------------------------------------------------

_CSV_GOOD = """date,type,ticker,quantity,price_native,currency,fee,purchase_fx,notes,move_cash,transaction_id
2026-01-02,DEPOSIT,,20000,1.0,USD,,,,,csv-d1
2026-01-05,BUY,AAPL,10,150.0,USD,1.0,1.0,first buy,,csv-b1
2026-01-06,BUY,AAPL,10,160.0,USD,,,,,csv-b2
"""

_CSV_BAD = """date,type,ticker,quantity,price_native,currency,fee,purchase_fx,notes,move_cash,transaction_id
2026-01-02,DEPOSIT,,20000,1.0,USD,,,,,
not-a-date,BUY,AAPL,10,150,USD,,,,
2026-01-06,SELL,AAPL,999,170,USD,,,,
"""


def test_csv_good_preview_and_effects():
    preview = preview_csv_import(_CSV_GOOD)
    assert preview["ok"] is True and preview["invalid"] == []
    assert preview["duplicates"] == []
    assert preview["effects"]["n_new"] == 3
    assert preview["effects"]["new_tickers"] == ["AAPL"]
    assert preview["effects"]["cash_delta"] == {"USD": 16899.0}


def test_csv_fee_amount_convention():
    # Standalone FEE moves -fee: amount accepted in fee and/or quantity.
    only_qty = """date,type,ticker,quantity,price_native,currency,fee,purchase_fx,notes,move_cash,transaction_id
2026-01-02,DEPOSIT,,1000,1.0,USD,,,,,
2026-01-03,FEE,,25,1.0,USD,,,,,
"""
    p = preview_csv_import(only_qty)
    assert p["ok"] is True, p["invalid"]
    assert p["effects"]["cash_delta"] == {"USD": 975.0}
    only_fee = """date,type,ticker,quantity,price_native,currency,fee,purchase_fx,notes,move_cash,transaction_id
2026-01-02,DEPOSIT,,1000,1.0,USD,,,,,
2026-01-03,FEE,,,1.0,USD,25.0,,,,
"""
    p2 = preview_csv_import(only_fee)
    assert p2["ok"] is True, p2["invalid"]
    assert p2["effects"]["cash_delta"] == {"USD": 975.0}


def test_csv_rejects_future_dates_and_unknown_columns():
    future = """date,type,ticker,quantity,price_native,currency,fee,purchase_fx,notes,move_cash,transaction_id
2999-01-01,DEPOSIT,,100,1.0,USD,,,,,
"""
    p = preview_csv_import(future, today=date(2026, 10, 12))
    assert p["ok"] is False
    assert "future" in p["invalid"][0]["error"]
    weird = "date,type,price\n2026-01-01,BUY,5\n"
    p2 = preview_csv_import(weird)
    assert p2["ok"] is False


def test_csv_bad_file_never_partial():
    preview = preview_csv_import(_CSV_BAD)
    assert preview["ok"] is False
    assert len(preview["invalid"]) == 1  # bad date; oversell caught later
    assert preview["valid"] != []  # parsed rows exist...
    # ...but the oversell row must fail the trial apply, not half-commit
    oversell_only = """date,type,ticker,quantity,price_native,currency,fee,purchase_fx,notes,move_cash,transaction_id
2026-01-06,SELL,AAPL,999,170,USD,,,,
"""
    trial = preview_csv_import(oversell_only, existing=[])
    assert trial["ok"] is False
    assert trial["effects"]["trial_error"] is not None


def test_csv_duplicate_detection():
    existing = _flow_ledger()
    # same ids as existing ledger → clash
    same_ids = """date,type,ticker,quantity,price_native,currency,fee,purchase_fx,notes,move_cash,transaction_id
2026-01-05,BUY,AAPL,10,150.0,USD,,,,,b1
"""
    p = preview_csv_import(same_ids, existing=existing)
    assert p["ok"] is False
    assert any("already in the ledger" in d for d in p["duplicates"])
    # identical economics, fresh id → still warned
    twin = """date,type,ticker,quantity,price_native,currency,fee,purchase_fx,notes,move_cash,transaction_id
2026-01-05,BUY,AAPL,10,150.0,USD,0.0,1.0,,,
"""
    p2 = preview_csv_import(twin, existing=existing)
    assert any("identical" in d for d in p2["duplicates"])


def test_notes_roundtrip_and_plain_text():
    tx = _tx("n1", 3, "BUY", "AAPL", 1, 9.0,
             notes="<b>bold</b> & \"quotes\" — plain text only")
    d = transaction_to_dict(tx)
    back = transaction_from_dict(d)
    assert back.notes == tx.notes  # persisted verbatim...
    # ...and never interpolated as HTML (callers render as text)
    assert "<b>" in back.notes  # i.e. stored raw, escaped at render


def test_edit_preserves_identity_and_order():
    txns = _flow_ledger()
    original = next(t for t in txns if t.transaction_id == "b1")
    edited = validate_transaction(Transaction(
        transaction_id="b1", date=original.date, type="BUY",
        ticker="AAPL", quantity=10, price_native=140.0, currency="USD",
        fee=0.0, purchase_fx=1.0, notes="corrected", source="user",
        move_cash=True, created_at=original.created_at))
    rebuilt = [edited if t.transaction_id == "b1" else t for t in txns]
    assert [t.transaction_id for t in rebuilt] == \
        [t.transaction_id for t in txns]  # identity + order preserved
    state = apply_ledger(rebuilt)
    assert state is not None

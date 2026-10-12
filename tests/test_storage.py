"""Tests for storage/ (database, migrations, repositories)."""

from datetime import date

import pytest

from analytics.ledger import Transaction
from storage import database as db
from storage import migrations as mig
from storage import repositories as repo


def _mem():
    conn = db.connect(":memory:")
    return conn


def _tx(tid, day, type, ticker=None, qty=1.0, px=1.0, ccy="USD"):
    return Transaction(transaction_id=tid, date=date(2026, 1, day),
                       type=type, ticker=ticker, quantity=qty,
                       price_native=px, currency=ccy, purchase_fx=1.0)


def test_fresh_db_and_idempotent_migrations(tmp_path):
    path = str(tmp_path / "t.db")
    c1 = db.connect(path)
    assert mig.schema_version(c1) == 2
    c1.close()
    c2 = db.connect(path)  # re-open migrates idempotently
    assert mig.schema_version(c2) == 2
    c2.close()


def test_create_rename_validation():
    conn = _mem()
    with pytest.raises(ValueError):
        repo.create_portfolio(conn, "   ")
    with pytest.raises(ValueError):
        repo.create_portfolio(conn, "X", base_currency="XX")
    p = repo.create_portfolio(conn, "Main", "USD", "SPY", 0.05)
    assert p.name == "Main" and p.archived is False
    assert repo.rename_portfolio(conn, p.portfolio_id, "Core").name == "Core"
    with pytest.raises(ValueError):
        repo.rename_portfolio(conn, p.portfolio_id, "  ")
    with pytest.raises(ValueError):
        repo.get_portfolio(conn, "missing")


def test_clone_archive_delete_lifecycle():
    conn = _mem()
    a = repo.create_portfolio(conn, "A")
    repo.set_meta(conn, "x", "1")
    b = repo.duplicate_portfolio(conn, a.portfolio_id, "B")
    assert b.name == "B" and b.base_currency == a.base_currency
    assert b.portfolio_id != a.portfolio_id
    assert [p.name for p in repo.list_portfolios(conn)] == ["A", "B"]
    repo.set_active_portfolio_id(conn, a.portfolio_id)
    with pytest.raises(ValueError):
        repo.set_archived(conn, a.portfolio_id, True)  # active first
    repo.set_active_portfolio_id(conn, b.portfolio_id)
    repo.set_archived(conn, a.portfolio_id, True)
    assert [p.name for p in repo.list_portfolios(conn)] == ["B"]
    assert [p.name for p in repo.list_portfolios(conn, True)] == ["A", "B"]
    with pytest.raises(ValueError):
        repo.delete_portfolio(conn, b.portfolio_id)  # still active
    repo.set_active_portfolio_id(conn, a.portfolio_id) if False else None
    # unarchive A to switch back, then delete B is still blocked (active)
    repo.set_archived(conn, a.portfolio_id, False)
    repo.set_active_portfolio_id(conn, a.portfolio_id)
    repo.delete_portfolio(conn, b.portfolio_id)
    assert [p.name for p in repo.list_portfolios(conn, True)] == ["A"]
    with pytest.raises(ValueError):
        repo.delete_portfolio(conn, "default")


def test_portfolio_isolation_and_transactions():
    conn = _mem()
    a = repo.create_portfolio(conn, "A")
    b = repo.create_portfolio(conn, "B")
    assert repo.replace_transactions(
        conn, a.portfolio_id,
        [_tx("d", 1, "DEPOSIT", qty=100.0),
         _tx("b", 2, "BUY", "AAPL", 1, 10.0)]) == 2
    assert [t.transaction_id for t in repo.list_transactions(
        conn, a.portfolio_id)] == ["d", "b"]
    assert repo.list_transactions(conn, b.portfolio_id) == []
    # invalid batch writes nothing (validated first)
    with pytest.raises(ValueError):
        repo.replace_transactions(
            conn, a.portfolio_id, [_tx("ok", 1, "DEPOSIT", qty=1.0),
                                   _tx("bad", 1, "HOLD", "AAPL", 1, 1.0)])
    assert len(repo.list_transactions(conn, a.portfolio_id)) == 2
    assert repo.get_watchlist(conn, a.portfolio_id) == []
    assert repo.set_watchlist(conn, a.portfolio_id, ["msft", "AAPL", "msft"]) \
        == ["AAPL", "MSFT"]
    with pytest.raises(ValueError):
        repo.set_watchlist(conn, a.portfolio_id, ["!!!"])


def test_json_import_replace_or_create_new(tmp_path):
    from analytics.persistence import (
        export_portfolio_v2, validate_portfolio_import)
    conn = _mem()
    a = repo.create_portfolio(conn, "A")
    repo.replace_transactions(
        conn, a.portfolio_id, [_tx("d", 1, "DEPOSIT", qty=50.0)])
    doc = export_portfolio_v2({}, "USD", repo.list_transactions(
        conn, a.portfolio_id), ["TSLA"], {})
    back = validate_portfolio_import(doc)
    # replace-current path
    repo.replace_transactions(conn, a.portfolio_id, back.transactions)
    repo.set_watchlist(conn, a.portfolio_id, back.watchlist)
    assert len(repo.list_transactions(conn, a.portfolio_id)) == 1
    # create-new path keeps the source untouched
    c = repo.create_portfolio(conn, "Imported")
    repo.replace_transactions(conn, c.portfolio_id, back.transactions)
    assert len(repo.list_transactions(conn, c.portfolio_id)) == 1
    assert len(repo.list_transactions(conn, a.portfolio_id)) == 1


def test_saved_analyses_crud_and_isolation():
    conn = _mem()
    a = repo.create_portfolio(conn, "A")
    b = repo.create_portfolio(conn, "B")
    with pytest.raises(ValueError):
        repo.save_analysis(conn, a.portfolio_id, "  ", "scenario", {}, {})
    with pytest.raises(ValueError):
        repo.save_analysis(conn, a.portfolio_id, "X", "nope", {}, {})
    r = repo.save_analysis(conn, a.portfolio_id, "Bear", "scenario",
                           {"h": 5}, {"ev": 1.1}, tags="Risk",
                           notes="n1")
    assert r.name == "Bear"
    assert repo.list_analyses(conn, b.portfolio_id) == []
    assert repo.rename_analysis(conn, r.analysis_id, "Bear v2").name == "Bear v2"
    assert repo.set_analysis_notes(
        conn, r.analysis_id, "n2").notes == "n2"
    repo.delete_analysis(conn, r.analysis_id)
    assert repo.list_analyses(conn, a.portfolio_id) == []
    with pytest.raises(ValueError):
        repo.get_analysis(conn, r.analysis_id)


def test_revisions_append_only():
    conn = _mem()
    a = repo.create_portfolio(conn, "A")
    r1 = repo.record_revision(conn, a.portfolio_id, "t1", "CREATE",
                              None, {"a": 1})
    r2 = repo.record_revision(conn, a.portfolio_id, "t1", "UPDATE",
                              {"a": 1}, {"a": 2})
    assert [r.action for r in repo.list_revisions(
        conn, a.portfolio_id, "t1")] == ["CREATE", "UPDATE"]
    assert len(repo.list_revisions(conn, a.portfolio_id)) == 2
    with pytest.raises(ValueError):
        repo.record_revision(conn, a.portfolio_id, "t1", "MERGE")
    with pytest.raises(ValueError):
        repo.record_revision(conn, a.portfolio_id, "t1", "CREATE",
                             None, None, revision_id=r1.revision_id)


def test_atomic_rollback_on_failed_import():
    import sqlite3
    conn = _mem()
    a = repo.create_portfolio(conn, "A")
    repo.replace_transactions(conn, a.portfolio_id,
                              [_tx("d", 1, "DEPOSIT", qty=5.0)])
    class Boom(Exception):
        pass
    try:
        with db.atomic(conn=conn):
            conn.execute("DELETE FROM transactions WHERE portfolio_id = ?",
                         (a.portfolio_id,))
            raise Boom()
    except Boom:
        pass
    # rolled back: the original row survives
    assert len(repo.list_transactions(conn, a.portfolio_id)) == 1
    with pytest.raises(sqlite3.Error):
        with db.atomic(conn=conn):
            conn.execute("INSERT INTO portfolios (portfolio_id) VALUES (NULL)")


def test_no_secrets_tables_or_columns():
    conn = _mem()
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "secrets" not in tables and "credentials" not in tables
    cols = set()
    for t in tables:
        cols.update(r[1] for r in conn.execute(
            f"PRAGMA table_info({t})").fetchall())
    lowered = " ".join(cols).lower()
    assert "secret" not in lowered and "api_key" not in lowered \
        and "password" not in lowered


def _pos(shares=10.0, **kw):
    d = {"shares": shares, "purchase_date": "2026-01-05",
         "purchase_price_native": 100.0, "purchase_price_source": "estimate",
         "native_currency": "USD", "quote_scale": 1.0}
    d.update(kw)
    return d


def test_positions_roundtrip_and_validation():
    conn = _mem()
    a = repo.create_portfolio(conn, "A")
    assert repo.get_positions(conn, a.portfolio_id) == {}
    assert repo.replace_positions(conn, a.portfolio_id, {
        "AAPL": _pos(), "BARC.L": _pos(5, purchase_price_native=512.0,
                                       native_currency="GBX",
                                       quote_unit="GBX", quote_scale=0.01),
    }) == 2
    back = repo.get_positions(conn, a.portfolio_id)
    assert set(back) == {"AAPL", "BARC.L"}
    assert back["BARC.L"]["quote_scale"] == pytest.approx(0.01)
    with pytest.raises(ValueError):
        repo.replace_positions(conn, a.portfolio_id,
                               {"AAPL": _pos(), "BAD!": _pos()})
    # failed replace keeps the old rows
    assert set(repo.get_positions(conn, a.portfolio_id)) == {"AAPL", "BARC.L"}
    with pytest.raises(ValueError):
        repo.replace_positions(conn, a.portfolio_id, {"AAPL": {"shares": 0}})


def test_duplicate_copies_positions_and_delete_clears_them():
    conn = _mem()
    a = repo.create_portfolio(conn, "A")
    repo.replace_positions(conn, a.portfolio_id, {"AAPL": _pos()})
    b = repo.duplicate_portfolio(conn, a.portfolio_id, "B")
    assert set(repo.get_positions(conn, b.portfolio_id)) == {"AAPL"}
    repo.set_active_portfolio_id(conn, a.portfolio_id)
    repo.delete_portfolio(conn, b.portfolio_id)
    assert set(repo.get_positions(conn, a.portfolio_id)) == {"AAPL"}


def test_save_and_load_workspace_roundtrip():
    conn = _mem()
    a = repo.create_portfolio(conn, "A")
    txns = [_tx("d", 1, "DEPOSIT", qty=100.0),
            _tx("b", 2, "BUY", "AAPL", 1, 10.0)]
    repo.save_workspace(conn, a.portfolio_id,
                        positions={"AAPL": _pos(1)}, transactions=txns,
                        watchlist=["msft"], base_currency="EUR",
                        benchmark="QQQ", risk_free=0.03)
    ws = repo.load_workspace(conn, a.portfolio_id)
    assert set(ws["positions"]) == {"AAPL"}
    assert [t.transaction_id for t in ws["transactions"]] == ["d", "b"]
    assert ws["watchlist"] == ["MSFT"]
    assert ws["portfolio"].base_currency == "EUR"
    assert ws["portfolio"].benchmark == "QQQ"
    assert ws["portfolio"].risk_free == pytest.approx(0.03)
    # invalid workspace (bad ledger) writes nothing
    with pytest.raises(ValueError):
        repo.save_workspace(
            conn, a.portfolio_id, positions={"AAPL": _pos(1)},
            transactions=[_tx("x", 1, "HOLD", "AAPL", 1, 1.0)],
            watchlist=[])
    assert len(repo.load_workspace(
        conn, a.portfolio_id)["transactions"]) == 2

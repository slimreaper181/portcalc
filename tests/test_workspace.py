"""Tests for ui.workspace (DB/session translation, no Streamlit)."""

from datetime import date

from analytics.ledger import Transaction
from storage import database as db
from storage import repositories as repo
from ui.workspace import (
    legacy_seed_state,
    session_is_empty,
    workspace_signature,
    workspace_to_session,
)


def _tx(tid, day, type, ticker=None, qty=1.0):
    return Transaction(transaction_id=tid, date=date(2026, 1, day),
                       type=type, ticker=ticker, quantity=qty,
                       price_native=10.0, currency="USD", purchase_fx=1.0)


def test_signature_stable_and_change_sensitive():
    pos = {"AAPL": {"shares": 1.0}}
    base = workspace_signature(pos, [], ["MSFT"], "USD", "SPY")
    assert workspace_signature(pos, [], ["MSFT"], "USD", "SPY") == base
    # watchlist order-insensitive...
    assert workspace_signature(pos, [], ["MSFT", "TSLA"], "USD", "SPY") == \
        workspace_signature(pos, [], ["TSLA", "MSFT"], "USD", "SPY")
    # ...but real changes flip it
    assert workspace_signature(pos, [], ["MSFT"], "EUR", "SPY") != base
    assert workspace_signature({"AAPL": {"shares": 2.0}}, [], ["MSFT"],
                               "USD", "SPY") != base


def test_workspace_to_session_shapes():
    conn = db.connect(":memory:")
    a = repo.create_portfolio(conn, "A", "EUR", "QQQ", 0.03)
    repo.replace_positions(conn, a.portfolio_id, {
        "AAPL": {"shares": 1, "purchase_date": "2026-01-05",
                 "purchase_price_native": 100.0,
                 "purchase_price_source": "estimate",
                 "native_currency": "USD"}})
    repo.replace_transactions(conn, a.portfolio_id,
                              [_tx("d", 1, "DEPOSIT", qty=5.0)])
    repo.set_watchlist(conn, a.portfolio_id, ["tsla"])
    state = workspace_to_session(repo.load_workspace(conn, a.portfolio_id))
    assert set(state["portfolio"]) == {"AAPL"}
    assert isinstance(state["ledger_txns"][0], dict)
    assert state["ledger_txns"][0]["transaction_id"] == "d"  # JSON-safe dicts
    assert state["watchlist"] == ["TSLA"]
    assert state["base_currency"] == "EUR"
    assert state["bt_benchmark"] == "QQQ"
    assert not session_is_empty(state["portfolio"], state["ledger_txns"])
    assert session_is_empty({}, [])


def test_legacy_seed_state_migrates_lots():
    seed = legacy_seed_state(
        {"AAPL": {"shares": 2, "purchase_date": "2026-01-05",
                  "purchase_price_native": 100.0,
                  "purchase_price_source": "estimate",
                  "native_currency": "USD"},
         "NODATE": {"shares": 1, "avg_cost": 5.0}}, "GBP")
    assert set(seed["portfolio"]) == {"AAPL", "NODATE"}  # nothing dropped
    assert seed["base_currency"] == "GBP"
    assert seed["bt_benchmark"] == "SPY"
    # dated aggregate became a cash-exempt opening lot (dict form)
    lots = [d for d in seed["ledger_txns"] if d.get("ticker") == "AAPL"]
    assert len(lots) == 1 and lots[0]["move_cash"] is False


def test_seed_roundtrips_through_save_workspace():
    conn = db.connect(":memory:")
    a = repo.create_portfolio(conn, "A")
    seed = legacy_seed_state(
        {"AAPL": {"shares": 2, "purchase_date": "2026-01-05",
                  "purchase_price_native": 100.0,
                  "purchase_price_source": "estimate",
                  "native_currency": "USD"}}, "USD")
    from analytics.persistence import transaction_from_dict
    repo.save_workspace(
        conn, a.portfolio_id, positions=seed["portfolio"],
        transactions=[transaction_from_dict(d)
                      for d in seed["ledger_txns"]],
        watchlist=seed["watchlist"], base_currency=seed["base_currency"],
        benchmark=seed["bt_benchmark"])
    back = workspace_to_session(repo.load_workspace(conn, a.portfolio_id))

    def _sig(state):
        return workspace_signature(
            state["portfolio"], state["ledger_txns"], state["watchlist"],
            state["base_currency"], state["bt_benchmark"])

    # Save → load → save → load is idempotent: reloading a persisted
    # workspace never shows phantom modifications (the property the
    # shell's hash-guarded sync relies on).
    from analytics.persistence import transaction_from_dict as _from_dict
    repo.save_workspace(
        conn, a.portfolio_id, positions=back["portfolio"],
        transactions=[_from_dict(d) for d in back["ledger_txns"]],
        watchlist=back["watchlist"], base_currency=back["base_currency"],
        benchmark=back["bt_benchmark"])
    back2 = workspace_to_session(repo.load_workspace(conn, a.portfolio_id))
    assert _sig(back2) == _sig(back)
    assert set(back2["portfolio"]) == {"AAPL"}
    assert len(back2["ledger_txns"]) == 1

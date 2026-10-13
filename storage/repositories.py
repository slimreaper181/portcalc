"""storage.repositories — typed data access (parameterised SQL only).

Every public function takes an explicit ``sqlite3.Connection`` (tests pass
tmp files; the app passes its singleton). Multi-row mutations that must
not half-apply take ``atomic_conn`` semantics via :func:`db_transaction`
helpers below — callers wrap with ``storage.database.atomic``.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from storage.models import Portfolio, Revision, SavedAnalysis

DEFAULT_PORTFOLIO_ID = "default"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# meta (active portfolio pointer)
# ---------------------------------------------------------------------------

def get_meta(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    """Read a meta value (never raises on missing keys)."""
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = ?",
                           (key,)).fetchone()
    except sqlite3.Error:
        return default
    return row["value"] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Upsert a meta value."""
    conn.execute("INSERT INTO meta (key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                 (str(key), str(value)))


def get_active_portfolio_id(conn: sqlite3.Connection) -> str | None:
    """Active portfolio id, or None when never chosen."""
    value = get_meta(conn, "active_portfolio_id", "")
    return value or None


def set_active_portfolio_id(conn: sqlite3.Connection, portfolio_id: str) -> None:
    """Point the workspace at a portfolio (must exist, not archived)."""
    row = conn.execute("SELECT archived FROM portfolios WHERE portfolio_id = ?",
                       (portfolio_id,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown portfolio {portfolio_id!r}.")
    if int(row["archived"]):
        raise ValueError("Cannot activate an archived portfolio (unarchive first).")
    set_meta(conn, "active_portfolio_id", portfolio_id)


# ---------------------------------------------------------------------------
# portfolios
# ---------------------------------------------------------------------------

def _row_to_portfolio(row: sqlite3.Row) -> Portfolio:
    return Portfolio(
        portfolio_id=row["portfolio_id"], name=row["name"],
        base_currency=row["base_currency"], benchmark=row["benchmark"],
        risk_free=float(row["risk_free"]), created_at=row["created_at"],
        updated_at=row["updated_at"], archived=bool(row["archived"]))


def create_portfolio(conn: sqlite3.Connection, name: str,
                     base_currency: str = "USD", benchmark: str = "SPY",
                     risk_free: float = 0.05,
                     portfolio_id: str | None = None) -> Portfolio:
    """Create a portfolio (name required, non-blank)."""
    from analytics.currency import validate_base_currency
    label = str(name or "").strip()
    if not label:
        raise ValueError("Portfolio name must be non-blank.")
    base = validate_base_currency(base_currency)
    bench = str(benchmark or "SPY").strip().upper() or "SPY"
    try:
        rf = float(risk_free)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid risk-free rate: {risk_free!r}.") from e
    if not (0.0 <= rf <= 1.0):
        raise ValueError("Risk-free rate must be between 0 and 1.")
    pid = portfolio_id or _new_id()
    now = _now()
    try:
        conn.execute(
            "INSERT INTO portfolios (portfolio_id, name, base_currency, "
            "benchmark, risk_free, created_at, updated_at, archived) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            (pid, label, base, bench, rf, now, now))
    except sqlite3.IntegrityError as e:
        raise ValueError(f"Portfolio id already exists: {pid}.") from e
    return get_portfolio(conn, pid)


def get_portfolio(conn: sqlite3.Connection, portfolio_id: str) -> Portfolio:
    """Fetch one portfolio by id (raises when missing)."""
    row = conn.execute("SELECT * FROM portfolios WHERE portfolio_id = ?",
                       (portfolio_id,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown portfolio {portfolio_id!r}.")
    return _row_to_portfolio(row)


def list_portfolios(conn: sqlite3.Connection,
                    include_archived: bool = False) -> list[Portfolio]:
    """All portfolios (active first by creation order)."""
    query = "SELECT * FROM portfolios"
    if not include_archived:
        query += " WHERE archived = 0"
    query += " ORDER BY rowid"
    return [_row_to_portfolio(r) for r in conn.execute(query).fetchall()]


def rename_portfolio(conn: sqlite3.Connection, portfolio_id: str,
                     name: str) -> Portfolio:
    """Rename (blank names rejected)."""
    label = str(name or "").strip()
    if not label:
        raise ValueError("Portfolio name must be non-blank.")
    get_portfolio(conn, portfolio_id)
    conn.execute("UPDATE portfolios SET name = ?, updated_at = ? "
                 "WHERE portfolio_id = ?", (label, _now(), portfolio_id))
    return get_portfolio(conn, portfolio_id)


def set_archived(conn: sqlite3.Connection, portfolio_id: str,
                 archived: bool) -> Portfolio:
    """Archive/unarchive (archived portfolios leave the switcher)."""
    get_portfolio(conn, portfolio_id)
    if archived and get_active_portfolio_id(conn) == portfolio_id:
        raise ValueError("Archive another portfolio first: this one is active.")
    conn.execute("UPDATE portfolios SET archived = ?, updated_at = ? "
                 "WHERE portfolio_id = ?", (1 if archived else 0, _now(),
                                            portfolio_id))
    return get_portfolio(conn, portfolio_id)


def delete_portfolio(conn: sqlite3.Connection, portfolio_id: str) -> None:
    """Delete a portfolio and ALL its rows (positions, transactions,
    watchlist, research, revisions). Refuses the active or default
    portfolio without an explicit replacement — callers confirm in UI."""
    get_portfolio(conn, portfolio_id)
    if get_active_portfolio_id(conn) == portfolio_id:
        raise ValueError("Switch active portfolio before deleting this one.")
    if portfolio_id == DEFAULT_PORTFOLIO_ID:
        raise ValueError("The default portfolio cannot be deleted "
                         "(archive it instead).")
    conn.execute("DELETE FROM revisions WHERE portfolio_id = ?",
                 (portfolio_id,))
    conn.execute("DELETE FROM saved_analyses WHERE portfolio_id = ?",
                 (portfolio_id,))
    conn.execute("DELETE FROM watchlist WHERE portfolio_id = ?",
                 (portfolio_id,))
    conn.execute("DELETE FROM transactions WHERE portfolio_id = ?",
                 (portfolio_id,))
    conn.execute("DELETE FROM positions WHERE portfolio_id = ?",
                 (portfolio_id,))
    conn.execute("DELETE FROM portfolios WHERE portfolio_id = ?",
                 (portfolio_id,))


def duplicate_portfolio(conn: sqlite3.Connection, portfolio_id: str,
                        name: str,
                        include_research: bool = False) -> Portfolio:
    """Clone settings + positions + transactions + watchlist.

    Saved research is NOT copied unless ``include_research`` is set (the
    user opts in explicitly); copies get fresh ids under the clone.
    """
    src = get_portfolio(conn, portfolio_id)
    label = str(name or "").strip()
    if not label:
        raise ValueError("Clone needs a non-blank name.")
    now = _now()
    clone = create_portfolio(conn, label, src.base_currency, src.benchmark,
                             src.risk_free)
    conn.execute(
        "INSERT INTO positions (portfolio_id, ticker, shares, purchase_date,"
        " purchase_price_native, purchase_price_source, manual_purchase_price,"
        " native_currency, quote_unit, quote_scale, legacy_avg_cost) "
        "SELECT ?, ticker, shares, purchase_date, purchase_price_native, "
        "purchase_price_source, manual_purchase_price, native_currency, "
        "quote_unit, quote_scale, legacy_avg_cost FROM positions "
        "WHERE portfolio_id = ?", (clone.portfolio_id, portfolio_id))
    conn.execute(
        "INSERT INTO transactions (portfolio_id, transaction_id, date, type, "
        "ticker, quantity, price_native, currency, fee, purchase_fx, notes, "
        "source, move_cash, created_at) "
        "SELECT ?, transaction_id, date, type, ticker, quantity, "
        "price_native, currency, fee, purchase_fx, notes, source, move_cash, "
        "created_at FROM transactions WHERE portfolio_id = ?",
        (clone.portfolio_id, portfolio_id))
    conn.execute(
        "INSERT INTO watchlist (portfolio_id, ticker, added_at) "
        "SELECT ?, ticker, added_at FROM watchlist WHERE portfolio_id = ?",
        (clone.portfolio_id, portfolio_id))
    if include_research:
        for row in conn.execute(
                "SELECT name, type, config_json, result_json, tags, notes "
                "FROM saved_analyses WHERE portfolio_id = ?",
                (portfolio_id,)).fetchall():
            conn.execute(
                "INSERT INTO saved_analyses (analysis_id, portfolio_id, "
                "name, type, config_json, result_json, tags, notes, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?)",
                (_new_id(), clone.portfolio_id, row["name"], row["type"],
                 row["config_json"], row["result_json"], row["tags"],
                 row["notes"], now, now))
    conn.execute("UPDATE portfolios SET updated_at = ? WHERE portfolio_id = ?",
                 (now, clone.portfolio_id))
    return get_portfolio(conn, clone.portfolio_id)


# ---------------------------------------------------------------------------
# transactions (ledger persistence)
# ---------------------------------------------------------------------------

_TX_COLS = ("transaction_id", "date", "type", "ticker", "quantity",
            "price_native", "currency", "fee", "purchase_fx", "notes",
            "source", "move_cash", "created_at")


def _tx_to_row(tx) -> tuple:
    from analytics.persistence import transaction_to_dict
    d = transaction_to_dict(tx)
    return (d["transaction_id"], d["date"], d["type"], d["ticker"],
            d["quantity"], d["price_native"], d["currency"], d["fee"],
            d["purchase_fx"], d["notes"], d["source"],
            1 if d.get("move_cash", True) else 0, d.get("created_at"))


def _row_to_tx(row: sqlite3.Row):
    from analytics.persistence import transaction_from_dict
    return transaction_from_dict({
        "transaction_id": row["transaction_id"], "date": row["date"],
        "type": row["type"], "ticker": row["ticker"],
        "quantity": row["quantity"], "price_native": row["price_native"],
        "currency": row["currency"], "fee": row["fee"],
        "purchase_fx": row["purchase_fx"], "notes": row["notes"] or "",
        "source": row["source"] or "user",
        "move_cash": bool(row["move_cash"]),
        "created_at": row["created_at"]})


def replace_ledger_with_audit(conn: sqlite3.Connection,
                                portfolio_id: str,
                                transactions) -> dict:
    """Replace a ledger AND record CREATE/UPDATE/DELETE revisions.

    Diffs the new ledger against stored rows by stable transaction id
    (payloads compared as JSON-safe dicts): identical ids with changed
    payloads → UPDATE (before+after); new ids → CREATE; missing ids →
    DELETE. A byte-identical replace records nothing. Validation runs
    first — failures write neither rows nor revisions (atomic).
    """
    from analytics.ledger import Transaction, validate_transaction
    from analytics.persistence import transaction_to_dict
    from storage import database as _db
    get_portfolio(conn, portfolio_id)
    if not isinstance(transactions, list) or any(
            not isinstance(t, Transaction) for t in transactions):
        raise ValueError("Transactions must be a list of Transaction.")
    fresh = list(transactions)
    for tx in fresh:
        validate_transaction(tx)  # raises before anything is touched
    old_rows = {t.transaction_id: transaction_to_dict(t)
                for t in list_transactions(conn, portfolio_id)}
    new_rows = {t.transaction_id: transaction_to_dict(t) for t in fresh}
    counts = {"created": 0, "updated": 0, "deleted": 0}
    with _db.atomic(conn=conn):
        for tid, payload in new_rows.items():
            if tid not in old_rows:
                record_revision(conn, portfolio_id, tid, "CREATE",
                                None, payload)
                counts["created"] += 1
            elif payload != old_rows[tid]:
                record_revision(conn, portfolio_id, tid, "UPDATE",
                                old_rows[tid], payload)
                counts["updated"] += 1
        for tid, payload in old_rows.items():
            if tid not in new_rows:
                record_revision(conn, portfolio_id, tid, "DELETE",
                                payload, None)
                counts["deleted"] += 1
        replace_transactions(conn, portfolio_id, fresh)
    return counts


def replace_transactions(conn: sqlite3.Connection, portfolio_id: str,
                         transactions) -> int:
    """Replace a portfolio's whole ledger atomically (validated first).

    Returns the stored count. Raises on the first invalid transaction with
    NOTHING written (callers: validate-then-commit call this inside
    ``storage.database.atomic``).
    """
    get_portfolio(conn, portfolio_id)
    rows = []
    for tx in transactions or []:
        from analytics.ledger import validate_transaction
        if hasattr(tx, "transaction_id"):
            validate_transaction(tx)
        rows.append(_tx_to_row(tx))
    conn.execute("DELETE FROM transactions WHERE portfolio_id = ?",
                 (portfolio_id,))
    conn.executemany(
        "INSERT INTO transactions (portfolio_id, transaction_id, date, type, "
        "ticker, quantity, price_native, currency, fee, purchase_fx, notes, "
        "source, move_cash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?)",
        [(portfolio_id,) + r for r in rows])
    conn.execute("UPDATE portfolios SET updated_at = ? WHERE portfolio_id = ?",
                 (_now(), portfolio_id))
    return len(rows)


def list_transactions(conn: sqlite3.Connection,
                      portfolio_id: str) -> list:
    """Ledger transactions in deterministic (date, created, id) order."""
    get_portfolio(conn, portfolio_id)
    rows = conn.execute(
        "SELECT * FROM transactions WHERE portfolio_id = ? "
        "ORDER BY date, COALESCE(created_at, ''), transaction_id",
        (portfolio_id,)).fetchall()
    return [_row_to_tx(r) for r in rows]


# ---------------------------------------------------------------------------
# watchlist
# ---------------------------------------------------------------------------

def get_watchlist(conn: sqlite3.Connection, portfolio_id: str) -> list[str]:
    """Watchlist tickers (sorted)."""
    get_portfolio(conn, portfolio_id)
    rows = conn.execute("SELECT ticker FROM watchlist WHERE portfolio_id = ? "
                        "ORDER BY ticker", (portfolio_id,)).fetchall()
    return [r["ticker"] for r in rows]


def set_watchlist(conn: sqlite3.Connection, portfolio_id: str,
                  tickers: list[str]) -> list[str]:
    """Replace the watchlist (validated tickers, sorted, deduped)."""
    from analytics.validation import validate_ticker_symbol
    get_portfolio(conn, portfolio_id)
    clean = []
    for sym in tickers or []:
        w = validate_ticker_symbol(sym)
        if w not in clean:
            clean.append(w)
    conn.execute("DELETE FROM watchlist WHERE portfolio_id = ?",
                 (portfolio_id,))
    now = _now()
    conn.executemany(
        "INSERT INTO watchlist (portfolio_id, ticker, added_at) "
        "VALUES (?, ?, ?)",
        [(portfolio_id, t, now) for t in sorted(clean)])
    conn.execute("UPDATE portfolios SET updated_at = ? WHERE portfolio_id = ?",
                 (_now(), portfolio_id))
    return sorted(clean)


# ---------------------------------------------------------------------------
# research workspace (saved analyses + notes)
# ---------------------------------------------------------------------------

_ANALYSIS_TYPES = ("scenario", "optimisation", "backtest", "oos",
                   "walkforward", "robustness", "stress")


def save_analysis(conn: sqlite3.Connection, portfolio_id: str, name: str,
                  type: str, config: dict, result: dict,
                  tags: str = "", notes: str = "",
                  analysis_id: str | None = None) -> SavedAnalysis:
    """Create or replace a saved analysis (config must be JSON-safe dict)."""
    get_portfolio(conn, portfolio_id)
    label = str(name or "").strip()
    if not label:
        raise ValueError("Analysis name must be non-blank.")
    atype = str(type or "").strip().lower()
    if atype not in _ANALYSIS_TYPES:
        raise ValueError(f"Unknown analysis type {type!r}.")
    try:
        config_json = json.dumps(dict(config or {}), sort_keys=True)
        result_json = json.dumps(dict(result or {}), sort_keys=True)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Analysis payloads must be JSON-safe: {e}")
    aid = analysis_id or _new_id()
    now = _now()
    conn.execute(
        "INSERT INTO saved_analyses (analysis_id, portfolio_id, name, type, "
        "config_json, result_json, tags, notes, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(analysis_id) DO UPDATE SET name = excluded.name, "
        "type = excluded.type, config_json = excluded.config_json, "
        "result_json = excluded.result_json, tags = excluded.tags, "
        "notes = excluded.notes, updated_at = excluded.updated_at",
        (aid, portfolio_id, label, atype, config_json, result_json,
         str(tags or ""), str(notes or ""), now, now))
    return get_analysis(conn, aid)


def _row_to_analysis(row: sqlite3.Row) -> SavedAnalysis:
    return SavedAnalysis(
        analysis_id=row["analysis_id"], portfolio_id=row["portfolio_id"],
        name=row["name"], type=row["type"], config_json=row["config_json"],
        result_json=row["result_json"], tags=row["tags"], notes=row["notes"],
        created_at=row["created_at"], updated_at=row["updated_at"])


def get_analysis(conn: sqlite3.Connection, analysis_id: str) -> SavedAnalysis:
    """Fetch one saved analysis (raises when missing)."""
    row = conn.execute("SELECT * FROM saved_analyses WHERE analysis_id = ?",
                       (analysis_id,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown analysis {analysis_id!r}.")
    return _row_to_analysis(row)


def list_analyses(conn: sqlite3.Connection, portfolio_id: str,
                  type: str | None = None) -> list[SavedAnalysis]:
    """Saved analyses for a portfolio (newest first, optional type filter)."""
    get_portfolio(conn, portfolio_id)
    if type is None:
        rows = conn.execute(
            "SELECT * FROM saved_analyses WHERE portfolio_id = ? "
            "ORDER BY updated_at DESC, analysis_id", (portfolio_id,)).fetchall()
    else:
        atype = str(type).strip().lower()
        if atype not in _ANALYSIS_TYPES:
            raise ValueError(f"Unknown analysis type {type!r}.")
        rows = conn.execute(
            "SELECT * FROM saved_analyses WHERE portfolio_id = ? AND type = ? "
            "ORDER BY updated_at DESC, analysis_id",
            (portfolio_id, atype)).fetchall()
    return [_row_to_analysis(r) for r in rows]


def rename_analysis(conn: sqlite3.Connection, analysis_id: str,
                    name: str) -> SavedAnalysis:
    """Rename a saved analysis (blank rejected)."""
    label = str(name or "").strip()
    if not label:
        raise ValueError("Analysis name must be non-blank.")
    get_analysis(conn, analysis_id)
    conn.execute("UPDATE saved_analyses SET name = ?, updated_at = ? "
                 "WHERE analysis_id = ?", (label, _now(), analysis_id))
    return get_analysis(conn, analysis_id)


def set_analysis_notes(conn: sqlite3.Connection, analysis_id: str,
                       notes: str) -> SavedAnalysis:
    """Replace notes text (plain text; UI escapes on render)."""
    get_analysis(conn, analysis_id)
    conn.execute("UPDATE saved_analyses SET notes = ?, updated_at = ? "
                 "WHERE analysis_id = ?", (str(notes or ""), _now(),
                                           analysis_id))
    return get_analysis(conn, analysis_id)


def delete_analysis(conn: sqlite3.Connection, analysis_id: str) -> None:
    """Delete one saved analysis (missing ids raise)."""
    get_analysis(conn, analysis_id)
    conn.execute("DELETE FROM saved_analyses WHERE analysis_id = ?",
                 (analysis_id,))


# ---------------------------------------------------------------------------
# revisions (audit trail; Milestone 2 writes, schema ready now)
# ---------------------------------------------------------------------------

def record_revision(conn: sqlite3.Connection, portfolio_id: str,
                    transaction_id: str, action: str,
                    before: dict | None = None, after: dict | None = None,
                    revision_id: str | None = None) -> Revision:
    """Append an immutable revision entry (CREATE/UPDATE/DELETE)."""
    get_portfolio(conn, portfolio_id)
    act = str(action or "").strip().upper()
    if act not in ("CREATE", "UPDATE", "DELETE"):
        raise ValueError(f"Unknown revision action {action!r}.")
    if not transaction_id or not str(transaction_id).strip():
        raise ValueError("Revisions need a transaction id.")
    try:
        before_json = json.dumps(before, sort_keys=True) \
            if before is not None else None
        after_json = json.dumps(after, sort_keys=True) \
            if after is not None else None
    except (TypeError, ValueError) as e:
        raise ValueError(f"Revision payloads must be JSON-safe: {e}")
    rid = revision_id or _new_id()
    now = _now()
    try:
        conn.execute(
            "INSERT INTO revisions (revision_id, transaction_id, "
            "portfolio_id, action, before_json, after_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rid, str(transaction_id), portfolio_id, act, before_json,
             after_json, now))
    except sqlite3.IntegrityError as e:
        raise ValueError(f"Revision id already recorded: {rid}.") from e
    row = conn.execute("SELECT * FROM revisions WHERE revision_id = ?",
                       (rid,)).fetchone()
    return Revision(revision_id=row["revision_id"],
                    transaction_id=row["transaction_id"],
                    portfolio_id=row["portfolio_id"], action=row["action"],
                    before_json=row["before_json"],
                    after_json=row["after_json"],
                    created_at=row["created_at"])


def list_revisions(conn: sqlite3.Connection, portfolio_id: str,
                   transaction_id: str | None = None) -> list[Revision]:
    """Revision history (insertion order; optional per-transaction filter)."""
    get_portfolio(conn, portfolio_id)
    if transaction_id is None:
        rows = conn.execute(
            "SELECT * FROM revisions WHERE portfolio_id = ? "
            "ORDER BY rowid", (portfolio_id,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM revisions WHERE portfolio_id = ? AND "
            "transaction_id = ? ORDER BY rowid",
            (portfolio_id, transaction_id)).fetchall()
    return [Revision(revision_id=r["revision_id"],
                     transaction_id=r["transaction_id"],
                     portfolio_id=r["portfolio_id"], action=r["action"],
                     before_json=r["before_json"],
                     after_json=r["after_json"],
                     created_at=r["created_at"]) for r in rows]


# ---------------------------------------------------------------------------
# positions (aggregate holdings per portfolio; app-dict shape)
# ---------------------------------------------------------------------------

def _position_row_to_dict(row: sqlite3.Row) -> dict:
    pos: dict = {"shares": float(row["shares"])}
    if row["purchase_date"]:
        pos["purchase_date"] = row["purchase_date"]
    if row["purchase_price_native"] is not None:
        pos["purchase_price_native"] = float(row["purchase_price_native"])
    if row["purchase_price_source"]:
        pos["purchase_price_source"] = row["purchase_price_source"]
    pos["manual_purchase_price"] = bool(row["manual_purchase_price"])
    if row["native_currency"]:
        pos["native_currency"] = row["native_currency"]
    if row["quote_unit"]:
        pos["quote_unit"] = row["quote_unit"]
    pos["quote_scale"] = float(row["quote_scale"] or 1.0)
    if row["legacy_avg_cost"] is not None:
        pos["avg_cost"] = float(row["legacy_avg_cost"])
    return pos


def get_positions(conn: sqlite3.Connection,
                  portfolio_id: str) -> dict[str, dict]:
    """Aggregate holdings for a portfolio (app ``ticker -> dict`` shape)."""
    get_portfolio(conn, portfolio_id)
    rows = conn.execute("SELECT * FROM positions WHERE portfolio_id = ? "
                        "ORDER BY ticker", (portfolio_id,)).fetchall()
    return {r["ticker"]: _position_row_to_dict(r) for r in rows}


def replace_positions(conn: sqlite3.Connection, portfolio_id: str,
                      positions: dict) -> int:
    """Replace all aggregate holdings (validated first — nothing on error)."""
    from analytics.positions import position_from_dict, position_to_dict
    get_portfolio(conn, portfolio_id)
    if not isinstance(positions, dict):
        raise ValueError("Positions must be a mapping.")
    parsed = []
    for ticker, data in positions.items():
        if not isinstance(data, dict):
            raise ValueError(f"Position {ticker!r} must be a mapping.")
        pos = position_from_dict(ticker, data)  # raises on any bad row
        parsed.append((pos.ticker, position_to_dict(pos)))
    conn.execute("DELETE FROM positions WHERE portfolio_id = ?",
                 (portfolio_id,))
    for ticker, clean in parsed:
        conn.execute(
            "INSERT INTO positions (portfolio_id, ticker, shares, "
            "purchase_date, purchase_price_native, purchase_price_source, "
            "manual_purchase_price, native_currency, quote_unit, "
            "quote_scale, legacy_avg_cost) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (portfolio_id, ticker,
             clean["shares"], clean["purchase_date"],
             clean["purchase_price_native"],
             clean["purchase_price_source"],
             1 if clean["manual_purchase_price"] else 0,
             clean["native_currency"], clean["quote_unit"],
             clean.get("quote_scale") or 1.0, clean.get("avg_cost")))
    return len(parsed)


# ---------------------------------------------------------------------------
# workspace load/save (single atomic snapshot per portfolio)
# ---------------------------------------------------------------------------

def load_workspace(conn: sqlite3.Connection,
                   portfolio_id: str) -> dict:
    """Full editable state for a portfolio (positions + ledger + prefs)."""
    record = get_portfolio(conn, portfolio_id)
    return {
        "portfolio": record,
        "positions": get_positions(conn, portfolio_id),
        "transactions": list_transactions(conn, portfolio_id),
        "watchlist": get_watchlist(conn, portfolio_id),
    }


def save_workspace(conn: sqlite3.Connection, portfolio_id: str, *,
                   positions: dict, transactions: list,
                   watchlist: list, base_currency: str | None = None,
                   benchmark: str | None = None,
                   risk_free: float | None = None) -> None:
    """Persist the whole editable state atomically (validate-then-write)."""
    from analytics.ledger import Transaction, validate_transaction
    from storage import database as _db
    record = get_portfolio(conn, portfolio_id)
    if not isinstance(transactions, list) or any(
            not isinstance(t, Transaction) for t in transactions):
        raise ValueError("Transactions must be a list of Transaction.")
    for tx in transactions:
        validate_transaction(tx)  # raises on bad ledger; nothing written yet
    # Validate positions/watchlist BEFORE touching any table.
    from analytics.positions import position_from_dict
    from analytics.validation import validate_ticker_symbol
    for ticker, data in (positions or {}).items():
        position_from_dict(ticker, data)
    clean_watch: list[str] = []
    for sym in watchlist or []:
        w = validate_ticker_symbol(sym)
        if w not in clean_watch:
            clean_watch.append(w)
    if base_currency is not None:
        from analytics.currency import validate_base_currency
        validate_base_currency(base_currency)
    if risk_free is not None and not (0.0 <= float(risk_free) <= 1.0):
        raise ValueError("Risk-free rate must be between 0 and 1.")
    with _db.atomic(conn=conn):
        replace_positions(conn, portfolio_id, positions or {})
        replace_ledger_with_audit(conn, portfolio_id, list(transactions))
        set_watchlist(conn, portfolio_id, clean_watch)
        updates: dict = {}
        if base_currency is not None:
            from analytics.currency import validate_base_currency
            updates["base_currency"] = validate_base_currency(base_currency)
        if benchmark is not None:
            updates["benchmark"] = str(benchmark or "SPY").strip().upper()
        if risk_free is not None:
            updates["risk_free"] = float(risk_free)
        if updates:
            sets = ", ".join(f"{k} = ?" for k in updates)
            conn.execute(f"UPDATE portfolios SET {sets}, updated_at = ? "
                         "WHERE portfolio_id = ?",
                         (*updates.values(), _now(), portfolio_id))
    void = record  # keep linters quiet about the pre-check read
    assert void is not None

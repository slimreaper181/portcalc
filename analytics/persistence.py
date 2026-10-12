"""
persistence.py
--------------
Versioned portfolio documents (schema v2) + legacy v1 compatibility.

* v1 (existing): ``{schema_version: 1, base_currency, positions: [...]}`` —
  aggregate positions only. Read support is permanent; the v1 reader in
  :mod:`analytics.positions` is untouched.
* v2 (new): adds ``transactions`` (ledger), ``cash`` note, ``watchlist``
  and ``settings``. Full float precision + ISO dates so exports reproduce
  state exactly (Programme 174).

Policies (documented, tested):
* Import REPLACES portfolio state (never merges) — importing the same file
  twice cannot duplicate ledger entries (Programme 176).
* Malformed documents are rejected with the offending field named; unknown
  future top-level keys are ignored (forward compatibility), unknown
  schema versions rejected.
* Migration v1 → ledger is deterministic: aggregate position →
  synthetic opening BUY (``source="migrate"``, id ``mig-v1-{TICKER}-{k}``)
  using existing shares/date/price/currency. Positions lacking a dated
  price (e.g. bare legacy averages) are SKIPPED and reported, never
  invented. After migration the ledger is canonical for accounting; the
  aggregate positions remain the universe definition for historical
  analytics (Programme 168).
* No credentials ever serialised (validated by test).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from analytics.ledger import Transaction, new_transaction_id, validate_transaction
from analytics.positions import position_from_dict
from analytics.validation import validate_ticker_symbol
from analytics.currency import validate_base_currency

PORTFOLIO_SCHEMA_VERSION = 2


@dataclass
class PortfolioDocument:
    """Validated v1/v2 import result (app-native shapes)."""

    schema_version: int
    base_currency: str
    positions: dict = field(default_factory=dict)
    transactions: list = field(default_factory=list)  # Transaction
    watchlist: list = field(default_factory=list)     # tickers
    settings: dict = field(default_factory=dict)
    skipped_migration: list = field(default_factory=list)


def _iso(d) -> str | None:
    if d is None:
        return None
    if isinstance(d, date) and not isinstance(d, datetime):
        return d.isoformat()
    if isinstance(d, datetime):
        return d.date().isoformat()
    return str(d)


def _parse_date(value, field_name: str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        parts = str(value).strip().split("-")
        if len(parts) != 3:
            raise ValueError
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, TypeError, AttributeError) as e:
        raise ValueError(f"Bad date in {field_name}: {value!r}.") from e


def transaction_to_dict(tx: Transaction) -> dict:
    """Serialise one transaction (full precision, ISO date)."""
    return {
        "transaction_id": str(tx.transaction_id),
        "date": _iso(tx.date),
        "type": tx.type,
        "ticker": tx.ticker,
        "quantity": float(tx.quantity),
        "price_native": float(tx.price_native),
        "currency": tx.currency,
        "fee": float(tx.fee),
        "purchase_fx": (float(tx.purchase_fx)
                        if tx.purchase_fx is not None else None),
        "notes": str(tx.notes or ""),
        "source": str(tx.source or "user"),
    }


def transaction_from_dict(data: dict) -> Transaction:
    """Parse + validate one transaction dict (untrusted input)."""
    if not isinstance(data, dict):
        raise ValueError("Transaction must be a mapping.")
    try:
        tid = str(data.get("transaction_id") or "").strip()
        if not tid:
            raise ValueError("missing transaction_id")
        tx = Transaction(
            transaction_id=tid,
            date=_parse_date(data.get("date"), "transaction.date"),
            type=data.get("type"),
            ticker=data.get("ticker"),
            quantity=data.get("quantity", 0.0),
            price_native=data.get("price_native", 0.0),
            currency=data.get("currency", "USD"),
            fee=data.get("fee", 0.0),
            purchase_fx=data.get("purchase_fx"),
            notes=data.get("notes", ""),
            source=data.get("source", "user") or "user")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Malformed transaction: {e}")
    return validate_transaction(tx)


def export_portfolio_v2(positions: dict, base_currency: str,
                        transactions: list | None = None,
                        watchlist: list | None = None,
                        settings: dict | None = None) -> dict:
    """Build a schema-v2 export document (never includes secrets)."""
    from analytics.positions import export_portfolio as _export_v1
    base = validate_base_currency(base_currency)
    v1 = _export_v1(positions or {}, base)
    txns = []
    for tx in transactions or []:
        if not isinstance(tx, Transaction):
            raise ValueError("Transactions must be Transaction objects.")
        txns.append(transaction_to_dict(validate_transaction(tx)))
    watch = []
    for sym in watchlist or []:
        w = validate_ticker_symbol(sym)
        if w not in watch:
            watch.append(w)
    doc = {"schema_version": PORTFOLIO_SCHEMA_VERSION,
           "base_currency": base,
           "positions": v1["positions"],
           "transactions": txns,
           "watchlist": watch,
           "settings": dict(settings or {}),
           "exported_at": datetime.now(timezone.utc).isoformat()}
    return doc


def validate_portfolio_import(data: dict) -> PortfolioDocument:
    """Validate a v1 or v2 portfolio document (untrusted input)."""
    if not isinstance(data, dict):
        raise ValueError("Imported portfolio must be a JSON object.")
    version = data.get("schema_version")
    if version not in (1, PORTFOLIO_SCHEMA_VERSION):
        raise ValueError(
            f"Unsupported portfolio schema version: {version!r} "
            f"(expected 1 or {PORTFOLIO_SCHEMA_VERSION}).")
    base = validate_base_currency(data.get("base_currency", ""))
    raw_positions = data.get("positions")
    if not isinstance(raw_positions, list):
        raise ValueError("Imported portfolio has no positions list.")
    positions: dict[str, dict] = {}
    for i, entry in enumerate(raw_positions):
        if not isinstance(entry, dict):
            raise ValueError(f"Position #{i} must be a mapping.")
        sym = validate_ticker_symbol(entry.get("ticker", ""))
        if sym in positions:
            raise ValueError(f"Duplicate ticker in import: {sym}.")
        pos = position_from_dict(sym, entry)
        from analytics.positions import position_to_dict
        stored = position_to_dict(pos)
        if pos.native_currency is not None and not (
                len(pos.native_currency) == 3
                and pos.native_currency.isalpha()):
            raise ValueError(f"Unknown currency for {sym}.")
        positions[sym] = stored
    transactions: list = []
    if version == PORTFOLIO_SCHEMA_VERSION:
        raw_txns = data.get("transactions", [])
        if not isinstance(raw_txns, list):
            raise ValueError("v2 transactions must be a list.")
        ids = set()
        for i, entry in enumerate(raw_txns):
            try:
                tx = transaction_from_dict(entry)
            except ValueError as e:
                raise ValueError(f"Transaction #{i}: {e}")
            if tx.transaction_id in ids:
                raise ValueError(
                    f"Duplicate transaction id: {tx.transaction_id}.")
            ids.add(tx.transaction_id)
            transactions.append(tx)
    watchlist: list = []
    for sym in data.get("watchlist", []) or []:
        w = validate_ticker_symbol(sym)
        if w not in watchlist:
            watchlist.append(w)
    settings = data.get("settings", {}) or {}
    if not isinstance(settings, dict):
        raise ValueError("Settings must be a mapping.")
    return PortfolioDocument(schema_version=int(version),
                             base_currency=base, positions=positions,
                             transactions=transactions, watchlist=watchlist,
                             settings={str(k): v for k, v in settings.items()})


def rebase_ledger_fx(transactions: list, new_base: str,
                   fx_history, limit: int = 30,
                   only_missing: bool = True) -> tuple[list, list]:
    """Recompute stored native→base FX for a base-currency switch.

    Native economics (quantity/price/currency) are never mutated — only the
    presentation ``purchase_fx`` legs are recomputed via date-matched
    historical FX (``rate_on_date``). With ``only_missing`` (default),
    user-recorded legs are preserved and only unknown legs resolve; pass
    False to recompute every non-base leg. Unresolvable legs become None
    (base unknown, disclosed downstream — never invented).
    Returns ``(adjusted_transactions, unresolved_ids)``.
    """
    from analytics.currency import rate_on_date
    new_base = validate_base_currency(new_base)
    out, unresolved = [], []
    for tx in transactions or []:
        if not isinstance(tx, Transaction):
            continue
        if tx.currency == new_base:
            tx.purchase_fx = 1.0
            out.append(tx)
            continue
        if only_missing and tx.purchase_fx is not None:
            out.append(tx)
            continue
        rate = None
        try:
            if fx_history is not None:
                rate = rate_on_date(fx_history, tx.currency, new_base,
                                    tx.date, limit=limit)
        except ValueError:
            rate = None
        tx.purchase_fx = rate
        if rate is None:
            unresolved.append(tx.transaction_id)
        out.append(tx)
    return out, unresolved


def migrate_v1_to_ledger(positions: dict) -> tuple[list, list]:
    """Aggregate v1 positions → synthetic opening BUY transactions.

    Returns ``(transactions, skipped)`` where skipped lists tickers that
    lack a dated execution price (never invented). Deterministic ids
    (``mig-v1-{TICKER}-{k}``); purchase FX stays None (resolved later via
    purchase-date FX with an estimated label, or cost stays unknown).
    """
    txns, skipped = [], []
    for k, (ticker, data) in enumerate(sorted((positions or {}).items())):
        try:
            sym = validate_ticker_symbol(ticker)
            pos = position_from_dict(sym, data if isinstance(data, dict)
                                     else {})
        except ValueError:
            skipped.append(str(ticker))
            continue
        if pos.purchase_date is None or pos.purchase_price_native is None \
                or not pos.purchase_price_native > 0 \
                or pos.native_currency is None or pos.shares is None \
                or not pos.shares > 0:
            skipped.append(sym)
            continue
        txns.append(validate_transaction(Transaction(
            transaction_id=f"mig-v1-{sym}-{k}",
            date=pos.purchase_date, type="BUY", ticker=sym,
            quantity=float(pos.shares),
            price_native=float(pos.purchase_price_native),
            currency=pos.native_currency, fee=0.0, purchase_fx=None,
            notes="Migrated aggregate position (estimated cost basis).",
            source="migrate")))
    return txns, skipped

"""analytics.ledger_csv — atomic bulk transaction CSV import (pure).

CSV SCHEMA (header row required, exact names, case-sensitive):

    date,type,ticker,quantity,price_native,currency,fee,purchase_fx,
    notes,move_cash,transaction_id

* ``date`` — ISO ``YYYY-MM-DD``; future dates are rejected (policy: the
  ledger never books tomorrow's trades).
* ``type`` — one of BUY SELL DIVIDEND FEE DEPOSIT WITHDRAWAL INTEREST.
* ``ticker`` — required for BUY/SELL/DIVIDEND/INTEREST, blank otherwise.
* ``quantity`` — positive number (shares, or cash amount for
  DEPOSIT/WITHDRAWAL/FEE).
* ``price_native`` — positive execution price for BUY/SELL (per-share
  amount ≥ 0 for DIVIDEND/INTEREST, 1.0 convention for cash flows).
* ``currency`` — transaction/cash currency (e.g. USD, GBP, EUR).
* ``fee`` — ≥ 0, blank means 0 (for a standalone FEE the amount may
  be given here or in ``quantity`` when only one is filled).
* ``purchase_fx`` — native→base rate at transaction date; blank means
  unknown (base legs stay unknown, never invented).
* ``notes`` — optional plain text (rendered as text, never raw HTML).
* ``move_cash`` — optional true/false (default true); false is only
  valid for BUY/SELL (cash-exempt lots, like classic aggregate adds).
* ``transaction_id`` — optional stable id; minted when blank. Clashes
  with the existing ledger or within the file are reported, never
  silently double-imported.

Import is all-or-nothing: :func:`preview_csv_import` validates every
row AND trial-applies the combined ledger (oversells, negative cash
and duplicate ids fail the whole file). Nothing is written until the
caller commits the returned transactions.
"""

from __future__ import annotations

import csv
import io
import math
import uuid
from datetime import date

from analytics.ledger import Transaction, apply_ledger, validate_transaction
from analytics.persistence import transaction_to_dict

CSV_COLUMNS = ("date", "type", "ticker", "quantity", "price_native",
               "currency", "fee", "purchase_fx", "notes", "move_cash",
               "transaction_id")

_REQUIRED_COLUMNS = ("date", "type", "quantity")

_TYPE_SET = ("BUY", "SELL", "DIVIDEND", "FEE", "DEPOSIT", "WITHDRAWAL",
             "INTEREST")
_TICKER_TYPES = ("BUY", "SELL", "DIVIDEND", "INTEREST")


def _parse_bool(raw: str, line: int, field: str) -> bool:
    text = str(raw or "").strip().lower()
    if text in ("", "true", "1", "yes", "y"):
        return True
    if text in ("false", "0", "no", "n"):
        return False
    raise ValueError(f"Line {line}: {field} must be true/false.")


def _parse_float(raw: str, line: int, field: str, *,
                 allow_blank: bool = False,
                 minimum: float | None = None,
                 exclusive_minimum: bool = False) -> float | None:
    text = str(raw or "").strip()
    if not text:
        if allow_blank:
            return None
        raise ValueError(f"Line {line}: {field} is required.")
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise ValueError(f"Line {line}: {field} must be a number.") from None
    if not math.isfinite(value):
        raise ValueError(f"Line {line}: {field} must be finite.")
    if minimum is not None:
        bad = value <= minimum if exclusive_minimum else value < minimum
        if bad:
            raise ValueError(f"Line {line}: {field} out of range.")
    return value


def parse_csv_rows(text: str) -> tuple[list[dict], list[dict]]:
    """Parse raw CSV text → (row dicts with _line, header errors).

    Never raises on data problems: malformed rows surface as error
    dicts ``{"line": n, "error": msg}`` so the preview can show every
    problem at once. Unknown extra columns are rejected (typos in the
    header must not silently drop a field).
    """
    if text is None or not str(text).strip():
        return [], [{"line": 0, "error": "CSV file is empty."}]
    try:
        reader = csv.DictReader(io.StringIO(str(text)))
    except csv.Error as e:
        return [], [{"line": 0, "error": f"Unreadable CSV: {e}"}]
    header = reader.fieldnames or []
    seen = [h for h in header if h is not None]
    if [h for h in seen if h not in CSV_COLUMNS]:
        unknown = sorted({h for h in seen if h not in CSV_COLUMNS})
        return [], [{"line": 1,
                     "error": f"Unknown column(s): {', '.join(unknown)}."}]
    missing = [c for c in _REQUIRED_COLUMNS if c not in seen]
    if missing:
        return [], [{"line": 1,
                     "error": f"Missing required column(s): "
                              f"{', '.join(missing)}."}]
    rows = []
    for n, raw in enumerate(reader, start=2):
        if raw is None:
            continue
        values = {k: ("" if v is None else str(v)) for k, v in raw.items()
                  if k in CSV_COLUMNS}
        if all(not v.strip() for v in values.values()):
            continue  # skip blank lines
        values["_line"] = n
        rows.append(values)
    if not rows:
        return [], [{"line": 0, "error": "CSV has a header but no rows."}]
    return rows, []


def row_to_transaction(row: dict,
                       today: date | None = None) -> Transaction:
    """Convert one parsed row → validated Transaction (raises ValueError)."""
    from analytics.persistence import _parse_date as _parse_tx_date
    line = row.get("_line", "?")
    ttype = str(row.get("type") or "").strip().upper()
    if ttype not in _TYPE_SET:
        raise ValueError(f"Line {line}: unknown type "
                         f"{row.get('type')!r}.")
    try:
        day = _parse_tx_date(row.get("date"), "date")
    except ValueError:
        raise ValueError(f"Line {line}: bad date "
                         f"{row.get('date')!r} (use YYYY-MM-DD).") from None
    ref = today or date.today()
    if day > ref:
        raise ValueError(f"Line {line}: future dates are rejected "
                         f"({day.isoformat()}).")
    ticker = str(row.get("ticker") or "").strip().upper() or None
    if ttype in _TICKER_TYPES and not ticker:
        raise ValueError(f"Line {line}: {ttype} needs a ticker.")
    if ttype not in _TICKER_TYPES and ticker:
        raise ValueError(f"Line {line}: {ttype} must not name a ticker.")
    qty = _parse_float(row.get("quantity"), line, "quantity",
                       minimum=0.0, exclusive_minimum=True,
                       allow_blank=(ttype == "FEE"))
    fee = _parse_float(row.get("fee"), line, "fee", allow_blank=True,
                       minimum=0.0)
    if ttype == "FEE":
        # Engine convention (pinned by tests): a standalone fee moves
        # -fee, with quantity mirroring the amount. Accept the amount in
        # either column when only one is given; an explicit fee wins.
        if fee is None:
            fee = float(qty) if qty is not None else None
        if qty is None:
            qty = float(fee) if fee is not None else None
        if qty is None or fee is None:
            raise ValueError(f"Line {line}: FEE needs an amount "
                             "(quantity and/or fee).")
    fee = fee or 0.0
    if ttype in ("BUY", "SELL"):
        price = _parse_float(row.get("price_native"), line,
                             "price_native", minimum=0.0,
                             exclusive_minimum=True)
    elif ttype in ("DIVIDEND", "INTEREST"):
        price = _parse_float(row.get("price_native"), line,
                             "price_native", minimum=0.0) or 0.0
    else:
        price = _parse_float(row.get("price_native") or "1.0", line,
                             "price_native", minimum=0.0,
                             exclusive_minimum=True)
    ccy = str(row.get("currency") or "USD").strip().upper() or "USD"
    fx = _parse_float(row.get("purchase_fx"), line, "purchase_fx",
                      allow_blank=True, minimum=0.0,
                      exclusive_minimum=True)
    move_cash = _parse_bool(row.get("move_cash", ""), line, "move_cash")
    tid = str(row.get("transaction_id") or "").strip() or uuid.uuid4().hex
    return validate_transaction(Transaction(
        transaction_id=tid, date=day, type=ttype, ticker=ticker,
        quantity=float(qty), price_native=float(price), currency=ccy,
        fee=float(fee), purchase_fx=fx,
        notes=str(row.get("notes") or ""), source="import",
        move_cash=move_cash))


def _economic_key(tx: Transaction) -> tuple:
    return (tx.date.isoformat(), tx.type, tx.ticker or "",
            round(float(tx.quantity), 6), round(float(tx.price_native), 6),
            tx.currency, round(float(tx.fee), 6))


def preview_csv_import(text: str,
                       existing: list[Transaction] | None = None,
                       today: date | None = None) -> dict:
    """Validate a whole file → preview dict (nothing applied).

    Returns ``{"valid": [Transaction...], "invalid": [{"line","error"}],
    "duplicates": [str...], "effects": {...}, "ok": bool}``. ``ok`` is
    true only when every row parses AND the combined ledger trial-applies
    cleanly. Effects summarise the trial (new rows, new tickers, cash
    deltas per currency, realised events added) or carry the trial error.
    """
    existing = list(existing or [])
    rows, header_errors = parse_csv_rows(text)
    invalid = list(header_errors)
    valid: list[Transaction] = []
    if not header_errors:
        for row in rows:
            try:
                valid.append(row_to_transaction(row, today=today))
            except ValueError as e:
                invalid.append({"line": row.get("_line", "?"),
                                "error": str(e)})
    duplicates: list[str] = []
    if valid:
        seen_ids: set[str] = set()
        for tx in valid:
            if tx.transaction_id in seen_ids:
                duplicates.append(
                    f"Line id {tx.transaction_id[:8]}… appears twice in "
                    "the file.")
            seen_ids.add(tx.transaction_id)
        existing_ids = {t.transaction_id for t in existing
                        if isinstance(t, Transaction)}
        for tx in valid:
            if tx.transaction_id in existing_ids:
                duplicates.append(
                    f"Id {tx.transaction_id[:8]}… already in the ledger "
                    "(re-import would double-count).")
        existing_keys = {_economic_key(t) for t in existing
                         if isinstance(t, Transaction)}
        for tx in valid:
            if _economic_key(tx) in existing_keys:
                duplicates.append(
                    f"{tx.date.isoformat()} {tx.type} {tx.ticker or ''} "
                    "looks identical to an existing entry.")
    effects: dict = {"n_new": 0, "new_tickers": [], "cash_delta": {},
                     "realised_added": 0, "trial_error": None}
    ok = not invalid and not duplicates and bool(valid)
    if ok:
        try:
            before = apply_ledger(existing)
            after = apply_ledger(existing + valid)
        except ValueError as e:
            ok = False
            effects["trial_error"] = str(e)
        else:
            effects["n_new"] = len(valid)
            old_t = {t.ticker for t in existing if t.ticker}
            effects["new_tickers"] = sorted(
                {t.ticker for t in valid if t.ticker} - old_t)
            effects["cash_delta"] = {
                c: round(after.cash.get(c, 0.0) - before.cash.get(c, 0.0), 2)
                for c in sorted(set(after.cash) | set(before.cash))}
            effects["realised_added"] = len(after.realised) - len(
                before.realised)
    return {"valid": valid, "invalid": invalid, "duplicates": duplicates,
            "effects": effects, "ok": ok}


def preview_to_dicts(preview: dict) -> list[dict]:
    """Validated preview transactions → JSON-safe session dicts."""
    return [transaction_to_dict(t) for t in preview.get("valid", [])]

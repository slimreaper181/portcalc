"""
ledger.py
---------
Transaction ledger + FIFO multi-lot accounting + cash engine (pure).

What it answers
---------------
"What do I actually own (which lots), what did I realise, what cash do I
hold — derived deterministically from the transaction record?"

Design rules (frozen):
* The ledger (transaction list) is the ONLY mutable input; holdings, lots,
  realised P&L, income and cash are DERIVED by :func:`apply_ledger` — never
  stored as competing truths, never edited by hand.
* FIFO consumption: sells take oldest lots first (per ticker, ordered by
  date then transaction id). No average-cost overwriting of realised P&L
  (weighted-average cost is a separate informational display).
* Fees: a BUY fee increases that lot's cost basis; a SELL fee reduces
  proceeds; a standalone FEE only moves cash. Documented, never silent.
* Cash is currency-keyed. BUY/SELL/DIVIDEND/FEE/DEPOSIT/WITHDRAWAL move
  the matching currency balance. Negative cash is REJECTED by default
  (no silent leverage) unless explicitly allowed.
* Native transaction economics are preserved; Base Currency conversions
  use the FX supplied per transaction (purchase-date FX for cost,
  sale-date FX for proceeds). Switching the reporting base recomputes
  presentation only — stored native amounts never change.
* Precision: full float internally; rounding is presentation-only.

Transaction model (Programme 19 fields): transaction_id (stable uuid;
migration uses deterministic ``mig-…`` ids — never row position), date,
type, ticker (None for pure cash flows), quantity, price_native, currency,
fee (in ``currency``), purchase_fx (native→base at transaction date; None
means base cost stays unknown rather than invented), notes/source.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

TX_TYPES = ("BUY", "SELL", "DIVIDEND", "FEE", "DEPOSIT", "WITHDRAWAL",
            "INTEREST")

_CASH_ONLY = ("FEE", "DEPOSIT", "WITHDRAWAL")
_SECURITY_FLOWS = ("BUY", "SELL", "DIVIDEND", "INTEREST")


@dataclass
class Transaction:
    """One ledger entry (validated by :func:`validate_transaction`)."""

    transaction_id: str
    date: date
    type: str
    ticker: str | None = None
    quantity: float = 0.0
    price_native: float = 0.0
    currency: str = "USD"
    fee: float = 0.0
    purchase_fx: float | None = None  # native→base at txn date
    notes: str = ""
    source: str = "user"  # user | aggregate | import | migrate
    move_cash: bool = True  # False: lot-only entry (classic aggregate adds
    # carry no cash movement; full-form flows always move cash)
    created_at: datetime | None = None  # entry timestamp (same-date order)


@dataclass
class Lot:
    """One open FIFO lot (derived, never hand-edited)."""

    lot_id: str
    ticker: str
    purchase_date: date
    shares_original: float
    shares_remaining: float
    execution_price_native: float
    native_currency: str
    purchase_fx: float | None
    cost_basis_native: float      # remaining native cost (incl. buy fee share)
    cost_basis_base: float | None  # remaining base cost (None if FX unknown)
    source: str
    transaction_id: str
    created_at: datetime | None = None  # entry order tiebreak (same-date lots)


@dataclass
class RealisedEvent:
    """One closed sale (or partial sale) with reconciled P&L."""

    transaction_id: str
    date: date
    ticker: str
    shares_sold: float
    proceeds_native: float
    cost_native: float
    fee_native: float
    realised_pnl_native: float
    proceeds_base: float | None
    cost_base: float | None
    fee_base: float | None
    realised_pnl_base: float | None
    fx_effect_base: float | None  # total_base − local×sale_fx (None if unknown)
    lots_consumed: list = field(default_factory=list)  # (lot_id, shares)


@dataclass
class LedgerState:
    """Fully derived ledger outcome (single source of accounting truth)."""

    open_lots: list = field(default_factory=list)          # Lot
    cash: dict = field(default_factory=dict)               # ccy → balance
    realised: list = field(default_factory=list)           # RealisedEvent
    dividend_income_native: dict = field(default_factory=dict)
    dividend_income_base: float | None = None
    interest_income_base: float | None = None
    fees_paid_base: float | None = None
    n_transactions: int = 0


def new_transaction_id() -> str:
    """Stable unique id for user-created transactions."""
    return uuid.uuid4().hex


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def validate_transaction(tx: Transaction) -> Transaction:
    """Validate one transaction in isolation (no ledger state needed)."""
    if not isinstance(tx, Transaction):
        raise ValueError("Not a Transaction.")
    if not tx.transaction_id or not isinstance(tx.transaction_id, str):
        raise ValueError("Transaction needs a non-empty string id.")
    if not isinstance(tx.date, date) or isinstance(tx.date, datetime):
        raise ValueError(f"Transaction date must be a date, got {tx.date!r}.")
    ttype = str(tx.type or "").strip().upper()
    if ttype not in TX_TYPES:
        raise ValueError(f"Unknown transaction type {tx.type!r}; "
                         f"expected one of {list(TX_TYPES)}.")
    tx.type = ttype
    if ttype in _SECURITY_FLOWS:
        if not tx.ticker or not str(tx.ticker).strip():
            raise ValueError(f"{ttype} needs a ticker.")
        tx.ticker = str(tx.ticker).strip().upper()
    else:
        if tx.ticker is not None and str(tx.ticker).strip():
            raise ValueError(f"{ttype} must not name a ticker.")
        tx.ticker = None
    if not _finite(tx.quantity) or float(tx.quantity) <= 0:
        raise ValueError(f"{ttype} quantity must be positive, "
                         f"got {tx.quantity!r}.")
    if ttype in ("BUY", "SELL"):
        if not _finite(tx.price_native) or float(tx.price_native) <= 0:
            raise ValueError(f"{ttype} price must be positive, "
                             f"got {tx.price_native!r}.")
    elif ttype in ("DIVIDEND", "INTEREST"):
        # quantity = entitled shares, price_native = cash per share.
        if not _finite(tx.price_native) or float(tx.price_native) < 0:
            raise ValueError(f"{ttype} per-share amount must be >= 0.")
    elif ttype in ("DEPOSIT", "WITHDRAWAL"):
        if not _finite(tx.price_native) or float(tx.price_native) <= 0:
            raise ValueError("Cash-flow price must be positive (1.0 for cash).")
    if not _finite(tx.fee) or float(tx.fee) < 0:
        raise ValueError(f"Fee must be >= 0, got {tx.fee!r}.")
    tx.currency = str(tx.currency or "USD").strip().upper()
    if not tx.currency:
        raise ValueError("Transaction needs a currency.")
    if tx.purchase_fx is not None:
        if not _finite(tx.purchase_fx) or float(tx.purchase_fx) <= 0:
            raise ValueError("purchase_fx must be positive when supplied.")
        tx.purchase_fx = float(tx.purchase_fx)
    tx.quantity = float(tx.quantity)
    tx.price_native = float(tx.price_native)
    tx.fee = float(tx.fee)
    if not isinstance(tx.move_cash, bool):
        raise ValueError("move_cash must be true/false.")
    if not tx.move_cash and tx.type not in ("BUY", "SELL"):
        raise ValueError("Cash-exempt entries are only valid for BUY/SELL.")
    return tx


def _order_key(tx: Transaction):
    # Deterministic: date, then entry timestamp (preserves same-day entry
    # order), then id (stable for migrated/imported batches without times).
    created = tx.created_at
    if not isinstance(created, datetime):
        created = datetime.min.replace(tzinfo=timezone.utc)
    elif created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return (tx.date, created, tx.transaction_id)


def _open_shares(open_lots: list, ticker: str) -> float:
    return sum(l.shares_remaining for l in open_lots if l.ticker == ticker)


def _prepare(transactions):
    """Validate (on copies — caller objects never mutated), dedupe, sort."""
    txns = [validate_transaction(_copy_tx(t)) for t in (transactions or [])]
    ids = [t.transaction_id for t in txns]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate transaction ids in ledger.")
    txns.sort(key=_order_key)
    return txns


def _new_state(n: int) -> LedgerState:
    state = LedgerState(n_transactions=n)
    state.dividend_income_base = 0.0
    state.interest_income_base = 0.0
    state.fees_paid_base = 0.0
    return state


def _move_cash(state: LedgerState, ccy: str, amount: float,
               tx: Transaction, allow_negative_cash: bool):
    state.cash[ccy] = state.cash.get(ccy, 0.0) + amount
    if state.cash[ccy] < -1e-9 and not allow_negative_cash:
        raise ValueError(
            f"Transaction {tx.transaction_id} ({tx.type} {tx.date}) "
            f"would leave {ccy} cash at {state.cash[ccy]:.2f}: negative "
            "cash is rejected (deposit first or allow explicitly).")


def _base_leg(amount_native: float, fx: float | None):
    # Base leg is None (never invented) when the transaction FX is
    # unknown; income/fee totals below therefore sum known legs only.
    if fx is None:
        return None
    return float(amount_native) * float(fx)


def _advance(state: LedgerState, tx: Transaction,
             allow_negative_cash: bool = False) -> None:
    """Apply ONE validated transaction to a ledger state (in place).

    This is the single accounting truth shared by :func:`apply_ledger`
    (whole-ledger derive) and :func:`trace_ledger` (running snapshots):
    both validate, order and step identically, so a trace prefix can
    never disagree with a from-scratch replay.
    """
    ccy = tx.currency
    if tx.type == "BUY":
        gross = tx.quantity * tx.price_native
        if tx.move_cash:
            _move_cash(state, ccy, -(gross + tx.fee), tx,
                       allow_negative_cash)
        _fee_base(state, tx.fee, tx.purchase_fx)
        fx = tx.purchase_fx
        cost_base = _base_leg(gross + tx.fee, fx)
        state.open_lots.append(Lot(
            lot_id=f"lot-{tx.transaction_id}",
            ticker=tx.ticker, purchase_date=tx.date,
            shares_original=tx.quantity, shares_remaining=tx.quantity,
            execution_price_native=tx.price_native,
            native_currency=ccy, purchase_fx=fx,
            cost_basis_native=gross + tx.fee,
            cost_basis_base=cost_base,
            source=tx.source, transaction_id=tx.transaction_id,
            created_at=tx.created_at))
    elif tx.type == "SELL":
        avail = _open_shares(state.open_lots, tx.ticker)
        if tx.quantity > avail + 1e-9:
            raise ValueError(
                f"Transaction {tx.transaction_id} sells {tx.quantity:g} "
                f"{tx.ticker} but only {avail:g} shares are available "
                "(shorting is not supported).")
        proceeds_native = tx.quantity * tx.price_native - tx.fee
        if tx.move_cash:
            _move_cash(state, ccy, proceeds_native, tx,
                       allow_negative_cash)
        _fee_base(state, tx.fee, tx.purchase_fx)
        fx = tx.purchase_fx
        need = tx.quantity
        cost_n, cost_b, base_known_leg, consumed = 0.0, 0.0, True, []

        def _lot_key(lot):
            created = lot.created_at
            if not isinstance(created, datetime):
                created = datetime.min.replace(tzinfo=timezone.utc)
            elif created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            return (lot.purchase_date, created, lot.lot_id)

        for lot in sorted(
                [l for l in state.open_lots if l.ticker == tx.ticker],
                key=_lot_key):
            if need <= 1e-12:
                break
            take = min(lot.shares_remaining, need)
            unit_n = (lot.cost_basis_native / lot.shares_remaining
                      if lot.shares_remaining > 0 else 0.0)
            cost_n += unit_n * take
            lot.cost_basis_native -= unit_n * take
            if lot.cost_basis_base is None:
                base_known_leg = False
            elif lot.shares_remaining > 0:
                unit_b = lot.cost_basis_base / lot.shares_remaining
                cost_b += unit_b * take
                lot.cost_basis_base -= unit_b * take
            lot.shares_remaining -= take
            need -= take
            consumed.append((lot.lot_id, take))
        state.open_lots = [l for l in state.open_lots
                           if l.shares_remaining > 1e-9]
        proceeds_base = _base_leg(proceeds_native, fx)
        fee_base = _base_leg(tx.fee, fx)
        cost_b_final = (cost_b if (base_known_leg
                                   and proceeds_base is not None)
                        else None)
        realised_n = proceeds_native - cost_n
        realised_b = (proceeds_base - cost_b_final
                      if (proceeds_base is not None
                          and cost_b_final is not None) else None)
        # FX-effect identity: realised_base − native realised at sale FX.
        # Fee treatment is symmetric (inside proceeds on both legs).
        fx_effect = (realised_b - realised_n * float(fx)
                     if (realised_b is not None and fx is not None)
                     else None)
        state.realised.append(RealisedEvent(
            transaction_id=tx.transaction_id, date=tx.date,
            ticker=tx.ticker, shares_sold=tx.quantity,
            proceeds_native=proceeds_native, cost_native=cost_n,
            fee_native=tx.fee, realised_pnl_native=realised_n,
            proceeds_base=proceeds_base, cost_base=cost_b_final,
            fee_base=fee_base, realised_pnl_base=realised_b,
            fx_effect_base=fx_effect, lots_consumed=consumed))
    elif tx.type in ("DIVIDEND", "INTEREST"):
        amount_native = tx.quantity * tx.price_native
        _move_cash(state, ccy, amount_native, tx, allow_negative_cash)
        amount_base = _base_leg(amount_native, tx.purchase_fx)
        if tx.type == "DIVIDEND":
            state.dividend_income_native[tx.ticker] = \
                state.dividend_income_native.get(tx.ticker, 0.0) \
                + amount_native
            if amount_base is not None:
                state.dividend_income_base += amount_base
        else:
            if amount_base is not None:
                state.interest_income_base += amount_base
    elif tx.type == "FEE":
        # Standalone fee: pass quantity=fee, price=1.0 (amount=fee).
        _move_cash(state, ccy, -tx.fee, tx, allow_negative_cash)
        _fee_base(state, tx.fee, tx.purchase_fx)
    elif tx.type == "DEPOSIT":
        _move_cash(state, ccy, tx.quantity * tx.price_native, tx,
                   allow_negative_cash)
    elif tx.type == "WITHDRAWAL":
        _move_cash(state, ccy, -(tx.quantity * tx.price_native), tx,
                   allow_negative_cash)


def _fee_base(state: LedgerState, amount_native: float,
              fx: float | None):
    fb = _base_leg(amount_native, fx)
    if fb is not None:
        state.fees_paid_base += fb


def apply_ledger(transactions: list[Transaction],
                 allow_negative_cash: bool = False) -> LedgerState:
    """Derive holdings, realised P&L, income and cash from transactions.

    Deterministic order: (date, transaction_id). Raises ``ValueError``
    naming the offending transaction on oversell (beyond available FIFO
    shares) or unsupported negative cash. Either the whole ledger applies
    or nothing does (validated up front, applied in order).
    """
    txns = _prepare(transactions)
    state = _new_state(len(txns))
    for tx in txns:
        _advance(state, tx, allow_negative_cash)
    return state


def trace_ledger(transactions: list[Transaction],
                 allow_negative_cash: bool = False) -> list[dict]:
    """Running snapshots after every transaction (same truth as apply).

    Returns one dict per transaction in deterministic ledger order with
    ``transaction_id``/``date``/``type``/``ticker``, ``cash_after``
    (currency → balance copy) and ``shares_after`` (ticker → open
    shares). Raises exactly where :func:`apply_ledger` would.
    """
    txns = _prepare(transactions)
    state = _new_state(len(txns))
    out = []
    for tx in txns:
        _advance(state, tx, allow_negative_cash)
        out.append({
            "transaction_id": tx.transaction_id,
            "date": tx.date, "type": tx.type, "ticker": tx.ticker,
            "cash_after": dict(state.cash),
            "shares_after": {t: s for t, s in
                             open_shares_by_ticker(state).items()},
        })
    return out


def sell_consumption(transactions: list[Transaction],
                     realised: RealisedEvent) -> list[dict]:
    """FIFO lots consumed by one sale, with origin economics.

    Joins ``realised.lots_consumed`` (lot_id, shares) back to the
    originating BUY transactions (``lot-{transaction_id}``) so the UI
    can show lot date, execution price and consumed cost — all derived
    from the same ledger, never hand-entered.
    """
    by_id = {t.transaction_id: t for t in (transactions or [])
             if isinstance(t, Transaction)}
    rows = []
    for lot_id, shares in (realised.lots_consumed or []):
        origin_id = str(lot_id)[4:] if str(lot_id).startswith("lot-") \
            else str(lot_id)
        origin = by_id.get(origin_id)
        rows.append({
            "lot_id": lot_id,
            "shares": float(shares),
            "lot_date": origin.date if origin is not None else None,
            "execution_price_native": origin.price_native
            if origin is not None else None,
            "currency": origin.currency if origin is not None else None,
        })
    return rows


def lot_market_values(open_lots: list, marks_native: dict,
                      fx_now: dict) -> list[dict]:
    """Per-lot market value / unrealised / FX effect (display legs).

    ``marks_native`` maps ticker → current native price (quote-unit
    adjusted); ``fx_now`` maps currency → native→base rate. Legs are
    None (never zeroed) when a mark or rate is missing or invalid.
    FX effect = native cost restated at current FX minus booked base
    cost (how much currency movement changed the booked cost basis).
    """
    rows = []
    for lot in open_lots or []:
        px = (marks_native or {}).get(lot.ticker)
        rate = (fx_now or {}).get(lot.native_currency)
        valid = (px is not None and rate is not None
                 and math.isfinite(float(px)) and float(px) > 0
                 and math.isfinite(float(rate)) and float(rate) > 0)
        market_native = lot.shares_remaining * float(px) if valid else None
        market_base = market_native * float(rate) \
            if valid else None
        unrealised = (market_base - lot.cost_basis_base
                      if (valid and lot.cost_basis_base is not None)
                      else None)
        fx_effect = ((lot.cost_basis_native * float(rate)
                      - lot.cost_basis_base)
                     if (valid and lot.cost_basis_base is not None)
                     else None)
        rows.append({
            "lot_id": lot.lot_id, "ticker": lot.ticker,
            "purchase_date": lot.purchase_date,
            "shares_original": lot.shares_original,
            "shares_remaining": lot.shares_remaining,
            "execution_price_native": lot.execution_price_native,
            "native_currency": lot.native_currency,
            "cost_basis_native": lot.cost_basis_native,
            "cost_basis_base": lot.cost_basis_base,
            "market_native": market_native, "market_base": market_base,
            "unrealised_base": unrealised, "fx_effect_base": fx_effect,
            "source": lot.source, "transaction_id": lot.transaction_id,
        })
    return rows


def open_shares_by_ticker(state: LedgerState) -> dict[str, float]:
    """Remaining shares per ticker (must equal position shares)."""
    out: dict[str, float] = {}
    for lot in state.open_lots:
        out[lot.ticker] = out.get(lot.ticker, 0.0) + lot.shares_remaining
    return out


def cost_by_ticker(state: LedgerState) -> dict[str, tuple]:
    """Per-ticker (native_cost, base_cost|None, native_ccy|None)."""
    out: dict[str, list] = {}
    for lot in state.open_lots:
        row = out.setdefault(lot.ticker, [0.0, 0.0, lot.native_currency, True])
        row[0] += lot.cost_basis_native
        if lot.cost_basis_base is None:
            row[3] = False
        else:
            row[1] += lot.cost_basis_base
    return {t: (n, (b if ok else None), c) for t, (n, b, c, ok) in out.items()}


def average_cost_per_share(state: LedgerState) -> dict[str, tuple]:
    """Informational weighted-average native cost per remaining share."""
    out = {}
    for t, (native_cost, _b, ccy) in cost_by_ticker(state).items():
        shares = open_shares_by_ticker(state).get(t, 0.0)
        out[t] = (native_cost / shares if shares > 0 else None, ccy)
    return out


def unrealised_by_ticker(state: LedgerState, current_native: dict,
                         fx_now: dict, base_currency: str) -> dict:
    """Value open lots at explicit marks (pure; maps supplied by caller).

    Returns ticker → dict(market_native, market_base, cost_base|None,
    unrealised_base|None). Missing marks raise (never silently zeroed).
    """
    base_currency = str(base_currency).strip().upper()
    out = {}
    for lot in state.open_lots:
        if lot.ticker not in current_native:
            raise ValueError(f"No current mark supplied for {lot.ticker}.")
        px = current_native[lot.ticker]
        if px is None or not math.isfinite(float(px)) or float(px) <= 0:
            raise ValueError(f"Invalid current mark for {lot.ticker}.")
        mv_native = lot.shares_remaining * float(px)
        rate = fx_now.get(lot.native_currency)
        if rate is None or not math.isfinite(float(rate)) \
                or float(rate) <= 0:
            raise ValueError(
                f"No current FX for {lot.native_currency}.")
        mv_base = mv_native * float(rate)
        row = out.setdefault(lot.ticker, {
            "market_native": 0.0, "market_base": 0.0, "cost_base": 0.0,
            "base_known": True, "native_currency": lot.native_currency})
        row["market_native"] += mv_native
        row["market_base"] += mv_base
        if lot.cost_basis_base is None:
            row["base_known"] = False
        else:
            row["cost_base"] += lot.cost_basis_base
    for t, row in out.items():
        if row["base_known"]:
            row["cost_base"] = row["cost_base"]
            row["unrealised_base"] = row["market_base"] - row["cost_base"]
        else:
            row["cost_base"] = None
            row["unrealised_base"] = None
    return out


def _copy_tx(tx: Transaction) -> Transaction:
    return Transaction(
        transaction_id=tx.transaction_id, date=tx.date, type=tx.type,
        ticker=tx.ticker, quantity=tx.quantity, price_native=tx.price_native,
        currency=tx.currency, fee=tx.fee, purchase_fx=tx.purchase_fx,
        notes=tx.notes, source=tx.source, move_cash=tx.move_cash,
        created_at=tx.created_at)

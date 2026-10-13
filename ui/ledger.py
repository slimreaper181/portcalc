"""ledger UI helpers — session transaction parsing + derived-state builds.

Thin glue between Streamlit session state and :mod:`analytics.ledger` /
:mod:`analytics.persistence` (all maths stays in analytics). Every helper
is deterministic and side-effect free except where noted.
"""

from __future__ import annotations

from analytics.ledger import (
    LedgerState,
    Transaction,
    apply_ledger,
    open_shares_by_ticker,
)
from analytics.persistence import (
    migrate_v1_to_ledger,
    rebase_ledger_fx,
    transaction_from_dict,
    transaction_to_dict,
)


def parse_session_transactions(raw_list) -> tuple[list, str | None]:
    """Parse session txn dicts → (transactions, error|None)."""
    txns = []
    for d in raw_list or []:
        if not isinstance(d, dict):
            return [], "Ledger holds a malformed entry (not a mapping)."
        try:
            txns.append(transaction_from_dict(d))
        except ValueError as e:
            return [], f"Ledger entry rejected: {e}"
    return txns, None


def build_ledger_view(txn_dicts, legacy_positions, base_ccy, fx_hist,
                      cleared: bool = False):
    """Migrate (if needed), resolve FX, apply → view dict for shell/sidebar.

    Returns dict with ``transactions`` (Transaction list), ``state``
    (LedgerState|None), ``error`` (str|None), ``notes`` (list),
    ``migrated`` (bool), ``open_shares`` and ``cash`` dicts. Pure apart
    from reading its arguments (never touches session state).
    """
    notes: list = []
    raw = [t for t in (txn_dicts or []) if isinstance(t, dict)]
    migrated = False
    if not raw and not cleared and legacy_positions:
        mig, skipped = migrate_v1_to_ledger(legacy_positions)
        raw = [transaction_to_dict(t) for t in mig]
        migrated = True
        if skipped:
            notes.append(
                "Migration skipped undated entries (reported, never "
                f"invented): {', '.join(skipped)}.")
    txns, error = parse_session_transactions(raw)
    state = None
    if error is None and txns:
        txns, unresolved = rebase_ledger_fx(txns, base_ccy, fx_hist,
                                            only_missing=True)
        if unresolved:
            notes.append(
                f"Purchase-date FX unavailable for {len(unresolved)} "
                "ledger leg(s); base cost stays unknown there.")
        try:
            state = apply_ledger(txns)
        except ValueError as e:
            error = f"Ledger inconsistent: {e}"
            state = None
    return {"transactions": txns, "state": state, "error": error,
            "notes": notes, "migrated": migrated,
            "open_shares": open_shares_by_ticker(state) if state else {},
            "cash": dict(state.cash) if state else {}}

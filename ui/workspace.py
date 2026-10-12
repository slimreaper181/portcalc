"""ui.workspace — database/session translation (pure, no Streamlit).

The app shell keeps the working copy in ``st.session_state`` (positions
dict, JSON-safe ledger dicts, watchlist, base currency, benchmark) while
SQLite is the durable store. These helpers translate between the two so
the shell stays thin and the logic stays unit-tested:

* :func:`workspace_signature` — stable hash of the editable state; the
  shell skips DB writes when nothing changed.
* :func:`workspace_to_session` — DB workspace record → session shapes.
* :func:`legacy_seed_state` — first-run seed from ``portfolio.json`` /
  ``settings.json`` (legacy aggregates + migrated opening lots).
"""

from __future__ import annotations

import hashlib
import json


def workspace_signature(positions: dict,
                        ledger_txn_dicts: list,
                        watchlist: list,
                        base_currency: str,
                        benchmark: str) -> str:
    """Stable hash of everything the workspace sync persists."""
    payload = {
        "positions": positions or {},
        "ledger": ledger_txn_dicts or [],
        "watchlist": sorted(watchlist or []),
        "base": base_currency or "USD",
        "benchmark": benchmark or "SPY",
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def workspace_to_session(workspace: dict) -> dict:
    """Translate ``repositories.load_workspace`` output to session shapes."""
    from analytics.persistence import transaction_to_dict
    record = workspace["portfolio"]
    return {
        "portfolio": dict(workspace.get("positions") or {}),
        "ledger_txns": [transaction_to_dict(t)
                        for t in workspace.get("transactions") or []],
        "watchlist": list(workspace.get("watchlist") or []),
        "base_currency": record.base_currency or "USD",
        "bt_benchmark": record.benchmark or "SPY",
    }


def legacy_seed_state(legacy_positions: dict,
                      legacy_base: str) -> dict:
    """First-run session shapes from legacy files (same shape as above)."""
    from analytics.persistence import (
        migrate_v1_to_ledger, transaction_to_dict)
    positions = dict(legacy_positions or {})
    txns, _skipped = migrate_v1_to_ledger(positions)
    return {
        "portfolio": positions,
        "ledger_txns": [transaction_to_dict(t) for t in txns],
        "watchlist": [],
        "base_currency": legacy_base or "USD",
        "bt_benchmark": "SPY",
    }


def session_is_empty(session_portfolio: dict,
                     session_ledger: list) -> bool:
    """True when the active workspace holds nothing tradeable."""
    return not (session_portfolio or session_ledger)

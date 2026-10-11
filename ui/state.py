"""state — session-state helpers: defaults, flash-adjacent keys, result cache.

Result cache rule: the key must include EVERY mathematically relevant input
(tickers, dates, lookbacks, methods, bounds, costs, currency, capital,
benchmark, risk-free). Display-only choices (chart size, expanders) must not
be part of cache keys — and must not invalidate results.
"""

from __future__ import annotations

import streamlit as st


def init_defaults(defaults: dict) -> None:
    """Set session keys only when absent (never overwrite user state)."""
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


class ResultCache:
    """Configuration-hash result store on top of session state."""

    def __init__(self, namespace: str, store=None):
        self.namespace = str(namespace)
        self.store = store if store is not None else st.session_state

    def _slot(self) -> str:
        return f"_pc_cache_{self.namespace}"

    def get(self, key):
        """Return cached value when the stored key matches, else None."""
        try:
            slot = self.store.get(self._slot())
        except Exception:
            return None
        if isinstance(slot, dict) and slot.get("key") == key:
            return slot.get("value")
        return None

    def put(self, key, value) -> None:
        """Store value under key (overwrites any previous entry)."""
        try:
            self.store[self._slot()] = {"key": key, "value": value}
        except Exception:
            pass

    def key_changed(self, key) -> bool:
        """True when inputs changed since the cached run (needs refresh)."""
        try:
            slot = self.store.get(self._slot())
        except Exception:
            return True
        if not isinstance(slot, dict) or "key" not in slot:
            return True
        return slot.get("key") != key

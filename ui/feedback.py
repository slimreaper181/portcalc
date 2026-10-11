"""feedback — session-backed flash messages that survive st.rerun().

Streamlit drops per-run deltas (st.success/...) on ``st.rerun()``, so
confirmations rendered just before a rerun never display. Flash messages
persist in session state, render once on the next run, then clear.
"""

from __future__ import annotations

import streamlit as st

_FLASH_KEY = "_pc_flash"

_KINDS = ("success", "info", "warning", "error")


def flash(message: str, kind: str = "success", store=None) -> None:
    """Queue a flash message (default store: Streamlit session state)."""
    if kind not in _KINDS:
        raise ValueError(f"Unknown flash kind {kind!r}.")
    text = str(message)
    if not text:
        return
    if store is None:
        store = st.session_state
    try:
        queue = store.get(_FLASH_KEY, [])
    except Exception:
        queue = []
    queue = list(queue) + [{"kind": kind, "text": text}]
    try:
        store[_FLASH_KEY] = queue
    except Exception:
        pass


def take_flash(store=None) -> list:
    """Drain queued messages (render-once semantics)."""
    if store is None:
        store = st.session_state
    try:
        queue = list(store.get(_FLASH_KEY, []))
    except Exception:
        return []
    try:
        store[_FLASH_KEY] = []
    except Exception:
        pass
    return [m for m in queue
            if isinstance(m, dict) and m.get("kind") in _KINDS]


def render_flash(store=None) -> int:
    """Display and clear queued messages. Returns the count shown."""
    shown = 0
    for m in take_flash(store):
        kind, text = m["kind"], m["text"]
        {"success": st.success, "info": st.info,
         "warning": st.warning, "error": st.error}[kind](text)
        shown += 1
    return shown

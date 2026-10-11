"""errors — reusable section-level error cards (no raw tracebacks).

Expected, user-actionable failures render as compact cards: what happened,
why, and what to do. Unexpected exceptions render the same card plus a
collapsed technical-details expander (never a bare traceback).
"""

from __future__ import annotations

import traceback

import streamlit as st


def error_card(title: str, what: str = "", why: str = "",
               action: str = "") -> None:
    """Render a professional error card for an expected failure."""
    st.error(f"**{title}**" + (f" — {what}" if what else ""))
    with st.expander("What happened and what to do", expanded=False):
        if why:
            st.markdown(f"**Why:** {why}")
        if action:
            st.markdown(f"**Next step:** {action}")


def unexpected_error_card(section: str, exc: BaseException) -> None:
    """Render an unexpected-failure card with collapsed technical details."""
    st.error(f"**{section} could not be displayed.** "
             "This looks like an application issue rather than your inputs.")
    with st.expander("Technical details", expanded=False):
        st.code(f"{type(exc).__name__}: {exc}\n\n"
                + "".join(traceback.format_exception(exc))[-2000:])


def render_section(name: str, fn, *args, **kwargs) -> bool:
    """Run one UI section with an error boundary.

    Returns True when the section rendered. Expected ``ValueError`` shows
    the message as a card; anything else shows the unexpected card. Either
    way the exception never propagates to sibling sections — this replaces
    the cross-tab ``st.stop()`` cascade for section-local failures.
    """
    try:
        fn(*args, **kwargs)
        return True
    except ValueError as e:
        error_card(f"{name} unavailable", str(e) or "Invalid inputs.",
                   action="Adjust the section inputs and try again.")
        return False
    except Exception as e:  # never let one section kill the page
        unexpected_error_card(name, e)
        return False

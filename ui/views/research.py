"""views.research — Research library tab (persistent analyses, Milestone 8).

Saved analyses live in the Milestone-1 ``saved_analyses`` table per
portfolio (config + key result summary as JSON; never giant arrays).
This view renders the library (Name/Type/Created/Last Run/Tags), detail
with plain-text notes, rename/delete, Saved-vs-Recomputed re-runs on
current data, and compatible-type comparison with exact config
differences. User content is shown through text widgets/frames only —
never interpolated into raw HTML.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from analytics.research import (
    ANALYSIS_LABELS,
    ANALYSIS_TYPES,
    check_compatible,
    check_json_safe,
    config_diff,
    parse_tags,
    rerun_analysis,
    results_comparison,
    summarise_result,
    validate_analysis_type,
)
from storage import database as ws_db
from storage import repositories as ws_repo
from storage.database import workspace_db_path


def _db():
    """Open workspace DB + active portfolio id (raises with UI message)."""
    if not st.session_state.get("_db_ok"):
        raise ValueError("Research persistence needs the database "
                         "workspace (file mode keeps the working copy only).")
    pid = st.session_state.get("_ws_pid")
    if not pid:
        raise ValueError("No active portfolio.")
    return ws_db.connect(workspace_db_path()), pid


def save_analysis_button(atype: str, config: dict, result: dict,
                         key: str, label: str = "Save to Research"):
    """Compact saver for other views: name/tags/notes + Save (DB-backed).

    Returns the saved row or None. Config/result must be JSON-safe
    (dates as ISO strings).
    """
    try:
        atype = validate_analysis_type(atype)
        check_json_safe(dict(config or {}), "config")
        check_json_safe(dict(result or {}), "result")
    except ValueError as e:
        st.error(f"Cannot save research: {e}")
        return None
    with st.expander(label, expanded=False):
        name = st.text_input("Analysis name", value="", key=f"{key}_name",
                             placeholder="e.g. Q3 MinVar review")
        tags = st.text_input("Tags (comma-separated)", value="",
                             key=f"{key}_tags",
                             placeholder="Risk, Quarterly")
        notes = st.text_area("Notes", value="", key=f"{key}_notes",
                             placeholder="Plain-text research note.")
        if st.button("Save analysis", key=f"{key}_go", type="primary"):
            if not str(name or "").strip():
                st.error("Give the analysis a name first.")
                return None
            try:
                conn, pid = _db()
                try:
                    row = ws_repo.save_analysis(
                        conn, pid, str(name).strip(), atype,
                        dict(config or {}), dict(result or {}),
                        tags=str(tags or ""), notes=str(notes or ""))
                finally:
                    conn.close()
            except ValueError as e:
                st.error(f"Save failed: {e}")
                return None
            st.success(f"Saved '{row.name}' to the research library.")
            return row
    return None


def _current_panels(tickers: list[str], base_ccy: str) -> dict:
    """Base-currency panels for saved tickers (fresh fetch, honest errors)."""
    from data.pipeline import (
        base_price_panel,
        load_backtest_data,
        load_fx_history,
        load_instruments,
    )
    native = load_backtest_data(list(tickers))
    instruments = load_instruments(list(tickers))
    fx_hist = load_fx_history()
    panel = base_price_panel(native, list(tickers), instruments, fx_hist,
                             str(base_ccy or "USD"))
    return {"prices": panel, "base_panel": panel}


def render(ctx):
    """Render the Research tab from the shared context."""
    base_ccy = ctx.get("base_ccy", "USD")
    st.markdown("## Research Library")
    st.caption("Saved analyses persist per portfolio (SQLite). Re-runs use "
               "current market data — Saved vs Recomputed stay distinct until "
               "you explicitly overwrite.")
    try:
        conn, pid = _db()
    except ValueError as e:
        st.warning(str(e))
        return
    try:
        try:
            pname = ws_repo.get_portfolio(conn, pid).name
            rows = ws_repo.list_analyses(conn, pid)
        finally:
            conn.close()
    except ValueError as e:
        st.warning(f"Research library unavailable: {e}")
        return
    st.caption(f"Portfolio: {pname} — research is isolated per portfolio "
               "(cloning copies it only when you opt in).")
    if not rows:
        st.caption("No saved analyses yet — use “Save to Research” in "
                   "Optimise, Backtest, Scenario or Stress Lab.")
        return
    table = pd.DataFrame([{
        "Name": r.name,
        "Type": ANALYSIS_LABELS.get(r.type, r.type),
        "Portfolio": pname,
        "Created": str(r.created_at)[:19],
        "Last Run": str(r.updated_at)[:19],
        "Tags": r.tags or "",
    } for r in rows])
    st.dataframe(table, width="stretch", hide_index=True)
    by_name = {r.name: r for r in rows}
    pick = st.selectbox("Open analysis", list(by_name), key="rs_open")
    row = by_name[pick]
    try:
        config = json.loads(row.config_json or "{}")
        result = json.loads(row.result_json or "{}")
    except ValueError:
        st.error("Stored payloads are corrupt (invalid JSON).")
        return
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("#### Saved Result")
        summ = summarise_result(result)
        if summ.empty:
            st.caption("No numeric summary stored.")
        else:
            st.dataframe(summ, width="stretch", hide_index=True)
    with d2:
        st.markdown("#### Configuration")
        if config:
            st.dataframe(pd.DataFrame(
                [{"Setting": k, "Value": str(v)[:80]} for k, v in
                 sorted(config.items())]), width="stretch", hide_index=True)
        else:
            st.caption("No configuration stored.")

    # ---- notes / tags / rename / delete ----
    with st.expander("Notes & tags", expanded=False):
        notes = st.text_area("Notes (plain text)", value=row.notes or "",
                             key=f"rs_notes_{row.analysis_id}")
        tags = st.text_input("Tags", value=row.tags or "",
                             key=f"rs_tags_{row.analysis_id}")
        if st.button("Save notes & tags", key=f"rs_nt_{row.analysis_id}"):
            try:
                conn2, _ = _db()
                try:
                    ws_repo.set_analysis_notes(conn2, row.analysis_id, notes)
                    conn2.execute(
                        "UPDATE saved_analyses SET tags = ? WHERE "
                        "analysis_id = ?",
                        (",".join(parse_tags(tags)), row.analysis_id))
                finally:
                    conn2.close()
            except ValueError as e:
                st.error(f"Update failed: {e}")
            else:
                st.success("Notes & tags saved.")
                st.rerun()
    with st.expander("Rename / delete", expanded=False):
        new_name = st.text_input("Rename to", value=row.name,
                                 key=f"rs_rn_{row.analysis_id}")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Rename", key=f"rs_rngo_{row.analysis_id}"):
                try:
                    conn2, _ = _db()
                    try:
                        ws_repo.rename_analysis(conn2, row.analysis_id,
                                                new_name)
                    finally:
                        conn2.close()
                except ValueError as e:
                    st.error(f"Rename failed: {e}")
                else:
                    st.success("Renamed.")
                    st.rerun()
        with c2:
            if st.checkbox("Confirm delete", key=f"rs_del_{row.analysis_id}"):
                if st.button("Delete permanently",
                             key=f"rs_delgo_{row.analysis_id}"):
                    try:
                        conn2, _ = _db()
                        try:
                            ws_repo.delete_analysis(conn2, row.analysis_id)
                        finally:
                            conn2.close()
                    except ValueError as e:
                        st.error(f"Delete failed: {e}")
                    else:
                        st.success("Deleted.")
                        st.rerun()

    # ---- re-run on current data ----
    with st.expander("Re-run on current data", expanded=False):
        st.caption("Recomputes from current market data. The saved result is "
                   "kept until you explicitly overwrite it.")
        if st.button("Recompute now", key=f"rs_rr_{row.analysis_id}"):
            try:
                tickers = [str(t) for t in (config.get("tickers") or [])]
                panels = _current_panels(
                    tickers, config.get("base_ccy", base_ccy))
                fresh = rerun_analysis(row.type, config, panels)
                st.session_state[f"rs_fresh_{row.analysis_id}"] = fresh
            except ValueError as e:
                st.error(f"Re-run failed: {e}")
            except Exception:
                st.error("Re-run failed on current data (network or "
                         "coverage error). The saved result is untouched.")
        fresh = st.session_state.get(f"rs_fresh_{row.analysis_id}")
        if fresh:
            f1, f2 = st.columns(2)
            with f1:
                st.markdown("**Saved**")
                st.dataframe(summarise_result(result), width="stretch",
                             hide_index=True)
            with f2:
                st.markdown("**Recomputed**")
                st.dataframe(summarise_result(fresh), width="stretch",
                             hide_index=True)
            st.markdown("**Metric comparison**")
            st.dataframe(results_comparison(result, fresh), width="stretch",
                         hide_index=True)
            if st.button("Overwrite saved result with recomputed",
                         key=f"rs_ow_{row.analysis_id}"):
                try:
                    conn2, pid2 = _db()
                    try:
                        ws_repo.save_analysis(
                            conn2, pid2, row.name, row.type, config, fresh,
                            tags=row.tags or "", notes=row.notes or "",
                            analysis_id=row.analysis_id)
                    finally:
                        conn2.close()
                except ValueError as e:
                    st.error(f"Overwrite failed: {e}")
                else:
                    st.session_state.pop(f"rs_fresh_{row.analysis_id}", None)
                    st.success("Saved result updated.")
                    st.rerun()

    # ---- compare ----
    with st.expander("Compare with another analysis", expanded=False):
        others = [r.name for r in rows if r.analysis_id != row.analysis_id]
        if not others:
            st.caption("Save a second analysis to compare.")
        else:
            other = st.selectbox("Compare with", others, key="rs_cmp")
            brow = by_name[other]
            try:
                check_compatible(row.type, brow.type)
            except ValueError as e:
                st.warning(str(e))
            else:
                try:
                    bconfig = json.loads(brow.config_json or "{}")
                    bresult = json.loads(brow.result_json or "{}")
                except ValueError:
                    st.error("Comparison payloads are corrupt.")
                else:
                    st.markdown(f"**{row.name} (A) vs {brow.name} (B)**")
                    st.dataframe(results_comparison(result, bresult),
                                 width="stretch", hide_index=True)
                    diff = config_diff(config, bconfig)
                    if diff.empty:
                        st.caption("Configurations are identical.")
                    else:
                        st.markdown("**Configuration differences**")
                        st.dataframe(diff, width="stretch", hide_index=True)

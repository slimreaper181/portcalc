"""views.accounting — Transaction Accounting 2.0 surface (M2).

Rendered inside Overview's ledger expander (M9 will promote it to the
PORTFOLIO area). Everything here edits the session working copy only;
the shell's audited end-of-run sync persists to the DB workspace and
records CREATE/UPDATE/DELETE revisions automatically.

* Edit form (identity + entry order preserved, full-ledger trial apply).
* Two-click delete with confirmation.
* Running cash / running position from trace_ledger (one accounting truth).
* FIFO consumption detail per SELL, valued lot drilldown.
* Atomic CSV import with preview + duplicate detection.
* Audit-trail viewer with restore (appends new revisions; history is
  never overwritten).
"""

from __future__ import annotations

import json
from datetime import date, datetime

import pandas as pd
import streamlit as st

from analytics.ledger import (
    Transaction,
    apply_ledger,
    lot_market_values,
    sell_consumption,
    trace_ledger,
    validate_transaction,
)
from analytics.ledger_csv import preview_csv_import, preview_to_dicts
from analytics.persistence import transaction_from_dict, transaction_to_dict
from analytics.validation import validate_ticker_symbol
from storage import database as ws_db
from storage import repositories as ws_repo
from storage.database import workspace_db_path
from ui.feedback import flash
from ui.ledger import parse_session_transactions

_CASH_CCYS = ["USD", "GBP", "EUR"]


def _session_txns() -> list[dict]:
    return [t for t in (st.session_state.get("ledger_txns") or [])
            if isinstance(t, dict)]


def _trial_replace(tx_dicts: list[dict]) -> tuple[list, str | None]:
    """Parse + full-ledger trial apply → (transactions, error|None)."""
    txns, err = parse_session_transactions(tx_dicts)
    if err is not None:
        return [], err
    try:
        apply_ledger(txns)
    except ValueError as e:
        return [], f"Ledger inconsistent: {e}"
    return txns, None


def _commit_session(tx_dicts: list[dict], message: str, kind="success"):
    st.session_state.ledger_txns = tx_dicts
    flash(message, kind)
    for _k in ("tx_edit_id", "tx_del_armed", "ac_csv_pending"):
        st.session_state.pop(_k, None)
    st.rerun()


# ---------------------------------------------------------------------------
# edit form
# ---------------------------------------------------------------------------

def _render_edit_form(all_dicts: list[dict]):
    edit_id = st.session_state.get("tx_edit_id")
    if not edit_id:
        return
    current = next((d for d in all_dicts
                    if d.get("transaction_id") == edit_id), None)
    if current is None:
        st.session_state.pop("tx_edit_id", None)
        return
    st.markdown(f"#### Edit transaction `{edit_id[:8]}…`")
    st.caption("Identity and entry order are preserved; the whole ledger "
               "is trial-applied before anything changes. Type cannot be "
               "changed here — delete and re-record for that.")
    try:
        orig = transaction_from_dict(current)
    except ValueError as e:
        st.error(f"Stored entry is malformed: {e}")
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("Type", value=str(orig.type), disabled=True,
                      key="ac_edit_type")
        new_date = st.date_input("Date", value=orig.date,
                                 max_value=date.today(), key="ac_edit_date")
        _ct = orig.created_at
        _dflt = (_ct.time() if isinstance(_ct, datetime) else
                 datetime(2026, 1, 1, 12, 0).time())
        new_time = st.time_input("Entry time (same-day order)",
                                 value=_dflt, key="ac_edit_time")
    with c2:
        needs_ticker = orig.type in ("BUY", "SELL", "DIVIDEND", "INTEREST")
        new_ticker = st.text_input(
            "Ticker", value=str(orig.ticker or ""),
            disabled=not needs_ticker, key="ac_edit_ticker")
        new_qty = st.number_input("Quantity", min_value=0.0,
                                  value=float(orig.quantity),
                                  key="ac_edit_qty")
        show_price = orig.type in ("BUY", "SELL", "DIVIDEND", "INTEREST")
        new_price = st.number_input(
            "Price (native)", min_value=0.0,
            value=float(orig.price_native) if show_price else 1.0,
            key="ac_edit_price") if show_price else 1.0
    with c3:
        _ccys = _CASH_CCYS if orig.currency not in _CASH_CCYS \
            else _CASH_CCYS
        try:
            _ccy_idx = _ccys.index(orig.currency)
        except ValueError:
            _ccys = [orig.currency] + _ccys
            _ccy_idx = 0
        new_ccy = st.selectbox("Currency", _ccys, index=_ccy_idx,
                               key="ac_edit_ccy")
        new_fee = st.number_input(
            "Fee (native)", min_value=0.0, value=float(orig.fee),
            key="ac_edit_fee") \
            if orig.type in ("BUY", "SELL", "FEE") else 0.0
        new_move = st.checkbox(
            "Move cash", value=bool(orig.move_cash),
            key="ac_edit_move",
            help="Cash-exempt lots behave like classic aggregate adds.") \
            if orig.type in ("BUY", "SELL") else orig.move_cash
    new_notes = st.text_area("Notes", value=str(orig.notes or ""),
                             key="ac_edit_notes", max_chars=500)
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Save correction", key="ac_edit_save",
                     type="primary"):
            try:
                sym = (validate_ticker_symbol(new_ticker)
                       if needs_ticker else None)
                if float(new_qty) <= 0:
                    raise ValueError("Quantity must be positive.")
                if orig.type in ("BUY", "SELL") and \
                        float(new_price) <= 0:
                    raise ValueError("Execution price must be positive.")
                if orig.type == "FEE" and float(new_qty) <= 0:
                    raise ValueError("Fee amount must be positive.")
                fx = orig.purchase_fx
                if new_date != orig.date or new_ccy != orig.currency:
                    # Stored FX would be stale: clear it and let the
                    # render-time rebase re-resolve (or report unknown).
                    fx = None
                candidate = validate_transaction(Transaction(
                    transaction_id=orig.transaction_id,
                    date=new_date, type=orig.type, ticker=sym,
                    quantity=float(new_qty),
                    price_native=float(new_price),
                    currency=new_ccy,
                    fee=float(new_qty) if orig.type == "FEE"
                    else float(new_fee),
                    purchase_fx=fx, notes=str(new_notes or ""),
                    source=orig.source, move_cash=bool(new_move),
                    created_at=datetime.combine(new_date, new_time)))
                rebuilt = [transaction_to_dict(candidate)
                           if d.get("transaction_id") == edit_id else d
                           for d in all_dicts]
                _txns, _err = _trial_replace(rebuilt)
                if _err is not None:
                    raise ValueError(_err)
            except ValueError as e:
                st.error(str(e))
            else:
                _commit_session(
                    rebuilt,
                    f"Correction saved ({orig.type} {edit_id[:8]}…) — "
                    "ledger rebuilt.", "success")
    with b2:
        if st.button("Cancel", key="ac_edit_cancel"):
            st.session_state.pop("tx_edit_id", None)
            st.rerun()
    st.markdown("---")


# ---------------------------------------------------------------------------
# main section
# ---------------------------------------------------------------------------

def render_ledger_section(ctx):
    """Transactions, lots, CSV import and audit trail (M2 surface)."""
    base_ccy = ctx.get("base_ccy", "USD")
    ccy = ctx.get("ccy", {})
    instruments = ctx.get("instruments", {})
    valuation_prices = ctx.get("valuation_prices", {})
    ledger_state = ctx.get("ledger_state")
    ledger_error = ctx.get("ledger_error")
    for _n in ctx.get("ledger_notes", []):
        st.caption(str(_n))
    if ledger_error:
        st.warning(f"Ledger problem: {ledger_error} Correct it below — "
                   "derived lots, cash and P&L rebuild deterministically.")

    all_dicts = _session_txns()
    _render_edit_form(all_dicts)
    if not all_dicts:
        st.caption("No ledger transactions yet — record one in the "
                   "sidebar, or add a holding (auto-migrated).")
    else:
        # ---- filters ----
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            ft = st.selectbox(
                "Ticker", ["All"] + sorted(
                    {str(t.get("ticker") or "") for t in all_dicts
                     if t.get("ticker")}), key="ac_ticker")
        with f2:
            fy = st.selectbox(
                "Type",
                ["All", "BUY", "SELL", "DIVIDEND", "FEE", "DEPOSIT",
                 "WITHDRAWAL", "INTEREST"], key="ac_type")
        with f3:
            fd_from = st.date_input("From", value=None, key="ac_from")
        with f4:
            fd_to = st.date_input("To", value=None, key="ac_to")
        fq = st.text_input("Search notes", value="", key="ac_q",
                           placeholder="free text…").strip().lower()

        txns, perr = parse_session_transactions(all_dicts)
        trace = []
        if perr is None and txns:
            try:
                trace = trace_ledger(txns)
            except ValueError:
                trace = []  # apply error already surfaced above
        by_id = {t["transaction_id"]: t for t in trace}
        realised_by_id = {}
        if ledger_state is not None:
            realised_by_id = {e.transaction_id: e
                              for e in ledger_state.realised}
        by_txn = {t.transaction_id: t for t in txns} if perr is None \
            else {}

        rows, row_ids = [], []
        for d in sorted(all_dicts,
                        key=lambda x: (str(x.get("date", "")),
                                       str(x.get("transaction_id", "")))):
            if ft != "All" and d.get("ticker") != ft:
                continue
            if fy != "All" and d.get("type") != fy:
                continue
            if fd_from and str(d.get("date", "")) < fd_from.isoformat():
                continue
            if fd_to and str(d.get("date", "")) > fd_to.isoformat():
                continue
            if fq and fq not in str(d.get("notes", "")).lower():
                continue
            snap = by_id.get(d.get("transaction_id"), {})
            cash_after = snap.get("cash_after", {})
            moved = str(d.get("currency", ""))
            row = {
                "Date": str(d.get("date", ""))[:10],
                "Type": d.get("type"),
                "Ticker": d.get("ticker") or "—",
                "Qty": d.get("quantity"),
                "Price": d.get("price_native"),
                "Ccy": moved,
                "Fee": d.get("fee"),
                f"Cash {moved} after": (
                    f"{float(cash_after.get(moved, 0.0)):,.2f}"
                    if moved in cash_after else "—"),
                "Notes": (str(d.get("notes", ""))[:40]),
            }
            if ft != "All":
                shares = (snap.get("shares_after", {}) or {}).get(ft)
                row["Shares after"] = (
                    f"{float(shares):g}" if shares is not None else "—")
            rows.append(row)
            row_ids.append(d.get("transaction_id"))

        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("Delete corrects the ledger (derived lots/P&L/cash "
                   "rebuild deterministically; never hand-edited). Cash "
                   "after = balance in the moved currency. Filter to one "
                   "ticker for shares-after-event.")

        # ---- per-row actions ----
        armed = st.session_state.get("tx_del_armed")
        for d, tid in zip(
                [x for x in all_dicts
                 if x.get("transaction_id") in set(row_ids)], row_ids):
            _d = next((x for x in all_dicts
                       if x.get("transaction_id") == tid), None)
            if _d is None:
                continue
            lbl = (f"{str(_d.get('date', ''))[:10]} {_d.get('type')} "
                   f"{_d.get('ticker') or ''} `{str(tid)[:6]}…`")
            a1, a2, a3 = st.columns([6, 1, 1])
            with a1:
                st.markdown(lbl)
            with a2:
                if st.button("Edit", key=f"ac_ed_{tid}"):
                    st.session_state.tx_edit_id = tid
                    st.session_state.pop("tx_del_armed", None)
                    st.rerun()
            with a3:
                if armed == tid:
                    if st.button("Confirm", key=f"ac_del_yes_{tid}",
                                 type="primary"):
                        kept = [x for x in all_dicts
                                if x.get("transaction_id") != tid]
                        _txns, _err = _trial_replace(kept)
                        if _err is not None:
                            st.error(_err)
                        else:
                            _commit_session(
                                kept, "Transaction deleted — ledger "
                                "rebuilt.", "info")
                else:
                    if st.button("Delete", key=f"ac_del_{tid}"):
                        st.session_state.tx_del_armed = tid
                        st.session_state.pop("tx_edit_id", None)
                        st.rerun()
        if armed:
            if st.button("Cancel delete", key="ac_del_cancel"):
                st.session_state.pop("tx_del_armed", None)
                st.rerun()

        # ---- FIFO consumption per SELL ----
        sells = [t for t in (txns if perr is None else [])
                 if t.type == "SELL" and t.transaction_id in set(row_ids)]
        if sells and ledger_state is not None:
            with st.expander("FIFO consumption (SELL detail)",
                             expanded=False):
                for s in sells:
                    ev = realised_by_id.get(s.transaction_id)
                    if ev is None:
                        continue
                    st.markdown(
                        f"**SELL {s.quantity:g} {s.ticker}** on "
                        f"{s.date.isoformat()} — realised "
                        f"{ev.realised_pnl_native:+,.2f} "
                        f"{s.currency} (fee {ev.fee_native:g}).")
                    _detail = [{
                        "Lot date": (r["lot_date"].isoformat()
                                     if r["lot_date"] else "—"),
                        "Price": r["execution_price_native"],
                        "Ccy": r["currency"],
                        "Shares": r["shares"],
                    } for r in sell_consumption(txns, ev)]
                    st.dataframe(pd.DataFrame(_detail), width="stretch",
                                 hide_index=True)
                st.caption("Oldest lots first (date, entry time, id). "
                           "Fees adjust proceeds; lots keep their own "
                           "purchase FX.")

    # ---- open lots drilldown ----
    if ledger_state is not None and ledger_state.open_lots:
        with st.expander("Open lots (drilldown)", expanded=False):
            _norm = {}
            for t, px in (valuation_prices or {}).items():
                try:
                    _inst = instruments.get(t)
                    _scale = float(_inst.quote_scale or 1.0)
                    _norm[t] = float(px) * _scale
                except (TypeError, ValueError, AttributeError):
                    continue
            _fx_now = (ccy or {}).get("fx_now", {}) or {}
            _lrows = []
            for r in lot_market_values(ledger_state.open_lots, _norm,
                                       _fx_now):
                _mkt = (f"{r['market_base']:,.2f} {base_ccy}"
                        if r["market_base"] is not None else "—")
                _upl = (f"{r['unrealised_base']:+,.2f}"
                        if r["unrealised_base"] is not None else "—")
                _fxe = (f"{r['fx_effect_base']:+,.2f}"
                        if r["fx_effect_base"] is not None else "—")
                _cb = (f"{r['cost_basis_base']:,.2f}"
                       if r["cost_basis_base"] is not None
                       else f"{r['cost_basis_native']:,.2f} "
                            f"{r['native_currency']} (base ?)")

                def _fmt_day(v):
                    return v.isoformat() if hasattr(v, "isoformat") \
                        else str(v)
                _lrows.append({
                    "Ticker": r["ticker"],
                    "Bought": _fmt_day(r["purchase_date"]),
                    "Orig": round(r["shares_original"], 6),
                    "Left": round(r["shares_remaining"], 6),
                    "Exec": f"{r['execution_price_native']:,.2f} "
                            f"{r['native_currency']}",
                    "Cost (base)": _cb,
                    f"Market ({base_ccy})": _mkt,
                    "Unrealised": _upl,
                    "FX effect": _fxe,
                    "Source": r["source"],
                })
            st.dataframe(pd.DataFrame(_lrows), width="stretch",
                         hide_index=True)
            st.caption("Unrealised = market − booked base cost. FX effect "
                       "= native cost restated at current FX minus booked "
                       "base cost. Missing marks show — (never zeroed).")

    # ---- CSV import ----
    with st.expander("Bulk CSV import (atomic)", expanded=False):
        st.caption("Schema: date,type,ticker,quantity,price_native,"
                   "currency,fee,purchase_fx,notes,move_cash,"
                   "transaction_id. Entire file validates before anything "
                   "is added; duplicates warn; future dates rejected.")
        up = st.file_uploader("Transaction CSV", type=["csv"],
                              key="ac_csv")
        if up is not None:
            try:
                text = up.getvalue().decode("utf-8-sig")
            except (ValueError, AttributeError, UnicodeDecodeError):
                st.error("Unreadable file (expect UTF-8 CSV).")
                text = None
            if text is not None:
                if len(text) > 2_000_000:
                    st.error("File too large (2 MB limit).")
                else:
                    existing, _e = parse_session_transactions(all_dicts)
                    preview = preview_csv_import(
                        text, existing=existing if _e is None else [])
                    if preview["invalid"]:
                        st.markdown("**Invalid rows**")
                        st.dataframe(pd.DataFrame(preview["invalid"]),
                                     width="stretch", hide_index=True)
                    if preview["duplicates"]:
                        st.markdown("**Possible duplicates**")
                        for _d in preview["duplicates"]:
                            st.warning(_d)
                    eff = preview["effects"]
                    e1, e2, e3 = st.columns(3)
                    e1.metric("New rows", len(preview["valid"]))
                    e2.metric("New tickers",
                              ", ".join(eff["new_tickers"]) or "—")
                    e3.metric("Cash delta",
                              "; ".join(f"{c} {v:+,.2f}"
                                        for c, v in
                                        eff["cash_delta"].items()) or "—")
                    if eff["trial_error"]:
                        st.error(f"Trial apply failed: "
                                 f"{eff['trial_error']}")
                    if preview["ok"]:
                        st.success(
                            f"{len(preview['valid'])} rows valid — "
                            "nothing applied yet.")
                        if st.button(
                                f"Import {len(preview['valid'])} "
                                "transactions", key="ac_csv_go",
                                type="primary"):
                            pending = preview_to_dicts(preview)
                            merged = all_dicts + pending
                            _txns, _err = _trial_replace(merged)
                            if _err is not None:
                                st.error(f"Import aborted: {_err}")
                            else:
                                _commit_session(
                                    merged,
                                    f"Imported {len(pending)} "
                                    "transactions.", "success")
                    else:
                        st.info("Fix the file — nothing was applied.")

    # ---- audit trail ----
    with st.expander("Audit trail (revisions)", expanded=False):
        if not st.session_state.get("_db_ok"):
            st.caption("Revision history needs the database workspace "
                       "(file mode keeps the working copy only).")
            return
        try:
            _c = ws_db.connect(workspace_db_path())
            try:
                _pid = st.session_state.get("_ws_pid")
                _revs = list(reversed(
                    ws_repo.list_revisions(_c, _pid)))[-80:]
                _revs = list(reversed(_revs))
            finally:
                _c.close()
        except (ValueError, TypeError) as e:
            st.warning(f"Audit unavailable: {e}")
            return
        _flt = st.text_input("Filter by transaction id", value="",
                             key="ac_audit_f").strip()
        if _flt:
            _revs = [r for r in _revs if _flt in r.transaction_id]
        if not _revs:
            st.caption("No revisions recorded yet.")
            return
        for r in _revs[:50]:
            c1, c2, c3 = st.columns([5, 2, 2])
            with c1:
                st.markdown(f"`{r.transaction_id[:8]}…` **{r.action}** · "
                            f"{str(r.created_at)[:19]}")
            with c2:
                if r.action in ("UPDATE", "DELETE") and r.before_json:
                    if st.button("Restore", key=f"ac_rs_{r.revision_id}"):
                        try:
                            payload = json.loads(r.before_json)
                            restored = transaction_to_dict(
                                transaction_from_dict(payload))
                        except (ValueError, TypeError) as e:
                            st.error(f"Cannot restore: {e}")
                        else:
                            ids = {d.get("transaction_id")
                                   for d in all_dicts}
                            if r.transaction_id in ids:
                                merged = [
                                    restored if d.get("transaction_id")
                                    == r.transaction_id else d
                                    for d in all_dicts]
                            else:
                                merged = all_dicts + [restored]
                            _txns, _err = _trial_replace(merged)
                            if _err is not None:
                                st.error(f"Restore blocked: {_err}")
                            else:
                                _commit_session(
                                    merged,
                                    "Version restored as a new revision "
                                    "(history preserved).", "success")
            with c3:
                with st.popover("Detail"):
                    if r.before_json:
                        st.markdown("Before")
                        st.json(json.loads(r.before_json))
                    if r.after_json:
                        st.markdown("After")
                        st.json(json.loads(r.after_json))
        st.caption("Restoring appends a new revision — earlier entries "
                   "are never overwritten.")

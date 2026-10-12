"""
app.py
------
Portfolio Analytics & Risk Dashboard — Streamlit entry point.

Tabs:
  1. 📈 Overview      — positions, returns, correlation/covariance
  2. ⚠️  Risk          — VaR (parametric, historical, Monte Carlo) + CVaR
  3. 🎯 Optimise      — efficient frontier, max Sharpe, min variance, target constraints
  4. 📊 Scenario      — Monte Carlo forward simulation with stress scenarios
  5. 📉 Performance   — benchmark comparison, drawdowns, downside risk, rolling stats
  6. 🔍 Security Detail — single-holding analytics + TradingView chart

Run locally:
    streamlit run app.py

The app is designed to be modular so the analytics modules can be imported
and used independently of Streamlit (e.g. in a desktop GUI or CLI).

Numerical conventions:
  * Market-data returns are daily LOG returns (see analytics.returns).
  * Weight vectors, mu vectors and covariance matrices always share the
    canonical (sorted) ticker order defined in analytics.validation.
"""

import os
import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import streamlit as st

from analytics.validation import (
    canonical_tickers,
    validate_shares,
    validate_ticker_symbol,
)
from analytics.currency import rate_on_date, validate_base_currency
from analytics.ledger import (
    Transaction,
    apply_ledger,
    new_transaction_id,
    open_shares_by_ticker,
    validate_transaction,
)
from analytics.persistence import (
    export_portfolio_v2,
    migrate_v1_to_ledger,
    validate_portfolio_import,
)
from storage import database as ws_db
from storage import repositories as ws_repo
from ui.workspace import (
    legacy_seed_state,
    workspace_signature,
    workspace_to_session,
)
from data.market_data import fetch_current_prices
from analytics.positions import (
    format_money,
    normalise_quote,
    position_from_dict,
    resolve_purchase_price,
)
from data.fx import SUPPORTED_BASE_CURRENCIES, get_instrument
from data.pipeline import (
    build_core_analytics,
    build_currency_layer,
    build_position_valuations,
    load_alpaca_trades,
    load_backtest_data,
    load_benchmark_data,
    load_current_fx,
    load_dividends_history,
    load_exchange_hint,
    load_factor_data,
    load_fx_history,
    load_instruments,
    load_market_data,
    load_raw_history,
    load_splits,
    resolve_valuation_prices,
)
from ui.errors import render_section
from ui.feedback import flash, render_flash
from ui.ledger import build_ledger_view, parse_session_transactions
from analytics.persistence import transaction_to_dict
from ui.views import backtest as views_backtest
from ui.views import factors as views_factors
from ui.views import performance as views_performance
from ui.views import security as views_security
from ui.views import optimise as views_optimise
from ui.views import scenario as views_scenario
from ui.views import overview as views_overview
from ui.views import risk as views_risk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "portfolio.json")
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")


def workspace_db_path() -> str:
    """Workspace database location (PORTCALC_DB overrides for tests)."""
    override = os.environ.get("PORTCALC_DB")
    if override:
        return override
    return os.path.join(os.path.dirname(__file__), "portcalc.db")

DEFAULT_PORTFOLIO = {
    "AAPL":  {"shares": 10, "avg_cost": 150.0},
    "MSFT":  {"shares": 5,  "avg_cost": 300.0},
    "GOOGL": {"shares": 3,  "avg_cost": 130.0},
    "AMZN":  {"shares": 4,  "avg_cost": 180.0},
    "NVDA":  {"shares": 8,  "avg_cost": 500.0},
}

# ---------------------------------------------------------------------------
# Page configuration + visual theme (single design system in ui/theme.py)
# ---------------------------------------------------------------------------

from ui.theme import inject_theme
inject_theme()


# ---------------------------------------------------------------------------
# Portfolio persistence helpers
# ---------------------------------------------------------------------------

def load_portfolio() -> dict:
    """Load portfolio from JSON file, falling back to defaults."""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                data = json.load(f)
            # Normalise keys on load (upper-case, stripped).
            cleaned = {}
            for k, v in data.items():
                try:
                    sym = validate_ticker_symbol(k)
                except ValueError:
                    continue
                cleaned[sym] = v
            if cleaned:
                return cleaned
        except (json.JSONDecodeError, IOError):
            pass
    return DEFAULT_PORTFOLIO.copy()


def save_portfolio(portfolio: dict) -> None:
    """Legacy JSON write — file mode only (DB mode syncs at end of run)."""
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(portfolio, f, indent=2)
    except IOError as e:
        st.warning(f"Could not save portfolio: {e}")


def load_settings() -> dict:
    """Load UI settings (base currency); defaults when missing/invalid."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                try:
                    base = validate_base_currency(data.get("base_currency", "USD"))
                except ValueError:
                    base = "USD"
                return {"base_currency": base}
        except (json.JSONDecodeError, IOError, ValueError):
            pass
    return {"base_currency": "USD"}


def save_settings(settings: dict) -> None:
    """Legacy JSON write — file mode only (DB mode syncs at end of run)."""
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump({"schema_version": 1,
                       "base_currency": settings.get("base_currency", "USD")},
                      f, indent=2)
    except IOError as e:
        st.warning(f"Could not save settings: {e}")


def _persist_portfolio_now() -> None:
    """Write-through for file mode; DB mode persists via end-of-run sync."""
    if not st.session_state.get("_db_ok"):
        save_portfolio(st.session_state.portfolio)


def _persist_settings_now() -> None:
    """Write-through for file mode; DB mode persists via end-of-run sync."""
    if not st.session_state.get("_db_ok"):
        save_settings({"base_currency": st.session_state.get(
            "base_currency", "USD")})


def _complete_add_position(sym, shares, purchase_date, price_native,
                           source, manual, inst):
    """Store one aggregate position (single purchase date, no tax lots).

    ``price_native`` is per current split-adjusted share; ``source`` is one
    of ``estimate``/``manual``/``legacy``. An existing ticker is overwritten
    (with a warning) — multiple purchase lots are not supported.
    """
    if sym in st.session_state.portfolio:
        st.warning(f"{sym} already held — updating it in place. Multiple "
                   "purchase lots are not supported yet.")
    # Never silently drop cash-moving (full-form) lots through the classic
    # aggregate path: those belong to the Transaction form.
    try:
        _existing, _err = parse_session_transactions(
            st.session_state.ledger_txns)
        _foreign = [t.transaction_id for t in _existing
                    if t.ticker == sym and t.source not in ("aggregate",
                                                            "migrate")]
        if _err is None and _foreign:
            st.warning(
                f"{sym} has full-form ledger lots — update it with a Sell/Buy "
                "in Record Transaction instead (classic update would orphan "
                "their cash flows).")
            return
    except Exception:
        pass
    st.session_state.portfolio[sym] = {
        "shares": float(shares),
        "purchase_date": (purchase_date.isoformat()
                          if hasattr(purchase_date, "isoformat")
                          and purchase_date is not None else None),
        "purchase_price_native": float(price_native),
        "purchase_price_source": source,
        "manual_purchase_price": bool(manual),
        "native_currency": inst.get("native_currency")
        if isinstance(inst, dict) else getattr(inst, "native_currency", None),
        "quote_unit": inst.get("quote_unit")
        if isinstance(inst, dict) else getattr(inst, "quote_unit", None),
        "quote_scale": float(inst.get("quote_scale", 1.0))
        if isinstance(inst, dict) else float(
            getattr(inst, "quote_scale", 1.0)),
    }
    _persist_portfolio_now()
    st.session_state.data_loaded = False
    _label = {"estimate": "estimated close", "manual": "actual execution",
              "legacy": "legacy cost"}.get(source, source)
    # Mirror into the ledger as a cash-exempt lot (classic aggregate adds
    # carry no cash movement; full-form BUYs move cash). Replaces any prior
    # lots for this ticker (single aggregate acquisition, like the dict).
    try:
        _keep = [t for t in st.session_state.ledger_txns
                 if not (isinstance(t, dict)
                         and str(t.get("ticker", "")).upper() == sym)]
        _nccy = (inst.get("native_currency")
                 if isinstance(inst, dict)
                 else getattr(inst, "native_currency", None)) or "USD"
        _base_now = st.session_state.get("base_currency", "USD")
        _fx = 1.0 if _nccy == _base_now else None
        if _fx is None:
            try:
                _fxh = load_fx_history()
                _fx = rate_on_date(
                    _fxh, _nccy, _base_now,
                    purchase_date if hasattr(purchase_date, "isoformat")
                    and purchase_date is not None else date.today())
            except ValueError:
                _fx = None
        _lot = validate_transaction(Transaction(
            transaction_id=new_transaction_id(),
            date=(purchase_date if hasattr(purchase_date, "isoformat")
                  and purchase_date is not None else date.today()),
            type="BUY", ticker=sym, quantity=float(shares),
            price_native=float(price_native), currency=_nccy, fee=0.0,
            purchase_fx=_fx, notes=f"Classic add ({_label}).",
            source="aggregate", move_cash=False))
        _keep.append(transaction_to_dict(_lot))
        st.session_state.ledger_txns = _keep
    except (ValueError, AttributeError):
        pass  # aggregate dict stays canonical; ledger keeps prior lots
    # Flash (not st.success): per-run deltas are dropped by st.rerun().
    flash(f"Added {sym} ({_label}).", "success")
    st.rerun()


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Workspace bootstrap — SQLite is the durable store (Task 10, Milestone 1).
# Session holds the working copy; every mutation persists via
# _sync_workspace_to_db (hash-guarded, called post-sidebar and end of run).
# JSON files are first-run seed only. When the database is unavailable the
# shell degrades to the legacy file behaviour (single portfolio).
# ---------------------------------------------------------------------------

def _workspace_seed_default(conn) -> None:
    """Create the default portfolio from legacy files on a fresh database."""
    legacy_pos = load_portfolio()
    legacy_base = load_settings().get("base_currency", "USD")
    seed = legacy_seed_state(legacy_pos, legacy_base)
    ws_repo.create_portfolio(
        conn, "My Portfolio", seed["base_currency"],
        seed["bt_benchmark"], 0.05,
        portfolio_id=ws_repo.DEFAULT_PORTFOLIO_ID)
    from analytics.persistence import transaction_from_dict
    ws_repo.save_workspace(
        conn, ws_repo.DEFAULT_PORTFOLIO_ID,
        positions=seed["portfolio"],
        transactions=[transaction_from_dict(d)
                      for d in seed["ledger_txns"]],
        watchlist=seed["watchlist"],
        base_currency=seed["base_currency"],
        benchmark=seed["bt_benchmark"])
    ws_repo.set_active_portfolio_id(conn, ws_repo.DEFAULT_PORTFOLIO_ID)


def _workspace_apply_session(state: dict, portfolio_id: str) -> None:
    """Load a DB workspace into session (working copy)."""
    st.session_state.portfolio = state["portfolio"]
    st.session_state.ledger_txns = state["ledger_txns"]
    st.session_state.watchlist = state["watchlist"]
    st.session_state.base_currency = state["base_currency"]
    st.session_state.bt_benchmark = state["bt_benchmark"]
    st.session_state["_ws_pid"] = portfolio_id
    st.session_state["_ws_hash"] = workspace_signature(
        state["portfolio"], state["ledger_txns"], state["watchlist"],
        state["base_currency"], state["bt_benchmark"])


def _workspace_resolve_active(conn) -> str:
    """Active portfolio id: session choice wins, else DB pointer, else deflt."""
    wanted = st.session_state.get("_ws_pid")
    if wanted:
        try:
            _rec = ws_repo.get_portfolio(conn, wanted)
            if not _rec.archived:
                return wanted
        except ValueError:
            pass
    try:
        pointed = ws_repo.get_active_portfolio_id(conn)
        if pointed:
            _rec = ws_repo.get_portfolio(conn, pointed)
            if not _rec.archived:
                return pointed
    except ValueError:
        pass
    return ws_repo.DEFAULT_PORTFOLIO_ID


def _workspace_switch_to(portfolio_id: str) -> None:
    """Point session at another portfolio; next lines reload from the DB."""
    st.session_state["_ws_pid"] = portfolio_id
    for _k in ("portfolio", "ledger_txns", "watchlist", "base_currency",
               "bt_benchmark", "ws_switcher", "_ws_hash", "pending_buy",
               "confirm_reset"):
        st.session_state.pop(_k, None)
    st.session_state.data_loaded = False


def _sync_workspace_to_db() -> None:
    """Persist the working copy (no-op when unchanged, invalid or file mode).

    Invalid ledgers are never written: the run keeps going on the session
    copy and the next valid run persists. DB errors flash once in the
    sidebar instead of breaking the run.
    """
    if not st.session_state.get("_db_ok"):
        return
    _pid = st.session_state.get("_ws_pid")
    if not _pid:
        return
    try:
        _txns, _err = parse_session_transactions(
            st.session_state.ledger_txns)
        if _err is not None:
            return
        _sig = workspace_signature(
            st.session_state.portfolio, st.session_state.ledger_txns,
            st.session_state.watchlist,
            st.session_state.get("base_currency", "USD"),
            st.session_state.get("bt_benchmark", "SPY"))
        if _sig == st.session_state.get("_ws_hash"):
            return
        _conn = ws_db.connect(workspace_db_path())
        try:
            ws_repo.save_workspace(
                _conn, _pid, positions=st.session_state.portfolio,
                transactions=_txns,
                watchlist=list(st.session_state.watchlist),
                base_currency=st.session_state.get("base_currency", "USD"),
                benchmark=st.session_state.get("bt_benchmark", "SPY"))
        finally:
            _conn.close()
        st.session_state["_ws_hash"] = _sig
    except Exception as e:
        flash(f"Workspace save failed ({e}); your edits are kept for "
              "this session only.", "warning")


st.session_state["_db_ok"] = False
try:
    _boot_conn = ws_db.connect(workspace_db_path())
    try:
        if not ws_repo.list_portfolios(_boot_conn, include_archived=True):
            _workspace_seed_default(_boot_conn)
            _boot_conn.commit()
        _active_pid = _workspace_resolve_active(_boot_conn)
        try:
            ws_repo.set_active_portfolio_id(_boot_conn, _active_pid)
            _boot_conn.commit()
        except ValueError:
            pass
        if st.session_state.get("_ws_pid") != _active_pid or \
                "portfolio" not in st.session_state:
            _workspace_apply_session(
                workspace_to_session(
                    ws_repo.load_workspace(_boot_conn, _active_pid)),
                _active_pid)
        st.session_state["_db_ok"] = True
    finally:
        _boot_conn.close()
except Exception:
    st.session_state["_db_ok"] = False

if "portfolio" not in st.session_state:
    # File mode (database unavailable): legacy single-portfolio behaviour.
    st.session_state.portfolio = load_portfolio()

if "base_currency" not in st.session_state:
    st.session_state.base_currency = load_settings().get(
        "base_currency", "USD") if not st.session_state.get("_db_ok") \
        else "USD"

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# Task 9 ledger/watchlist session state (JSON-safe shapes only).
if "ledger_txns" not in st.session_state:
    st.session_state.ledger_txns = []
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "ledger_cleared" not in st.session_state:
    st.session_state.ledger_cleared = False
if "bt_benchmark" not in st.session_state:
    st.session_state.bt_benchmark = "SPY"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sidebar — Portfolio Editor
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📈 Portfolio Editor")
    render_flash()  # rerun-surviving confirmations render once, here

    if st.session_state.get("_db_ok"):
        st.markdown('<div class="pc-kicker">WORKSPACE</div>',
                    unsafe_allow_html=True)
        try:
            _ws_conn = ws_db.connect(workspace_db_path())
            try:
                _ws_all = ws_repo.list_portfolios(_ws_conn)
                _ws_arch = [p for p in ws_repo.list_portfolios(
                    _ws_conn, include_archived=True) if p.archived]
            finally:
                _ws_conn.close()
        except Exception as e:
            st.warning(f"Workspace unavailable: {e}")
            _ws_all, _ws_arch = [], []
        _ws_pid_now = st.session_state.get("_ws_pid")
        _ws_ids = [p.portfolio_id for p in _ws_all]
        if _ws_ids:
            if _ws_pid_now not in _ws_ids:
                _ws_pid_now = _ws_ids[0]
            _ws_names = {p.portfolio_id: p.name for p in _ws_all}
            _sel = st.selectbox(
                "Portfolio", _ws_ids, index=_ws_ids.index(_ws_pid_now),
                format_func=lambda _pid: _ws_names.get(_pid, _pid),
                key="ws_switcher",
                help="Each portfolio keeps its own positions, ledger, "
                     "watchlist and settings.")
            if _sel != st.session_state.get("_ws_pid"):
                try:
                    _sw_conn = ws_db.connect(workspace_db_path())
                    try:
                        ws_repo.set_active_portfolio_id(_sw_conn, _sel)
                    finally:
                        _sw_conn.close()
                except ValueError as e:
                    st.error(str(e))
                else:
                    _workspace_switch_to(_sel)
                    st.rerun()
        with st.expander("Manage portfolios", expanded=False):
            _new_name = st.text_input("New portfolio name", value="",
                                      key="ws_new_name",
                                      placeholder="e.g. ISA 2026")
            if st.button("Create + switch", key="ws_create"):
                _label = str(_new_name or "").strip()
                if not _label:
                    st.error("Name the new portfolio first.")
                else:
                    try:
                        _c = ws_db.connect(workspace_db_path())
                        try:
                            _np = ws_repo.create_portfolio(
                                _c, _label,
                                st.session_state.get("base_currency",
                                                     "USD"),
                                st.session_state.get("bt_benchmark",
                                                     "SPY"), 0.05)
                            ws_repo.set_active_portfolio_id(
                                _c, _np.portfolio_id)
                        finally:
                            _c.close()
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        _workspace_switch_to(_np.portfolio_id)
                        flash(f"Switched to '{_label}'.", "success")
                        st.rerun()
            _ren = st.text_input("Rename active", value=str(
                {p.portfolio_id: p.name for p in _ws_all}.get(
                    st.session_state.get("_ws_pid"), "")),
                key="ws_rename")
            if st.button("Rename", key="ws_rename_go"):
                try:
                    _c = ws_db.connect(workspace_db_path())
                    try:
                        ws_repo.rename_portfolio(
                            _c, st.session_state.get("_ws_pid"), _ren)
                    finally:
                        _c.close()
                except ValueError as e:
                    st.error(str(e))
                else:
                    flash("Portfolio renamed.", "success")
                    st.rerun()
            _dup = st.text_input("Duplicate active as", value="", key="ws_dup",
                                 placeholder="e.g. Core copy")
            if st.button("Duplicate + switch", key="ws_dup_go"):
                try:
                    _c = ws_db.connect(workspace_db_path())
                    try:
                        _cp = ws_repo.duplicate_portfolio(
                            _c, st.session_state.get("_ws_pid"), _dup)
                        ws_repo.set_active_portfolio_id(
                            _c, _cp.portfolio_id)
                    finally:
                        _c.close()
                except ValueError as e:
                    st.error(str(e))
                else:
                    _workspace_switch_to(_cp.portfolio_id)
                    flash(f"Switched to '{_cp.name}'.", "success")
                    st.rerun()
            _arch_opts = [p for p in _ws_all
                          if p.portfolio_id != st.session_state.get("_ws_pid")
                          and p.portfolio_id != ws_repo.DEFAULT_PORTFOLIO_ID]
            if _arch_opts:
                _arch_pick = st.selectbox(
                    "Archive (keeps data, hides here)",
                    [p.portfolio_id for p in _arch_opts],
                    format_func=lambda _pid: {
                        p.portfolio_id: p.name for p in _arch_opts
                    }.get(_pid, _pid), key="ws_archive_pick")
                if st.button("Archive", key="ws_archive_go"):
                    try:
                        _c = ws_db.connect(workspace_db_path())
                        try:
                            ws_repo.set_archived(_c, _arch_pick, True)
                        finally:
                            _c.close()
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        flash("Portfolio archived.", "info")
                        st.rerun()
            if _ws_arch:
                _un_pick = st.selectbox(
                    "Unarchive",
                    [p.portfolio_id for p in _ws_arch],
                    format_func=lambda _pid: {
                        p.portfolio_id: p.name for p in _ws_arch
                    }.get(_pid, _pid), key="ws_unarchive_pick")
                if st.button("Unarchive", key="ws_unarchive_go"):
                    try:
                        _c = ws_db.connect(workspace_db_path())
                        try:
                            ws_repo.set_archived(_c, _un_pick, False)
                        finally:
                            _c.close()
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        flash("Portfolio unarchived.", "success")
                        st.rerun()
            _del_opts = [p for p in _ws_all
                         if p.portfolio_id != st.session_state.get("_ws_pid")
                         and p.portfolio_id != ws_repo.DEFAULT_PORTFOLIO_ID]
            if _del_opts:
                _del_pick = st.selectbox(
                    "Delete (forever)",
                    [p.portfolio_id for p in _del_opts],
                    format_func=lambda _pid: {
                        p.portfolio_id: p.name for p in _del_opts
                    }.get(_pid, _pid), key="ws_delete_pick")
                _del_ok = st.checkbox(
                    "I understand this deletes its positions, ledger and "
                    "research forever.", key="ws_delete_ack")
                if st.button("Delete portfolio", key="ws_delete_go",
                             disabled=not _del_ok):
                    try:
                        _c = ws_db.connect(workspace_db_path())
                        try:
                            ws_repo.delete_portfolio(_c, _del_pick)
                        finally:
                            _c.close()
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        flash("Portfolio deleted.", "warning")
                        st.rerun()
    else:
        st.caption("File mode — database unavailable (single portfolio).")

    st.markdown('<div class="pc-kicker">PORTFOLIO</div>',
                unsafe_allow_html=True)
    # --- Base currency (global valuation currency) ---
    if st.session_state.get("base_currency") not in SUPPORTED_BASE_CURRENCIES:
        st.session_state.base_currency = "USD"
    _prev_base = st.session_state.base_currency
    st.selectbox(
        "Base Currency", list(SUPPORTED_BASE_CURRENCIES),
        key="base_currency",
        help="Currency used to value and aggregate the portfolio.")
    if st.session_state.base_currency != _prev_base:
        _persist_settings_now()

    st.markdown("---")

    # --- Add position: Ticker / Shares / Purchase Date ---
    with st.expander("➕ Add / Update Position", expanded=False):
        new_ticker = st.text_input("Ticker", placeholder="e.g. TSLA").upper().strip()
        new_shares = st.number_input("Shares", min_value=0.0, step=0.5, value=1.0)
        _today = date.today()
        buy_date = st.date_input(
            "Purchase Date",
            value=st.session_state.get(
                "new_buy_date", date.today() - timedelta(days=365)),
            max_value=date.today(), key="new_buy_date",
            help="Estimated using the market closing price on the selected "
                 "date. Your actual execution price may have differed.")
        use_override = st.checkbox(
            "Use actual execution price", value=False, key="new_use_override",
            help="Advanced: override the estimate with your broker fill "
                 "(in the security's native currency).")
        new_override = st.number_input(
            "Actual execution price (native)", min_value=0.0, step=0.5,
            value=0.0, key="new_override") if use_override else 0.0

        if st.button("Add Position"):
            try:
                sym = validate_ticker_symbol(new_ticker)
                sh = validate_shares(float(new_shares))
                if use_override and (
                        not np.isfinite(new_override) or new_override <= 0):
                    raise ValueError(
                        "Override price must be positive when enabled.")
            except ValueError as e:
                st.error(str(e))
            else:
                try:
                    inst = get_instrument(sym)
                except ValueError as e:
                    st.error(str(e))
                    inst = None
                if inst is not None:
                    if use_override:
                        _complete_add_position(
                            sym, sh, None, float(new_override), "manual",
                            True, inst)
                    else:
                        try:
                            raw = load_raw_history(sym)
                            splits = load_splits(sym)
                        except ValueError as e:
                            st.error(str(e))
                        else:
                            res = resolve_purchase_price(
                                raw, splits, buy_date, _today)
                            if res["status"] == "ok":
                                # res["price"] is in raw quote units (e.g. GBp)
                                # — normalise to native currency before storing.
                                _complete_add_position(
                                    sym, sh, res["used_date"],
                                    normalise_quote(res["price"],
                                                    inst.quote_scale),
                                    "estimate", False, inst)
                            elif res["status"] == "missing":
                                st.session_state.pending_buy = {
                                    "ticker": sym, "shares": sh,
                                    "inst": {"native_currency": inst.native_currency,
                                             "quote_unit": inst.quote_unit,
                                             "quote_scale": inst.quote_scale},
                                    "prev": [res["prev"][0].isoformat(),
                                             normalise_quote(
                                                 res["prev"][1],
                                                 inst.quote_scale)],
                                    "next": [res["next"][0].isoformat(),
                                             normalise_quote(
                                                 res["next"][1],
                                                 inst.quote_scale)],
                                }
                                st.rerun()
                            else:
                                st.error(res["message"])

        # Pending non-trading-date choice (explicit, never silent).
        pending = st.session_state.get("pending_buy")
        if pending:
            st.info(
                f"No market price exists on the selected date for "
                f"{pending['ticker']}. Choose explicitly:")
            _pdate, _pprice = pending["prev"]
            _ndate, _nprice = pending["next"]
            _nccy = pending["inst"]["native_currency"]
            _opts = {
                f"{_pdate} — prev close {format_money(_pprice, _nccy)}":
                    (_pdate, _pprice),
                f"{_ndate} — next close {format_money(_nprice, _nccy)}":
                    (_ndate, _nprice),
            }
            _choice = st.radio("Trading date", list(_opts),
                               key="pending_buy_choice")
            _c1, _c2 = st.columns(2)
            with _c1:
                if st.button("Use date", key="pending_buy_ok"):
                    _d, _px = _opts[_choice]
                    _complete_add_position(
                        pending["ticker"], pending["shares"],
                        date.fromisoformat(_d), float(_px), "estimate",
                        False, pending["inst"])
                    st.session_state.pending_buy = None
                    st.rerun()
            with _c2:
                if st.button("Cancel", key="pending_buy_cancel"):
                    st.session_state.pending_buy = None
                    st.rerun()

    st.markdown('<div class="pc-kicker">HOLDINGS</div>',
                unsafe_allow_html=True)
    # --- Current positions ---
    to_remove = []
    for ticker, pos in list(st.session_state.portfolio.items()):
        col1, col2 = st.columns([3, 1])
        try:
            _p = position_from_dict(ticker, pos)
            _buy = _p.purchase_date.isoformat() if _p.purchase_date else "no date"
            _src = {"estimate": "est", "manual": "actual",
                    "legacy": "legacy"}.get(
                        _p.purchase_price_source or "", "?")
            col1.markdown(
                f"**{ticker}** — {_p.shares:g} sh · buy {_buy} ({_src})")
        except ValueError:
            col1.markdown(f"**{ticker}** — (invalid entry, remove me)")
        if col2.button("✕", key=f"rm_{ticker}"):
            to_remove.append(ticker)

    for t in to_remove:
        # Lot-aware close: open ledger lots are sold at the current mark
        # (cash-exempt, like classic adds) so realised P&L is recorded and
        # no dangling lots survive. Failures block removal (no partials).
        try:
            _now_txns, _now_err = parse_session_transactions(
                st.session_state.ledger_txns)
            _open = 0.0
            if _now_err is None and _now_txns:
                _open = open_shares_by_ticker(
                    apply_ledger(_now_txns)).get(t, 0.0)
            if _open > 0:
                _marks = fetch_current_prices([t])
                _px = float(_marks.get(t, float("nan")))
                _inst_rm = get_instrument(t)
                _scale = float(_inst_rm.quote_scale or 1.0)
                _nccy = _inst_rm.native_currency or "USD"
                if not np.isfinite(_px) or _px <= 0:
                    raise ValueError("no current mark available")
                _base_now = st.session_state.get("base_currency", "USD")
                _fx = 1.0 if _nccy == _base_now else None
                if _fx is None:
                    _fx = rate_on_date(load_fx_history(), _nccy, _base_now,
                                       date.today())
                _close = validate_transaction(Transaction(
                    transaction_id=new_transaction_id(), date=date.today(),
                    type="SELL", ticker=t, quantity=float(_open),
                    price_native=float(_px) * _scale, currency=_nccy,
                    fee=0.0, purchase_fx=_fx,
                    notes="Holding removed — closed at current mark.",
                    source="remove-close", move_cash=False))
                _keep = list(st.session_state.ledger_txns) + [
                    transaction_to_dict(_close)]
                # Validate the resulting ledger before mutating anything.
                _check, _check_err = parse_session_transactions(_keep)
                if _check_err is None:
                    apply_ledger(_check)
                else:
                    raise ValueError(_check_err)
                st.session_state.ledger_txns = _keep
                flash(f"Removed {t} — {float(_open):g} shares closed at "
                      f"~{float(_px) * _scale:,.2f} {_nccy} (realised "
                      "recorded, cash untouched).", "info")
            else:
                flash(f"Removed {t}.", "info")
        except (ValueError, AttributeError):
            st.warning(f"Could not close {t}'s ledger lots safely — "
                       "holding kept. Use a Sell transaction instead.")
            continue
        del st.session_state.portfolio[t]
        _persist_portfolio_now()
        st.session_state.data_loaded = False
        st.rerun()

    st.markdown('<div class="pc-kicker">TRANSACTIONS</div>',
                unsafe_allow_html=True)
    with st.expander("Record transaction", expanded=False):
        st.caption(
            "Full-form flows move cash (BUY spends, SELL/DIVIDEND add). "
            "Classic Adds above are cash-exempt shortcuts.")
        tx_type = st.selectbox(
            "Type", ["BUY", "SELL", "DIVIDEND", "FEE", "DEPOSIT",
                     "WITHDRAWAL", "INTEREST"],
            key="tx_type",
            help="DIVIDEND/INTEREST record cash income. DEPOSIT/WITHDRAWAL "
                 "are external flows (never investment return).")
        tx_date = st.date_input(
            "Date", value=date.today(), max_value=date.today(), key="tx_date")
        _needs_ticker = tx_type in ("BUY", "SELL", "DIVIDEND", "INTEREST")
        tx_ticker = (st.text_input("Ticker", key="tx_ticker").upper().strip()
                     if _needs_ticker else "")
        if tx_type in ("DEPOSIT", "WITHDRAWAL"):
            tx_qty = st.number_input("Amount", min_value=0.0, step=100.0,
                                     key="tx_qty")
            tx_price, _show_price = 1.0, False
        elif tx_type == "FEE":
            tx_qty = st.number_input("Fee amount", min_value=0.0, step=1.0,
                                     key="tx_qty")
            tx_price, _show_price = 1.0, False
        elif tx_type in ("DIVIDEND", "INTEREST"):
            tx_qty = st.number_input("Shares entitled", min_value=0.0,
                                     step=1.0, key="tx_qty")
            tx_price = st.number_input("Per-share amount (native)",
                                       min_value=0.0, step=0.1,
                                       key="tx_price")
            _show_price = True
        else:
            tx_qty = st.number_input("Shares", min_value=0.0, step=0.5,
                                     key="tx_qty")
            tx_price = st.number_input("Execution price (native)",
                                       min_value=0.0, step=0.5,
                                       key="tx_price")
            _show_price = True
        tx_ccy = st.selectbox("Currency", ["USD", "GBP", "EUR"], key="tx_ccy",
                              help="Native transaction currency = cash "
                                   "currency moved.")
        tx_fee = st.number_input("Fee (native)", min_value=0.0, step=1.0,
                                 key="tx_fee") \
            if tx_type in ("BUY", "SELL") else 0.0
        if tx_type == "BUY" and tx_ticker:
            if st.button("Estimate price from history", key="tx_estimate"):
                try:
                    _e_inst = get_instrument(
                        validate_ticker_symbol(tx_ticker))
                    _e_raw = load_raw_history(
                        validate_ticker_symbol(tx_ticker))
                    _e_res = resolve_purchase_price(
                        _e_raw, load_splits(
                            validate_ticker_symbol(tx_ticker)),
                        tx_date, date.today())
                    if _e_res["status"] == "ok":
                        st.session_state.tx_price = float(normalise_quote(
                            _e_res["price"], _e_inst.quote_scale))
                        st.rerun()
                    else:
                        st.warning(
                            "No estimate: "
                            + _e_res.get("message", "unavailable"))
                except ValueError as e:
                    st.error(str(e))
        if tx_type == "SELL" and tx_ticker:
            try:
                _pv_txns, _pv_err = parse_session_transactions(
                    st.session_state.ledger_txns)
                _avail = open_shares_by_ticker(
                    apply_ledger(_pv_txns)).get(tx_ticker, 0.0) \
                    if _pv_err is None and _pv_txns else 0.0
                st.caption(f"Available: {_avail:g} shares of {tx_ticker}.")
            except ValueError:
                st.caption("Ledger preview unavailable.")
        if st.button("Record Transaction", key="tx_save"):
            try:
                _sym = (validate_ticker_symbol(tx_ticker)
                        if _needs_ticker else None)
                _base_now = st.session_state.get("base_currency", "USD")
                _fx = 1.0 if tx_ccy == _base_now else None
                if _fx is None:
                    try:
                        _fx = rate_on_date(load_fx_history(), tx_ccy,
                                           _base_now, tx_date)
                    except ValueError:
                        _fx = None
                _ntx = validate_transaction(Transaction(
                    transaction_id=new_transaction_id(), date=tx_date,
                    type=tx_type, ticker=_sym,
                    quantity=float(tx_qty),
                    price_native=float(tx_price) if _show_price else 1.0,
                    currency=tx_ccy, fee=float(tx_fee), purchase_fx=_fx,
                    notes="", source="user", move_cash=True,
                    created_at=datetime.now(timezone.utc)))
                _all, _parse_err = parse_session_transactions(
                    st.session_state.ledger_txns)
                if _parse_err is not None:
                    raise ValueError(_parse_err)
                _trial = _all + [_ntx]
                _st = apply_ledger(_trial)  # validates cash/sell limits first
                st.session_state.ledger_txns = [
                    transaction_to_dict(t) for t in _trial]
                _extra = ""
                if tx_type == "SELL":
                    _hit = [e for e in _st.realised
                            if e.transaction_id == _ntx.transaction_id]
                    if _hit:
                        _extra = (f" Realised "
                                  f"{_hit[0].realised_pnl_native:+,.2f} "
                                  f"{tx_ccy} (est.).")
                flash(f"Recorded {tx_type} {_extra}".rstrip() + ".", "success")
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    st.markdown('<div class="pc-kicker">WATCHLIST</div>',
                unsafe_allow_html=True)
    _wl_new = st.text_input("Add symbol", key="wl_new",
                            placeholder="e.g. TSLA").upper().strip()
    if st.button("Add to watchlist", key="wl_add"):
        try:
            _wl_sym = validate_ticker_symbol(_wl_new)
            if _wl_sym not in st.session_state.watchlist:
                st.session_state.watchlist = sorted(
                    st.session_state.watchlist + [_wl_sym])
                flash(f"Watching {_wl_sym}.", "success")
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    for _w in list(st.session_state.watchlist):
        _c1, _c2 = st.columns([3, 1])
        _c1.markdown(f"**{_w}**")
        if _c2.button("✕", key=f"wl_rm_{_w}"):
            st.session_state.watchlist = [
                x for x in st.session_state.watchlist if x != _w]
            flash(f"Stopped watching {_w}.", "info")
            st.rerun()

    st.markdown('<div class="pc-kicker">DATA</div>',
                unsafe_allow_html=True)
    with st.expander("Market data", expanded=False):
        if st.button("🔄 Refresh Market Data"):
            st.session_state.data_loaded = False
            for _cache_fn in (load_market_data, load_benchmark_data,
                              load_backtest_data, load_factor_data,
                              load_exchange_hint, load_alpaca_trades,
                              load_instruments, load_fx_history, load_current_fx,
                              load_raw_history, load_splits, load_dividends_history):
                try:
                    _cache_fn.clear()
                except Exception:
                    pass
            flash("Market data refreshed.", "success")
            st.rerun()
        st.caption(
            "Current US prices: Alpaca IEX (Yahoo fallback). Historical "
            "prices, FX: Yahoo Finance. Factors: Kenneth French library. "
            "Charts: TradingView (display only).")

    st.markdown('<div class="pc-kicker">FILES</div>',
                unsafe_allow_html=True)
    # --- Export / Import (schema v2: positions + ledger + watchlist) ---
    with st.expander("Import / export / reset", expanded=False):
        st.markdown("### Portfolio Data")
        try:
            _exp_txns, _exp_err = parse_session_transactions(
                st.session_state.ledger_txns)
            if _exp_err is not None:
                raise ValueError(_exp_err)
            _export_doc = export_portfolio_v2(
                st.session_state.portfolio,
                st.session_state.get("base_currency", "USD"),
                _exp_txns, list(st.session_state.watchlist),
                {"benchmark": st.session_state.get("bt_benchmark", "SPY")})
            st.download_button(
                "Export Portfolio (JSON)", data=json.dumps(_export_doc, indent=2),
                file_name="portcalc-portfolio-v2.json", mime="application/json",
                key="export_json")
        except ValueError as e:
            st.warning(f"Export unavailable: {e}")
        _uploaded = st.file_uploader("Import Portfolio (JSON)", type=["json"],
                                     key="import_json")
        if _uploaded is not None:
            if st.session_state.get("_db_ok"):
                _imp_mode = st.radio(
                    "Import target",
                    ["Replace active portfolio", "Create new portfolio"],
                    key="import_mode",
                    help="Replace overwrites this portfolio only — other "
                         "portfolios are untouched. Create keeps everything "
                         "and switches to the new copy.")
                _imp_new_name = st.text_input(
                    "New portfolio name", value="Imported",
                    key="import_new_name") \
                    if _imp_mode == "Create new portfolio" else ""
            else:
                _imp_mode, _imp_new_name = "Replace active portfolio", ""
            if st.button("Apply import", key="import_apply"):
                try:
                    _doc = json.load(_uploaded)
                    _imp = validate_portfolio_import(_doc)
                except (ValueError, json.JSONDecodeError) as e:
                    st.error(f"Import rejected: {e}")
                except Exception:
                    st.error("Import rejected: unreadable file.")
                else:
                    # Import REPLACES all state (Programme 176: no merges,
                    # so re-importing can never duplicate ledger entries).
                    _imp_base = _imp.base_currency if (
                        _imp.base_currency in SUPPORTED_BASE_CURRENCIES
                    ) else None
                    _imp_bench = str(
                        _imp.settings.get("benchmark") or "SPY")
                    if _imp_mode == "Create new portfolio" and \
                            st.session_state.get("_db_ok"):
                        _label = str(_imp_new_name or "").strip() or "Imported"
                        try:
                            _c = ws_db.connect(workspace_db_path())
                            try:
                                _nip = ws_repo.create_portfolio(
                                    _c, _label, _imp_base or
                                    st.session_state.get("base_currency",
                                                         "USD"),
                                    _imp_bench, 0.05)
                                ws_repo.save_workspace(
                                    _c, _nip.portfolio_id,
                                    positions=_imp.positions,
                                    transactions=list(_imp.transactions),
                                    watchlist=list(_imp.watchlist),
                                    base_currency=_imp_base,
                                    benchmark=_imp_bench)
                                ws_repo.set_active_portfolio_id(
                                    _c, _nip.portfolio_id)
                            finally:
                                _c.close()
                        except ValueError as e:
                            st.error(f"Import rejected: {e}")
                        else:
                            _workspace_switch_to(_nip.portfolio_id)
                            flash(f"Imported into new portfolio '{_label}' "
                                  f"({len(_imp.positions)} positions, "
                                  f"{len(_imp.transactions)} transactions).",
                                  "success")
                            st.rerun()
                    else:
                        st.session_state.portfolio = _imp.positions
                        st.session_state.ledger_txns = [
                            transaction_to_dict(t)
                            for t in _imp.transactions]
                        st.session_state.watchlist = list(_imp.watchlist)
                        st.session_state.ledger_cleared = False
                        if _imp_base:
                            st.session_state.base_currency = _imp_base
                            _persist_settings_now()
                        if _imp.settings.get("benchmark"):
                            st.session_state.bt_benchmark = _imp_bench
                        _persist_portfolio_now()
                        st.session_state.data_loaded = False
                        flash(f"Imported {len(_imp.positions)} positions, "
                              f"{len(_imp.transactions)} transactions "
                              f"(base {_imp.base_currency}).", "success")
                        st.rerun()

        # --- Reset with intentional confirmation (active portfolio only) ---
        _reset_name = "this portfolio"
        if st.session_state.get("_db_ok"):
            try:
                _rc = ws_db.connect(workspace_db_path())
                try:
                    _reset_name = ws_repo.get_portfolio(
                        _rc, st.session_state.get("_ws_pid")).name
                finally:
                    _rc.close()
            except (ValueError, TypeError):
                pass
        if st.button(f"Reset '{_reset_name}'", key="reset_start"):
            st.session_state.confirm_reset = True
            st.rerun()
        if st.session_state.get("confirm_reset"):
            st.warning(f"Delete ALL positions, transactions, cash and "
                       f"watchlist in '{_reset_name}'? Other portfolios are "
                       "untouched. This cannot be undone.")
            _r1, _r2 = st.columns(2)
            with _r1:
                if st.button("Confirm reset", key="reset_confirm"):
                    st.session_state.portfolio = {}
                    st.session_state.ledger_txns = []
                    st.session_state.watchlist = []
                    st.session_state.ledger_cleared = False
                    _persist_portfolio_now()
                    st.session_state.confirm_reset = False
                    st.session_state.data_loaded = False
                    flash(f"'{_reset_name}' reset — positions, ledger and "
                          "watchlist removed.", "warning")
                    st.rerun()
            with _r2:
                if st.button("Cancel", key="reset_cancel"):
                    st.session_state.confirm_reset = False
                    st.rerun()

    st.markdown('<div class="pc-kicker">ADVANCED</div>',
                unsafe_allow_html=True)
    with st.expander("Assumptions", expanded=False):
        st.caption(
            "Adjusted closes (splits/dividends per Yahoo). End-of-day "
            "execution approximation; fractional shares; no market impact "
            "beyond explicit bps costs. Risk-free rate fixed at 5% across "
            "historical windows. French factors are US datasets. "
            "TradingView charts are display-only. "
            "Marks: fresh bid/ask mid, else last trade, else Yahoo "
            "fallback (previous close when needed) — never invented. "
            "Ledger: FIFO lots; fees adjust basis/proceeds; cash is "
            "currency-keyed; negative cash rejected. TWR excludes external "
            "flows; XIRR needs both flow signs. No orders are ever sent.")


# Persist sidebar mutations before any early stop below (empty portfolio
# or market-data failure must not lose the working copy).
_sync_workspace_to_db()

try:
    tickers = canonical_tickers(list(st.session_state.portfolio.keys()))
except ValueError as e:
    st.warning(f"Portfolio problem: {e} Add at least one valid position in the sidebar.")
    st.stop()

if not tickers:
    st.warning("Add at least one position in the sidebar to get started.")
    st.stop()

# De-duplicated tickers are canonicalised; warn if the raw keys differed.
if len(tickers) != len(st.session_state.portfolio):
    st.info("Duplicate tickers were merged (symbols are case-insensitive).")

with st.spinner("Loading market data…"):
    # Default history period — can be overridden per-tab
    try:
        prices, current_prices, rf_rate = load_market_data(tickers, "2y")
    except ValueError as e:
        st.error(f"Market data error: {e}")
        st.stop()
    except Exception:
        st.error(
            "Could not load market data (network or API error). "
            "Check your connection and click Refresh Market Data to retry."
        )
        st.stop()

# Current valuation prices: Alpaca latest trade for supported US equities,
# Yahoo fallback otherwise. Historical `prices` series are untouched.
valuation_prices, price_view, alpaca_status = resolve_valuation_prices(
    tickers, current_prices)

# Multi-currency layer: quote-normalise native data, convert history and
# current prices into the base currency. All downstream analytics operate
# on the base-currency panel — never on mixed native currencies.
base_ccy = st.session_state.get("base_currency", "USD")
try:
    ccy = build_currency_layer(tickers, prices, valuation_prices, base_ccy)
except ValueError as e:
    st.error(f"Currency setup: {e}")
    st.stop()
native_panel = ccy["native_panel"]
instruments = ccy["instruments"]

try:
    core = build_core_analytics(tickers, ccy["base_panel"],
                                ccy["base_current"], rf_rate)
except ValueError as e:
    st.error(f"Data error: {e}")
    st.stop()

prices = core["prices"]
current_prices = core["current_prices"]
rf_rate = core["rf_rate"]
weights = core["weights"]
portfolio_value = core["portfolio_value"]
rets = core["rets"]
mu = core["mu"]
cov = core["cov"]
corr = core["corr"]
port_return = core["port_return"]
port_vol = core["port_vol"]
port_sharpe = core["port_sharpe"]
port_daily_returns = core["port_daily_returns"]
rc = core["rc"]

# Central position valuations (single path reused by every tab).
# Native current prices, quote-normalised; base values come from the engine.
_native_current_norm = {
    t: float(valuation_prices.get(t, float("nan")))
    * ccy["instruments"][t].quote_scale for t in tickers}
try:
    positions, valuations, val_agg, val_warnings = build_position_valuations(
        tickers, ccy["instruments"], _native_current_norm, price_view,
        ccy["fx_hist"], ccy["fx_now"], base_ccy)
except ValueError as e:
    st.error(f"Valuation error: {e}")
    st.stop()
val_by_ticker = {v.ticker: v for v in valuations}
pos_by_ticker = positions

# ---------------------------------------------------------------------------
# Task 9 — ledger build (migration + FX resolution + derived state)
# ---------------------------------------------------------------------------

_ledger_view = build_ledger_view(
    st.session_state.ledger_txns, st.session_state.portfolio, base_ccy,
    ccy.get("fx_hist"), cleared=st.session_state.ledger_cleared)
if _ledger_view["migrated"]:
    st.session_state.ledger_txns = [
        transaction_to_dict(t) for t in _ledger_view["transactions"]]
ledger_notes: list = list(_ledger_view["notes"])
ledger_error: str | None = _ledger_view["error"]
_ledger_txns = _ledger_view["transactions"]
ledger_state = _ledger_view["state"]
ledger_open_shares = _ledger_view["open_shares"]
ledger_cash = _ledger_view["cash"]

# ---------------------------------------------------------------------------
# Task 9 — live mark snapshot (one consistent read for all current cards)
# ---------------------------------------------------------------------------

from data.pipeline import build_live_snapshot, load_previous_closes

try:
    _prev_closes = load_previous_closes(tuple(tickers))
except ValueError:
    _prev_closes = {}
try:
    _watch_syms = [w for w in st.session_state.watchlist
                   if w not in tickers]
    live_snapshot = build_live_snapshot(
        tickers + _watch_syms, current_prices, _prev_closes)
except Exception:
    live_snapshot = {"_status": {"feed": None,
                                 "feed_note": "snapshot unavailable",
                                 "n_symbols": 0, "at": ""}}

# Live securities value from marks (native marks × current FX) + cash = NAV.
# Legacy Yahoo-based valuations above stay untouched for cost/P&L analytics.
live_holdings: dict = {}
live_day_pnl_native: dict = {}
for _t in tickers:
    _m = live_snapshot.get(_t, {})
    _mark = _m.get("mark")
    _sh = float(ledger_open_shares.get(
        _t, st.session_state.portfolio.get(_t, {}).get("shares", 0.0)) or 0.0)
    _inst = ccy["instruments"][_t]
    _fx_now = ccy["fx_now"].get(_inst.native_currency)
    if _mark is None or not np.isfinite(float(_mark)) or _sh <= 0 \
            or _fx_now is None or not np.isfinite(float(_fx_now)) \
            or float(_fx_now) <= 0:
        live_holdings[_t] = {"native_value": None, "base_value": None,
                             "day_pnl_base": None, "shares": _sh}
        continue
    _native_value = _sh * float(_mark) * float(_inst.quote_scale)
    live_holdings[_t] = {
        "native_value": _native_value,
        "base_value": _native_value * float(_fx_now),
        "shares": _sh,
        "day_pnl_base": None,
    }
    _prev = _m.get("prev_close")
    if _prev is not None and float(_prev) > 0:
        _day_native = (float(_mark) - float(_prev)) * _sh \
            * float(_inst.quote_scale)
        live_holdings[_t]["day_pnl_base"] = _day_native * float(_fx_now)
_cash_base = 0.0
for _ccy, _bal in ledger_cash.items():
    _r = ccy["fx_now"].get(_ccy)
    if _r is not None and np.isfinite(float(_r)) and float(_r) > 0:
        _cash_base += float(_bal) * float(_r)
_live_sec_base = sum(v["base_value"] for v in live_holdings.values()
                    if v["base_value"] is not None)
portfolio_nav = _live_sec_base + _cash_base
day_pnl_base = sum(v["day_pnl_base"] for v in live_holdings.values()
                   if v["day_pnl_base"] is not None)
live_incomplete = [t for t, v in live_holdings.items()
                   if v["base_value"] is None]

# Shared render context for ui/views tabs (single source of shell state).
# Overview may return an updated dict (history-period rebuild) merged back.
ctx = {
    "tickers": tickers,
    "base_ccy": base_ccy,
    "ccy": ccy,
    "prices": prices,
    "current_prices": current_prices,
    "rf_rate": rf_rate,
    "weights": weights,
    "portfolio_value": portfolio_value,
    "rets": rets,
    "mu": mu,
    "cov": cov,
    "corr": corr,
    "port_return": port_return,
    "port_vol": port_vol,
    "port_sharpe": port_sharpe,
    "port_daily_returns": port_daily_returns,
    "rc": rc,
    "positions": positions,
    "valuations": valuations,
    "val_agg": val_agg,
    "val_warnings": val_warnings,
    "val_by_ticker": val_by_ticker,
    "pos_by_ticker": pos_by_ticker,
    "valuation_prices": valuation_prices,
    "price_view": price_view,
    "alpaca_status": alpaca_status,
    "instruments": instruments,
    "native_panel": native_panel,
    "core": core,
    "live_snapshot": live_snapshot,
    "live_holdings": live_holdings,
    "ledger_state": ledger_state,
    "ledger_error": ledger_error,
    "ledger_notes": ledger_notes,
    "ledger_cash": ledger_cash,
    "portfolio_nav": portfolio_nav,
    "day_pnl_base": day_pnl_base,
    "live_incomplete": live_incomplete,
    "watchlist": list(st.session_state.watchlist),
}

# ---------------------------------------------------------------------------
# Application shell header (compact context — details live in the tabs)
# ---------------------------------------------------------------------------

from ui.formatting import format_date, format_money, format_signed_money

_header_bench = st.session_state.get("bt_benchmark", "SPY")
try:
    _header_through = format_date(prices.index[-1])
except Exception:
    _header_through = "—"
st.markdown("## Portfolio Calc")
st.caption("Portfolio Research & Risk")
_h1, _h2, _h3, _h4 = st.columns(4)
_h1.metric("NAV", format_money(portfolio_nav, base_ccy),
           help="Live securities value (mark prices) + cash.")
_h2.metric("Day P&L", format_signed_money(day_pnl_base, base_ccy)
           if live_holdings else "—",
           help="Mark vs previous official close (pre/post-market marks "
                "compared against the last close; see methodology).")
_h3.metric("Cash", format_money(_cash_base, base_ccy),
           help="Currency-keyed cash from the transaction ledger.")
_h4.metric("Base Currency", base_ccy)
_live_note = live_snapshot.get("_status", {}).get("feed_note", "")
st.caption(f"Benchmark {str(_header_bench)} · Data through {_header_through} "
           f"· Live: {_live_note}"
           + (" · Partial marks: " + ", ".join(live_incomplete)
              if live_incomplete else ""))
st.markdown("---")

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_overview, tab_risk, tab_optimise, tab_scenario, tab_perf, tab_security, tab_backtest, tab_factors = st.tabs([
    "📈 Overview",
    "⚠️  Risk",
    "🎯 Optimise",
    "📊 Scenario Analysis",
    "📉 Performance",
    "🔍 Security Detail",
    "🧪 Backtest",
    "📐 Factor Analysis",
])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tab_overview:
    _ov_out = render_section("Overview", views_overview.render, ctx)
    if isinstance(_ov_out, dict):
        ctx.update(_ov_out)
# TAB 2 — RISK
# ============================================================
with tab_risk:
    render_section("Risk", views_risk.render, ctx)
# TAB 3 — OPTIMISE
# ============================================================
with tab_optimise:
    render_section("Optimise", views_optimise.render, ctx)
# TAB 4 — SCENARIO ANALYSIS
# ============================================================
with tab_scenario:
    render_section("Scenario Analysis", views_scenario.render, ctx)
# TAB 5 — PERFORMANCE vs BENCHMARK
# ============================================================
with tab_perf:
    render_section("Performance", views_performance.render, ctx)
# TAB 6 — SECURITY DETAIL
# ============================================================
with tab_security:
    render_section("Security Detail", views_security.render, ctx)
# TAB 7 — BACKTEST
# ============================================================
with tab_backtest:
    render_section("Backtest", views_backtest.render, ctx)
# TAB 8 — FACTOR ANALYSIS
# ============================================================
with tab_factors:
    render_section("Factor Analysis", views_factors.render, ctx)

# End-of-run persist: ledger auto-migration and in-tab edits (Overview
# delete) land here; hash-guarded so clean runs are a cheap no-op.
_sync_workspace_to_db()

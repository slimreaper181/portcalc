"""storage.database — connection factory, paths, atomic transactions."""

from __future__ import annotations

import contextlib
import os
import sqlite3
from pathlib import Path

from storage.migrations import migrate


def default_db_path() -> str:
    """Predictable local location (env override for tests/portability)."""
    override = os.environ.get("PORTCALC_DB", "").strip()
    if override:
        return override
    home = Path.home()
    try:
        target = home / ".portcalc" / "portcalc.db"
        target.parent.mkdir(parents=True, exist_ok=True)
        return str(target)
    except OSError:
        # Last resort: repo-local file (documented; keep out of git).
        return str(Path(__file__).resolve().parent.parent / ".portcalc.db")


def connect(path: str | None = None) -> sqlite3.Connection:
    """Open the database (creating + migrating schema as needed)."""
    db_path = path or default_db_path()
    parent = os.path.dirname(os.path.abspath(db_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Autocommit: single-statement repository writes are durable across the
    # short-lived connections the app opens per operation; multi-statement
    # batches go through atomic() (explicit BEGIN / savepoint + commit).
    conn.isolation_level = None
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    migrate(conn)
    return conn


@contextlib.contextmanager
def atomic(path: str | None = None, conn: sqlite3.Connection | None = None):
    """Atomic multi-row writes: commit on success, rollback on any error.

    Uses a savepoint when the connection already holds a transaction
    (nested use from tests/UI flows) and a full BEGIN otherwise, so the
    helper is safe in both situations.
    """
    own = False
    if conn is None:
        conn = connect(path)
        own = True
    if conn.in_transaction:
        sp = "portcalc_sp"
        conn.execute(f"SAVEPOINT {sp}")
        try:
            yield conn
        except Exception:
            try:
                conn.execute(f"ROLLBACK TO {sp}")
                conn.execute(f"RELEASE {sp}")
            except Exception:
                pass
            raise
        else:
            try:
                conn.execute(f"RELEASE {sp}")
            except Exception:
                pass
        finally:
            if own:
                conn.close()
        return
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        if own:
            conn.close()

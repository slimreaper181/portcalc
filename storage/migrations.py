"""storage.migrations — deterministic, idempotent schema migrations."""

from __future__ import annotations

SCHEMA_V1 = """CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    base_currency TEXT NOT NULL DEFAULT 'USD',
    benchmark TEXT NOT NULL DEFAULT 'SPY',
    risk_free REAL NOT NULL DEFAULT 0.05,
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    archived INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS positions (
    portfolio_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    shares REAL NOT NULL,
    purchase_date TEXT,
    purchase_price_native REAL,
    purchase_price_source TEXT,
    manual_purchase_price INTEGER NOT NULL DEFAULT 0,
    native_currency TEXT,
    quote_unit TEXT,
    quote_scale REAL NOT NULL DEFAULT 1.0,
    legacy_avg_cost REAL,
    PRIMARY KEY (portfolio_id, ticker)
);
CREATE TABLE IF NOT EXISTS transactions (
    rowid_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id TEXT NOT NULL,
    transaction_id TEXT NOT NULL,
    date TEXT NOT NULL,
    type TEXT NOT NULL,
    ticker TEXT,
    quantity REAL NOT NULL,
    price_native REAL NOT NULL,
    currency TEXT NOT NULL,
    fee REAL NOT NULL DEFAULT 0,
    purchase_fx REAL,
    notes TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'user',
    move_cash INTEGER NOT NULL DEFAULT 1,
    created_at TEXT,
    UNIQUE (portfolio_id, transaction_id)
);
CREATE TABLE IF NOT EXISTS watchlist (
    portfolio_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    added_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (portfolio_id, ticker)
);
CREATE TABLE IF NOT EXISTS saved_analyses (
    analysis_id TEXT PRIMARY KEY,
    portfolio_id TEXT NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT NOT NULL DEFAULT '{}',
    tags TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_tx_portfolio ON transactions (portfolio_id);
CREATE INDEX IF NOT EXISTS idx_tx_ticker ON transactions (ticker);
CREATE INDEX IF NOT EXISTS idx_saved_portfolio ON saved_analyses (portfolio_id);
"""

SCHEMA_V2 = """CREATE TABLE IF NOT EXISTS revisions (
    revision_id TEXT PRIMARY KEY,
    transaction_id TEXT NOT NULL,
    portfolio_id TEXT NOT NULL,
    action TEXT NOT NULL,
    before_json TEXT,
    after_json TEXT,
    created_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_rev_txn ON revisions (transaction_id);
CREATE INDEX IF NOT EXISTS idx_rev_portfolio ON revisions (portfolio_id);
"""

# (version, ddl) in ascending order. Each entry must be idempotent
# (IF NOT EXISTS) so migrate() is safe to re-run.
MIGRATIONS: list[tuple[int, str]] = [
    (1, SCHEMA_V1),
    (2, SCHEMA_V2),
]


def migrate(conn, target: int | None = None) -> int:
    """Apply pending migrations in order; returns the schema version."""
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")
    cur.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version")
    current = int(cur.fetchone()[0])
    latest = max(v for v, _ in MIGRATIONS)
    stop = latest if target is None else int(target)
    for version, ddl in sorted(MIGRATIONS):
        if current < version <= stop:
            cur.executescript(ddl)
            cur.execute("INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                        (version,))
            current = version
    conn.commit()
    return current


def schema_version(conn) -> int:
    """Current schema version (0 when never migrated)."""
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)")
    cur.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version")
    return int(cur.fetchone()[0])

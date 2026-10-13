"""storage — durable local persistence (SQLite, no ORM, no secrets).

The database is canonical for portfolios, ledger transactions, watchlist
and (later) saved research. Session state mirrors the active portfolio;
every mutation writes through. Portfolio JSON files remain the portable
export/import format — the database never traps user data.
"""

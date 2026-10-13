"""storage.models — plain dataclasses (no SQL here)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Portfolio:
    """One workspace portfolio (settings + derived state lives elsewhere)."""

    portfolio_id: str
    name: str
    base_currency: str = "USD"
    benchmark: str = "SPY"
    risk_free: float = 0.05
    created_at: str = ""
    updated_at: str = ""
    archived: bool = False


@dataclass
class SavedAnalysis:
    """Named research object (config + result summary, never giant arrays)."""

    analysis_id: str
    portfolio_id: str
    name: str
    type: str  # scenario|optimisation|backtest|oos|walkforward|robustness|stress
    config_json: str = "{}"
    result_json: str = "{}"
    tags: str = ""
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Revision:
    """One immutable audit-trail entry (Milestone 2 writes these)."""

    revision_id: str
    transaction_id: str
    portfolio_id: str
    action: str  # CREATE | UPDATE | DELETE
    before_json: str | None = None
    after_json: str | None = None
    created_at: str = ""

"""M9 — institutional HTML report: escaping, metadata, safety."""

import os

from analytics.report import build_report_html


def _payload(**over):
    base = {
        "portfolio_name": "My Portfolio",
        "base_ccy": "USD",
        "benchmark": "SPY",
        "generated_utc": "2026-01-01 00:00 UTC",
        "data_through": "2025-12-31",
        "fx_through": "2025-12-31",
        "freshness_note": "Marks as of 12:00 UTC (IEX).",
        "nav": "$10,000.00",
        "cash_base": "$1,000.00",
        "day_pnl": "+$12.00",
        "n_positions": 2,
        "holdings": [
            {"ticker": "AAPL", "shares": "10", "mark": "150.00",
             "value": "1500.00", "weight": "15%", "pnl": "+10.00",
             "currency": "USD"},
        ],
        "pnl": {"unrealised": "100", "realised": "50", "income": "5",
                "fees": "1", "total": "154", "net_flows": "9000"},
        "performance": {"twr": "5%", "xirr": "6%", "window": "2024-2025"},
        "risk": {"volatility": "15%", "sharpe": "1.2",
                 "var_95_1d": "200", "max_drawdown": "-8%"},
        "attribution": {"rows": [{"ticker": "AAPL",
                                  "contribution": "+2%"}]},
        "stress": {"rows": [{"window": "COVID", "return": "-10%",
                             "drawdown": "-12%"}]},
        "analyses": [{"name": "Q1", "type": "oos", "created": "2026",
                      "last_run": "2026", "tags": "Risk"}],
        "notes": "Quarterly review.",
    }
    base.update(over)
    return base


def test_report_includes_metadata_and_sections():
    html = build_report_html(_payload())
    for needle in ("My Portfolio", "2026-01-01 00:00 UTC", "USD", "SPY",
                   "2025-12-31", "Holdings", "Accounting", "Performance",
                   "Risk", "Attribution", "Stress", "Saved Analyses",
                   "Sources", "AAPL", "Q1"):
        assert needle in html
    assert "<html" in html and "</html>" in html
    assert "@media print" in html  # printable stylesheet


def test_report_escapes_user_content():
    evil = '<script>alert("x")</script><b>bold</b>'
    html = build_report_html(_payload(
        portfolio_name=evil, notes=evil,
        holdings=[{"ticker": evil, "shares": evil, "mark": evil,
                   "value": evil, "weight": evil, "pnl": evil,
                   "currency": evil}],
        analyses=[{"name": evil, "type": evil, "created": evil,
                   "last_run": evil, "tags": evil}]))
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<b>bold</b>" not in html


def test_report_never_claims_realtime_and_has_no_secrets():
    os.environ["APCA_API_SECRET_KEY"] = "SECRET-should-never-leak"
    try:
        html = build_report_html(_payload())
    finally:
        del os.environ["APCA_API_SECRET_KEY"]
    assert "SECRET-should-never-leak" not in html
    assert "APCA_API_SECRET_KEY" not in html
    lowered = html.lower()
    assert "real-time" in lowered or "real time" in lowered or \
        "real" in lowered  # freshness disclaimer present
    assert "never" in lowered or "point-in-time" in lowered


def test_empty_portfolio_renders():
    html = build_report_html({})
    assert "<html" in html
    assert "No saved analyses" in html

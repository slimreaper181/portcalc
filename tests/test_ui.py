"""Tests for ui/ presentation helpers (all pure — no Streamlit context)."""

import pandas as pd
import pytest

from ui.charts import (base_layout, centered_bars, horizontal_bars,
                       line_chart, palette, stacked_area)
from ui.errors import error_card  # import-time only (render needs context)
from ui.feedback import flash, render_flash, take_flash
from ui.formatting import (format_date, format_int, format_money,
                           format_number, format_percent, format_ratio,
                           format_signed_money, format_signed_percent)
from ui.insights import MAX_INSIGHTS, build_insights
from ui.state import ResultCache
from ui.tables import money_table, with_currency
from ui.theme import CHART_THEME, TOKENS, tokens


def test_format_money_edges():
    assert format_money(1234.5, "USD") == "$1,234.50"
    assert format_money(-42.1, "GBP").startswith("-")
    assert format_money(None, "USD") == "—"
    assert format_money(float("nan"), "USD") == "—"
    assert format_money(float("inf"), "USD") == "—"
    assert format_money("garbage", "USD") == "—"


def test_format_signed_and_percent():
    assert format_signed_money(1.2, "USD").startswith("+")
    assert "−" in format_signed_money(-0.4, "USD")
    assert format_signed_money(None, "USD") == "—"
    assert format_percent(0.1234) == "12.34%"
    assert format_signed_percent(0.0321).startswith("+")
    assert "−" in format_signed_percent(-0.0105)
    assert format_percent(None) == "—"
    assert format_ratio(1.234) == "1.23"
    assert format_ratio(float("nan")) == "—"
    assert format_number(1234567.891) == "1,234,567.89"
    assert format_int(42.9) == "42"
    assert format_date("2026-08-13") == "2026-08-13"
    assert format_date(None) == "—"


def test_flash_survives_and_clears_once():
    store = {}
    flash("Position added", "success", store=store)
    flash("Heads up", "warning", store=store)
    assert len(store["_pc_flash"]) == 2
    first = take_flash(store=store)
    assert [m["kind"] for m in first] == ["success", "warning"]
    assert take_flash(store=store) == []  # rendered once
    with pytest.raises(ValueError):
        flash("x", "celebration", store=store)
    flash("", "info", store=store)  # empty messages ignored
    assert take_flash(store=store) == []


def test_result_cache_keys():
    store = {}
    cache = ResultCache("wf", store=store)
    assert cache.get(("a", 1)) is None
    assert cache.key_changed(("a", 1)) is True
    cache.put(("a", 1), {"v": 5})
    assert cache.get(("a", 1)) == {"v": 5}
    assert cache.key_changed(("a", 1)) is False
    assert cache.key_changed(("a", 2)) is True  # any input change invalidates
    assert cache.get(("a", 2)) is None


def test_insights_capped_ordered_and_advice_free():
    facts = {
        "n_assets": 5, "largest_weight": 0.35, "largest_ticker": "NVDA",
        "top3_weight": 0.67, "max_rc_ticker": "NVDA", "max_rc_share": 0.45,
        "max_rc_weight": 0.20, "max_corr_pair": "AAPL/MSFT",
        "max_corr_value": 0.91, "beta": 1.3, "benchmark": "SPY",
        "max_drawdown": -0.18, "fx_top_ccy": "USD", "fx_top_share": 0.76,
        "effective_holdings": 2.4,
    }
    out = build_insights(facts)
    assert 1 <= len(out) <= MAX_INSIGHTS
    rank = {"notable": 0, "watch": 1, "info": 2}
    assert [rank[i.severity] for i in out] == sorted(
        rank[i.severity] for i in out)
    blob = " ".join(i.text.lower() for i in out)
    for word in ("buy", "sell", "recommendation", "should"):
        assert word not in blob
    assert build_insights({}) == []
    assert build_insights(None) == []


def test_insights_advice_guard_raises_without_assert():
    # Audit regression: the advice screen must raise ValueError (production
    # invariant), never rely on `assert` (stripped under `python -O`).
    from ui.insights import _assert_no_advice
    _assert_no_advice("NVDA is 35% of capital.")
    with pytest.raises(ValueError):
        _assert_no_advice("You should buy NVDA now.")
    import ui.insights as _ins
    with pytest.raises(ValueError):
        _ins._assert_no_advice("Strong buy signal detected.")


def test_insights_cash_day_attribution_drag():
    out = build_insights({"cash_weight": 0.18, "day_pnl": 120.0,
                          "nav": 10000.0, "top_contributor": "NVDA",
                          "top_contributor_share": 0.031,
                          "top_detractor": "INTC",
                          "top_detractor_share": -0.012,
                          "cost_drag": 47.18,
                          "cost_drag_method": "ERC"}, limit=8)
    texts = " | ".join(i.text for i in out)
    assert "Cash represents 18.0%" in texts
    assert "Day P&L is +1.20%" in texts
    assert "NVDA added +3.10%" in texts
    assert "INTC detracted 1.20%" in texts
    assert "47.18" in texts and "ERC" in texts
    for word in ("buy", "sell", "should", "recommendation"):
        assert word not in texts.lower()
    capped = build_insights({"cash_weight": 0.5, "day_pnl": 999.0,
                             "nav": 1000.0}, limit=1)
    assert len(capped) == 1


def test_charts_return_figures():
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 1.0, 2.5]},
                      index=pd.bdate_range("2024-01-01", periods=3))
    for fig in (line_chart(df, "t"), stacked_area(df, "t"),
                horizontal_bars(["A", "B"], [0.6, 0.4]),
                centered_bars(["Mkt", "SMB"], [0.5, -0.2])):
        assert fig.layout.height in (320, 480, 640)
    assert palette(0) == palette(8)  # cycle wraps
    layout = base_layout("T", "Y", "compact")
    assert layout["height"] == 320


def test_tables_pure_and_formatted():
    df = with_currency(pd.DataFrame({
        "Ticker": ["A", "B"], "Value": [1234.5, -42.1],
        "Weight": [0.6, 0.4], "N": [3.0, 4.0]}), "USD")
    out = money_table(df, money_cols=["Value"], pct_cols=["Weight"],
                      int_cols=["N"])
    assert df["Value"].iloc[0] == 1234.5  # input untouched
    assert out["Value"].iloc[0] == "$1,234.50"
    assert out["Weight"].iloc[1] == "40.00%"
    assert out["N"].iloc[0] == "3"


def test_theme_tokens_stable():
    t = tokens()
    for key in ("background", "surface", "border", "text_primary", "accent",
                "positive", "negative", "warning"):
        assert t[key].startswith("#")
    assert len(CHART_THEME) > 0
    t["accent"] = "neon"
    assert tokens()["accent"] != "neon"  # registry immutable via copy

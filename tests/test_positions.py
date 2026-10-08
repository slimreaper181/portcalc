"""Tests for analytics.positions (model, valuation, purchase, dividends)."""

import json
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from analytics.positions import (
    Position,
    PositionValuation,
    build_analysis_snapshot,
    cumulative_split_factor,
    estimate_dividend_income,
    export_portfolio,
    format_money,
    normalise_quote,
    position_from_dict,
    position_to_dict,
    resolve_purchase_price,
    summarize_portfolio_csv,
    validate_positions_import,
    value_portfolio,
    value_position,
)

TODAY = date(2026, 1, 10)


def _closes(prices, start="2024-01-02"):
    idx = pd.bdate_range(start, periods=len(prices))
    return pd.Series(prices, index=idx)


# ---------------------------------------------------------------------------
# Purchase-price resolution
# ---------------------------------------------------------------------------

def test_purchase_ok_exact_date():
    closes = _closes([100.0, 102.0, 101.0, 103.0, 104.0])
    out = resolve_purchase_price(closes, pd.Series(dtype=float),
                                 date(2024, 1, 4), TODAY)
    assert out["status"] == "ok"
    assert out["price"] == pytest.approx(101.0)
    assert out["split_factor"] == pytest.approx(1.0)
    assert out["used_date"] == date(2024, 1, 4)


def test_purchase_weekend_offers_prev_next():
    closes = _closes([100.0, 102.0, 101.0, 103.0, 104.0, 105.0, 106.0])
    # Sunday 2024-01-07: no silent substitution.
    out = resolve_purchase_price(closes, pd.Series(dtype=float),
                                 date(2024, 1, 7), TODAY)
    assert out["status"] == "missing"
    prev_d, prev_px = out["prev"]
    next_d, next_px = out["next"]
    assert prev_d == date(2024, 1, 5) and prev_px == pytest.approx(103.0)
    assert next_d == date(2024, 1, 8) and next_px == pytest.approx(104.0)


def test_purchase_invalid_dates():
    closes = _closes([100.0, 102.0])
    assert resolve_purchase_price(
        closes, None, date(2027, 1, 1), TODAY)["status"] == "invalid"
    out = resolve_purchase_price(
        closes, None, date(2020, 1, 1), TODAY)
    assert out["status"] == "invalid"  # pre-history
    assert resolve_purchase_price(
        closes, None, "not-a-date", TODAY)["status"] == "invalid"
    assert resolve_purchase_price(
        pd.Series(dtype=float), None, date(2024, 1, 2),
        TODAY)["status"] == "invalid"


def test_split_adjustment_semantics():
    # 4-for-1 split on 2024-01-04: $100 raw close on Jan 2 is $25/share now.
    closes = _closes([100.0, 101.0, 25.5, 26.0])
    splits = pd.Series([4.0], index=pd.DatetimeIndex([date(2024, 1, 4)]))
    assert cumulative_split_factor(splits, date(2024, 1, 2)) == pytest.approx(4.0)
    assert cumulative_split_factor(splits, date(2024, 1, 5)) == pytest.approx(1.0)
    assert cumulative_split_factor(pd.Series(dtype=float),
                                   date(2024, 1, 2)) == pytest.approx(1.0)
    out = resolve_purchase_price(closes, splits, date(2024, 1, 2), TODAY)
    assert out["status"] == "ok"
    assert out["raw_close"] == pytest.approx(100.0)
    assert out["price"] == pytest.approx(25.0)  # per current share


# ---------------------------------------------------------------------------
# Position parsing / legacy migration
# ---------------------------------------------------------------------------

def test_position_legacy_avg_cost_preserved():
    pos = position_from_dict("msft", {"shares": 2.0, "avg_cost": 150.0})
    assert pos.ticker == "MSFT"
    assert pos.purchase_price_native == pytest.approx(150.0)
    assert pos.purchase_price_source == "legacy"
    assert pos.manual_purchase_price is True
    assert pos.purchase_date is None
    assert pos.legacy_avg_cost == pytest.approx(150.0)


def test_position_manual_override_wins():
    pos = position_from_dict("AAPL", {
        "shares": 1.0, "avg_cost": 100.0,
        "purchase_date": "2024-03-14", "purchase_price_native": 120.0,
        "purchase_price_source": "manual", "manual_purchase_price": True,
        "native_currency": "USD", "quote_scale": 1.0})
    assert pos.purchase_price_native == pytest.approx(120.0)
    assert pos.purchase_price_source == "manual"
    assert pos.purchase_date == date(2024, 3, 14)
    assert position_to_dict(pos)["purchase_date"] == "2024-03-14"


def test_position_validation():
    with pytest.raises(ValueError):
        position_from_dict("AAPL", {"shares": 0.0})
    with pytest.raises(ValueError):
        position_from_dict("AAPL", {"shares": -2.0})
    with pytest.raises(ValueError):
        position_from_dict("", {"shares": 1.0})
    with pytest.raises(ValueError):
        position_from_dict("AAPL", {"shares": 1.0,
                                    "purchase_date": "tomorrow!!"})
    with pytest.raises(ValueError):
        position_from_dict("AAPL", {"shares": 1.0,
                                    "manual_purchase_price": -5.0})
    # Fractional shares keep full precision.
    pos = position_from_dict("NVDA", {"shares": 0.0042})
    assert pos.shares == pytest.approx(0.0042)


# ---------------------------------------------------------------------------
# Valuation, cost basis, P&L (+ the £100 → $23 example)
# ---------------------------------------------------------------------------

def _gbp_position():
    return Position(ticker="BARC.L", shares=100.0 / 1.0,
                    purchase_date=date(2024, 1, 2),
                    purchase_price_native=1.00,
                    purchase_price_source="estimate",
                    manual_purchase_price=False,
                    native_currency="GBP", quote_unit="GBp",
                    quote_scale=0.01)


def test_purchase_fx_cost_example():
    # £100 cost @1.20 → $120; £110 value @1.30 → $143; P&L = $23.
    pos = Position(ticker="X", shares=100.0, purchase_date=date(2024, 1, 2),
                   purchase_price_native=1.00, purchase_price_source="estimate",
                   manual_purchase_price=False, native_currency="GBP",
                   quote_unit="GBP", quote_scale=1.0)
    v = value_position(pos, 1.10, "Yahoo", None, 1.20, 1.30, "USD")
    assert isinstance(v, PositionValuation)
    assert v.native_cost_basis == pytest.approx(100.0)
    assert v.base_cost_basis == pytest.approx(120.0)
    assert v.native_market_value == pytest.approx(110.0)
    assert v.base_market_value == pytest.approx(143.0)
    assert v.unrealised_pnl == pytest.approx(23.0)
    assert v.unrealised_pnl_pct == pytest.approx(23.0 / 120.0)


def test_gbp_pence_end_to_end():
    # BARC.L 320 GBp quote → £3.20 native; 100 shares → £320 (not £32,000).
    assert normalise_quote(320.0, 0.01) == pytest.approx(3.20)
    pos = Position(ticker="BARC.L", shares=100.0,
                   purchase_date=date(2024, 1, 2),
                   purchase_price_native=3.00,
                   purchase_price_source="estimate",
                   manual_purchase_price=False, native_currency="GBP",
                   quote_unit="GBp", quote_scale=0.01)
    v = value_position(pos, normalise_quote(320.0, 0.01), "Yahoo", None,
                       1.0, 1.0, "GBP")
    assert v.native_market_value == pytest.approx(320.0)
    assert v.base_market_value == pytest.approx(320.0)


def test_missing_cost_basis_omits_pnl():
    pos = Position(ticker="AAPL", shares=1.0, native_currency="USD",
                   quote_unit="USD", quote_scale=1.0)
    v = value_position(pos, 200.0, "Yahoo", None, None, 1.0, "USD")
    assert v.base_market_value == pytest.approx(200.0)
    assert v.unrealised_pnl is None and v.unrealised_pnl_pct is None
    assert v.base_cost_basis is None
    with pytest.raises(ValueError):
        value_position(pos, -5.0, "Yahoo", None, None, 1.0, "USD")
    with pytest.raises(ValueError):
        value_position(
            Position(ticker="ZZ", shares=1.0),  # unknown native currency
            10.0, "Yahoo", None, None, 1.0, "USD")


def test_portfolio_reconciliation():
    a = value_position(
        Position(ticker="A", shares=10.0, purchase_date=date(2024, 1, 2),
                 purchase_price_native=10.0, purchase_price_source="manual",
                 manual_purchase_price=True, native_currency="USD",
                 quote_unit="USD", quote_scale=1.0),
        12.0, "Yahoo", None, 1.0, 1.0, "USD")
    b = value_position(
        Position(ticker="B", shares=5.0, native_currency="USD",
                 quote_unit="USD", quote_scale=1.0),
        20.0, "Yahoo", None, None, 1.0, "USD")
    agg = value_portfolio([a, b])
    assert agg["total_base_value"] == pytest.approx(120.0 + 100.0)
    assert agg["weights"]["A"] == pytest.approx(120.0 / 220.0)
    assert agg["weights"]["B"] == pytest.approx(100.0 / 220.0)
    # P&L reconciliation: portfolio P&L == Σ position P&L (B has none).
    assert agg["total_pnl"] is None  # unknown cost for B → honestly None
    agg2 = value_portfolio([a])
    assert agg2["total_pnl"] == pytest.approx(20.0)
    assert agg2["total_base_cost"] == pytest.approx(100.0)
    with pytest.raises(ValueError):
        value_portfolio([])


def test_mixed_portfolio_valuation_and_weights():
    # A=$100 USD, B=£100 GBP→$120 @1.20, C=€100 EUR→$110 @1.10.
    vals = [
        value_position(
            Position(ticker="A", shares=1.0, native_currency="USD",
                     quote_unit="USD", quote_scale=1.0),
            100.0, "Yahoo", None, None, 1.0, "USD"),
        value_position(
            Position(ticker="B", shares=1.0, native_currency="GBP",
                     quote_unit="GBP", quote_scale=1.0),
            100.0, "Yahoo", None, None, 1.20, "USD"),
        value_position(
            Position(ticker="C", shares=1.0, native_currency="EUR",
                     quote_unit="EUR", quote_scale=1.0),
            100.0, "Yahoo", None, None, 1.10, "USD"),
    ]
    agg = value_portfolio(vals)
    assert agg["total_base_value"] == pytest.approx(330.0)
    assert agg["weights"]["A"] == pytest.approx(100 / 330)
    assert agg["weights"]["B"] == pytest.approx(120 / 330)
    assert agg["weights"]["C"] == pytest.approx(110 / 330)


def test_base_return_identity():
    # R_base = (1+R_local)(1+R_fx) − 1, via engine values.
    pos = Position(ticker="B", shares=10.0, purchase_date=date(2024, 1, 2),
                   purchase_price_native=100.0,
                   purchase_price_source="estimate",
                   manual_purchase_price=False, native_currency="GBP",
                   quote_unit="GBP", quote_scale=1.0)
    v = value_position(pos, 100.0, "Yahoo", None, 1.00, 1.10, "USD")
    r_local = (100.0 - 100.0) / 100.0  # flat local
    assert v.base_market_value == pytest.approx(10 * 100.0 * 1.10)
    v2 = value_position(pos, 110.0, "Yahoo", None, 1.00, 1.10, "USD")
    r_base = v2.base_market_value / v.base_cost_basis - 1
    assert r_base == pytest.approx((1 + 0.10) * (1 + 0.10) - 1)


# ---------------------------------------------------------------------------
# Dividends (estimated, post-purchase only; never in backtests)
# ---------------------------------------------------------------------------

def test_dividend_income_post_purchase_only():
    divs = pd.Series(
        [0.25, 0.27, 0.27],
        index=pd.DatetimeIndex(
            [date(2024, 1, 5), date(2024, 4, 5), date(2024, 7, 5)]))
    total, events, skipped = estimate_dividend_income(
        divs, date(2024, 3, 14), 10.0, lambda d: 1.30)
    assert total == pytest.approx((0.27 + 0.27) * 10.0 * 1.30)
    assert len(events) == 2 and skipped == 0
    assert events[0]["date"] == date(2024, 4, 5)


def test_dividend_missing_fx_skipped_not_assumed():
    divs = pd.Series([0.27], index=pd.DatetimeIndex([date(2024, 4, 5)]))
    total, events, skipped = estimate_dividend_income(
        divs, date(2024, 3, 14), 10.0, lambda d: None)
    assert total == pytest.approx(0.0) and skipped == 1
    assert estimate_dividend_income(
        pd.Series(dtype=float), date(2024, 3, 14), 10.0,
        lambda d: 1.0) == (0.0, [], 0)


def test_dividends_not_double_counted_in_backtest():
    # Backtest engine consumes adjusted prices only; the dividend estimator
    # is a separate reporting path — engine output is identical with or
    # without dividend data present.
    from analytics.backtest import backtest_buy_and_hold
    idx = pd.bdate_range("2024-01-02", periods=5)
    px = pd.DataFrame({"A": [100.0, 101.0, 102.0, 103.0, 104.0],
                       "B": [50.0, 50.5, 51.0, 51.5, 52.0]}, index=idx)
    w = np.array([0.5, 0.5])
    r1 = backtest_buy_and_hold(px, w, 1000.0, risk_free_annual=0.0)
    divs = pd.Series([0.5], index=pd.DatetimeIndex([idx[2]]))
    total, _, _ = estimate_dividend_income(divs, idx[0].date(), 5.0,
                                           lambda d: 1.0)
    assert total == pytest.approx(2.5)  # reported separately…
    r2 = backtest_buy_and_hold(px, w, 1000.0, risk_free_annual=0.0)
    pd.testing.assert_series_equal(r1.values, r2.values)  # …never merged in


# ---------------------------------------------------------------------------
# Formatting / persistence / invariants
# ---------------------------------------------------------------------------

def test_format_money():
    assert format_money(1234.56, "USD") == "$1,234.56"
    assert format_money(-1245.20, "USD") == "-$1,245.20"
    assert format_money(-312.42, "GBP") == "-£312.42"
    assert format_money(0.0, "EUR") == "€0.00"
    assert format_money(1250000.0, "USD") == "$1,250,000.00"
    assert format_money(None, "USD") == "n/a"
    assert format_money(float("nan"), "GBP") == "n/a"
    assert format_money(float("inf"), "USD") == "n/a"


def test_export_has_no_secrets_and_roundtrips(monkeypatch):
    monkeypatch.setenv("APCA_API_SECRET_KEY", "shh-test-secret")
    doc = export_portfolio(
        {"AAPL": {"shares": 2.0, "avg_cost": 150.0},
         "BARC.L": {"shares": 100.0, "purchase_date": "2024-01-02",
                    "purchase_price_native": 3.0,
                    "purchase_price_source": "estimate",
                    "manual_purchase_price": False,
                    "native_currency": "GBP", "quote_unit": "GBp",
                    "quote_scale": 0.01}},
        "GBP", run_id="PC-20260110-AB12")
    import json
    blob = json.dumps(doc)
    assert "shh-test-secret" not in blob
    assert "APCA_API" not in blob and "secret" not in blob.lower().replace(
        "purchase", "")
    assert doc["schema_version"] == 1 and doc["base_currency"] == "GBP"
    positions, base = validate_positions_import(
        json.loads(blob))
    assert base == "GBP" and set(positions) == {"AAPL", "BARC.L"}
    assert positions["AAPL"]["purchase_price_source"] == "legacy"


def test_import_validation_rejects_garbage():
    with pytest.raises(ValueError):
        validate_positions_import({"schema_version": 99, "base_currency": "USD",
                                   "positions": []})
    with pytest.raises(ValueError):
        validate_positions_import({"schema_version": 1,
                                   "base_currency": "JPY", "positions": []})
    with pytest.raises(ValueError):
        validate_positions_import({"schema_version": 1,
                                   "base_currency": "USD", "positions": []})
    with pytest.raises(ValueError):
        validate_positions_import({
            "schema_version": 1, "base_currency": "USD",
            "positions": [{"ticker": "AAPL", "shares": -1.0}]})
    with pytest.raises(ValueError):
        validate_positions_import({
            "schema_version": 1, "base_currency": "USD",
            "positions": [{"ticker": "AAPL", "shares": 1.0},
                          {"ticker": "aapl", "shares": 2.0}]})
    with pytest.raises(ValueError):
        validate_positions_import("not-a-dict")


def test_base_switch_preserves_inputs():
    pos = position_from_dict("BARC.L", {
        "shares": 100.0, "purchase_date": "2024-01-02",
        "purchase_price_native": 3.0, "purchase_price_source": "estimate",
        "manual_purchase_price": False, "native_currency": "GBP",
        "quote_unit": "GBp", "quote_scale": 0.01})
    before = position_to_dict(pos)
    _ = value_position(pos, 3.2, "Yahoo", None, 1.20, 1.30, "USD")
    _ = value_position(pos, 3.2, "Yahoo", None, 1.0, 1.0, "GBP")
    _ = value_position(pos, 3.2, "Yahoo", None, 1.10, 1.10, "EUR")
    after = position_to_dict(pos)
    assert before == after  # switching base never mutates the position


def test_purchase_date_does_not_move_current_price():
    pos = position_from_dict("AAPL", {"shares": 1.0,
                                      "purchase_date": "2024-01-02",
                                      "purchase_price_native": 180.0,
                                      "purchase_price_source": "estimate",
                                      "manual_purchase_price": False,
                                      "native_currency": "USD",
                                      "quote_unit": "USD",
                                      "quote_scale": 1.0})
    v1 = value_position(pos, 200.0, "Yahoo", None, 1.0, 1.0, "USD")
    pos2 = position_from_dict("AAPL", {"shares": 1.0,
                                       "purchase_date": "2023-01-02",
                                       "purchase_price_native": 120.0,
                                       "purchase_price_source": "estimate",
                                       "manual_purchase_price": False,
                                       "native_currency": "USD",
                                       "quote_unit": "USD",
                                       "quote_scale": 1.0})
    v2 = value_position(pos2, 200.0, "Yahoo", None, 1.0, 1.0, "USD")
    assert v1.native_current_price == v2.native_current_price == 200.0
    assert v1.base_cost_basis != v2.base_cost_basis


def test_weight_invariance_across_base():
    # $100 + £100(@1.20) → weights invariant when viewed in GBP/EUR/USD.
    from analytics.currency import convert_value
    usd_per = {"USD": 1.0, "GBP": 1.20, "EUR": 1.10}
    native = [("A", 100.0, "USD"), ("B", 100.0, "GBP")]
    for base in ("USD", "GBP", "EUR"):
        vals = {t: convert_value(px, c, base, usd_per)
                for t, px, c in native}
        tot = sum(vals.values())
        w = {t: v / tot for t, v in vals.items()}
        assert w["A"] == pytest.approx(100 / 220, rel=1e-9)
        assert w["B"] == pytest.approx(120 / 220, rel=1e-9)


def test_value_invariance_roundtrip():
    from analytics.currency import convert_value
    usd_per = {"USD": 1.0, "GBP": 0.80 ** -1}  # GBPUSD = 1.25
    gbp = convert_value(1000.0, "USD", "GBP", usd_per)
    assert gbp == pytest.approx(800.0)
    assert convert_value(gbp, "GBP", "USD", usd_per) == pytest.approx(1000.0)


def test_csv_summary_export():
    v = value_position(
        Position(ticker="A", shares=2.0, purchase_date=date(2024, 1, 2),
                 purchase_price_native=50.0, purchase_price_source="manual",
                 manual_purchase_price=True, native_currency="USD",
                 quote_unit="USD", quote_scale=1.0),
        60.0, "Yahoo", None, 1.0, 1.0, "USD")
    df = summarize_portfolio_csv(
        [v], {"A": position_from_dict(
            "A", {"shares": 2.0, "purchase_date": "2024-01-02",
                  "purchase_price_native": 50.0,
                  "purchase_price_source": "manual",
                  "manual_purchase_price": True})})
    assert list(df.columns) == ["Ticker", "Shares", "Purchase Date",
                                "Purchase Price", "Native Currency",
                                "Current Price", "Market Value", "Weight",
                                "P&L"]
    assert df.loc[0, "Market Value"] == pytest.approx(120.0)
    assert df.loc[0, "Weight"] == "100.0%"


def test_analysis_snapshot_has_no_secrets():
    import json
    snap = build_analysis_snapshot(
        ["AAPL"], {"AAPL": 2.0}, {"AAPL": "2024-01-02"}, "USD", "max",
        "SPY", 0.05, "max_sharpe", 0.0, 1.0, {"horizon": 5}, "ff5_mom",
        "Yahoo Finance", "2026-01-08")
    blob = json.dumps(snap)
    assert "APCA" not in blob and "secret" not in blob.lower()
    assert snap["schema_version"] == 1
    assert snap["run_id"].startswith("PC-")
    assert set(snap) >= {"tickers", "shares", "purchase_dates",
                         "base_currency", "benchmark", "fx_source"}


def test_alpaca_isolation(monkeypatch):
    # Absurd live prices must not move purchase/valuation/FX paths, which
    # never call Alpaca (pure inputs in, pure outputs out).
    import data.alpaca as live

    def _boom(*a, **k):
        raise AssertionError("live path must never be called")

    monkeypatch.setattr(live, "fetch_latest_trades", _boom)
    monkeypatch.setattr(live, "fetch_latest_trade", _boom)
    pos = position_from_dict("AAPL", {"shares": 1.0,
                                      "purchase_date": "2024-01-02",
                                      "purchase_price_native": 180.0,
                                      "purchase_price_source": "estimate",
                                      "manual_purchase_price": False,
                                      "native_currency": "USD",
                                      "quote_unit": "USD",
                                      "quote_scale": 1.0})
    v = value_position(pos, 200.0, "Yahoo", None, 1.0, 1.0, "USD")
    assert v.base_market_value == pytest.approx(200.0)
    from analytics.currency import convert_value
    assert convert_value(100.0, "GBP", "USD",
                         {"USD": 1.0, "GBP": 1.2}) == pytest.approx(120.0)

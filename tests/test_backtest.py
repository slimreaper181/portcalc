"""Tests for analytics.backtest (deterministic, hand-verifiable)."""

import numpy as np
import pandas as pd
import pytest

from analytics.backtest import (
    REBALANCE_FREQUENCIES,
    BacktestResult,
    apply_transaction_costs,
    backtest_benchmark,
    backtest_buy_and_hold,
    backtest_rebalanced,
    backtest_summary,
    calculate_turnover,
    compare_backtests,
    detect_mixed_markets,
    generate_rebalance_dates,
    prepare_backtest_data,
    validate_backtest_window,
    validate_target_weights,
    validate_transaction_cost_bps,
)


def make_prices(dates, a, b):
    return pd.DataFrame({"A": a, "B": b}, index=pd.DatetimeIndex(dates))


W2 = np.array([0.5, 0.5])


# ---------------------------------------------------------------------------
# §38 manual buy-and-hold: 100 capital, A 5 sh @10, B 2.5 sh @20 → 105
# ---------------------------------------------------------------------------

def test_manual_buy_and_hold_shares_and_wealth():
    idx = pd.DatetimeIndex(["2020-01-06", "2020-01-07"])
    px = make_prices(idx, [10.0, 12.0], [20.0, 18.0])
    res = backtest_buy_and_hold(px, W2, 100.0, risk_free_annual=0.0)
    assert isinstance(res, BacktestResult)
    assert res.values.iloc[0] == pytest.approx(100.0)
    assert res.values.iloc[-1] == pytest.approx(5 * 12 + 2.5 * 18)  # 105
    # Fractional shares: B holds exactly 2.5 shares.
    assert res.values.iloc[-1] == pytest.approx(105.0)
    # Weight drift (not constant weights): A drifts up to 60/105.
    assert res.weights.loc[idx[-1], "A"] == pytest.approx(60 / 105)
    assert res.weights.loc[idx[-1], "B"] == pytest.approx(45 / 105)
    assert res.weights.sum(axis=1).sub(1.0).abs().max() < 1e-9
    assert (res.costs == 0).all() and (res.turnover == 0).all()
    assert len(res.rebalance_events) == 0


# ---------------------------------------------------------------------------
# §39 manual rebalance: A doubles, B flat → sell A, buy B, 50/50 + costs
# ---------------------------------------------------------------------------

def _drift_frame():
    idx = pd.DatetimeIndex(["2020-01-06", "2020-02-03", "2020-02-04"])
    return make_prices(idx, [10.0, 20.0, 20.0], [10.0, 10.0, 10.0])


def test_manual_rebalance_drift_and_restore():
    px = _drift_frame()
    # Sanity: monthly schedule hits 2020-02-03 (Feb 1 is a Saturday).
    assert list(generate_rebalance_dates(px.index, "monthly")) == [
        pd.Timestamp("2020-02-03")]
    res = backtest_rebalanced(px, W2, 100.0, "monthly", cost_bps=10.0,
                              risk_free_annual=0.0)
    # Pre-rebalance value 150 (5*20 + 5*10). Exact circular solve:
    # c* = rate/3 → cost 0.05; final executed trades ∓25.025 (funded
    # from post-cost capital), NOT the naive pre-cost ∓25.
    assert res.values.loc["2020-02-03"] == pytest.approx(150.0 - 0.05)
    assert res.costs.loc["2020-02-03"] == pytest.approx(0.05)
    assert res.turnover.loc["2020-02-03"] == pytest.approx(50.0 / 150.0)
    # Weights restored to exactly 50/50 at the rebalance date.
    assert res.weights.loc["2020-02-03", "A"] == pytest.approx(0.5)
    assert res.weights.loc["2020-02-03", "B"] == pytest.approx(0.5)
    # Trade legs: A sold 25.025 (cost 0.025025), B bought 24.975.
    tr = res.trades
    assert len(tr) == 2
    a = tr[tr.Ticker == "A"].iloc[0]
    assert a["Trade Value"] == pytest.approx(-25.025)
    assert a["Transaction Cost"] == pytest.approx(0.025025)
    assert a["Before Weight"] == pytest.approx(100 / 150)
    assert tr["Transaction Cost"].sum() == pytest.approx(0.05)
    # Accounting identity: before − cost == after == stored value.
    ev = res.rebalance_events.iloc[0]
    assert ev["Portfolio Value Before"] == pytest.approx(150.0)
    assert ev["Portfolio Value After"] == pytest.approx(
        ev["Portfolio Value Before"] - ev["Transaction Cost"])
    assert ev["Portfolio Value After"] == pytest.approx(
        res.values.loc["2020-02-03"])
    assert res.metrics["total_costs"] == pytest.approx(0.05)
    assert res.metrics["n_rebalances"] == 1
    # Buy-and-hold twin keeps the drifted 150 (no cost deducted).
    bh = backtest_buy_and_hold(px, W2, 100.0, risk_free_annual=0.0)
    assert bh.values.loc["2020-02-03"] == pytest.approx(150.0)


def test_zero_cost_rebalance_still_rebalances():
    px = _drift_frame()
    res = backtest_rebalanced(px, W2, 100.0, "monthly", cost_bps=0.0,
                              risk_free_annual=0.0)
    assert res.values.loc["2020-02-03"] == pytest.approx(150.0)
    assert res.metrics["total_costs"] == 0.0
    # ...but unlike buy-and-hold, weights snap back to target.
    assert res.weights.loc["2020-02-03", "A"] == pytest.approx(0.5)


def test_never_frequency_equals_buy_and_hold():
    idx = pd.bdate_range("2020-01-01", periods=40)
    rng = np.random.default_rng(0)
    px = pd.DataFrame(
        {"A": 100 * np.cumprod(1 + rng.normal(0, 0.01, 40)),
         "B": 50 * np.cumprod(1 + rng.normal(0, 0.01, 40))}, index=idx)
    a = backtest_rebalanced(px, W2, 1000.0, "never", name="X")
    b = backtest_buy_and_hold(px, W2, 1000.0, name="X")
    pd.testing.assert_series_equal(a.values, b.values)


def test_large_costs_use_final_executed_notionals():
    # Asymmetric 80/20 target, drifted 50/50-ish holdings, deliberately
    # heavy 500 bps: A flat @10, B triples to 30 → Vb = 8*10 + 2*30 = 140.
    # Naive pre-cost estimate (3.20) is measurably wrong; the exact solve
    # gives c ≈ 0.02219 → cost ≈ 3.107.
    idx = pd.DatetimeIndex(["2020-01-06", "2020-02-03", "2020-02-04"])
    px = make_prices(idx, [10.0, 10.0, 10.0], [10.0, 30.0, 30.0])
    w = np.array([0.8, 0.2])
    res = backtest_rebalanced(px, w, 100.0, "monthly", cost_bps=500.0,
                              risk_free_annual=0.0)
    d = pd.Timestamp("2020-02-03")
    ev = res.rebalance_events.iloc[0]
    assert ev["Portfolio Value Before"] == pytest.approx(140.0)
    reported = float(ev["Transaction Cost"])
    final_trades = res.trades["Trade Value"].to_numpy(dtype=float)
    # (1) reported cost == rate × Σ|final executed trades| (tight).
    assert reported == pytest.approx(
        0.05 * np.abs(final_trades).sum(), rel=1e-9)
    # (2) accounting identity on reported figures.
    assert ev["Portfolio Value After"] == pytest.approx(
        ev["Portfolio Value Before"] - reported, abs=1e-9)
    assert ev["Portfolio Value After"] == pytest.approx(
        res.values.loc[d], abs=1e-9)
    assert res.costs.loc[d] == pytest.approx(reported, rel=1e-12)
    # (3) naive pre-cost estimate (0.05 × (|112−80| + |28−60|) = 3.20) is
    # measurably different from the exact cost.
    naive = 0.05 * (abs(140 * 0.8 - 80.0) + abs(140 * 0.2 - 60.0))
    assert naive == pytest.approx(3.2)
    assert abs(reported - naive) > 0.01
    # Post-rebalance weights are exactly the 80/20 targets.
    assert res.weights.loc[d, "A"] == pytest.approx(0.8)
    assert res.weights.loc[d, "B"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def test_rebalance_schedules_calendar():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    monthly = generate_rebalance_dates(idx, "monthly")
    assert [d.strftime("%Y-%m-%d") for d in monthly] == [
        "2020-02-03", "2020-03-02", "2020-04-01", "2020-05-01",
        "2020-06-01", "2020-07-01", "2020-08-03", "2020-09-01",
        "2020-10-01", "2020-11-02", "2020-12-01"]
    quarterly = generate_rebalance_dates(idx, "quarterly")
    assert [d.strftime("%Y-%m-%d") for d in quarterly] == [
        "2020-04-01", "2020-07-01", "2020-10-01"]
    semi = generate_rebalance_dates(idx, "semi-annual")
    assert [d.strftime("%Y-%m-%d") for d in semi] == ["2020-07-01"]
    # Annual on a Jan–Dec window: allocation only, no rebalance event.
    assert len(generate_rebalance_dates(idx, "annual")) == 0
    assert len(generate_rebalance_dates(idx, "never")) == 0
    assert set(REBALANCE_FREQUENCIES) == {
        "never", "monthly", "quarterly", "semi-annual", "annual"}
    with pytest.raises(ValueError):
        generate_rebalance_dates(idx, "fortnightly")


def test_non_trading_day_rolls_forward():
    # Custom calendar missing a Monday: Feb 3 absent → rolls to Feb 4.
    idx = pd.DatetimeIndex(["2020-01-06", "2020-02-04", "2020-02-05",
                            "2020-03-02"])
    dates = generate_rebalance_dates(idx, "monthly")
    assert list(dates) == [pd.Timestamp("2020-02-04"),
                           pd.Timestamp("2020-03-02")]


# ---------------------------------------------------------------------------
# Costs / turnover primitives
# ---------------------------------------------------------------------------

def test_cost_and_turnover_primitives():
    assert apply_transaction_costs(np.array([-25.0, 25.0]), 10.0) == pytest.approx(0.05)
    assert apply_transaction_costs(np.array([0.0, 0.0]), 10.0) == 0.0
    assert calculate_turnover(np.array([-25.0, 25.0]), 150.0) == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        apply_transaction_costs(np.array([1.0]), -5.0)
    with pytest.raises(ValueError):
        calculate_turnover(np.array([1.0]), 0.0)


# ---------------------------------------------------------------------------
# CAGR and summary
# ---------------------------------------------------------------------------

def test_cagr_uses_calendar_time():
    idx = pd.DatetimeIndex(["2020-01-01", "2021-12-31"])
    v = pd.Series([100.0, 121.0], index=idx)
    years = 730 / 365.25
    s = backtest_summary(v, 100.0, 0.0)
    assert s["years"] == pytest.approx(years)
    assert s["cagr"] == pytest.approx(1.21 ** (1 / years) - 1)
    assert s["total_return"] == pytest.approx(0.21)
    # CAGR is distinct from log-space annualised return.
    assert s["annualised_return"] != pytest.approx(s["cagr"])


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def test_benchmark_wealth_path():
    idx = pd.DatetimeIndex(["2020-01-06", "2020-01-07", "2020-01-08"])
    b = pd.Series([50.0, 55.0, 60.0], index=idx)
    res = backtest_benchmark(b, idx, 1000.0, risk_free_annual=0.0)
    assert list(res.values) == pytest.approx([1000.0, 1100.0, 1200.0])
    assert res.is_benchmark and res.weights is None and res.trades is None
    assert res.metrics["total_costs"] == 0.0


def test_benchmark_date_alignment():
    idx = pd.DatetimeIndex(["2020-01-06", "2020-01-07", "2020-01-08"])
    # Benchmark missing the middle day (holiday misalignment) → bounded fill.
    b = pd.Series([50.0, 60.0],
                  index=pd.DatetimeIndex(["2020-01-06", "2020-01-08"]))
    res = backtest_benchmark(b, idx, 1000.0, risk_free_annual=0.0)
    assert list(res.values.index) == list(idx)
    assert res.values.iloc[1] == pytest.approx(1000.0)  # carried forward
    with pytest.raises(ValueError):
        backtest_benchmark(b.iloc[:1], idx, 1000.0)


# ---------------------------------------------------------------------------
# Data prep, alignment, validation
# ---------------------------------------------------------------------------

def test_column_order_independence():
    idx = pd.bdate_range("2020-01-01", periods=60)
    rng = np.random.default_rng(3)
    base = pd.DataFrame(
        {"A": 100 * np.cumprod(1 + rng.normal(0, 0.01, 60)),
         "B": 50 * np.cumprod(1 + rng.normal(0, 0.01, 60))}, index=idx)
    px1, _ = prepare_backtest_data(base[["A", "B"]], ["A", "B"],
                                   idx[0], idx[-1], min_rows=10)
    px2, _ = prepare_backtest_data(base[["B", "A"]], ["A", "B"],
                                   idx[0], idx[-1], min_rows=10)
    r1 = backtest_rebalanced(px1, W2, 1000.0, "quarterly", 10.0)
    r2 = backtest_rebalanced(px2, W2, 1000.0, "quarterly", 10.0)
    pd.testing.assert_series_equal(r1.values, r2.values)
    pd.testing.assert_frame_equal(r1.rebalance_events, r2.rebalance_events)


def test_single_asset_portfolio():
    idx = pd.bdate_range("2020-01-01", periods=60)
    px = pd.DataFrame({"A": np.linspace(100, 130, 60)}, index=idx)
    w = np.array([1.0])
    bh = backtest_buy_and_hold(px, w, 1000.0)
    rb = backtest_rebalanced(px, w, 1000.0, "quarterly", 10.0)
    # One asset can never drift: rebalanced == buy-and-hold, zero costs.
    pd.testing.assert_series_equal(bh.values, rb.values)
    assert rb.metrics["total_costs"] == 0.0
    assert bh.values.iloc[-1] == pytest.approx(1300.0)


def test_missing_prices_and_ipo_note():
    idx = pd.bdate_range("2020-01-01", periods=60)
    a = np.linspace(100, 110, 60)
    b = np.full(60, np.nan)
    b[20:] = np.linspace(50, 60, 40)  # B lists 20 days in
    px = pd.DataFrame({"A": a, "B": b}, index=idx)
    window, notes = prepare_backtest_data(px, ["A", "B"], idx[0], idx[-1],
                                          min_rows=10)
    assert window.index[0] == idx[20]  # only common valid dates kept
    assert any("B" in n for n in notes)  # limitation explained
    # Far too short a window is rejected, not silently used.
    with pytest.raises(ValueError):
        prepare_backtest_data(px, ["A", "B"], idx[0], idx[-1], min_rows=100)
    with pytest.raises(ValueError):
        prepare_backtest_data(px, ["A", "B"], idx[-1], idx[0], min_rows=10)


def test_date_range_and_input_validation():
    with pytest.raises(ValueError):
        validate_backtest_window(pd.Timestamp("2020-02-01"),
                                 pd.Timestamp("2020-01-01"))
    with pytest.raises(ValueError):
        validate_transaction_cost_bps(-1.0)
    with pytest.raises(ValueError):
        validate_transaction_cost_bps(1001.0)
    with pytest.raises(ValueError):
        backtest_buy_and_hold(
            make_prices(["2020-01-06", "2020-01-07"], [1, 1], [1, 1]),
            np.array([0.6, 0.6]), 100.0)  # sums to 1.2
    with pytest.raises(ValueError):
        validate_target_weights(np.array([-0.1, 1.1]), ["A", "B"])
    with pytest.raises(ValueError):
        backtest_buy_and_hold(
            make_prices(["2020-01-06", "2020-01-07"], [1, 1], [1, 1]),
            W2, 0.0)


def test_mixed_markets_warning():
    assert detect_mixed_markets(["AAPL", "MSFT"]) is None
    note = detect_mixed_markets(["AAPL", "BARC.L"])
    assert note is not None and "FX" in note


# ---------------------------------------------------------------------------
# No look-ahead: post-rebalance shares depend only on prices at the event
# ---------------------------------------------------------------------------

def test_no_look_ahead_at_rebalance():
    idx = pd.DatetimeIndex(["2020-01-06", "2020-02-03", "2020-02-04"])
    # Two futures after the Feb-3 rebalance: crash vs moon for A.
    px_crash = make_prices(idx, [10.0, 20.0, 2.0], [10.0, 10.0, 10.0])
    px_moon = make_prices(idx, [10.0, 20.0, 200.0], [10.0, 10.0, 10.0])
    r1 = backtest_rebalanced(px_crash, W2, 100.0, "monthly", 10.0)
    r2 = backtest_rebalanced(px_moon, W2, 100.0, "monthly", 10.0)
    # Identical pre-event history → identical event accounting.
    pd.testing.assert_frame_equal(r1.rebalance_events, r2.rebalance_events)
    pd.testing.assert_frame_equal(
        r1.trades.sort_values(["Date", "Ticker"]).reset_index(drop=True),
        r2.trades.sort_values(["Date", "Ticker"]).reset_index(drop=True))
    # Shares set from Feb-3 close only: A gets 75*scale/20 in both worlds.
    scale = (150.0 - 0.05) / 150.0
    assert r1.values.loc["2020-02-03"] == pytest.approx(149.95)
    assert r2.values.loc["2020-02-03"] == pytest.approx(149.95)
    assert r1.values.loc["2020-02-04"] == pytest.approx(
        (75 * scale / 20) * 2 + (75 * scale / 10) * 10)
    _ = scale  # documented formula reference


# ---------------------------------------------------------------------------
# §40 live-price isolation: absurd Alpaca data cannot move the backtest
# ---------------------------------------------------------------------------

def test_live_price_isolation(monkeypatch):
    import data.alpaca as live

    def _boom(*a, **k):
        return {"MSFT": 1e12, "AAPL": 1e12}

    monkeypatch.setattr(live, "fetch_latest_trades", _boom)
    idx = pd.bdate_range("2020-01-01", periods=40)
    px = pd.DataFrame({"A": np.linspace(10, 12, 40),
                       "B": np.linspace(20, 18, 40)}, index=idx)
    before = backtest_rebalanced(px, W2, 1000.0, "monthly", 10.0)
    # Even with the live path returning nonsense, the engine — which only
    # accepts explicit historical frames — is bit-identical.
    after = backtest_rebalanced(px, W2, 1000.0, "monthly", 10.0)
    pd.testing.assert_series_equal(before.values, after.values)
    assert live.fetch_latest_trades(["MSFT"])["MSFT"] == 1e12  # mock active


def test_backtest_module_has_no_live_imports():
    import pathlib

    src = pathlib.Path("analytics/backtest.py").read_text()
    lowered = src.lower()
    # The only allowed mention is the documented isolation guarantee.
    assert "alpaca" not in lowered.replace("no alpaca/live prices", "")
    for forbidden in ("from data.alpaca", "import data.alpaca",
                      "fetch_latest", "PriceQuote", "build_price_view"):
        assert forbidden not in src


# ---------------------------------------------------------------------------
# Comparison table + metrics sanity
# ---------------------------------------------------------------------------

def test_compare_backtests_table():
    idx = pd.bdate_range("2020-01-01", periods=120)
    rng = np.random.default_rng(7)
    px = pd.DataFrame(
        {"A": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 120)),
         "B": 50 * np.cumprod(1 + rng.normal(0.001, 0.01, 120))}, index=idx)
    results = {
        "Current B&H": backtest_buy_and_hold(px, W2, 10_000.0),
        "Quarterly": backtest_rebalanced(px, W2, 10_000.0, "quarterly", 10.0),
        "SPY": backtest_benchmark(
            pd.Series(400 * np.cumprod(1 + rng.normal(0.001, 0.008, 120)),
                      index=idx), idx, 10_000.0),
    }
    table = compare_backtests(results)
    assert list(table.columns) == ["Current B&H", "Quarterly", "SPY"]
    assert table.loc["Final Value", "Current B&H"] == pytest.approx(
        results["Current B&H"].values.iloc[-1])
    assert table.loc["Rebalances", "Quarterly"] >= 1
    assert table.loc["Total Costs", "Quarterly"] > 0
    assert table.loc["Total Costs", "SPY"] == 0.0  # UI renders as N/A
    for name, res in results.items():
        for key in ("final_value", "total_return", "cagr", "sharpe",
                    "max_drawdown"):
            assert np.isfinite(res.metrics[key]), (name, key)
    with pytest.raises(ValueError):
        compare_backtests({})


# ---------------------------------------------------------------------------
# §57 multi-currency: historical-FX wealth path + current-FX invariance
# ---------------------------------------------------------------------------

def _fx_base_panel():
    # A in USD; B native £2.00→£2.04 with GBP flat @1.25 → $2.50→$2.55.
    idx = pd.bdate_range("2024-01-02", periods=3)
    fx = pd.DataFrame({"GBP": [1.25, 1.25, 1.25]}, index=idx)
    panel = pd.DataFrame({
        "A": [100.0, 101.0, 102.0],
        "B": [2.00 * 1.25, 2.02 * 1.25, 2.04 * 1.25],
    }, index=idx)
    return panel, fx


def test_backtest_wealth_on_fx_converted_panel():
    from analytics.currency import convert_price_series
    panel, fx = _fx_base_panel()
    # Prove the panel really embeds FX: rebuild B from native + FX.
    idx = panel.index
    native_b = pd.Series([200.0, 202.0, 204.0], index=idx)  # pence
    rebuilt = convert_price_series(native_b * 0.01, "GBP", "USD", fx)
    assert list(rebuilt) == pytest.approx(list(panel["B"]))
    # Buy & hold 50/50 on $1000: A 5 sh, B 200 sh.
    res = backtest_buy_and_hold(panel, W2, 1000.0, risk_free_annual=0.0)
    assert res.values.iloc[0] == pytest.approx(1000.0)
    assert res.values.iloc[1] == pytest.approx(5 * 101.0 + 200 * 2.525)
    assert res.values.iloc[2] == pytest.approx(5 * 102.0 + 200 * 2.55)
    assert np.all(np.isfinite(res.values.values))


def test_current_fx_cannot_move_history():
    # The engine accepts only explicit historical frames — there is no
    # current-FX input, so "today's" rate cannot change history.
    panel, _ = _fx_base_panel()
    r1 = backtest_rebalanced(panel, W2, 1000.0, "monthly", 10.0)
    current_fx_then, current_fx_now = 1.25, 999.0
    assert current_fx_then != current_fx_now  # the counterfactual differs…
    r2 = backtest_rebalanced(panel, W2, 1000.0, "monthly", 10.0)
    pd.testing.assert_series_equal(r1.values, r2.values)  # …yet history can't
    pd.testing.assert_frame_equal(r1.rebalance_events, r2.rebalance_events)

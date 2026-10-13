"""Tests for analytics.walkforward (true walk-forward optimisation).

Every estimation slice must satisfy ``date < execution date`` — the
execution-day close may execute already-computed targets but never
estimate them. Failure behaviour: initial failure → structured failure
with no wealth path; later failure → hold, skip, record, continue.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date

from analytics.allocation import (
    annualised_covariance_simple,
    annualised_mean_returns_simple,
    diversification_ratio,
    equal_risk_contribution,
    maximum_diversification,
    risk_contributions_simple,
    simple_returns_from_prices,
)
from analytics.backtest import backtest_rebalanced
from analytics.currency import convert_price_series
from analytics.walkforward import (
    WalkForwardConfig,
    earliest_feasible_test_start,
    estimation_slice,
    estimate_event_weights,
    run_walk_forward,
    turnover_diagnostics,
    walk_forward_schedule,
)

TICKERS = ["A", "B", "C"]


def _panel(n=2100, seed=11, vols=(0.010, 0.012, 0.008), start="2015-01-01"):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed)
    data = {}
    for t, v in zip(TICKERS, vols):
        data[t] = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, v, n)))
    return pd.DataFrame(data, index=idx)


def _cfg(**kw):
    base = dict(method="erc", test_start=date(2020, 1, 2),
                test_end=date(2022, 12, 31), lookback_years=3.0,
                window_type="rolling", rebalance_frequency="quarterly",
                transaction_cost_bps=10.0)
    base.update(kw)
    return WalkForwardConfig(**base)


def _const_estimator(weights, tickers=None):
    tickers = list(tickers) if tickers else list(TICKERS)
    w = pd.Series(np.asarray(weights, dtype=float), index=tickers)

    def _est(est, exec_date):
        return w.copy(), {"est_return": 0.0, "est_volatility": 0.0,
                          "est_sharpe": 0.0, "diversification_ratio": 1.0}, "stub"
    return _est


# ---------------------------------------------------------------------------
# Scheduling + windows
# ---------------------------------------------------------------------------

def test_schedule_initial_plus_quarterly_marks():
    idx = pd.bdate_range("2020-01-01", periods=400)
    e0, rebs = walk_forward_schedule(idx, "quarterly")
    assert e0 == idx[0]
    assert len(rebs) > 0
    assert all(d in idx for d in rebs)
    assert rebs[0] > e0 and rebs[-1] < idx[-1]  # last bar never an event
    # Calendar marks roll forward to trading days: Apr/Jul/Oct 2020 present.
    assert pd.Timestamp("2020-04-01") in set(rebs)
    assert pd.Timestamp("2020-07-01") in set(rebs)


def test_schedule_rejects_nonperiodic():
    idx = pd.bdate_range("2020-01-01", periods=100)
    with pytest.raises(ValueError):
        walk_forward_schedule(idx, "never")
    with pytest.raises(ValueError):
        walk_forward_schedule(pd.DatetimeIndex([]), "quarterly")


def test_estimation_slice_strictly_before_cutoff():
    px = _panel()
    cut = pd.Timestamp("2020-06-15")
    est, info = estimation_slice(px, TICKERS, cut, 1.0, "rolling", 60)
    assert est.index.max() < cut
    assert info["est_end"] < cut.date()
    assert (cut - est.index[0]).days >= 360  # full 1y lookback honoured


def test_estimation_slice_expanding_uses_all_prior():
    px = _panel()
    cut = pd.Timestamp("2020-06-15")
    est, info = estimation_slice(px, TICKERS, cut, None, "expanding", 60)
    assert est.index.max() < cut
    assert est.index[0] == px.index[0]
    assert len(est) > 1000


def test_earliest_feasible_test_start_rolling():
    px = _panel()
    earliest = earliest_feasible_test_start(px, TICKERS, 3.0, "rolling", 60)
    assert earliest == (px.index[0] + pd.DateOffset(years=3)).date()
    # Starting earlier is refused with the earliest date in the message.
    cfg = _cfg(test_start=date(2016, 1, 4))
    res = run_walk_forward(px, TICKERS, cfg, 10000.0)
    assert res.success is False
    assert not np.isfinite(res.values).any()
    assert str(earliest) in res.message


def test_insufficient_history_expanding():
    # Only ~43 common observations before 2015-03-01 (< 60 required), while
    # the test window itself is covered: the history gate (not the window
    # slicer) must refuse with the deficit explained.
    px = _panel(n=400)
    cfg = _cfg(window_type="expanding", test_start=date(2015, 3, 1),
               test_end=date(2016, 1, 4))
    res = run_walk_forward(px, TICKERS, cfg, 10000.0)
    assert res.success is False
    assert "at least" in res.message


def test_config_validation():
    with pytest.raises(ValueError):
        _cfg(method="nope").validated(TICKERS)
    with pytest.raises(ValueError):
        _cfg(rebalance_frequency="never").validated(TICKERS)
    with pytest.raises(ValueError):
        _cfg(window_type="sideways").validated(TICKERS)
    with pytest.raises(ValueError):
        _cfg(min_weight=0.6, max_weight=0.5).validated(TICKERS)
    with pytest.raises(ValueError):
        _cfg(test_start=date(2022, 1, 1),
             test_end=date(2020, 1, 1)).validated(TICKERS)
    with pytest.raises(ValueError):
        _cfg(method="black_litterman", bl_ref="retrospective_current",
             bl_ref_weights=None).validated(TICKERS)


# ---------------------------------------------------------------------------
# No-look-ahead core
# ---------------------------------------------------------------------------

def test_estimator_never_sees_execution_day_or_later():
    # Recording stub captures exactly what each event was allowed to see.
    px = _panel()
    cfg = _cfg()
    seen = {}

    def _rec(est, d):
        seen[pd.Timestamp(d)] = est.index.max()
        return _const_estimator([1 / 3] * 3)(est, d)

    res = run_walk_forward(px, TICKERS, cfg, 10000.0, estimator=_rec)
    assert res.success
    assert len(seen) >= 3
    for d, mx in seen.items():
        assert mx < pd.Timestamp(d), (d, mx)
    # Initial event sees only pre-test-start data (never [T, E0]).
    e0 = res.events["Execution Date"].iloc[0]
    assert seen[pd.Timestamp(e0)] < pd.Timestamp(cfg.test_start)


def test_future_prices_cannot_move_targets():
    # Crash vs moon AFTER D-1: identical targets at every shared event.
    px = _panel()
    cfg = _cfg()
    base = run_walk_forward(px, TICKERS, cfg, 10000.0, estimator=None)
    assert base.success
    d1 = base.events["Execution Date"].iloc[1]
    crash, moon = px.copy(), px.copy()
    crash.loc[crash.index >= d1] *= np.linspace(1.0, 0.2, (crash.index >= d1).sum())[:, None]
    moon.loc[moon.index >= d1] *= np.linspace(1.0, 5.0, (moon.index >= d1).sum())[:, None]
    r_crash = run_walk_forward(crash, TICKERS, _cfg(), 10000.0)
    r_moon = run_walk_forward(moon, TICKERS, _cfg(), 10000.0)
    pd.testing.assert_frame_equal(
        r_crash.target_weights.loc[[d1]], r_moon.target_weights.loc[[d1]])
    assert r_crash.metrics["final_value"] != pytest.approx(
        r_moon.metrics["final_value"])


def test_execution_day_price_moves_shares_not_targets():
    # Identical through D-1, wildly different closes ON D and after.
    px = _panel()
    cfg = _cfg()
    d1 = run_walk_forward(px, TICKERS, cfg, 10000.0).events[
        "Execution Date"].iloc[1]
    alt = px.copy()
    mask = alt.index >= d1
    alt.loc[mask] *= 7.0
    r_base = run_walk_forward(px, TICKERS, _cfg(), 10000.0)
    r_alt = run_walk_forward(alt, TICKERS, _cfg(), 10000.0)
    pd.testing.assert_frame_equal(
        r_base.target_weights.loc[[d1]], r_alt.target_weights.loc[[d1]])
    assert r_base.metrics["final_value"] != pytest.approx(
        r_alt.metrics["final_value"])


def test_rolling_window_ignores_ancient_history():
    px = _panel()
    cfg = _cfg(lookback_years=1.0)
    d1 = run_walk_forward(px, TICKERS, cfg, 10000.0).events[
        "Execution Date"].iloc[1]
    old = px.copy()
    old.iloc[:100] *= 50.0  # dramatic change far outside the 1y window
    r_base = run_walk_forward(px, TICKERS, _cfg(lookback_years=1.0), 10000.0)
    r_old = run_walk_forward(old, TICKERS, _cfg(lookback_years=1.0), 10000.0)
    pd.testing.assert_frame_equal(
        r_base.target_weights.loc[[d1]], r_old.target_weights.loc[[d1]])


def test_expanding_window_excludes_cutoff_and_later():
    px = _panel()
    cut = pd.Timestamp("2020-06-15")
    est, _ = estimation_slice(px, TICKERS, cut, None, "expanding", 60)
    assert est.index.max() < cut
    # ...but every earlier observation is eligible.
    assert est.index[0] == px.index[0]


# ---------------------------------------------------------------------------
# Dynamic behaviour
# ---------------------------------------------------------------------------

def test_reestimation_changes_targets_on_regime_shift():
    # Asset A calm then wild while B/C stay calm: ERC must rotate away from
    # A as its risk contribution explodes (uniform vol scaling would NOT move
    # risk parity — the shift has to be differential).
    idx = pd.bdate_range("2015-01-01", periods=2100)
    rng = np.random.default_rng(4)
    a = np.concatenate([rng.normal(0.0004, 0.005, 1150),
                        rng.normal(0.0002, 0.030, 950)])
    b = rng.normal(0.0004, 0.008, 2100)
    c = rng.normal(0.0003, 0.008, 2100)
    px = pd.DataFrame(100 * np.exp(np.cumsum(np.column_stack([a, b, c]),
                                             axis=0)),
                      index=idx, columns=TICKERS)
    res = run_walk_forward(px, TICKERS, _cfg(), 10000.0)
    assert res.success
    first = res.target_weights.iloc[0].values
    last = res.target_weights.iloc[-1].values
    assert np.abs(first - last).max() > 0.05


def test_shares_fixed_between_events_wealth_is_shares_times_prices():
    # Zero costs, constant targets: within each inter-event segment wealth
    # must equal the shares established at that segment's start × prices
    # (shares reset at rebalances even when targets are unchanged, because
    # drifted weights are traded back to target).
    px = _panel()
    w = np.array([0.5, 0.3, 0.2])
    res = run_walk_forward(px, TICKERS, _cfg(transaction_cost_bps=0.0),
                           10000.0, estimator=_const_estimator(w))
    assert res.success
    bounds = [res.values.index[0]] + list(
        res.events["Execution Date"].iloc[1:]) + [res.values.index[-1]]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        seg = px.loc[lo:hi]
        shares = float(res.values.loc[lo]) * w / seg.iloc[0].values
        expect = (seg * shares).sum(axis=1)
        pd.testing.assert_series_equal(
            res.values.loc[lo:hi], expect.rename("value"),
            check_names=False, check_freq=False)


def test_static_equivalence_with_constant_targets():
    # Frozen estimator targets == static periodic backtest (same conventions).
    px = _panel()
    cfg = _cfg()
    w = np.array([0.5, 0.3, 0.2])
    dyn = run_walk_forward(px, TICKERS, cfg, 10000.0,
                           estimator=_const_estimator(w))
    assert dyn.success
    from analytics.backtest import prepare_backtest_data
    window, _ = prepare_backtest_data(
        px, TICKERS, cfg.test_start, cfg.test_end)
    static = backtest_rebalanced(window, w, 10000.0, "quarterly", 10.0)
    pd.testing.assert_series_equal(
        dyn.values, static.values.rename("value"), check_names=False,
        rtol=1e-9, atol=1e-6)


def test_cost_accounting_exact_at_each_event():
    px = _panel()
    res = run_walk_forward(px, TICKERS, _cfg(), 10000.0)
    assert res.success
    ev = res.events[res.events["Status"] == "success"]
    assert len(ev) >= 2
    dev = (ev["Portfolio Value Before"] - ev["Transaction Cost"]
           - ev["Portfolio Value After"]).abs()
    assert (dev < 1e-6).all()
    # Reported cost == rate × Σ|final executed trades| per event date.
    for d, grp in res.trades.groupby("Date"):
        row = ev[ev["Execution Date"] == pd.Timestamp(d)].iloc[0]
        expect = float(np.abs(grp["Trade Value"]).sum() * 10.0 / 10_000.0)
        assert row["Transaction Cost"] == pytest.approx(expect)
    assert res.metrics["total_costs"] == pytest.approx(res.costs.sum())


def test_zero_trade_cost_when_targets_unchanged():
    # Single asset, constant 100% target: every event needs no trade, so no
    # calendar event may charge a cost.
    px1 = _panel()[["A"]]
    cfg = _cfg()
    res = run_walk_forward(px1, ["A"], cfg, 10000.0,
                           estimator=_const_estimator([1.0], ["A"]))
    assert res.success
    assert (res.turnover == 0.0).all()
    assert (res.costs == 0.0).all()
    assert res.metrics["total_costs"] == pytest.approx(0.0)


def test_failed_intermediate_rebalance_holds_and_continues():
    px = _panel()
    cfg = _cfg()
    e1 = run_walk_forward(px, TICKERS, cfg, 10000.0,
                          estimator=_const_estimator([1 / 3] * 3)
                          ).events["Execution Date"].iloc[1]

    def _flaky(est, d):
        if pd.Timestamp(d) == pd.Timestamp(e1):
            return None, {}, "solver exploded"
        return _const_estimator([1 / 3] * 3)(est, d)

    res = run_walk_forward(px, TICKERS, _cfg(), 10000.0, estimator=_flaky)
    assert res.success  # run survives
    fail = res.events[res.events["Status"] == "failed/skipped"]
    assert len(fail) == 1 and fail["Execution Date"].iloc[0] == e1
    assert fail["Transaction Cost"].iloc[0] == 0.0
    assert fail["Turnover"].iloc[0] == 0.0
    assert "solver exploded" in fail["Reason"].iloc[0]
    assert len(res.failures) == 1
    later = res.events[res.events["Status"] == "success"]
    assert (later["Execution Date"] > e1).any()  # keeps optimising after
    assert np.isfinite(res.values).all()


def test_failed_initial_allocation_has_no_wealth_path():
    px = _panel()

    def _dead(est, d):
        return None, {}, "no convergence ever"

    res = run_walk_forward(px, TICKERS, _cfg(), 10000.0, estimator=_dead)
    assert res.success is False
    assert len(res.values) == 0
    assert "no wealth path" in res.message
    assert len(res.failures) == 1


def test_column_order_independence():
    px = _panel()
    shuffled = px[["C", "A", "B"]]
    r1 = run_walk_forward(px, TICKERS, _cfg(), 10000.0)
    r2 = run_walk_forward(shuffled, TICKERS, _cfg(), 10000.0)
    pd.testing.assert_frame_equal(r1.target_weights, r2.target_weights)
    pd.testing.assert_series_equal(r1.values, r2.values.rename("value"),
                                   check_names=False)


def test_weight_bounds_respected():
    px = _panel()
    res = run_walk_forward(px, TICKERS, _cfg(min_weight=0.2, max_weight=0.6),
                           10000.0)
    assert res.success
    assert ((res.target_weights >= 0.2 - 1e-9)
            & (res.target_weights <= 0.6 + 1e-9)).all().all()


def test_cost_drag_is_gross_minus_net():
    px = _panel()
    net = run_walk_forward(px, TICKERS, _cfg(transaction_cost_bps=25.0),
                           10000.0)
    gross = run_walk_forward(px, TICKERS, _cfg(transaction_cost_bps=0.0),
                             10000.0)
    assert net.success and gross.success
    pd.testing.assert_frame_equal(net.target_weights, gross.target_weights)
    drag = gross.metrics["final_value"] - net.metrics["final_value"]
    assert drag >= -1e-6  # costs never create wealth
    assert net.metrics["total_costs"] > 0


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def test_all_methods_run_end_to_end():
    px = _panel()
    for method in ("min_variance", "max_sharpe", "erc",
                   "max_diversification", "equal_weight"):
        res = run_walk_forward(px, TICKERS, _cfg(method=method), 10000.0)
        assert res.success, (method, res.message)
        assert (res.events["Status"] == "success").any()
        assert np.isfinite(res.values).all()
        assert abs(res.target_weights.sum(axis=1).iloc[0] - 1.0) < 1e-6


def test_black_litterman_fixed_view_evolves_with_covariance():
    idx = pd.bdate_range("2015-01-01", periods=2100)
    rng = np.random.default_rng(9)
    a = np.concatenate([rng.normal(0.0004, 0.005, 1150),
                        rng.normal(0.0002, 0.030, 950)])
    b = rng.normal(0.0004, 0.008, 2100)
    c = rng.normal(0.0003, 0.008, 2100)
    px = pd.DataFrame(100 * np.exp(np.cumsum(np.column_stack([a, b, c]),
                                             axis=0)),
                      index=idx, columns=TICKERS)
    views = [{"type": "absolute", "ticker": "A", "return": 0.15,
              "confidence": 0.7}]
    cfg = _cfg(method="black_litterman")
    cfg.bl_views = views
    res = run_walk_forward(px, TICKERS, cfg, 10000.0)
    assert res.success, res.message
    assert (res.events["Status"] == "success").sum() >= 2
    first = res.target_weights.iloc[0].values
    last = res.target_weights.iloc[-1].values
    assert np.abs(first - last).max() > 0.03  # prior/cov moved, view fixed


def test_erc_targets_equalise_event_risk():
    from analytics.walkforward import estimation_slice as _es
    px = _panel()
    res = run_walk_forward(px, TICKERS, _cfg(method="erc"), 10000.0)
    assert res.success
    for _, row in res.events[res.events["Status"] == "success"].iterrows():
        d = pd.Timestamp(row["Execution Date"])
        est, _ = _es(px, TICKERS, d, 3.0, "rolling", 60)
        cov = annualised_covariance_simple(
            simple_returns_from_prices(est, TICKERS))
        prc = risk_contributions_simple(
            res.target_weights.loc[d].values, cov, TICKERS)
        assert np.abs(prc.values - 1 / 3).max() <= 0.02


def test_maxdiv_targets_beat_equal_weight_dr():
    from analytics.walkforward import estimation_slice as _es
    px = _panel()
    res = run_walk_forward(
        px, TICKERS, _cfg(method="max_diversification"), 10000.0)
    assert res.success
    ew = np.ones(3) / 3
    for _, row in res.events[res.events["Status"] == "success"].iterrows():
        d = pd.Timestamp(row["Execution Date"])
        est, _ = _es(px, TICKERS, d, 3.0, "rolling", 60)
        cov = annualised_covariance_simple(
            simple_returns_from_prices(est, TICKERS))
        w = res.target_weights.loc[d].values
        assert diversification_ratio(w, cov) + 1e-9 >= diversification_ratio(
            ew, cov)


def test_bl_benchmark_delta_uses_only_training_window():
    from analytics.walkforward import _window_delta
    idx = pd.bdate_range("2018-01-01", periods=500)
    rng = np.random.default_rng(5)
    bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 500))),
                      index=idx)
    d1 = pd.Timestamp("2020-01-02")
    delta_a = _window_delta(bench, "2019-01-02", d1, 0.02)
    moved = bench.copy()
    moved.loc[moved.index >= d1] *= 3.0  # future benchmark shock
    delta_b = _window_delta(moved, "2019-01-02", d1, 0.02)
    assert delta_a == pytest.approx(delta_b)
    with pytest.raises(ValueError):
        _window_delta(None, "2019-01-02", d1, 0.02)


# ---------------------------------------------------------------------------
# Multi-currency + isolation
# ---------------------------------------------------------------------------

def _fx_panel(n=2100, seed=21):
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({
        "GBP": np.full(n, 1.25) + np.linspace(0, 0.05, n),
        "EUR": np.full(n, 1.10),
    }, index=idx)


def _mixed_base_panel():
    n = 2100
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(21)
    native_a = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.010, n)))
    native_b = 300 + np.cumsum(rng.normal(0.05, 1.5, n))   # pence
    native_c = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.009, n)))
    fx = _fx_panel(n)
    base = pd.DataFrame(index=idx)
    base["A"] = native_a  # USD
    base["B"] = convert_price_series(
        pd.Series(native_b * 0.01, index=idx), "GBP", "USD", fx)
    base["C"] = convert_price_series(
        pd.Series(native_c, index=idx), "EUR", "USD", fx)
    return base, fx


def test_mixed_currency_walk_forward_uses_base_covariance():
    base, _ = _mixed_base_panel()
    res = run_walk_forward(base, TICKERS, _cfg(), 10000.0)
    assert res.success, res.message
    # Manual base-panel computation matches the engine's first targets.
    d0 = res.events["Execution Date"].iloc[0]
    est, _ = estimation_slice(base, TICKERS, pd.Timestamp(d0), 3.0,
                              "rolling", 60)
    rets = simple_returns_from_prices(est, TICKERS)
    mu = annualised_mean_returns_simple(rets)
    cov = annualised_covariance_simple(rets)
    ref = equal_risk_contribution(TICKERS, mu.values, cov, 0.05, 0.0, 1.0)
    assert ref.success
    pd.testing.assert_series_equal(
        res.target_weights.loc[d0], ref.weights.reindex(TICKERS),
        check_names=False)


def test_future_fx_cannot_move_earlier_weights():
    base, fx = _mixed_base_panel()
    d1 = run_walk_forward(base, TICKERS, _cfg(), 10000.0).events[
        "Execution Date"].iloc[1]
    fx2 = fx.copy()
    fx2.loc[fx2.index >= pd.Timestamp(d1), "GBP"] *= 100.0
    idx = base.index
    alt = pd.DataFrame(index=idx)
    alt["A"] = base["A"]
    # Rebuild B from native with shocked FX: native GBP path is unchanged,
    # only the FX translation from D onward explodes.
    native_gbp = base["B"] / fx["GBP"].values  # £ path (exact pre-shock)
    alt["B"] = convert_price_series(native_gbp, "GBP", "USD", fx2)
    alt["C"] = base["C"]
    r_base = run_walk_forward(base, TICKERS, _cfg(), 10000.0)
    r_alt = run_walk_forward(alt, TICKERS, _cfg(), 10000.0)
    pd.testing.assert_frame_equal(
        r_base.target_weights.loc[[d1]], r_alt.target_weights.loc[[d1]])


def test_walkforward_module_has_no_live_imports():
    import ast
    from pathlib import Path
    tree = ast.parse(
        (Path(__file__).resolve().parent.parent / "analytics"
         / "walkforward.py").read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert not ({"yfinance", "alpaca", "streamlit", "data"} & mods), mods


def test_turnover_diagnostics_contract():
    px = _panel()
    res = run_walk_forward(px, TICKERS, _cfg(), 10000.0)
    d = turnover_diagnostics(res)
    assert d["n_scheduled"] == d["n_successful"] + d["n_failed"]
    assert d["n_successful"] >= 2
    assert d["total_costs"] == pytest.approx(res.metrics["total_costs"])
    assert d["cumulative_turnover"] == pytest.approx(
        res.metrics["cumulative_turnover"])
    bad = turnover_diagnostics(None)
    assert bad["n_scheduled"] == 0

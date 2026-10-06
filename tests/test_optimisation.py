"""Tests for analytics.optimisation incl. constraint validation (FIX 6/9)."""

import numpy as np
import pytest

from analytics.optimisation import (
    efficient_frontier,
    feasible_return_range,
    feasible_volatility_range,
    max_sharpe,
    min_variance,
    rebalance_trades,
    target_return,
    target_volatility,
)

MU = np.array([0.12, 0.08, 0.05, 0.03])
COV = np.array([
    [0.0400, 0.0080, 0.0040, 0.0020],
    [0.0080, 0.0225, 0.0030, 0.0015],
    [0.0040, 0.0030, 0.0100, 0.0010],
    [0.0020, 0.0015, 0.0010, 0.0064],
])
RF = 0.03


def test_min_variance_sums_to_one_and_respects_bounds():
    res = min_variance(MU, COV, RF, 0.0, 1.0)
    assert res.success
    assert res.weights.sum() == pytest.approx(1.0)
    assert ((res.weights >= 0.0) & (res.weights <= 1.0)).all()


def test_max_sharpe_beats_equal_weight():
    res = max_sharpe(MU, COV, RF, 0.0, 1.0)
    assert res.success
    w_eq = np.ones(4) / 4
    sr_eq = (w_eq @ MU - RF) / np.sqrt(w_eq @ COV @ w_eq)
    assert res.sharpe >= sr_eq - 1e-6


def test_target_return_feasible_and_infeasible():
    lo, hi = feasible_return_range(MU, 0.0, 1.0)
    assert lo == pytest.approx(0.03)  # min asset return attainable
    assert hi == pytest.approx(0.12)  # max asset return attainable
    mid = (lo + hi) / 2
    res = target_return(MU, COV, mid, RF, 0.0, 1.0)
    assert res.success
    assert res.expected_return == pytest.approx(mid, abs=1e-4)
    bad = target_return(MU, COV, hi + 0.50, RF, 0.0, 1.0)
    assert not bad.success
    assert "outside the feasible range" in bad.message
    assert np.all(np.isfinite(bad.weights))


def test_target_volatility_feasible_and_infeasible():
    lo, hi = feasible_volatility_range(MU, COV, 0.0, 1.0)
    assert lo > 0 and hi > lo
    mid = (lo + hi) / 2
    res = target_volatility(MU, COV, mid, RF, 0.0, 1.0)
    assert res.success
    assert res.volatility == pytest.approx(mid, abs=1e-3)
    bad = target_volatility(MU, COV, hi + 0.50, RF, 0.0, 1.0)
    assert not bad.success
    assert np.all(np.isfinite(bad.weights))


def test_infeasible_weight_bounds_short_circuit():
    # 3 assets capped at 20% can sum to at most 60%.
    r1 = min_variance(np.ones(3) * 0.05, np.eye(3) * 0.04, RF, 0.0, 0.20)
    assert not r1.success
    assert "at most" in r1.message
    # 10 assets floored at 20% already sum to 200%.
    r2 = max_sharpe(np.ones(10) * 0.05, np.eye(10) * 0.04, RF, 0.20, 1.0)
    assert not r2.success
    assert "already sum" in r2.message
    # Malformed ordering.
    r3 = min_variance(MU, COV, RF, 0.5, 0.2)
    assert not r3.success
    assert np.all(np.isfinite(r3.weights)) and r3.weights.sum() == pytest.approx(1.0)


def test_efficient_frontier_and_rebalance():
    ef = efficient_frontier(MU, COV, RF, n_points=10)
    assert not ef.empty
    assert list(ef.columns) == ["return", "volatility", "sharpe"]
    assert np.all(np.isfinite(ef.values))
    trades = rebalance_trades(
        np.ones(4) / 4, np.array([0.4, 0.3, 0.2, 0.1]),
        ["A", "B", "C", "D"], 100_000.0,
    )
    assert set(trades["Action"]) <= {"BUY", "SELL", "HOLD"}
    assert trades["Trade ($)"].sum() == pytest.approx(0.0, abs=1e-6)

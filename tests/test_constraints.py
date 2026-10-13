"""M5 — real-world optimisation constraints: bounds, turnover, cost, cash,
currency caps and custom risk budgets (all engines, feasibility honest)."""

import numpy as np
import pandas as pd
import pytest

from analytics.allocation import (
    black_litterman_allocation,
    black_litterman_posterior,
    equal_risk_contribution,
    maximum_diversification,
)
from analytics.constraints import (
    OptimizationConstraints,
    check_constraints,
    max_single_name_shift,
    min_achievable_turnover,
    turnover_fraction,
    validate_risk_budgets,
)
from analytics.optimisation import max_sharpe, min_variance

T = ["AAA", "BBB", "CCC"]
MU = np.array([0.08, 0.10, 0.06])
COV = pd.DataFrame(
    [[0.0400, 0.0080, 0.0020],
     [0.0080, 0.0625, 0.0040],
     [0.0020, 0.0040, 0.0225]],
    index=T, columns=T)


def test_turnover_convention():
    w = np.array([0.5, 0.3, 0.2])
    w0 = np.array([1 / 3, 1 / 3, 1 / 3])
    assert turnover_fraction(w, w0) == pytest.approx(
        (abs(0.5 - 1 / 3) + abs(0.3 - 1 / 3) + abs(0.2 - 1 / 3)) / 2)
    assert max_single_name_shift(w, w0) == pytest.approx(0.5 - 1 / 3)


def test_per_asset_bounds_enforced():
    c = OptimizationConstraints(per_asset_max={"AAA": 0.10})
    r = min_variance(MU, COV.values, constraints=c, tickers=T)
    assert r.success, r.message
    assert r.weights[0] <= 0.10 + 1e-6
    assert abs(r.weights.sum() - 1.0) < 1e-6


def test_infeasible_bounds_explained():
    c = OptimizationConstraints(
        per_asset_max={"AAA": 0.2, "BBB": 0.2, "CCC": 0.2})
    r = min_variance(MU, COV.values, constraints=c, tickers=T)
    assert not r.success and r.weights is None
    assert "cannot reach" in r.message
    assert check_constraints(T, c)


def test_turnover_limit_respected():
    w0 = np.array([1 / 3, 1 / 3, 1 / 3])
    c = OptimizationConstraints(turnover_limit=0.10,
                                turnover_reference=w0)
    r = min_variance(MU, COV.values, constraints=c, tickers=T)
    assert r.success, r.message
    assert turnover_fraction(r.weights, w0) <= 0.10 + 1e-4


def test_turnover_floor_explained():
    w0 = np.array([0.0, 0.5, 0.5])
    c = OptimizationConstraints(
        per_asset_min={"AAA": 0.9},
        turnover_limit=0.01, turnover_reference=w0)
    floor = min_achievable_turnover(T, c)
    assert floor == pytest.approx(0.90)
    assert floor > 0.01
    r = max_sharpe(MU, COV.values, constraints=c, tickers=T)
    assert not r.success and "turnover" in r.message.lower()


def test_cost_penalty_reduces_turnover():
    w0 = np.array([1 / 3, 1 / 3, 1 / 3])
    plain = min_variance(MU, COV.values)
    pen = min_variance(
        MU, COV.values,
        constraints=OptimizationConstraints(
            transaction_cost_bps=500.0, turnover_reference=w0),
        tickers=T)
    assert plain.success and pen.success
    assert turnover_fraction(pen.weights, w0) <= \
        turnover_fraction(plain.weights, w0) + 1e-9


def test_cash_target_sums_and_stats():
    c = OptimizationConstraints(cash_target=0.20)
    r = min_variance(MU, COV.values, rf=0.03, constraints=c, tickers=T)
    assert r.success, r.message
    assert abs(r.weights.sum() - 0.80) < 1e-6
    assert r.expected_return == pytest.approx(
        float(r.weights @ MU) + 0.20 * 0.03)


def test_currency_cap_respected():
    c = OptimizationConstraints(
        asset_currencies=["USD", "USD", "GBP"],
        currency_caps={"USD": 0.50})
    r = min_variance(MU, COV.values, constraints=c, tickers=T)
    assert r.success, r.message
    assert r.weights[0] + r.weights[1] <= 0.50 + 1e-6


def test_custom_risk_budgets():
    budgets = {"AAA": 0.2, "BBB": 0.3, "CCC": 0.5}
    r = equal_risk_contribution(T, MU, COV, budgets=budgets)
    assert r.success, r.message
    prc = r.risk_contributions
    assert prc["AAA"] == pytest.approx(0.2, abs=0.02)
    assert prc["BBB"] == pytest.approx(0.3, abs=0.02)
    assert prc["CCC"] == pytest.approx(0.5, abs=0.02)
    # via the constraint model too
    r2 = equal_risk_contribution(
        T, MU, COV,
        constraints=OptimizationConstraints(risk_budgets=budgets))
    assert r2.success, r2.message


def test_invalid_budgets_rejected():
    assert validate_risk_budgets(None, T) is None
    bad_sum = {"AAA": 0.5, "BBB": 0.5, "CCC": 0.5}
    r = equal_risk_contribution(T, MU, COV, budgets=bad_sum)
    assert not r.success and "sum to 1" in r.message
    neg = {"AAA": -0.1, "BBB": 0.6, "CCC": 0.5}
    with pytest.raises(ValueError):
        validate_risk_budgets(neg, T)


def test_constraints_across_methods():
    w0 = np.array([0.4, 0.3, 0.3])
    c = OptimizationConstraints(
        per_asset_min={"CCC": 0.10},
        turnover_limit=0.40, turnover_reference=w0,
        asset_currencies=["USD", "GBP", "EUR"],
        currency_caps={"USD": 0.70})
    rs = [
        min_variance(MU, COV.values, constraints=c, tickers=T),
        max_sharpe(MU, COV.values, constraints=c, tickers=T),
        equal_risk_contribution(T, MU, COV, constraints=c),
        maximum_diversification(T, MU, COV, constraints=c),
    ]
    post = black_litterman_posterior(
        T, COV, np.ones(3) / 3, 2.5, 0.05, [])
    assert post.success
    rs.append(black_litterman_allocation(
        T, post.posterior, COV, 2.5, constraints=c))
    for r in rs:
        assert r.success, r.message
        assert turnover_fraction(r.weights if isinstance(
            r.weights, np.ndarray) else r.weights.values, w0) <= 0.40 + 1e-4


def test_dict_bounds_need_tickers():
    c = OptimizationConstraints(per_asset_max={"AAA": 0.5})
    r = min_variance(MU, COV.values, constraints=c)
    assert not r.success and "ticker" in r.message.lower()

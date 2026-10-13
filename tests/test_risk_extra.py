"""Tests for analytics.risk_extra (component/marginal VaR, diagnostics)."""

import math

import numpy as np
import pandas as pd
import pytest

from analytics.risk_extra import (
    component_var,
    distribution_stats,
    diversification_benefit,
    herfindahl,
    marginal_var,
    omega_ratio,
    rolling_beta,
    rolling_correlation,
    rolling_tracking_error,
    upside_downside_capture,
)

COV = pd.DataFrame(np.array([[0.04, 0.01], [0.01, 0.03]]),
                   index=["A", "B"], columns=["A", "B"])
W = np.array([0.6, 0.4])


def test_component_var_reconciles_to_total():
    var_p = 0.05
    comp = component_var(W, COV, var_p, value=10000.0)
    assert comp.sum() == pytest.approx(var_p * 10000.0, rel=1e-9)
    assert list(comp.index) == ["A", "B"]
    with pytest.raises(ValueError):
        component_var(W, np.zeros((2, 2)), var_p)


def test_marginal_var_scales_components():
    var_p, value = 0.05, 10000.0
    marg = marginal_var(W, COV, var_p, value)
    comp = component_var(W, COV, var_p, value)
    assert (marg.values * W) == pytest.approx(comp.values, rel=1e-9)
    with pytest.raises(ValueError):
        marginal_var([0.5], COV, var_p)


def test_diversification_benefit_nonnegative():
    d = diversification_benefit(W, COV)
    assert d["diversification_benefit"] >= -1e-12
    assert d["weighted_standalone_vol"] >= d["portfolio_vol"] - 1e-12
    assert herfindahl(W) == pytest.approx(0.52)
    assert herfindahl(np.array([1.0, 0.0])) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        herfindahl([])


def test_distribution_labels():
    rng = np.random.default_rng(2)
    r = rng.normal(0.001, 0.01, 2000)
    s = distribution_stats(r)
    assert abs(s["skewness"]) < 0.2
    assert abs(s["excess_kurtosis"]) < 0.3  # excess (Fisher), normal ≈ 0
    assert s["n_obs"] == 2000
    with pytest.raises(ValueError):
        distribution_stats([0.01, 0.02, 0.03])
    with pytest.raises(ValueError):
        distribution_stats(np.full(10, 0.01))


def test_omega_with_mar():
    r = np.array([0.02, -0.01, 0.03, -0.005, 0.015])
    assert omega_ratio(r, 0.0) == pytest.approx(0.065 / 0.015)
    assert omega_ratio(np.array([0.01, 0.02]), 0.0) == math.inf
    with pytest.raises(ValueError):
        omega_ratio([], 0.0)


def test_capture_uses_benchmark_subsets():
    b = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    p = 1.2 * b
    c = upside_downside_capture(p, b)
    assert c["upside_capture"] == pytest.approx(1.2)
    assert c["downside_capture"] == pytest.approx(1.2)
    assert (c["n_up"], c["n_down"]) == (3, 2)
    with pytest.raises(ValueError):
        upside_downside_capture([0.01, 0.02], [0.01, 0.02])  # no down


def test_rolling_risk_no_future_leakage():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(3)
    b = pd.Series(rng.normal(0.0005, 0.01, 300), index=idx)
    p = pd.Series(1.1 * b.values + rng.normal(0, 0.002, 300), index=idx)
    beta = rolling_beta(p, b, 63)
    assert beta.dropna().iloc[0] == pytest.approx(
        np.cov(p.iloc[:63], b.iloc[:63])[0, 1] / np.var(b.iloc[:63], ddof=1),
        rel=1e-9)
    assert beta.iloc[-63:].notna().all() and beta.iloc[:62].isna().all()
    corr = rolling_correlation(p, b, 63)
    assert corr.dropna().between(-1, 1).all()
    te = rolling_tracking_error(p, b, 63)
    assert (te.dropna() >= 0).all()
    with pytest.raises(ValueError):
        rolling_beta(p, b, 1)

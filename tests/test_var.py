"""Tests for analytics.var: parametric, historical (rolling), Monte Carlo."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from analytics.var import historical_var, monte_carlo_var, parametric_var

rng = np.random.default_rng(11)
DAILY = pd.Series(rng.normal(0.0005, 0.01, size=300))


def test_parametric_var_manual():
    # z * sigma_daily * V * sqrt(T)
    v = parametric_var(0.01, 100_000.0, 0.95, 10)
    assert v == pytest.approx(norm.ppf(0.95) * 0.01 * 100_000.0 * np.sqrt(10))


def test_historical_var_1d_uses_log_to_simple():
    v = historical_var(DAILY, 100_000.0, 0.95, 1)
    tail_log = np.percentile(DAILY, 5)
    assert v == pytest.approx(-(np.exp(tail_log) - 1) * 100_000.0)


def test_historical_var_multiday_uses_rolling_not_sqrt():
    T = 10
    v = historical_var(DAILY, 100_000.0, 0.95, T)
    rolling = DAILY.rolling(window=T).sum().dropna()
    expected = -(np.exp(np.percentile(rolling, 5)) - 1) * 100_000.0
    assert v == pytest.approx(expected)
    # And it must differ from naive sqrt(T) scaling in general.
    naive = -(np.exp(np.percentile(DAILY, 5)) - 1) * 100_000.0 * np.sqrt(T)
    assert abs(v - naive) > 1e-6


def test_historical_var_insufficient_history_raises():
    short = pd.Series(np.linspace(-0.01, 0.01, 25))
    with pytest.raises(ValueError, match="Insufficient history"):
        historical_var(short, 100_000.0, 0.95, 10)
    with pytest.raises(ValueError):
        historical_var(pd.Series([], dtype=float), 100_000.0, 0.95, 1)


def test_monte_carlo_var_converts_log_to_simple():
    w = np.array([0.6, 0.4])
    mu_d = np.array([0.0004, 0.0002])
    cov_d = np.array([[0.0004, 0.0001], [0.0001, 0.0002]])
    var_usd, sim_rets = monte_carlo_var(
        w, mu_d, cov_d, 100_000.0, 0.95, 5, n_simulations=5_000, seed=42
    )
    # Returned sims are SIMPLE returns: var == -pct5 * V exactly.
    assert var_usd == pytest.approx(-np.percentile(sim_rets, 5) * 100_000.0)
    assert (sim_rets > -1.0).all()  # simple returns bounded below by -100%
    assert len(sim_rets) == 5_000


def test_monte_carlo_var_deterministic_seed():
    w = np.array([0.5, 0.5])
    mu_d = np.array([0.0, 0.0])
    cov_d = np.eye(2) * 0.0001
    v1, s1 = monte_carlo_var(w, mu_d, cov_d, 50_000.0, 0.95, 3, 2_000, seed=1)
    v2, s2 = monte_carlo_var(w, mu_d, cov_d, 50_000.0, 0.95, 3, 2_000, seed=1)
    assert v1 == v2
    assert np.array_equal(s1, s2)


def test_var_input_validation():
    with pytest.raises(ValueError):
        parametric_var(0.01, -5.0, 0.95, 1)
    with pytest.raises(ValueError):
        parametric_var(0.01, 100.0, 0.40, 1)
    with pytest.raises(ValueError):
        historical_var(DAILY, 100.0, 0.95, 0)
    with pytest.raises(ValueError):
        monte_carlo_var(np.array([1.0]), np.array([0.0, 0.0]),
                        np.eye(2) * 0.001, 100.0)

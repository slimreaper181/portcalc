"""Tests for analytics.scenario (FIX 4 contributions, FIX 5 memory safety)."""

import numpy as np
import pandas as pd
import pytest

from analytics.scenario import (
    calculate_outcome_distribution,
    count_deposits,
    explain_scenario_results,
    run_predefined_scenario,
    simulate_portfolio_paths,
    summarise_future_metrics,
    total_contributed_capital,
)

W = np.array([0.6, 0.4])
MU = np.array([0.08, 0.06])
COV = np.array([[0.04, 0.01], [0.01, 0.0225]])


def test_seeded_simulation_is_deterministic():
    p1 = simulate_portfolio_paths(W, MU, COV, 100_000.0, years=2, n_sims=500, seed=9)
    p2 = simulate_portfolio_paths(W, MU, COV, 100_000.0, years=2, n_sims=500, seed=9)
    assert np.array_equal(p1, p2)
    assert p1.shape == (500, 2 * 252 + 1)
    assert (p1[:, 0] == 100_000.0).all()


def test_simulation_does_not_mutate_inputs():
    w, mu, cov = W.copy(), MU.copy(), COV.copy()
    simulate_portfolio_paths(w, mu, cov, 10_000.0, years=1, n_sims=200, seed=3)
    assert np.array_equal(w, W) and np.array_equal(mu, MU) and np.array_equal(cov, COV)
    mu2, cov2 = MU.copy(), COV.copy()
    run_predefined_scenario(mu2, cov2, "Market Crash")
    assert np.array_equal(mu2, MU) and np.array_equal(cov2, COV)


def test_deposit_count_and_total_contributed():
    # 5y * 252 trading days, ~21-day months -> 60 deposits.
    assert count_deposits(5 * 252) == 60
    init, contrib, total = total_contributed_capital(100_000.0, 1_000.0, 5)
    assert contrib == 60_000.0
    assert total == 160_000.0


def test_contributions_raise_final_values_and_break_even():
    base = simulate_portfolio_paths(W, MU, COV, 100_000.0, years=2,
                                    n_sims=500, monthly_contrib=0.0, seed=5)
    with_c = simulate_portfolio_paths(W, MU, COV, 100_000.0, years=2,
                                      n_sims=500, monthly_contrib=500.0, seed=5)
    assert (with_c[:, -1] > base[:, -1]).all()
    m = calculate_outcome_distribution(with_c, 100_000.0,
                                       monthly_contrib=500.0, years=2, trading_days=252)
    n_dep = count_deposits(2 * 252)
    assert m["contributions"] == pytest.approx(500.0 * n_dep)
    assert m["total_contributed"] == pytest.approx(100_000.0 + 500.0 * n_dep)
    assert m["expected_profit_loss"] == pytest.approx(
        m["expected_value"] - m["total_contributed"])


def test_prob_loss_uses_total_contributed_not_initial():
    # Steep negative drift: finals land between initial and total invested.
    mu_neg = np.array([-0.05, -0.05])
    cov_tiny = np.eye(2) * 1e-10
    paths = simulate_portfolio_paths(W, mu_neg, cov_tiny,
                                     100_000.0, years=5, n_sims=300,
                                     monthly_contrib=1_000.0, seed=5)
    m = calculate_outcome_distribution(paths, 100_000.0,
                                       monthly_contrib=1_000.0, years=5)
    assert m["total_contributed"] == pytest.approx(160_000.0)
    # Every path is below total invested (loss), even though finals exceed
    # the initial 100k — the old `final < initial` logic would say ~0 loss.
    assert m["prob_loss"] == pytest.approx(1.0)
    assert (m["final_values"] > 100_000.0).all()


def test_predefined_scenarios_order_and_shocks():
    mu_crash, cov_crash, info_crash = run_predefined_scenario(MU, COV, "Market Crash")
    mu_bull, cov_bull, _ = run_predefined_scenario(MU, COV, "Bull Market")
    mu_hv, cov_hv, _ = run_predefined_scenario(MU, COV, "High Volatility")
    assert info_crash["return_shock"] == pytest.approx(-0.20)
    assert np.allclose(mu_crash, MU - 0.20)
    assert np.allclose(cov_hv, COV * 4.0)
    # Same seed: bull expected > normal > crash.
    kw = dict(years=3, n_sims=1_000, monthly_contrib=0.0, seed=11)
    e_bull = simulate_portfolio_paths(W, mu_bull, cov_bull, 50_000.0, **kw)[:, -1].mean()
    e_base = simulate_portfolio_paths(W, MU, COV, 50_000.0, **kw)[:, -1].mean()
    e_crash = simulate_portfolio_paths(W, mu_crash, cov_crash, 50_000.0, **kw)[:, -1].mean()
    assert e_bull > e_base > e_crash
    # High-volatility regime widens the distribution.
    e_hv = simulate_portfolio_paths(W, mu_hv, cov_hv, 50_000.0, **kw)[:, -1]
    e_nm = simulate_portfolio_paths(W, MU, COV, 50_000.0, **kw)[:, -1]
    assert e_hv.std() > e_nm.std()


def test_custom_scenario_and_validation():
    mu_c, cov_c, info = run_predefined_scenario(
        MU, COV, "Custom", return_shock=-0.1, vol_multiplier=1.5)
    assert np.allclose(mu_c, MU - 0.1)
    assert np.allclose(cov_c, COV * 2.25)
    assert info["label"] == "Custom Scenario"
    with pytest.raises(ValueError):
        run_predefined_scenario(MU, COV, "Custom", vol_multiplier=0.0)
    with pytest.raises(ValueError):
        simulate_portfolio_paths(W, MU, COV, 1_000.0, years=5, n_sims=60_000)
    with pytest.raises(ValueError):
        simulate_portfolio_paths(W, MU, COV, -100.0, years=5, n_sims=500)


def test_summarise_and_explain_with_contributions():
    paths = simulate_portfolio_paths(W, MU, COV, 100_000.0, years=2,
                                     n_sims=400, monthly_contrib=200.0, seed=21)
    metrics, pct_df = summarise_future_metrics(paths, 100_000.0, 2,
                                               monthly_contrib=200.0)
    assert list(pct_df.columns) == ["p5", "p25", "p50", "p75", "p95"]
    assert len(pct_df) == paths.shape[1]
    text = explain_scenario_results(
        metrics, {"label": "Normal Market", "return_shock": 0.0,
                  "vol_multiplier": 1.0},
        100_000.0, monthly_contrib=200.0, years=2,
    )
    assert "total capital" in text.lower() or "invested" in text.lower()

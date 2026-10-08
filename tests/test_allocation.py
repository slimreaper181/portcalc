"""Tests for analytics.allocation (ERC, MaxDiv, Black-Litterman, diagnostics)."""

import numpy as np
import pandas as pd
import pytest

from analytics.allocation import (
    AllocationResult,
    BlackLittermanResult,
    ViewSpec,
    allocation_summary,
    annualised_covariance_simple,
    annualised_mean_returns_simple,
    black_litterman_allocation,
    black_litterman_posterior,
    build_view_matrices,
    diversification_ratio,
    effective_number_of_holdings,
    equal_risk_contribution,
    estimate_risk_aversion_from_benchmark,
    estimation_window,
    implied_equilibrium_returns,
    maximum_diversification,
    risk_contribution_dispersion,
    risk_contributions_simple,
    simple_returns_from_prices,
    validate_confidence,
    validate_covariance,
    validate_delta,
    validate_tau,
    view_uncertainty_matrix,
)

T3 = ["A", "B", "C"]
W3 = np.array([0.5, 0.3, 0.2])


def _prices_3x() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    return pd.DataFrame(
        {"A": [100, 101, 102, 101, 103, 104, 103, 105, 106, 107],
         "B": [50, 50.5, 49.5, 50, 51, 50.5, 51.5, 52, 51, 52],
         "C": [200, 202, 201, 203, 202, 204, 206, 205, 207, 209]},
        index=idx,
    )


def _cov_3x() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.04, 0.008, 0.002],
         [0.008, 0.0225, 0.003],
         [0.002, 0.003, 0.0100]],
        index=T3, columns=T3,
    )


def _mu_3x() -> np.ndarray:
    return np.array([0.10, 0.07, 0.05])


# ---------------------------------------------------------------------------
# Simple-return statistics
# ---------------------------------------------------------------------------

def test_simple_returns_conversion():
    px = _prices_3x()
    rets = simple_returns_from_prices(px, T3)
    assert list(rets.columns) == T3
    assert rets["A"].iloc[0] == pytest.approx(101 / 100 - 1)
    assert len(rets) == len(px) - 1
    mu = annualised_mean_returns_simple(rets)
    assert mu["A"] == pytest.approx(rets["A"].mean() * 252)
    cov = annualised_covariance_simple(rets)
    pd.testing.assert_frame_equal(cov, rets.cov() * 252)
    with pytest.raises(ValueError):
        simple_returns_from_prices(px, ["A", "ZZZ"])
    bad = px.copy()
    bad.iloc[3, 1] = -5.0
    with pytest.raises(ValueError):
        simple_returns_from_prices(bad, T3)


# ---------------------------------------------------------------------------
# Covariance validation
# ---------------------------------------------------------------------------

def test_validate_covariance_ok_and_labels():
    cov, adjusted = validate_covariance(_cov_3x(), T3)
    assert adjusted is False
    assert list(cov.index) == T3
    # Shuffled labels are reindexed, not rejected.
    cov2, _ = validate_covariance(_cov_3x().reindex(
        index=T3[::-1], columns=T3[::-1]), T3)
    pd.testing.assert_frame_equal(cov, cov2)


def test_validate_covariance_rejects():
    with pytest.raises(ValueError):
        validate_covariance(np.eye(2), T3)  # wrong shape
    bad = _cov_3x().copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_covariance(bad, T3)
    bad2 = _cov_3x().copy()
    bad2.iloc[0, 1] += 0.05  # material asymmetry
    with pytest.raises(ValueError):
        validate_covariance(bad2, T3)
    bad3 = pd.DataFrame(np.eye(3), index=["A", "B", "ZZZ"],
                        columns=["A", "B", "ZZZ"])
    with pytest.raises(ValueError):
        validate_covariance(bad3, T3)


def test_validate_covariance_ridge_fix():
    # Known tiny negative eigenvalue (-1e-10): minimal ridge, still close.
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    C = pd.DataFrame(Q @ np.diag([0.04, 0.0225, -1e-10]) @ Q.T,
                     index=T3, columns=T3)
    C = (C + C.T) / 2
    fixed, adjusted = validate_covariance(C, T3)
    assert adjusted is True
    assert np.linalg.eigvalsh(fixed.values).min() >= -1e-12
    assert np.abs(fixed.values - C.values).max() < 1e-6
    # Materially indefinite → reject, never silently repair.
    bad = pd.DataFrame(np.diag([0.04, 0.0225, -0.5]),
                       index=T3, columns=T3)
    with pytest.raises(ValueError):
        validate_covariance(bad, T3)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_diversification_ratio_manual():
    cov = np.diag([0.04, 0.0225])
    w = np.array([0.5, 0.5])
    expected = (0.5 * 0.2 + 0.5 * 0.15) / np.sqrt(
        0.25 * 0.04 + 0.25 * 0.0225)
    assert diversification_ratio(w, cov) == pytest.approx(expected)
    with pytest.raises(ValueError):
        diversification_ratio(np.zeros(2), np.diag([0.04, 0.0225]))


def test_effective_holdings_and_dispersion():
    assert effective_number_of_holdings(np.ones(4) / 4) == pytest.approx(4.0)
    assert effective_number_of_holdings(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert risk_contribution_dispersion(np.array([0.25] * 4)) == pytest.approx(0.0)
    assert risk_contribution_dispersion(
        np.array([0.4, 0.3, 0.2, 0.1])) > 0.1
    with pytest.raises(ValueError):
        effective_number_of_holdings(np.zeros(3))


def test_risk_contributions_labelled_and_summing():
    prc = risk_contributions_simple(W3, _cov_3x(), T3)
    assert list(prc.index) == T3
    assert prc.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        risk_contributions_simple(np.array([0.5, 0.5]), _cov_3x(), T3)


def test_allocation_summary():
    s = allocation_summary(W3, _mu_3x(), _cov_3x(), 0.02, T3)
    assert s["expected_return"] == pytest.approx(W3 @ _mu_3x())
    assert s["volatility"] == pytest.approx(
        np.sqrt(W3 @ _cov_3x().values @ W3))
    assert s["sharpe"] == pytest.approx(
        (W3 @ _mu_3x() - 0.02) / np.sqrt(W3 @ _cov_3x().values @ W3))
    assert s["largest_position"] == pytest.approx(0.5)
    assert isinstance(s["risk_contributions"], pd.Series)
    assert set(s) >= {"expected_return", "volatility", "sharpe",
                      "diversification_ratio", "largest_position",
                      "effective_holdings", "rc_dispersion",
                      "risk_contributions"}


# ---------------------------------------------------------------------------
# ERC — §42 synthetic + properties
# ---------------------------------------------------------------------------

def test_erc_identical_uncorrelated_is_equal_weight():
    # Uncorrelated, identical vol → ERC must be exactly equal weights/RC.
    tickers = ["A", "B", "C"]
    cov = pd.DataFrame(np.diag([0.04] * 3), index=tickers, columns=tickers)
    mu = np.array([0.08, 0.08, 0.08])
    res = equal_risk_contribution(tickers, mu, cov, rf=0.02)
    assert isinstance(res, AllocationResult)
    assert res.success and res.weights is not None
    assert res.weights.values == pytest.approx(np.ones(3) / 3, abs=1e-4)
    assert res.risk_contributions.values == pytest.approx(
        np.ones(3) / 3, abs=1e-4)
    assert res.risk_contributions.sum() == pytest.approx(1.0)


def test_erc_equalises_risk_on_correlated_data():
    res = equal_risk_contribution(T3, _mu_3x(), _cov_3x(), rf=0.02)
    assert res.success and res.weights is not None
    assert res.weights.sum() == pytest.approx(1.0)
    assert ((res.weights.values >= 0.0) & (res.weights.values <= 1.0)).all()
    assert np.isfinite(res.volatility) and res.volatility > 0
    prc = res.risk_contributions.values
    assert prc.sum() == pytest.approx(1.0)
    assert np.abs(prc - 1 / 3).max() <= 0.02


def test_erc_is_not_inverse_volatility():
    # With differing correlations, ERC must differ from inverse-vol weights.
    cov = pd.DataFrame([[0.04, 0.024, 0.0],
                        [0.024, 0.0225, 0.0],
                        [0.0, 0.0, 0.01]], index=T3, columns=T3)
    res = equal_risk_contribution(T3, _mu_3x(), cov, rf=0.02)
    assert res.success
    vols = np.sqrt(np.diag(cov.values))
    inv_vol = (1 / vols) / (1 / vols).sum()
    assert np.abs(res.weights.values - inv_vol).max() > 1e-3


def test_erc_failure_philosophy():
    res = equal_risk_contribution(T3, _mu_3x(), _cov_3x(),
                                  min_weight=0.4, max_weight=1.0)
    assert isinstance(res, AllocationResult)
    assert res.success is False and res.weights is None
    assert res.message


# ---------------------------------------------------------------------------
# Maximum Diversification
# ---------------------------------------------------------------------------

def test_maxdiv_diagonal_is_equal_weight_with_sqrt_n():
    tickers = ["A", "B", "C", "D"]
    cov = pd.DataFrame(np.diag([0.04] * 4), index=tickers, columns=tickers)
    mu = np.full(4, 0.06)
    res = maximum_diversification(tickers, mu, cov, rf=0.02)
    assert res.success and res.weights is not None
    assert res.weights.values == pytest.approx(np.ones(4) / 4, abs=1e-4)
    assert res.diversification_ratio == pytest.approx(np.sqrt(4), abs=1e-3)


def test_maxdiv_properties_and_failure():
    res = maximum_diversification(T3, _mu_3x(), _cov_3x(), rf=0.02)
    assert res.success and res.weights is not None
    assert res.weights.sum() == pytest.approx(1.0)
    assert res.diversification_ratio >= 1.0 - 1e-6
    assert np.isfinite(res.volatility) and res.volatility > 0
    bad = maximum_diversification(T3, _mu_3x(), _cov_3x(),
                                  min_weight=0.5, max_weight=1.0)
    assert bad.success is False and bad.weights is None and bad.message


# ---------------------------------------------------------------------------
# Ordering independence
# ---------------------------------------------------------------------------

def test_order_independence():
    tickers = ["A", "B", "C"]
    cov = _cov_3x()
    mu = pd.Series(_mu_3x(), index=tickers)
    rev = tickers[::-1]
    erc1 = equal_risk_contribution(tickers, mu.values, cov, rf=0.02)
    erc2 = equal_risk_contribution(rev, mu.reindex(rev).values,
                                   cov.reindex(index=rev, columns=rev), rf=0.02)
    assert erc1.success and erc2.success
    pd.testing.assert_series_equal(
        erc1.weights.sort_index(), erc2.weights.sort_index())
    md1 = maximum_diversification(tickers, mu.values, cov, rf=0.02)
    md2 = maximum_diversification(rev, mu.reindex(rev).values,
                                  cov.reindex(index=rev, columns=rev), rf=0.02)
    pd.testing.assert_series_equal(
        md1.weights.sort_index(), md2.weights.sort_index())


# ---------------------------------------------------------------------------
# Black-Litterman (§39–§41)
# ---------------------------------------------------------------------------

def _bl_2asset():
    tickers = ["A", "B"]
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.0225]],
                       index=tickers, columns=tickers)
    w = np.array([0.6, 0.4])
    return tickers, cov, w


def test_implied_equilibrium_exact():
    tickers, cov, w = _bl_2asset()
    pi = implied_equilibrium_returns(cov, w, 2.5, tickers)
    # π = δΣw = 2.5 * [0.028, 0.015] = [0.07, 0.0375]
    assert pi["A"] == pytest.approx(0.07)
    assert pi["B"] == pytest.approx(0.0375)
    with pytest.raises(ValueError):
        implied_equilibrium_returns(cov, w, -1.0, tickers)


def test_bl_no_views_equals_prior():
    tickers, cov, w = _bl_2asset()
    res = black_litterman_posterior(tickers, cov, w, 2.5, 0.05, [])
    assert isinstance(res, BlackLittermanResult)
    assert res.success and res.n_views == 0
    pd.testing.assert_series_equal(res.posterior, res.prior)


def test_bl_absolute_view_moves_toward_q():
    tickers, cov, w = _bl_2asset()
    views = [{"type": "absolute", "ticker": "A", "return": 0.14,
              "confidence": 0.9}]
    res = black_litterman_posterior(tickers, cov, w, 2.5, 0.05, views)
    assert res.success
    prior_a = res.prior["A"]
    post_a = res.posterior["A"]
    assert prior_a < post_a < 0.14  # pulled toward Q, bounded by it
    assert abs(post_a - 0.14) < abs(prior_a - 0.14)


def test_bl_confidence_monotonicity():
    tickers, cov, w = _bl_2asset()
    dists = []
    for conf in (0.2, 0.5, 0.9):
        res = black_litterman_posterior(
            tickers, cov, w, 2.5, 0.05,
            [{"type": "absolute", "ticker": "A", "return": 0.14,
              "confidence": conf}])
        assert res.success
        dists.append(abs(res.posterior["A"] - 0.14))
    assert dists[0] > dists[1] > dists[2]


def test_bl_relative_view():
    tickers, cov, w = _bl_2asset()
    spec = build_view_matrices(
        [{"type": "relative", "long": "A", "short": "B", "return": 0.04,
          "confidence": 0.8}], tickers)
    assert isinstance(spec, ViewSpec)
    assert spec.P.tolist() == [[1.0, -1.0]]
    assert spec.Q.tolist() == [0.04]
    res = black_litterman_posterior(
        tickers, cov, w, 2.5, 0.05,
        [{"type": "relative", "long": "A", "short": "B", "return": 0.04,
          "confidence": 0.8}])
    assert res.success
    prior_spread = res.prior["A"] - res.prior["B"]
    post_spread = res.posterior["A"] - res.posterior["B"]
    assert abs(post_spread - 0.04) < abs(prior_spread - 0.04)


def test_view_matrices_validation():
    spec = build_view_matrices(
        [{"type": "absolute", "ticker": "A", "return": 0.1, "confidence": 0.7}],
        T3)
    assert spec.P.shape == (1, 3) and spec.Q.shape == (1,)
    assert spec.P[0].tolist() == [1.0, 0.0, 0.0]
    empty = build_view_matrices([], T3)
    assert empty.P.shape == (0, 3)
    with pytest.raises(ValueError):
        build_view_matrices(
            [{"type": "absolute", "ticker": "ZZZ", "return": 0.1,
              "confidence": 0.7}], T3)
    with pytest.raises(ValueError):
        build_view_matrices(
            [{"type": "absolute", "ticker": "A", "return": 0.1,
              "confidence": 1.0}], T3)
    with pytest.raises(ValueError):
        build_view_matrices(
            [{"type": "absolute", "ticker": "A", "return": 0.1,
              "confidence": 0.0}], T3)
    with pytest.raises(ValueError):
        build_view_matrices(
            [{"type": "relative", "long": "A", "short": "A", "return": 0.04,
              "confidence": 0.7}], T3)
    with pytest.raises(ValueError):
        build_view_matrices(
            [{"type": "mystery", "ticker": "A", "return": 0.1,
              "confidence": 0.7}], T3)


def test_view_uncertainty_monotonic_in_confidence():
    P = np.array([[1.0, 0.0, 0.0]])
    tau_cov = np.eye(3) * 0.002
    lo = view_uncertainty_matrix(P, tau_cov, np.array([0.2]))
    hi = view_uncertainty_matrix(P, tau_cov, np.array([0.9]))
    assert lo.shape == (1, 1) and hi.shape == (1, 1)
    assert hi[0, 0] < lo[0, 0]  # higher confidence → tighter Ω
    # Convention check: ((1−c)/c)·(PτΣP′) with (PτΣP′) = 0.002.
    assert hi[0, 0] == pytest.approx((0.1 / 0.9) * 0.002)


def test_estimate_risk_aversion():
    assert estimate_risk_aversion_from_benchmark(0.06, 0.04) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        estimate_risk_aversion_from_benchmark(0.06, 0.0)
    with pytest.raises(ValueError):
        estimate_risk_aversion_from_benchmark(-0.02, 0.04)
    with pytest.raises(ValueError):
        estimate_risk_aversion_from_benchmark(float("nan"), 0.04)


def test_bl_allocation_and_diagnostics():
    tickers, cov, w = _bl_2asset()
    post = black_litterman_posterior(
        tickers, cov, w, 2.5, 0.05,
        [{"type": "absolute", "ticker": "A", "return": 0.14,
          "confidence": 0.7}]).posterior
    res = black_litterman_allocation(tickers, post, cov, 2.5, rf=0.02)
    assert isinstance(res, AllocationResult)
    assert res.method == "Black-Litterman"
    assert res.success and res.weights is not None
    assert res.weights.sum() == pytest.approx(1.0)
    assert ((res.weights.values >= 0.0) & (res.weights.values <= 1.0)).all()
    assert np.isfinite(res.volatility) and res.volatility > 0
    assert np.isfinite(res.sharpe)
    # Failure philosophy: bad posterior → no fake weights.
    bad = black_litterman_allocation(
        tickers, np.array([np.nan, 0.05]), cov, 2.5)
    assert bad.success is False and bad.weights is None and bad.message
    bad2 = black_litterman_allocation(tickers, post, cov, -2.5)
    assert bad2.success is False and bad2.weights is None


def test_scalar_validators():
    assert validate_delta(2.5) == 2.5
    with pytest.raises(ValueError):
        validate_delta(0.0)
    with pytest.raises(ValueError):
        validate_delta(-3.0)
    assert validate_tau(0.05) == 0.05
    with pytest.raises(ValueError):
        validate_tau(0.0)
    assert validate_confidence(0.7) == 0.7
    with pytest.raises(ValueError):
        validate_confidence(1.0)
    with pytest.raises(ValueError):
        validate_confidence(0.0)
    with pytest.raises(ValueError):
        validate_confidence(1.5)


# ---------------------------------------------------------------------------
# Estimation window + look-ahead (§31, §38)
# ---------------------------------------------------------------------------

def _est_prices(n=120, seed=1):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    base = 100 + np.cumsum(rng.normal(0.05, 0.5, size=(n, 3)), axis=0)
    return pd.DataFrame(base, index=idx, columns=T3)


def test_estimation_window_strict_separation():
    px = _est_prices(200)
    T = px.index[100]
    est, info = estimation_window(px, T3, T, 3.0, min_rows=10)
    assert est.index.max() < T  # nothing dated >= T may enter
    assert info["test_start"] == T.date()
    assert info["n_obs"] == len(est)
    # Lookback is calendar-based from available data (never before cutoff).
    assert info["est_start"] >= (T - pd.DateOffset(years=3)).date()
    with pytest.raises(ValueError):
        estimation_window(px, T3, px.index[5], 3.0, min_rows=10)
    with pytest.raises(ValueError):
        estimation_window(px, T3, T, -1.0)
    with pytest.raises(ValueError):
        estimation_window(px, T3, T, 3.0, min_rows=10_000)


def test_look_ahead_weights_bit_identical():
    # Same training history, wildly different futures → identical estimates.
    idx = pd.bdate_range("2020-01-01", periods=150)
    rng = np.random.default_rng(11)
    train = 100 + np.cumsum(rng.normal(0.05, 0.5, size=(150, 3)), axis=0)
    fut_a = 100 + np.cumsum(rng.normal(0.0, 2.0, size=(30, 3)), axis=0)
    fut_b = 100 + np.cumsum(rng.normal(0.0, 20.0, size=(30, 3)), axis=0)
    cols = T3
    px_a = pd.DataFrame(np.vstack([train, fut_a]),
                        index=idx.append(pd.bdate_range(
                            idx[-1] + pd.Timedelta(days=1), periods=30)),
                        columns=cols)
    px_b = pd.DataFrame(np.vstack([train, fut_b]), index=px_a.index,
                        columns=cols)
    T = idx[-1] + pd.Timedelta(days=1)  # first future date
    est_a, _ = estimation_window(px_a, cols, T, 3.0, min_rows=10)
    est_b, _ = estimation_window(px_b, cols, T, 3.0, min_rows=10)
    pd.testing.assert_frame_equal(est_a, est_b)

    def _stats(px):
        r = px.pct_change().dropna(how="any")
        return r.mean().values * 252, r.cov().values * 252

    mu_a, cov_a = _stats(est_a)
    mu_b, cov_b = _stats(est_b)
    erc_a = equal_risk_contribution(cols, mu_a, cov_a)
    erc_b = equal_risk_contribution(cols, mu_b, cov_b)
    assert erc_a.success and erc_b.success
    assert np.array_equal(erc_a.weights.values, erc_b.weights.values)
    md_a = maximum_diversification(cols, mu_a, cov_a)
    md_b = maximum_diversification(cols, mu_b, cov_b)
    assert md_a.success and md_b.success
    assert np.array_equal(md_a.weights.values, md_b.weights.values)
    bl_a = black_litterman_posterior(
        cols, cov_a, np.ones(3) / 3, 2.5, 0.05,
        [{"type": "absolute", "ticker": "A", "return": 0.10,
          "confidence": 0.7}])
    bl_b = black_litterman_posterior(
        cols, cov_b, np.ones(3) / 3, 2.5, 0.05,
        [{"type": "absolute", "ticker": "A", "return": 0.10,
          "confidence": 0.7}])
    assert bl_a.success and bl_b.success
    assert np.array_equal(bl_a.posterior.values, bl_b.posterior.values)

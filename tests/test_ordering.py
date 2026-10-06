"""FIX 3: reordering source columns must not change portfolio results."""

import numpy as np
import pandas as pd
import pytest

from analytics.portfolio import (
    portfolio_expected_return,
    portfolio_std,
    risk_contributions,
    sharpe_ratio,
)
from analytics.returns import (
    annualised_cov_matrix,
    annualised_mean_returns,
    daily_returns,
)
from analytics.validation import (
    align_market_data,
    aligned_portfolio_returns,
    validate_expected_return_cov,
    validate_weights,
)

TICKERS = ["AAPL", "GOOGL", "MSFT"]
WEIGHTS = np.array([0.5, 0.3, 0.2])  # canonical (sorted) order


def _prices():
    """One fixed price panel; callers reorder columns to simulate yfinance."""
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    rng = np.random.default_rng(7)
    base = 100 + np.cumsum(rng.normal(0.05, 0.5, size=(len(idx), 3)), axis=0) + 100
    return pd.DataFrame(base, index=idx, columns=TICKERS)


def _stats(prices):
    r = daily_returns(prices)
    return (
        annualised_mean_returns(r),
        annualised_cov_matrix(r),
        r,
    )


def _canonical_results(mu, cov, rets):
    a = align_market_data(TICKERS, returns=rets, mu=mu, cov=cov)
    mu_a = a["mu"].values
    cov_a = a["cov"].values
    ret = portfolio_expected_return(WEIGHTS, mu_a)
    vol = portfolio_std(WEIGHTS, cov_a)
    return ret, vol, sharpe_ratio(ret, vol, 0.04), risk_contributions(WEIGHTS, cov_a)


def test_reordered_columns_give_identical_results():
    full = _prices()
    mu1, cov1, r1 = _stats(full[TICKERS])
    mu2, cov2, r2 = _stats(full[list(reversed(TICKERS))])
    res1 = _canonical_results(mu1, cov1, r1)
    res2 = _canonical_results(mu2, cov2, r2)
    for a, b in zip(res1, res2):
        assert np.allclose(a, b)


def test_aligned_portfolio_returns_order_invariant():
    full = _prices()
    _, _, r1 = _stats(full[TICKERS])
    _, _, r2 = _stats(full[list(reversed(TICKERS))])
    s1 = aligned_portfolio_returns(r1, WEIGHTS, TICKERS)
    s2 = aligned_portfolio_returns(r2, WEIGHTS, TICKERS)
    pd.testing.assert_series_equal(s1, s2)


def test_validation_catches_mismatch():
    with pytest.raises(ValueError):
        validate_weights(np.array([0.5, 0.5]), TICKERS)
    with pytest.raises(ValueError):
        validate_expected_return_cov(np.array([0.1, 0.2]), np.eye(3), TICKERS)
    with pytest.raises(ValueError):
        align_market_data(TICKERS, mu=pd.Series([0.1], index=["AAPL"]))

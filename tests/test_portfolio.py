"""Tests for analytics.portfolio with manually verifiable numbers."""

import numpy as np
import pandas as pd
import pytest

from analytics.portfolio import (
    portfolio_expected_return,
    portfolio_std,
    portfolio_variance,
    risk_contributions,
    sharpe_ratio,
    weights_from_shares,
)

W = np.array([0.6, 0.4])
MU = np.array([0.10, 0.05])
COV = np.array([[0.04, 0.01], [0.01, 0.0225]])
# var = .36*.04 + .16*.0225 + 2*.6*.4*.01 = .0144+.0036+.0048 = .0228
VAR = 0.0228


def test_expected_return():
    assert portfolio_expected_return(W, MU) == pytest.approx(0.08)


def test_variance_and_vol():
    assert portfolio_variance(W, COV) == pytest.approx(VAR)
    assert portfolio_std(W, COV) == pytest.approx(np.sqrt(VAR))


def test_sharpe():
    assert sharpe_ratio(0.08, np.sqrt(VAR), 0.02) == pytest.approx(
        (0.08 - 0.02) / np.sqrt(VAR)
    )
    assert sharpe_ratio(0.08, 0.0, 0.02) == 0.0


def test_risk_contributions_sum_to_one_and_match_manual():
    rc = risk_contributions(W, COV)
    assert rc.sum() == pytest.approx(1.0)
    # Manual: marg = [0.028, 0.015]; rc_raw = w*marg/sigma.
    sigma = np.sqrt(VAR)
    raw = W * np.array([0.028, 0.015]) / sigma
    assert rc == pytest.approx(raw / raw.sum())
    assert rc[0] > rc[1]  # first asset dominates risk


def test_risk_contributions_zero_vol():
    rc = risk_contributions(np.array([0.5, 0.5]), np.zeros((2, 2)))
    assert (rc == 0).all()


def test_weights_from_shares_canonical_order():
    w, tickers = weights_from_shares(
        {"MSFT": 5, "AAPL": 10}, {"AAPL": 100.0, "MSFT": 100.0}
    )
    assert tickers == ["AAPL", "MSFT"]
    assert w == pytest.approx([10 / 15, 5 / 15])


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        portfolio_expected_return(np.array([0.5, 0.5]), np.array([0.1]))
    with pytest.raises(ValueError):
        portfolio_variance(np.array([0.5, 0.5]), np.eye(3))
    with pytest.raises(ValueError):
        risk_contributions(np.array([0.5]), np.eye(2))
    with pytest.raises(ValueError):
        weights_from_shares({}, {})

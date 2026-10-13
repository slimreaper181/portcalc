"""Tests for analytics.security (single-holding analytics)."""

import numpy as np
import pandas as pd
import pytest

from analytics.security import (
    describe_correlation,
    period_return,
    period_returns,
    security_beta,
    security_portfolio_correlation,
    security_risk_contribution,
    security_summary,
)

IDX = pd.date_range("2024-01-01", periods=10, freq="B")
S = pd.Series([0.02, -0.01, 0.03, 0.004, -0.016, 0.024, -0.006, 0.008,
               -0.012, 0.014], index=IDX, name="SEC")
B = pd.Series([0.01, -0.005, 0.015, 0.002, -0.008, 0.012, -0.003, 0.004,
               -0.006, 0.007], index=IDX, name="benchmark")


def test_security_beta_reuses_aligned_maths():
    assert security_beta(S, B) == pytest.approx(2.0)
    # Missing benchmark dates do not shift the result.
    b_short = B.drop(IDX[4])
    s_al = S.drop(IDX[4])
    assert security_beta(S, b_short) == pytest.approx(
        security_beta(s_al, b_short))
    with pytest.raises(ValueError):
        security_beta(S, pd.Series(np.full(10, 0.001), index=IDX))


def test_correlation_and_bands():
    # S is exactly 2x B here, so correlation is 1.0.
    c = security_portfolio_correlation(S, B)
    assert c == pytest.approx(1.0)
    assert "Strong positive" in describe_correlation(c)
    assert "Moderate positive" in describe_correlation(0.5)
    assert describe_correlation(0.0) == "Low co-movement with the portfolio."
    assert "Moderate negative" in describe_correlation(-0.5)
    assert "Strong negative" in describe_correlation(-0.9)
    with pytest.raises(ValueError):
        security_portfolio_correlation(
            pd.Series(np.full(10, 0.001), index=IDX), B)
    with pytest.raises(ValueError):
        security_portfolio_correlation(S.iloc[:1], B.iloc[:1])
    # NaNs are dropped deliberately, not silently shifted.
    s_nan = S.copy()
    s_nan.iloc[0] = np.nan
    assert security_portfolio_correlation(s_nan, B) == pytest.approx(
        security_portfolio_correlation(S.iloc[1:], B.iloc[1:]))


def test_risk_contribution_matches_engine():
    w = np.array([0.5, 0.3, 0.2])
    cov = np.array([[0.04, 0.008, 0.002],
                    [0.008, 0.0225, 0.003],
                    [0.002, 0.003, 0.01]])
    from analytics.portfolio import risk_contributions
    rc = risk_contributions(w, cov)
    for i in range(3):
        assert security_risk_contribution(w, cov, i) == pytest.approx(rc[i])
    with pytest.raises(ValueError):
        security_risk_contribution(w, cov, 5)
    with pytest.raises(ValueError):
        security_risk_contribution(np.array([0.5]), cov, 0)


def test_period_returns_date_based():
    idx = pd.DatetimeIndex(["2023-06-15", "2023-12-29", "2024-01-15",
                            "2024-03-29", "2024-06-14"])
    px = pd.Series([100.0, 110.0, 105.0, 120.0, 132.0], index=idx)
    one_y = period_return(px, "1Y")
    # Cutoff 2023-06-14 → first bar on/after is 2023-06-15 @100.
    assert one_y["return"] == pytest.approx(132.0 / 100.0 - 1.0)
    assert one_y["start"] == pd.Timestamp("2023-06-15").date()
    ytd = period_return(px, "YTD")
    # 2024-01-01 cutoff → first bar 2024-01-15 @105.
    assert ytd["return"] == pytest.approx(132.0 / 105.0 - 1.0)
    with pytest.raises(ValueError):
        period_return(px, "2Y")
    # Too little history → None, not a crash.
    short = period_return(px.iloc[-1:], "1M")
    assert short["return"] is None
    all_p = period_returns(px)
    assert set(all_p) == {"1M", "3M", "6M", "YTD", "1Y"}
    assert all_p["1Y"]["return"] == pytest.approx(one_y["return"])


def test_security_summary_and_single_security_portfolio():
    w = np.array([1.0])
    cov = np.array([[0.04]])
    out = security_summary(S, w, cov, 0, 0.03,
                           portfolio_log_returns=S.copy(),
                           benchmark_log_returns=B)
    assert out["annualised_return"] == pytest.approx(S.mean() * 252)
    assert out["correlation_to_portfolio"] == pytest.approx(1.0)
    assert out["beta_vs_benchmark"] == pytest.approx(2.0)
    assert out["risk_contribution"] == pytest.approx(1.0)
    # No benchmark → beta None, not an exception.
    out2 = security_summary(S, w, cov, 0, 0.03,
                            portfolio_log_returns=S.copy())
    assert out2["beta_vs_benchmark"] is None
    # Constant benchmark → beta_error message, still no raise.
    out3 = security_summary(S, w, cov, 0, 0.03,
                            benchmark_log_returns=pd.Series(
                                np.full(10, 0.001), index=IDX))
    assert out3["beta_vs_benchmark"] is None
    assert out3["beta_error"]
    with pytest.raises(ValueError):
        security_summary(S, w, cov, 3, 0.03)

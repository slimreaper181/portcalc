"""Tests for analytics.returns (log-return conventions)."""

import numpy as np
import pandas as pd

from analytics.returns import (
    annualised_cov_matrix,
    annualised_mean_returns,
    correlation_matrix,
    cumulative_growth_from_log_returns,
    cumulative_return_from_log_returns,
    daily_returns,
    log_to_simple_returns,
)


def _prices():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {"A": [100.0, 110.0, 104.5, 104.5], "B": [50.0, 50.0, 52.5, 51.0]},
        index=idx,
    )


def test_daily_returns_are_log_returns():
    r = daily_returns(_prices())
    assert list(r.columns) == ["A", "B"]
    assert len(r) == 3
    # ln(110/100), ln(104.5/110), ln(104.5/104.5)
    assert r["A"].iloc[0] == np.log(110.0 / 100.0)
    assert r["A"].iloc[1] == np.log(104.5 / 110.0)
    assert r["A"].iloc[2] == 0.0
    assert r["B"].iloc[0] == 0.0


def test_annualised_mean_and_cov():
    r = daily_returns(_prices())
    mu = annualised_mean_returns(r)
    assert mu["A"] == r["A"].mean() * 252
    cov = annualised_cov_matrix(r)
    pd.testing.assert_frame_equal(cov, r.cov() * 252)
    corr = correlation_matrix(r)
    pd.testing.assert_frame_equal(corr, r.corr())


def test_cumulative_growth_matches_price_ratio():
    # Identity: exp(sum of log returns) == P_last / P_first.
    prices = _prices()
    r = daily_returns(prices)
    growth = cumulative_growth_from_log_returns(r)
    assert growth["A"].iloc[-1] == prices["A"].iloc[-1] / prices["A"].iloc[0]
    assert growth["B"].iloc[-1] == prices["B"].iloc[-1] / prices["B"].iloc[0]
    # First cumulative step equals the first price ratio.
    assert growth["A"].iloc[0] == prices["A"].iloc[1] / prices["A"].iloc[0]


def test_cumulative_return_is_growth_minus_one():
    r = daily_returns(_prices())
    ret = cumulative_return_from_log_returns(r)
    pd.testing.assert_frame_equal(ret, np.exp(r.cumsum()) - 1)
    # Sanity: (1 + r).cumprod() is WRONG for log returns and must differ.
    wrong = (1 + r).cumprod()
    assert not np.allclose(ret.values, (wrong - 1).values)


def test_log_to_simple_conversion():
    assert log_to_simple_returns(0.0) == 0.0
    assert log_to_simple_returns(np.log(1.1)) == 1.1 - 1
    s = pd.Series([0.0, np.log(2.0)])
    out = log_to_simple_returns(s)
    assert list(out) == [0.0, 1.0]

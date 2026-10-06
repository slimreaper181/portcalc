"""Tests for analytics.validation: tickers, alignment, scalar inputs, bounds."""

import numpy as np
import pandas as pd
import pytest

from analytics.validation import (
    align_market_data,
    canonical_tickers,
    check_history_length,
    sanitize_prices,
    validate_confidence,
    validate_horizon_years,
    validate_monthly_contrib,
    validate_portfolio_value,
    validate_risk_free_rate,
    validate_shares,
    validate_simulation_count,
    validate_ticker_symbol,
    validate_var_horizon,
    validate_weight_bounds,
)


def test_ticker_validation():
    assert validate_ticker_symbol("  aapl ") == "AAPL"
    assert validate_ticker_symbol("BRK.B") == "BRK.B"
    assert validate_ticker_symbol("^IRX") == "^IRX"
    with pytest.raises(ValueError):
        validate_ticker_symbol("")
    with pytest.raises(ValueError):
        validate_ticker_symbol("BAD TICKER!")


def test_canonical_tickers_sorted_unique():
    assert canonical_tickers(["msft", "AAPL", "aapl"]) == ["AAPL", "MSFT"]
    with pytest.raises(ValueError):
        canonical_tickers([])


def test_sanitize_prices_reorders_and_cleans():
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    df = pd.DataFrame(
        {"MSFT": np.linspace(100, 140, 40), "AAPL": np.linspace(50, 70, 40)},
        index=idx,
    )
    out = sanitize_prices(df, ["AAPL", "MSFT"])
    assert list(out.columns) == ["AAPL", "MSFT"]
    with pytest.raises(ValueError, match="No price history"):
        sanitize_prices(
            pd.DataFrame({"A": [np.nan, np.nan]}), ["A"]
        )
    with pytest.raises(ValueError):
        sanitize_prices(pd.DataFrame(), ["A"])


def test_check_history_length():
    check_history_length(300, 10)  # fine
    with pytest.raises(ValueError, match="Insufficient history"):
        check_history_length(25, 10)


def test_scalar_validators():
    assert validate_shares(3.0) == 3.0
    with pytest.raises(ValueError):
        validate_shares(0.0)
    with pytest.raises(ValueError):
        validate_risk_free_rate(0.99)
    assert validate_confidence(0.95) == 0.95
    with pytest.raises(ValueError):
        validate_confidence(0.4)
    assert validate_var_horizon(10) == 10
    with pytest.raises(ValueError):
        validate_var_horizon(0)
    with pytest.raises(ValueError):
        validate_portfolio_value(0.0)
    assert validate_simulation_count(5_000) == 5_000
    with pytest.raises(ValueError):
        validate_simulation_count(50)
    with pytest.raises(ValueError):
        validate_simulation_count(100_000)
    assert validate_horizon_years(5) == 5.0
    with pytest.raises(ValueError):
        validate_horizon_years(-1)
    assert validate_monthly_contrib(0.0) == 0.0
    with pytest.raises(ValueError):
        validate_monthly_contrib(-5)


def test_weight_bounds_validation():
    validate_weight_bounds(5, 0.0, 1.0)  # fine
    with pytest.raises(ValueError, match="at most"):
        validate_weight_bounds(3, 0.0, 0.20)
    with pytest.raises(ValueError, match="already sum"):
        validate_weight_bounds(10, 0.20, 1.0)
    with pytest.raises(ValueError, match="0 <="):
        validate_weight_bounds(4, 0.4, 0.2)


def test_align_market_data_missing_ticker():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    df = pd.DataFrame({"A": np.ones(10)}, index=idx)
    with pytest.raises(ValueError, match="missing tickers"):
        align_market_data(["A", "B"], prices=df)

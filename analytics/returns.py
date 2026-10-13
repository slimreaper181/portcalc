"""
returns.py
----------
Computes daily **log** returns and annualised statistics.

Convention used across this package
------------------------------------
* ``daily_returns`` returns **log returns**: ``r = ln(P_t / P_{t-1})``.
* Annualised means / covariances derived from them therefore also live in
  "log space". They are suitable for GBM-style simulation and for
  mean-variance maths (first-order), but they must be converted with
  ``exp(r) - 1`` before being interpreted as percentage P&L or applied to
  currency values. See :func:`log_to_simple_returns` and
  :func:`cumulative_return_from_log_returns`.
"""

import numpy as np
import pandas as pd


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from a price DataFrame.

    Args:
        prices: DataFrame of adjusted close prices (rows = dates, cols = tickers)

    Returns:
        DataFrame of daily **log** returns (first row dropped).
    """
    return np.log(prices / prices.shift(1)).dropna()


def annualised_mean_returns(daily_ret: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """
    Annualise the mean of daily log returns.

    Args:
        daily_ret:    DataFrame of daily **log** returns.
        trading_days: Number of trading days per year (default 252).

    Returns:
        Series of annualised mean (log) returns per ticker.
    """
    return daily_ret.mean() * trading_days


def annualised_cov_matrix(daily_ret: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    """
    Annualise the covariance matrix of daily log returns.

    Args:
        daily_ret:    DataFrame of daily **log** returns.
        trading_days: Number of trading days per year (default 252).

    Returns:
        Annualised covariance matrix as a DataFrame.
    """
    return daily_ret.cov() * trading_days


def correlation_matrix(daily_ret: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix of daily log returns.

    Args:
        daily_ret: DataFrame of daily log returns.

    Returns:
        Correlation matrix as a DataFrame.
    """
    return daily_ret.corr()


def log_to_simple_returns(log_returns) -> "pd.DataFrame | pd.Series | np.ndarray":
    """Convert log return(s) to simple (percentage) return(s): ``exp(r) - 1``.

    Accepts a scalar, Series, DataFrame or ndarray and preserves the type.
    """
    return np.exp(log_returns) - 1


def cumulative_growth_from_log_returns(log_returns) -> "pd.DataFrame | pd.Series":
    """Growth of $1 from a log-return series: ``exp(cumsum(r))``.

    This is the mathematically correct cumulative path for log returns.
    (Using ``(1 + r).cumprod()`` would treat log returns as simple returns
    and is wrong.)
    """
    return np.exp(log_returns.cumsum())


def cumulative_return_from_log_returns(log_returns) -> "pd.DataFrame | pd.Series":
    """Percentage cumulative return from log returns: ``exp(cumsum(r)) - 1``."""
    return cumulative_growth_from_log_returns(log_returns) - 1

"""
returns.py
----------
Computes daily log returns and annualised statistics.
"""

import numpy as np
import pandas as pd


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from a price DataFrame.

    Args:
        prices: DataFrame of adjusted close prices (rows = dates, cols = tickers)

    Returns:
        DataFrame of daily log returns (first row dropped).
    """
    return np.log(prices / prices.shift(1)).dropna()


def annualised_mean_returns(daily_ret: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """
    Annualise the mean of daily log returns.

    Args:
        daily_ret:    DataFrame of daily log returns.
        trading_days: Number of trading days per year (default 252).

    Returns:
        Series of annualised mean returns per ticker.
    """
    return daily_ret.mean() * trading_days


def annualised_cov_matrix(daily_ret: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    """
    Annualise the covariance matrix of daily log returns.

    Args:
        daily_ret:    DataFrame of daily log returns.
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

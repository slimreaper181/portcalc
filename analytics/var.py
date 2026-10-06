"""
var.py
------
Value-at-Risk calculations: parametric, historical, and Monte Carlo.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


def to_daily(annualised_value: float, trading_days: int = 252) -> float:
    """Convert annualised volatility to daily volatility."""
    return annualised_value / np.sqrt(trading_days)


def mean_to_daily(annualised_mean: float, trading_days: int = 252) -> float:
    """Convert annualised mean return to daily mean return."""
    return annualised_mean / trading_days


def parametric_var(
    port_std_daily: float,
    portfolio_value: float,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> float:
    """
    Parametric (variance-covariance) VaR.

    Assumes normally distributed returns.

    VaR = z * σ_daily * V * √T

    Args:
        port_std_daily:  Daily portfolio volatility.
        portfolio_value: Total portfolio value in USD.
        confidence:      Confidence level (e.g. 0.95).
        horizon_days:    VaR time horizon in trading days.

    Returns:
        VaR in USD (positive number = potential loss).
    """
    z = norm.ppf(confidence)
    return z * port_std_daily * portfolio_value * np.sqrt(horizon_days)


def historical_var(
    portfolio_daily_returns: pd.Series,
    portfolio_value: float,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> float:
    """
    Historical (empirical) VaR.

    Uses the empirical distribution of past portfolio returns.

    Args:
        portfolio_daily_returns: Series of daily portfolio log returns.
        portfolio_value:         Total portfolio value in USD.
        confidence:              Confidence level (e.g. 0.95).
        horizon_days:            VaR time horizon in trading days.

    Returns:
        VaR in USD (positive number = potential loss).
    """
    percentile = (1 - confidence) * 100
    daily_var_pct = np.percentile(portfolio_daily_returns, percentile)
    # Scale to horizon
    return -daily_var_pct * portfolio_value * np.sqrt(horizon_days)


def monte_carlo_var(
    weights: np.ndarray,
    mean_returns_daily: np.ndarray,
    cov_matrix_daily: np.ndarray,
    portfolio_value: float,
    confidence: float = 0.95,
    horizon_days: int = 1,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """
    Monte Carlo VaR via multivariate normal simulation.

    Args:
        weights:             Asset weight vector.
        mean_returns_daily:  Daily mean returns per asset.
        cov_matrix_daily:    Daily covariance matrix.
        portfolio_value:     Total portfolio value in USD.
        confidence:          Confidence level (e.g. 0.95).
        horizon_days:        VaR time horizon in trading days.
        n_simulations:       Number of simulated scenarios.
        seed:                Random seed for reproducibility.

    Returns:
        Tuple of (var_in_usd, simulated_portfolio_returns_array).
    """
    rng = np.random.default_rng(seed)

    # Simulate daily asset returns for each horizon day
    sim_asset_returns = rng.multivariate_normal(
        mean=mean_returns_daily,
        cov=cov_matrix_daily,
        size=(n_simulations, horizon_days),
    )  # shape: (n_sims, horizon_days, n_assets)

    # Portfolio return over horizon = sum of daily portfolio returns
    daily_port_returns = sim_asset_returns @ weights  # (n_sims, horizon_days)
    horizon_port_returns = daily_port_returns.sum(axis=1)  # (n_sims,)

    percentile = (1 - confidence) * 100
    var_return = np.percentile(horizon_port_returns, percentile)
    var_usd = -var_return * portfolio_value

    return float(var_usd), horizon_port_returns

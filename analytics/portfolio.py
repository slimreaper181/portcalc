"""
portfolio.py
------------
Core portfolio mathematics: expected return, variance, Sharpe ratio,
risk contributions, and Capital Market Line blending.
"""

import numpy as np
import pandas as pd


def portfolio_expected_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    """
    Compute weighted portfolio expected return.

    Args:
        weights:      Asset weight vector (must sum to 1).
        mean_returns: Annualised mean return per asset.

    Returns:
        Scalar expected return (annualised).
    """
    return float(np.dot(weights, mean_returns))


def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Compute portfolio variance = w' Σ w.

    Args:
        weights:    Asset weight vector.
        cov_matrix: Annualised covariance matrix.

    Returns:
        Scalar portfolio variance.
    """
    return float(weights @ cov_matrix @ weights)


def portfolio_std(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Compute portfolio standard deviation (volatility).

    Args:
        weights:    Asset weight vector.
        cov_matrix: Annualised covariance matrix.

    Returns:
        Scalar portfolio volatility.
    """
    return float(np.sqrt(portfolio_variance(weights, cov_matrix)))


def sharpe_ratio(port_return: float, port_std: float, risk_free_rate: float) -> float:
    """
    Compute the Sharpe ratio = (Rp - Rf) / σp.

    Args:
        port_return:    Annualised portfolio return.
        port_std:       Annualised portfolio volatility.
        risk_free_rate: Annualised risk-free rate.

    Returns:
        Scalar Sharpe ratio (0.0 if σp == 0).
    """
    if port_std == 0:
        return 0.0
    return (port_return - risk_free_rate) / port_std


def risk_contributions(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """
    Compute fractional risk contribution of each asset to portfolio volatility.

    RC_i = (w_i * (Σw)_i) / σp

    Args:
        weights:    Asset weight vector.
        cov_matrix: Annualised covariance matrix.

    Returns:
        Array of fractional risk contributions (sums to 1).
    """
    sigma = portfolio_std(weights, cov_matrix)
    if sigma == 0:
        return np.zeros_like(weights)
    marginal_rc = cov_matrix @ weights
    rc = weights * marginal_rc / sigma
    return rc / rc.sum()


def weights_from_shares(shares: dict[str, float], prices: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    """
    Convert share holdings to portfolio weights.

    Args:
        shares: Dict mapping ticker -> number of shares held.
        prices: Dict mapping ticker -> current price.

    Returns:
        Tuple of (weights_array, tickers_list) sorted by ticker.
    """
    tickers = sorted(shares.keys())
    values = np.array([shares[t] * prices.get(t, 0.0) for t in tickers], dtype=float)
    total = values.sum()
    if total == 0:
        return np.zeros(len(tickers)), tickers
    return values / total, tickers


def blended_portfolio_return(x: float, r1: float, rf: float) -> float:
    """
    Return of a blend between a risky portfolio and risk-free asset.

    rp = x * r1 + (1 - x) * rf

    Args:
        x:  Fraction invested in the risky portfolio.
        r1: Return of the risky portfolio.
        rf: Risk-free rate.

    Returns:
        Blended portfolio return.
    """
    return x * r1 + (1 - x) * rf


def blended_portfolio_variance(x: float, var_r1: float) -> float:
    """
    Variance of a blend between a risky portfolio and risk-free asset.

    Var(rp) = x^2 * Var(r1)

    Args:
        x:      Fraction invested in the risky portfolio.
        var_r1: Variance of the risky portfolio.

    Returns:
        Blended portfolio variance.
    """
    return (x ** 2) * var_r1


def blended_portfolio_std(x: float, sigma_r1: float) -> float:
    """
    Volatility of a blend between a risky portfolio and risk-free asset.

    σp = |x| * σ1

    Args:
        x:       Fraction invested in the risky portfolio.
        sigma_r1: Volatility of the risky portfolio.

    Returns:
        Blended portfolio volatility.
    """
    return abs(x) * sigma_r1


def sigma_p_from_target(target_rp: float, r1: float, rf: float, sigma_r1: float) -> float:
    """
    Derive the volatility on the CML given a target portfolio return.

    Args:
        target_rp: Desired portfolio return.
        r1:        Return of the risky portfolio.
        rf:        Risk-free rate.
        sigma_r1:  Volatility of the risky portfolio.

    Returns:
        Required portfolio volatility.
    """
    if r1 == rf:
        return 0.0
    x = (target_rp - rf) / (r1 - rf)
    return blended_portfolio_std(x, sigma_r1)

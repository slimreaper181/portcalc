"""
portfolio.py
------------
Core portfolio mathematics: expected return, variance, Sharpe ratio,
risk contributions, and Capital Market Line blending.

Ordering convention
-------------------
Weight vectors, expected-return vectors and covariance matrices must all
share one canonical ticker order (see ``analytics.validation``). Functions
here accept plain ``np.ndarray`` inputs for speed, but validate that
``len(weights) == len(tickers)``-equivalent dimensions hold whenever ticker
labels are available; callers should reindex labelled pandas structures to
the canonical order *before* calling ``.values``.
"""

import numpy as np
import pandas as pd

from .validation import (
    canonical_tickers,
    validate_expected_return_cov,
    validate_weights,
)


def _as_aligned_arrays(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate weight/mu/cov dimensions against each other.

    Raises:
        ValueError: on length/shape mismatch or non-finite values.
    """
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(mean_returns, dtype=float)
    if w.ndim != 1 or mu.ndim != 1 or w.shape != mu.shape:
        raise ValueError(
            f"Weights shape {np.shape(weights)} and expected-return shape "
            f"{np.shape(mean_returns)} must match."
        )
    if not np.all(np.isfinite(w)):
        raise ValueError("Portfolio weights must all be finite (no NaN/inf).")
    if not np.all(np.isfinite(mu)):
        raise ValueError("Expected returns must all be finite (no NaN/inf).")
    if cov_matrix is None:
        return w, mu, None
    cov = np.asarray(cov_matrix, dtype=float)
    n = w.shape[0]
    if cov.shape != (n, n):
        raise ValueError(
            f"Covariance matrix shape {cov.shape} does not match "
            f"{n} assets (expected {(n, n)})."
        )
    if not np.all(np.isfinite(cov)):
        raise ValueError("Covariance matrix must contain only finite values.")
    return w, mu, cov


def portfolio_expected_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    """
    Compute weighted portfolio expected return (``w . mu``).

    Args:
        weights:      Asset weight vector (must sum to 1; order must match
                      ``mean_returns`` and the canonical ticker order).
        mean_returns: Annualised mean return per asset (same order).

    Returns:
        Scalar expected return (annualised).
    """
    w, mu, _ = _as_aligned_arrays(weights, mean_returns)
    return float(np.dot(w, mu))


def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Compute portfolio variance = w' Σ w.

    Args:
        weights:    Asset weight vector (canonical ticker order).
        cov_matrix: Annualised covariance matrix (same order).

    Returns:
        Scalar portfolio variance.
    """
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    if w.ndim != 1 or cov.shape != (w.shape[0], w.shape[0]):
        raise ValueError(
            f"Weights shape {np.shape(weights)} incompatible with covariance "
            f"shape {np.shape(cov_matrix)}."
        )
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(cov)):
        raise ValueError("Weights and covariance must contain only finite values.")
    return float(w @ cov @ w)


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

    RC_i = (w_i * (Σw)_i) / σp, normalised to sum to 1.

    Args:
        weights:    Asset weight vector (canonical ticker order).
        cov_matrix: Annualised covariance matrix (same order).

    Returns:
        Array of fractional risk contributions (sums to 1; zeros if σp == 0).
    """
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    if w.ndim != 1 or cov.shape != (w.shape[0], w.shape[0]):
        raise ValueError(
            f"Weights shape {np.shape(weights)} incompatible with covariance "
            f"shape {np.shape(cov_matrix)}."
        )
    sigma = portfolio_std(w, cov)
    if sigma == 0:
        return np.zeros_like(w)
    marginal_rc = cov @ w
    rc = w * marginal_rc / sigma
    total = rc.sum()
    if total == 0:
        return np.zeros_like(w)
    return rc / total


def weights_from_shares(shares: dict[str, float], prices: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    """
    Convert share holdings to portfolio weights.

    Args:
        shares: Dict mapping ticker -> number of shares held.
        prices: Dict mapping ticker -> current price.

    Returns:
        Tuple of (weights_array, tickers_list) in canonical (sorted,
        upper-cased) ticker order. Weights sum to 1 unless total value is 0.

    Raises:
        ValueError: on empty holdings, negative shares, or invalid prices.
    """
    if not shares:
        raise ValueError("Holdings must contain at least one position.")
    tickers = canonical_tickers(list(shares.keys()))
    values = []
    for t in tickers:
        # Accept case-insensitive keys in the input dicts.
        sh = shares.get(t, shares.get(t.lower(), shares.get(t.upper())))
        px = prices.get(t, prices.get(t.lower(), prices.get(t.upper(), 0.0)))
        sh = float(sh)
        px = float(px)
        if not np.isfinite(sh) or sh < 0:
            raise ValueError(f"Shares for {t} must be finite and >= 0, got {sh!r}.")
        if not np.isfinite(px) or px < 0:
            raise ValueError(f"Price for {t} must be finite and >= 0, got {px!r}.")
        values.append(sh * px)
    values = np.array(values, dtype=float)
    total = values.sum()
    if total == 0:
        return np.zeros(len(tickers)), tickers
    return values / total, tickers


def validate_portfolio_inputs(
    weights: np.ndarray, tickers: list[str], mu: np.ndarray, cov: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a full (weights, mu, cov) triple against canonical tickers."""
    w = validate_weights(weights, tickers)
    mu_arr, cov_arr = validate_expected_return_cov(mu, cov, tickers)
    return w, mu_arr, cov_arr


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

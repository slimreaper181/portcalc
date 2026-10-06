"""
optimisation.py
---------------
Portfolio optimisation: minimum variance, maximum Sharpe, target return/vol,
efficient frontier, and rebalancing trade engine.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class OptimResult:
    """Container for a single portfolio optimisation result."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe: float
    success: bool
    message: str


def _make_constraints(n: int) -> list[dict]:
    """Weights sum to 1 constraint."""
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]


def _make_bounds(n: int, min_weight: float = 0.0, max_weight: float = 1.0) -> list[tuple]:
    """Per-asset weight bounds."""
    return [(min_weight, max_weight)] * n


def _port_ret(w, mu):
    return float(np.dot(w, mu))


def _port_vol(w, cov):
    return float(np.sqrt(w @ cov @ w))


def min_variance(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.05,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> OptimResult:
    """
    Find the minimum variance portfolio.

    Args:
        mu:         Annualised mean returns vector.
        cov:        Annualised covariance matrix.
        rf:         Risk-free rate (for Sharpe calculation).
        min_weight: Minimum weight per asset.
        max_weight: Maximum weight per asset.

    Returns:
        OptimResult with optimal weights.
    """
    n = len(mu)
    w0 = np.ones(n) / n

    res = minimize(
        fun=lambda w: _port_vol(w, cov),
        x0=w0,
        method="SLSQP",
        bounds=_make_bounds(n, min_weight, max_weight),
        constraints=_make_constraints(n),
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    w = res.x
    ret = _port_ret(w, mu)
    vol = _port_vol(w, cov)
    sr = (ret - rf) / vol if vol > 0 else 0.0

    return OptimResult(weights=w, expected_return=ret, volatility=vol, sharpe=sr,
                       success=res.success, message=res.message)


def max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.05,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> OptimResult:
    """
    Find the maximum Sharpe ratio portfolio.

    Args:
        mu:         Annualised mean returns vector.
        cov:        Annualised covariance matrix.
        rf:         Risk-free rate.
        min_weight: Minimum weight per asset.
        max_weight: Maximum weight per asset.

    Returns:
        OptimResult with optimal weights.
    """
    n = len(mu)
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        ret = _port_ret(w, mu)
        vol = _port_vol(w, cov)
        return -(ret - rf) / vol if vol > 0 else 0.0

    res = minimize(
        fun=neg_sharpe,
        x0=w0,
        method="SLSQP",
        bounds=_make_bounds(n, min_weight, max_weight),
        constraints=_make_constraints(n),
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    w = res.x
    ret = _port_ret(w, mu)
    vol = _port_vol(w, cov)
    sr = (ret - rf) / vol if vol > 0 else 0.0

    return OptimResult(weights=w, expected_return=ret, volatility=vol, sharpe=sr,
                       success=res.success, message=res.message)


def target_return(
    mu: np.ndarray,
    cov: np.ndarray,
    target: float,
    rf: float = 0.05,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> OptimResult:
    """
    Minimise variance subject to achieving a target return.

    Args:
        mu:         Annualised mean returns vector.
        cov:        Annualised covariance matrix.
        target:     Target annualised return.
        rf:         Risk-free rate.
        min_weight: Minimum weight per asset.
        max_weight: Maximum weight per asset.

    Returns:
        OptimResult with optimal weights.
    """
    n = len(mu)
    w0 = np.ones(n) / n

    constraints = _make_constraints(n) + [
        {"type": "eq", "fun": lambda w: _port_ret(w, mu) - target}
    ]

    res = minimize(
        fun=lambda w: _port_vol(w, cov),
        x0=w0,
        method="SLSQP",
        bounds=_make_bounds(n, min_weight, max_weight),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    w = res.x
    ret = _port_ret(w, mu)
    vol = _port_vol(w, cov)
    sr = (ret - rf) / vol if vol > 0 else 0.0

    return OptimResult(weights=w, expected_return=ret, volatility=vol, sharpe=sr,
                       success=res.success, message=res.message)


def target_volatility(
    mu: np.ndarray,
    cov: np.ndarray,
    target_vol: float,
    rf: float = 0.05,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> OptimResult:
    """
    Maximise return subject to a target volatility constraint.

    Args:
        mu:          Annualised mean returns vector.
        cov:         Annualised covariance matrix.
        target_vol:  Target annualised volatility.
        rf:          Risk-free rate.
        min_weight:  Minimum weight per asset.
        max_weight:  Maximum weight per asset.

    Returns:
        OptimResult with optimal weights.
    """
    n = len(mu)
    w0 = np.ones(n) / n

    constraints = _make_constraints(n) + [
        {"type": "eq", "fun": lambda w: _port_vol(w, cov) - target_vol}
    ]

    res = minimize(
        fun=lambda w: -_port_ret(w, mu),
        x0=w0,
        method="SLSQP",
        bounds=_make_bounds(n, min_weight, max_weight),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    w = res.x
    ret = _port_ret(w, mu)
    vol = _port_vol(w, cov)
    sr = (ret - rf) / vol if vol > 0 else 0.0

    return OptimResult(weights=w, expected_return=ret, volatility=vol, sharpe=sr,
                       success=res.success, message=res.message)


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.05,
    n_points: int = 50,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> pd.DataFrame:
    """
    Trace the efficient frontier by solving for minimum variance at each target return.

    Args:
        mu:         Annualised mean returns vector.
        cov:        Annualised covariance matrix.
        rf:         Risk-free rate.
        n_points:   Number of points along the frontier.
        min_weight: Minimum weight per asset.
        max_weight: Maximum weight per asset.

    Returns:
        DataFrame with columns [return, volatility, sharpe].
    """
    mv = min_variance(mu, cov, rf, min_weight, max_weight)
    r_min = mv.expected_return
    r_max = float(mu.max())

    if r_min >= r_max:
        r_max = r_min + 0.01

    target_returns = np.linspace(r_min, r_max, n_points)

    frontier = []
    for tr in target_returns:
        res = target_return(mu, cov, tr, rf, min_weight, max_weight)
        if res.success:
            frontier.append({
                "return": res.expected_return,
                "volatility": res.volatility,
                "sharpe": res.sharpe,
            })

    return pd.DataFrame(frontier)


def rebalance_trades(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    tickers: list[str],
    portfolio_value: float,
) -> pd.DataFrame:
    """
    Compute the trades required to rebalance from current to target weights.

    Args:
        current_weights: Current portfolio weight vector.
        target_weights:  Target portfolio weight vector.
        tickers:         List of asset tickers (same order as weights).
        portfolio_value: Total portfolio value in USD.

    Returns:
        DataFrame with columns [Ticker, Current Weight, Target Weight,
                                 Δ Weight, Trade ($), Action].
    """
    delta = target_weights - current_weights
    trade_usd = delta * portfolio_value

    rows = []
    for i, ticker in enumerate(tickers):
        action = "HOLD"
        if trade_usd[i] > 1:
            action = "BUY"
        elif trade_usd[i] < -1:
            action = "SELL"
        rows.append({
            "Ticker": ticker,
            "Current Weight": current_weights[i],
            "Target Weight": target_weights[i],
            "Δ Weight": delta[i],
            "Trade ($)": trade_usd[i],
            "Action": action,
        })

    return pd.DataFrame(rows)

"""
var.py
------
Value-at-Risk calculations: parametric, historical, and Monte Carlo.

Return-type conventions (read carefully)
-----------------------------------------
* ``portfolio_daily_returns`` inputs are daily **log** returns, i.e.
  ``r = ln(P_t / P_{t-1})`` as produced by
  :func:`analytics.returns.daily_returns`.
* A T-day log return is the *sum* of daily log returns. Before a log return
  is interpreted as P&L it is converted to a simple return with
  ``simple = exp(log) - 1``.
* All ``*_var`` functions return a **positive currency amount**
  (potential loss in USD).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .validation import (
    check_history_length,
    validate_confidence,
    validate_portfolio_value,
    validate_simulation_count,
    validate_var_horizon,
)


def to_daily(annualised_value: float, trading_days: int = 252) -> float:
    """Convert annualised volatility to daily volatility (divides by √T)."""
    return annualised_value / np.sqrt(trading_days)


def mean_to_daily(annualised_mean: float, trading_days: int = 252) -> float:
    """Convert annualised mean (log) return to a daily mean (log) return."""
    return annualised_mean / trading_days


def log_to_simple(log_return: float) -> float:
    """Convert a log return to a simple (percentage) return: ``exp(r) - 1``."""
    return float(np.exp(log_return) - 1)


def parametric_var(
    port_std_daily: float,
    portfolio_value: float,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> float:
    """
    Parametric (variance-covariance) VaR.

    Assumes normally distributed **simple** returns. In practice the daily
    volatility estimated from log returns is used as an approximation (the
    two agree to first order for small moves).

    VaR = z * σ_daily * V * √T

    Args:
        port_std_daily:  Daily portfolio volatility (decimal).
        portfolio_value: Total portfolio value in currency units.
        confidence:      Confidence level (e.g. 0.95).
        horizon_days:    VaR time horizon in trading days.

    Returns:
        VaR in currency units (positive number = potential loss).
    """
    confidence = validate_confidence(confidence)
    horizon_days = validate_var_horizon(horizon_days)
    portfolio_value = validate_portfolio_value(portfolio_value)
    if not np.isfinite(port_std_daily) or port_std_daily < 0:
        raise ValueError(
            f"Daily volatility must be finite and >= 0, got {port_std_daily!r}."
        )
    z = norm.ppf(confidence)
    return float(z * port_std_daily * portfolio_value * np.sqrt(horizon_days))


def historical_var(
    portfolio_daily_returns: pd.Series,
    portfolio_value: float,
    confidence: float = 0.95,
    horizon_days: int = 1,
    min_windows: int = 20,
) -> float:
    """
    Historical (empirical) VaR using genuine rolling T-day returns.

    For a multi-day horizon the function builds actual overlapping T-day
    **log** returns (rolling sums of daily log returns), converts the
    tail percentile to a simple return via ``exp(r) - 1``, and applies it
    to the portfolio value. No square-root-of-time scaling is used.

    Args:
        portfolio_daily_returns: Series of daily portfolio **log** returns.
        portfolio_value:         Total portfolio value in currency units.
        confidence:              Confidence level (e.g. 0.95).
        horizon_days:            VaR time horizon in trading days.
        min_windows:             Minimum overlapping T-day windows required.

    Returns:
        VaR in currency units (positive number = potential loss).

    Raises:
        ValueError: if inputs are invalid or history is too short for the
            requested horizon (no silent fallback number is produced).
    """
    import pandas as _pd

    confidence = validate_confidence(confidence)
    horizon_days = validate_var_horizon(horizon_days)
    portfolio_value = validate_portfolio_value(portfolio_value)

    if portfolio_daily_returns is None or len(portfolio_daily_returns) == 0:
        raise ValueError("No historical returns available for Historical VaR.")
    rets = _pd.Series(np.asarray(portfolio_daily_returns, dtype=float)).dropna()
    if len(rets) == 0 or not np.all(np.isfinite(rets)):
        raise ValueError("Historical returns contain no usable observations.")

    import analytics.validation as _v

    required = max(min_windows, _v.MIN_HISTORICAL_VAR_WINDOWS)
    check_history_length(len(rets), horizon_days)
    n_windows = len(rets) - horizon_days + 1
    if n_windows < required:
        raise ValueError(
            f"Insufficient history for {horizon_days}-day historical VaR: "
            f"{len(rets)} daily observations give only {n_windows} overlapping "
            f"{horizon_days}-day windows (need at least {required}). "
            "Shorten the horizon or load a longer history period."
        )

    if horizon_days == 1:
        t_day_log = rets
    else:
        # Genuine rolling T-day log returns (sums of daily logs).
        t_day_log = rets.rolling(window=horizon_days).sum().dropna()

    percentile = (1 - confidence) * 100
    tail_log_return = float(np.percentile(t_day_log, percentile))
    # Convert the log-return tail to a simple return before applying to value.
    tail_simple_return = float(np.exp(tail_log_return) - 1)
    return float(-tail_simple_return * portfolio_value)


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
    Monte Carlo VaR via multivariate normal simulation of daily **log** returns.

    Simulated daily asset log returns are aggregated to horizon log returns
    (sums), converted to simple returns with ``exp(r) - 1``, and the tail
    percentile is applied to the portfolio value.

    Args:
        weights:             Asset weight vector (must match mu/cov lengths).
        mean_returns_daily:  Daily mean **log** returns per asset.
        cov_matrix_daily:    Daily covariance matrix (log-return space).
        portfolio_value:     Total portfolio value in currency units.
        confidence:          Confidence level (e.g. 0.95).
        horizon_days:        VaR time horizon in trading days.
        n_simulations:       Number of simulated scenarios.
        seed:                Random seed for reproducibility.

    Returns:
        Tuple of (var_in_currency_units, simulated_horizon_simple_returns).
        The second element contains horizon **simple** (percentage) returns,
        suitable for direct P&L interpretation.
    """
    confidence = validate_confidence(confidence)
    horizon_days = validate_var_horizon(horizon_days)
    portfolio_value = validate_portfolio_value(portfolio_value)
    n_simulations = validate_simulation_count(n_simulations)

    w = np.asarray(weights, dtype=float)
    mu_d = np.asarray(mean_returns_daily, dtype=float)
    cov_d = np.asarray(cov_matrix_daily, dtype=float)
    n = w.shape[0]
    if mu_d.shape != (n,) or cov_d.shape != (n, n):
        raise ValueError(
            f"Weights shape {w.shape}, mean shape {mu_d.shape} and cov shape "
            f"{cov_d.shape} are inconsistent."
        )
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(mu_d)):
        raise ValueError("Weights and mean returns must be finite.")
    if not np.all(np.isfinite(cov_d)):
        raise ValueError("Covariance matrix must contain only finite values.")

    rng = np.random.default_rng(seed)

    # Simulate daily asset LOG returns for each horizon day.
    sim_asset_returns = rng.multivariate_normal(
        mean=mu_d,
        cov=cov_d,
        size=(n_simulations, horizon_days),
    )  # shape: (n_sims, horizon_days, n_assets)

    # Portfolio daily LOG returns, summed over the horizon.
    daily_port_returns = sim_asset_returns @ w  # (n_sims, horizon_days)
    horizon_log_returns = daily_port_returns.sum(axis=1)  # (n_sims,)

    # Convert summed LOG returns to SIMPLE returns before P&L use.
    horizon_simple_returns = np.exp(horizon_log_returns) - 1

    percentile = (1 - confidence) * 100
    var_simple_return = float(np.percentile(horizon_simple_returns, percentile))
    var_usd = -var_simple_return * portfolio_value

    return float(var_usd), np.asarray(horizon_simple_returns)

"""
optimisation.py
---------------
Portfolio optimisation: minimum variance, maximum Sharpe, target return/vol,
efficient frontier, and rebalancing trade engine.

All optimisers validate weight bounds up front (see
:func:`analytics.validation.validate_weight_bounds`) and return an
``OptimResult`` with ``success=False`` plus a human-readable message when
constraints are infeasible — they never propagate NaN weights or raise into
the UI. Inputs are expected in the canonical ticker order.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .validation import validate_weight_bounds


@dataclass
class OptimResult:
    """Container for a single portfolio optimisation result."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe: float
    success: bool
    message: str


_EQUAL_WEIGHTS_MSG = "equal-weight fallback"


def _make_constraints(n: int) -> list[dict]:
    """Weights sum to 1 constraint."""
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]


def _make_bounds(n: int, min_weight: float = 0.0, max_weight: float = 1.0) -> list[tuple]:
    """Per-asset weight bounds."""
    return [(min_weight, max_weight)] * n


def _port_ret(w, mu):
    return float(np.dot(w, mu))


def _port_vol(w, cov):
    return float(np.sqrt(max(float(w @ cov @ w), 0.0)))


def _check_inputs(mu: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate mu/cov shapes and finiteness; return (mu, cov, n)."""
    mu_arr = np.asarray(mu, dtype=float)
    cov_arr = np.asarray(cov, dtype=float)
    if mu_arr.ndim != 1 or mu_arr.shape[0] < 1:
        raise ValueError("mu must be a non-empty 1-D vector.")
    n = mu_arr.shape[0]
    if cov_arr.shape != (n, n):
        raise ValueError(
            f"Covariance shape {cov_arr.shape} inconsistent with {n} assets."
        )
    if not np.all(np.isfinite(mu_arr)) or not np.all(np.isfinite(cov_arr)):
        raise ValueError("mu and cov must contain only finite values.")
    return mu_arr, cov_arr, n


def _failure(n: int, mu: np.ndarray, cov: np.ndarray, rf: float, message: str) -> OptimResult:
    """Build a safe fallback result (equal weights, finite stats)."""
    w = np.ones(n) / n
    try:
        ret = _port_ret(w, mu)
        vol = _port_vol(w, cov)
    except Exception:
        ret, vol = 0.0, 0.0
    if not np.isfinite(ret):
        ret = 0.0
    if not np.isfinite(vol) or vol < 0:
        vol = 0.0
    sr = (ret - rf) / vol if vol > 0 else 0.0
    return OptimResult(
        weights=w, expected_return=float(ret), volatility=float(vol),
        sharpe=float(sr), success=False, message=message,
    )


def _validate_bounds_or_failure(
    n: int, mu_arr: np.ndarray, cov_arr: np.ndarray, rf: float,
    min_weight: float, max_weight: float,
) -> OptimResult | None:
    """Return a failure OptimResult if bounds are infeasible, else None."""
    try:
        validate_weight_bounds(n, min_weight, max_weight)
    except ValueError as e:
        return _failure(n, mu_arr, cov_arr, rf, str(e))
    return None


def feasible_return_range(
    mu: np.ndarray,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> tuple[float, float]:
    """Feasible [min, max] portfolio return under per-asset box bounds.

    With simplex + box constraints the return extremes are attained at
    vertices; allocating as much as possible to the lowest/highest-return
    assets gives the exact bounds via a greedy fill.
    """
    mu_arr = np.asarray(mu, dtype=float)
    n = mu_arr.shape[0]
    validate_weight_bounds(n, min_weight, max_weight)

    def _extreme(highest: bool) -> float:
        order = np.argsort(mu_arr)
        if highest:
            order = order[::-1]
        w = np.full(n, min_weight)
        remaining = 1.0 - min_weight * n
        for i in order:
            room = max_weight - min_weight
            add = min(room, remaining)
            w[i] += add
            remaining -= add
            if remaining <= 1e-12:
                break
        return float(w @ mu_arr)

    lo = _extreme(False)
    hi = _extreme(True)
    return (min(lo, hi), max(lo, hi))


def feasible_volatility_range(
    mu: np.ndarray,
    cov: np.ndarray,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> tuple[float, float]:
    """Feasible [min, max] annualised volatility under box bounds.

    The minimum comes from the min-variance solve; the maximum is found by
    maximising volatility (concave-down objective over a polytope, so the
    optimum is at a vertex) from several deterministic starts, seeded with
    concentrated single-asset-tilted portfolios.
    """
    mu_arr, cov_arr, n = _check_inputs(mu, cov)
    validate_weight_bounds(n, min_weight, max_weight)

    mv = min_variance(mu_arr, cov_arr, min_weight=min_weight, max_weight=max_weight)
    lo = mv.volatility if mv.success else 0.0

    best = lo
    # Deterministic starts: tilt towards each asset in turn.
    for k in range(n):
        w0 = np.full(n, min_weight)
        remaining = 1.0 - min_weight * n
        room_k = max_weight - min_weight
        add_k = min(room_k, remaining)
        w0[k] += add_k
        remaining -= add_k
        # Spread any leftover evenly (can only happen at boundary rounding).
        if remaining > 1e-12:
            for i in range(n):
                room = max_weight - w0[i]
                add = min(room, remaining)
                w0[i] += add
                remaining -= add
                if remaining <= 1e-12:
                    break
        try:
            res = minimize(
                fun=lambda w: -_port_vol(w, cov_arr),
                x0=w0,
                method="SLSQP",
                bounds=_make_bounds(n, min_weight, max_weight),
                constraints=_make_constraints(n),
                options={"ftol": 1e-12, "maxiter": 500},
            )
            if res.success:
                best = max(best, _port_vol(res.x, cov_arr))
            else:
                best = max(best, _port_vol(w0, cov_arr))
        except Exception:
            best = max(best, _port_vol(w0, cov_arr))
    # A lone-asset portfolio is only attainable when bounds permit full
    # concentration; the tilted starts above already cover that case.
    hi = best
    if max_weight >= 1.0 - 1e-12:
        single_vols = np.sqrt(np.maximum(np.diag(cov_arr), 0.0))
        hi = max(hi, float(np.max(single_vols)))
    if hi < lo:
        hi = lo
    # Guard against a degenerate zero range for UI sliders.
    if hi - lo < 1e-4:
        hi = lo + 1e-4
    return float(lo), float(hi)


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
        mu:         Annualised mean returns vector (canonical order).
        cov:        Annualised covariance matrix (same order).
        rf:         Risk-free rate (for Sharpe calculation).
        min_weight: Minimum weight per asset.
        max_weight: Maximum weight per asset.

    Returns:
        OptimResult with optimal weights (``success=False`` with a message
        and equal-weight fallback if infeasible or non-convergent).
    """
    mu_arr, cov_arr, n = _check_inputs(mu, cov)
    failed = _validate_bounds_or_failure(n, mu_arr, cov_arr, rf, min_weight, max_weight)
    if failed is not None:
        return failed
    w0 = np.ones(n) / n

    try:
        res = minimize(
            fun=lambda w: _port_vol(w, cov_arr),
            x0=w0,
            method="SLSQP",
            bounds=_make_bounds(n, min_weight, max_weight),
            constraints=_make_constraints(n),
            options={"ftol": 1e-12, "maxiter": 1000},
        )
    except Exception as e:  # never leak solver exceptions to the UI
        return _failure(n, mu_arr, cov_arr, rf, f"Min-variance optimisation failed: {e}")

    w = np.asarray(res.x, dtype=float)
    if not res.success or not np.all(np.isfinite(w)):
        msg = res.message if isinstance(res.message, str) else str(res.message)
        bad = _failure(n, mu_arr, cov_arr, rf, f"Min-variance did not converge: {msg}")
        return bad

    ret = _port_ret(w, mu_arr)
    vol = _port_vol(w, cov_arr)
    sr = (ret - rf) / vol if vol > 0 else 0.0
    return OptimResult(weights=w, expected_return=ret, volatility=vol, sharpe=sr,
                       success=True, message="Converged")


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
        mu:         Annualised mean returns vector (canonical order).
        cov:        Annualised covariance matrix (same order).
        rf:         Risk-free rate.
        min_weight: Minimum weight per asset.
        max_weight: Maximum weight per asset.

    Returns:
        OptimResult with optimal weights (fallback on failure, see
        :func:`min_variance`).
    """
    mu_arr, cov_arr, n = _check_inputs(mu, cov)
    failed = _validate_bounds_or_failure(n, mu_arr, cov_arr, rf, min_weight, max_weight)
    if failed is not None:
        return failed
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        ret = _port_ret(w, mu_arr)
        vol = _port_vol(w, cov_arr)
        return -(ret - rf) / vol if vol > 0 else 0.0

    try:
        res = minimize(
            fun=neg_sharpe,
            x0=w0,
            method="SLSQP",
            bounds=_make_bounds(n, min_weight, max_weight),
            constraints=_make_constraints(n),
            options={"ftol": 1e-12, "maxiter": 1000},
        )
    except Exception as e:
        return _failure(n, mu_arr, cov_arr, rf, f"Max-Sharpe optimisation failed: {e}")

    w = np.asarray(res.x, dtype=float)
    if not res.success or not np.all(np.isfinite(w)):
        msg = res.message if isinstance(res.message, str) else str(res.message)
        return _failure(n, mu_arr, cov_arr, rf, f"Max-Sharpe did not converge: {msg}")

    ret = _port_ret(w, mu_arr)
    vol = _port_vol(w, cov_arr)
    sr = (ret - rf) / vol if vol > 0 else 0.0
    return OptimResult(weights=w, expected_return=ret, volatility=vol, sharpe=sr,
                       success=True, message="Converged")


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

    The target is checked against the feasible return range first; an
    infeasible target returns ``success=False`` with a message instead of
    running a doomed optimisation.

    Args:
        mu:         Annualised mean returns vector (canonical order).
        cov:        Annualised covariance matrix (same order).
        target:     Target annualised return.
        rf:         Risk-free rate.
        min_weight: Minimum weight per asset.
        max_weight: Maximum weight per asset.

    Returns:
        OptimResult with optimal weights.
    """
    mu_arr, cov_arr, n = _check_inputs(mu, cov)
    failed = _validate_bounds_or_failure(n, mu_arr, cov_arr, rf, min_weight, max_weight)
    if failed is not None:
        return failed
    if not np.isfinite(target):
        return _failure(n, mu_arr, cov_arr, rf, f"Target return {target!r} is not finite.")
    lo, hi = feasible_return_range(mu_arr, min_weight, max_weight)
    if not lo - 1e-9 <= target <= hi + 1e-9:
        return _failure(
            n, mu_arr, cov_arr, rf,
            f"Target return {target:.2%} is outside the feasible range "
            f"[{lo:.2%}, {hi:.2%}] under the current weight bounds.",
        )
    w0 = np.ones(n) / n

    constraints = _make_constraints(n) + [
        {"type": "eq", "fun": lambda w: _port_ret(w, mu_arr) - target}
    ]

    try:
        res = minimize(
            fun=lambda w: _port_vol(w, cov_arr),
            x0=w0,
            method="SLSQP",
            bounds=_make_bounds(n, min_weight, max_weight),
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
    except Exception as e:
        return _failure(n, mu_arr, cov_arr, rf, f"Target-return optimisation failed: {e}")

    w = np.asarray(res.x, dtype=float)
    if not res.success or not np.all(np.isfinite(w)):
        msg = res.message if isinstance(res.message, str) else str(res.message)
        return _failure(n, mu_arr, cov_arr, rf, f"Target-return did not converge: {msg}")

    ret = _port_ret(w, mu_arr)
    vol = _port_vol(w, cov_arr)
    sr = (ret - rf) / vol if vol > 0 else 0.0
    return OptimResult(weights=w, expected_return=ret, volatility=vol, sharpe=sr,
                       success=True, message="Converged")


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

    The target is checked against the feasible volatility range first.

    Args:
        mu:          Annualised mean returns vector (canonical order).
        cov:         Annualised covariance matrix (same order).
        target_vol:  Target annualised volatility.
        rf:          Risk-free rate.
        min_weight:  Minimum weight per asset.
        max_weight:  Maximum weight per asset.

    Returns:
        OptimResult with optimal weights.
    """
    mu_arr, cov_arr, n = _check_inputs(mu, cov)
    failed = _validate_bounds_or_failure(n, mu_arr, cov_arr, rf, min_weight, max_weight)
    if failed is not None:
        return failed
    if not np.isfinite(target_vol) or target_vol <= 0:
        return _failure(
            n, mu_arr, cov_arr, rf,
            f"Target volatility must be a positive finite number, got {target_vol!r}.",
        )
    lo, hi = feasible_volatility_range(mu_arr, cov_arr, min_weight, max_weight)
    if not lo - 1e-9 <= target_vol <= hi + 1e-9:
        return _failure(
            n, mu_arr, cov_arr, rf,
            f"Target volatility {target_vol:.2%} is outside the feasible range "
            f"[{lo:.2%}, {hi:.2%}] under the current weight bounds.",
        )
    w0 = np.ones(n) / n

    constraints = _make_constraints(n) + [
        {"type": "eq", "fun": lambda w: _port_vol(w, cov_arr) - target_vol}
    ]

    try:
        res = minimize(
            fun=lambda w: -_port_ret(w, mu_arr),
            x0=w0,
            method="SLSQP",
            bounds=_make_bounds(n, min_weight, max_weight),
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
    except Exception as e:
        return _failure(n, mu_arr, cov_arr, rf, f"Target-volatility optimisation failed: {e}")

    w = np.asarray(res.x, dtype=float)
    if not res.success or not np.all(np.isfinite(w)):
        msg = res.message if isinstance(res.message, str) else str(res.message)
        return _failure(n, mu_arr, cov_arr, rf, f"Target-volatility did not converge: {msg}")

    ret = _port_ret(w, mu_arr)
    vol = _port_vol(w, cov_arr)
    sr = (ret - rf) / vol if vol > 0 else 0.0
    return OptimResult(weights=w, expected_return=ret, volatility=vol, sharpe=sr,
                       success=True, message="Converged")


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
        mu:         Annualised mean returns vector (canonical order).
        cov:        Annualised covariance matrix (same order).
        rf:         Risk-free rate.
        n_points:   Number of points along the frontier.
        min_weight: Minimum weight per asset.
        max_weight: Maximum weight per asset.

    Returns:
        DataFrame with columns [return, volatility, sharpe]; may be empty if
        no point converges (callers must handle the empty case).
    """
    try:
        mu_arr, cov_arr, n = _check_inputs(mu, cov)
        validate_weight_bounds(n, min_weight, max_weight)
    except ValueError:
        return pd.DataFrame(columns=["return", "volatility", "sharpe"])

    mv = min_variance(mu_arr, cov_arr, rf, min_weight, max_weight)
    if not mv.success:
        return pd.DataFrame(columns=["return", "volatility", "sharpe"])
    r_min = mv.expected_return
    try:
        _, r_max_feas = feasible_return_range(mu_arr, min_weight, max_weight)
    except ValueError:
        return pd.DataFrame(columns=["return", "volatility", "sharpe"])
    r_max = float(r_max_feas)

    if r_min >= r_max:
        r_max = r_min + 0.01

    target_returns = np.linspace(r_min, r_max, max(int(n_points), 2))

    frontier = []
    for tr in target_returns:
        res = target_return(mu_arr, cov_arr, float(tr), rf, min_weight, max_weight)
        if res.success and np.isfinite(res.volatility) and np.isfinite(res.expected_return):
            frontier.append({
                "return": res.expected_return,
                "volatility": res.volatility,
                "sharpe": res.sharpe,
            })

    return pd.DataFrame(frontier, columns=["return", "volatility", "sharpe"])


def rebalance_trades(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    tickers: list[str],
    portfolio_value: float,
) -> pd.DataFrame:
    """
    Compute the trades required to rebalance from current to target weights.

    Args:
        current_weights: Current portfolio weight vector (canonical order).
        target_weights:  Target portfolio weight vector (same order).
        tickers:         List of asset tickers (same order as weights).
        portfolio_value: Total portfolio value in currency units.

    Returns:
        DataFrame with columns [Ticker, Current Weight, Target Weight,
                                 Δ Weight, Trade ($), Action].
    """
    cur = np.asarray(current_weights, dtype=float)
    tgt = np.asarray(target_weights, dtype=float)
    if cur.shape != (len(tickers),) or tgt.shape != (len(tickers),):
        raise ValueError(
            f"Weight vectors {cur.shape}/{tgt.shape} do not match "
            f"{len(tickers)} tickers."
        )
    if not np.isfinite(portfolio_value) or portfolio_value <= 0:
        raise ValueError(f"Portfolio value must be positive, got {portfolio_value!r}.")
    delta = tgt - cur
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
            "Current Weight": cur[i],
            "Target Weight": tgt[i],
            "Δ Weight": delta[i],
            "Trade ($)": trade_usd[i],
            "Action": action,
        })

    return pd.DataFrame(rows)

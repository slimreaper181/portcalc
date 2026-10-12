"""
risk_extra.py
-------------
Institutional risk diagnostics beyond VaR/CVaR (pure functions).

Every metric documents its definition; nothing here is a score. Inputs are
simple-return-based annualised statistics (same units as
``analytics.allocation``) unless stated. Invalid input raises ``ValueError``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _check_w_cov(weights: np.ndarray, cov) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(weights, dtype=float)
    cov_arr = cov.values if isinstance(cov, pd.DataFrame) else np.asarray(
        cov, dtype=float)
    if w.ndim != 1 or cov_arr.shape != (w.shape[0], w.shape[0]):
        raise ValueError("Weights/covariance dimensions are inconsistent.")
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(cov_arr)):
        raise ValueError("Weights and covariance must be finite.")
    return w, cov_arr


def component_var(weights: np.ndarray, cov, portfolio_var: float,
                  value: float = 1.0) -> pd.Series:
    """Component VaR per asset: ``CVaR_i = w_i × (Σw)_i / σp × VaR_p``.

    Percentage components sum to (approximately) the total parametric VaR
    when ``portfolio_var`` is the matching parametric VaR — tested as the
    reconciliation contract. Monetary form multiplies by ``value``.
    """
    w, cov_arr = _check_w_cov(weights, cov)
    var_p = float(portfolio_var)
    if not math.isfinite(var_p):
        raise ValueError("Portfolio VaR must be finite.")
    port_var = float(w @ cov_arr @ w)
    if port_var <= 0:
        raise ValueError("Portfolio variance must be positive.")
    vol = math.sqrt(port_var)
    mrc = cov_arr @ w
    comp = w * mrc / vol / vol * var_p * float(value)
    n = len(w)
    tickers = (list(cov.index) if isinstance(cov, pd.DataFrame)
               else [f"a{i}" for i in range(n)])
    return pd.Series(comp, index=tickers)


def marginal_var(weights: np.ndarray, cov, portfolio_var: float,
                 value: float = 1.0) -> pd.Series:
    """Marginal VaR: ``∂VaR/∂w_i = (Σw)_i / σp × VaR_p`` (per unit weight)."""
    w, cov_arr = _check_w_cov(weights, cov)
    var_p = float(portfolio_var)
    if not math.isfinite(var_p):
        raise ValueError("Portfolio VaR must be finite.")
    port_var = float(w @ cov_arr @ w)
    if port_var <= 0:
        raise ValueError("Portfolio variance must be positive.")
    vol = math.sqrt(port_var)
    marg = cov_arr @ w / vol / vol * var_p * float(value)
    n = len(w)
    tickers = (list(cov.index) if isinstance(cov, pd.DataFrame)
               else [f"a{i}" for i in range(n)])
    return pd.Series(marg, index=tickers)


def diversification_benefit(weights: np.ndarray, cov) -> dict:
    """Weighted standalone volatility minus portfolio volatility (>= 0).

    ``benefit = Σ w_i σ_i − σp`` in the same annual units; long-only
    diversification cannot be negative (within floating tolerance).
    """
    w, cov_arr = _check_w_cov(weights, cov)
    vols = np.sqrt(np.maximum(np.diag(cov_arr), 0.0))
    port_vol = math.sqrt(max(float(w @ cov_arr @ w), 0.0))
    if port_vol <= 0:
        raise ValueError("Portfolio volatility must be positive.")
    weighted = float(w @ vols)
    return {"weighted_standalone_vol": weighted,
            "portfolio_vol": port_vol,
            "diversification_benefit": weighted - port_vol}


def herfindahl(weights: np.ndarray) -> float:
    """Herfindahl concentration ``Σ w_i²`` (1/N diversified → 1 concentrated)."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) == 0 or not np.all(np.isfinite(w)):
        raise ValueError("Weights must be a finite non-empty vector.")
    return float((w ** 2).sum())


def distribution_stats(returns) -> dict:
    """Skewness / excess kurtosis of a return series (both labelled).

    Uses population moments; ``kurtosis`` is EXCESS (Fisher, normal → 0).
    Raises on < 4 observations or zero dispersion.
    """
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 4:
        raise ValueError("Need at least 4 return observations.")
    if not np.all(np.isfinite(r.values)):
        raise ValueError("Returns must be finite.")
    std = float(r.std(ddof=0))
    scale = max(1.0, abs(float(r.mean())))
    if not math.isfinite(std) or std <= 1e-12 * scale:
        raise ValueError("Returns have (numerically) zero dispersion.")
    z = (r.values - float(r.mean())) / std
    return {"skewness": float((z ** 3).mean()),
            "excess_kurtosis": float((z ** 4).mean() - 3.0),
            "n_obs": int(len(r))}


def omega_ratio(returns, threshold: float = 0.0) -> float:
    """Omega: ``E[max(R−T,0)] / E[max(T−R,0)]`` at threshold ``T`` (MAR).

    Returns +inf when there is no downside (documented, not clipped);
    raises when both sides are empty.
    """
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) == 0:
        raise ValueError("No return observations.")
    if not np.all(np.isfinite(r.values)):
        raise ValueError("Returns must be finite.")
    if not math.isfinite(float(threshold)):
        raise ValueError("Threshold must be finite.")
    up = np.maximum(r.values - float(threshold), 0.0).mean()
    down = np.maximum(float(threshold) - r.values, 0.0).mean()
    if down <= 0:
        if up <= 0:
            raise ValueError("No dispersion around the threshold.")
        return math.inf
    return float(up / down)


def upside_downside_capture(portfolio_rets, benchmark_rets) -> dict:
    """Upside/downside capture vs benchmark (benchmark up/down subsets).

    Capture_up = mean(Rp | Rb > 0) / mean(Rb | Rb > 0); downside mirrors on
    Rb < 0. Raises when either subset is empty.
    """
    p = pd.Series(portfolio_rets, dtype=float)
    b = pd.Series(benchmark_rets, dtype=float)
    aligned = pd.DataFrame({"p": p, "b": b}).dropna()
    if aligned.empty:
        raise ValueError("No overlapping observations.")
    up = aligned[aligned["b"] > 0]
    down = aligned[aligned["b"] < 0]
    if up.empty or down.empty:
        raise ValueError("Need both up and down benchmark periods.")
    up_cap = float(up["p"].mean() / up["b"].mean()) if up["b"].mean() != 0 else math.inf
    down_cap = float(down["p"].mean() / down["b"].mean()) \
        if down["b"].mean() != 0 else math.inf
    return {"upside_capture": up_cap, "downside_capture": down_cap,
            "n_up": int(len(up)), "n_down": int(len(down))}


def rolling_beta(portfolio_rets, benchmark_rets, window: int) -> pd.Series:
    """Trailing rolling beta ``cov(Rp,Rb)/var(Rb)`` (no future leakage)."""
    p = pd.Series(portfolio_rets, dtype=float)
    b = pd.Series(benchmark_rets, dtype=float)
    if window is None or int(window) < 2:
        raise ValueError("Rolling window must be >= 2.")
    window = int(window)
    aligned = pd.DataFrame({"p": p, "b": b}).dropna()
    if len(aligned) < window:
        raise ValueError("Insufficient overlap for the rolling window.")
    cov_pb = aligned["p"].rolling(window).cov(aligned["b"])
    var_b = aligned["b"].rolling(window).var()
    return (cov_pb / var_b).rename("rolling_beta")


def rolling_correlation(a, b, window: int) -> pd.Series:
    """Trailing rolling correlation (no future leakage)."""
    sa = pd.Series(a, dtype=float)
    sb = pd.Series(b, dtype=float)
    if window is None or int(window) < 2:
        raise ValueError("Rolling window must be >= 2.")
    aligned = pd.DataFrame({"a": sa, "b": sb}).dropna()
    if len(aligned) < int(window):
        raise ValueError("Insufficient overlap for the rolling window.")
    return aligned["a"].rolling(int(window)).corr(
        aligned["b"]).rename("rolling_corr")


def rolling_tracking_error(portfolio_rets, benchmark_rets, window: int,
                           trading_days: int = 252) -> pd.Series:
    """Trailing annualised tracking error ``std(Rp − Rb) × √252``."""
    p = pd.Series(portfolio_rets, dtype=float)
    b = pd.Series(benchmark_rets, dtype=float)
    if window is None or int(window) < 2:
        raise ValueError("Rolling window must be >= 2.")
    aligned = pd.DataFrame({"p": p, "b": b}).dropna()
    if len(aligned) < int(window):
        raise ValueError("Insufficient overlap for the rolling window.")
    active = aligned["p"] - aligned["b"]
    return (active.rolling(int(window)).std()
            * math.sqrt(trading_days)).rename("rolling_te")

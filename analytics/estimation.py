"""
estimation.py
-------------
Institutional covariance / estimation layer (Task 10, Milestone 4).

Estimators (all consume **daily simple returns in Base Currency** and return
**annualised** covariances, matching ``analytics.allocation`` convention —
mean × 252, covariance × 252):

* ``sample`` — the existing estimator (``daily_ret.cov() * 252``), preserved
  bit-for-bit as the default so all existing results are unchanged.
* ``ewma`` — exponentially weighted covariance with mathematically correct
  normalisation: raw weights ``w_i = (1-λ)·λ^{T-1-i}`` (most recent last),
  normalised to sum to one; weighted mean; ``Σ w_i (r_i-μ)(r_i-μ)ᵀ × 252``.
  Configurable decay ``λ ∈ (0, 1)`` (RiskMetrics default 0.94).
* ``ledoit_wolf`` — Ledoit-Wolf (2004) shrinkage toward the scaled identity,
  using the scikit-learn ``ledoit_wolf_shrinkage`` maths (BSD-3-Clause;
  Gramfort/Varoquaux/Fritsch) ported to pure NumPy — identical formulae, no
  sklearn import (its compiled-extension chain is blocked in this runtime).
  Annualised. Handles ``T < p`` gracefully.

Expected returns are intentionally the sample mean only: §4.5 shrunk-mean
forecasting is deferred rather than inventing arbitrary prediction logic.

No-look-ahead: every function here is pure over the returns it is handed —
callers (OOS / walk-forward) pass training slices with strict ``date < D``.
:func:`estimation_window_slice` enforces the strict bound itself.

Column order: estimators are label-preserving and positional — permuting
input columns permutes the output identically (tested). PSD safety is
explicit: :func:`ensure_psd_cov` delegates to the canonical
``allocation.validate_covariance`` gate (minimal ridge for fp noise, raise
on materially invalid input) and :func:`covariance_diagnostics` never hides
a singular matrix (cond ``inf``, ``psd_ok=False``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

#: Available risk-model estimators (stable identifiers for UI + configs).
ESTIMATORS = ("sample", "ewma", "ledoit_wolf")

#: Human labels for the risk-model selector.
ESTIMATOR_LABELS = {
    "sample": "Sample",
    "ewma": "EWMA",
    "ledoit_wolf": "Ledoit-Wolf",
}

#: Estimation window shapes (prices → returns use strict date < end).
ESTIMATION_WINDOWS = ("expanding", "1y", "2y", "3y", "5y")

_WINDOW_YEARS = {"1y": 1.0, "2y": 2.0, "3y": 3.0, "5y": 5.0}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check_returns(daily_ret: pd.DataFrame) -> pd.DataFrame:
    """Validate a daily-returns frame; return it untouched (never reordered)."""
    if daily_ret is None or not isinstance(daily_ret, pd.DataFrame):
        raise ValueError("Daily returns must be a DataFrame.")
    if daily_ret.empty:
        raise ValueError("No daily returns supplied.")
    if len(daily_ret) < 2:
        raise ValueError(
            f"At least 2 return observations are required, got {len(daily_ret)}.")
    try:
        mat = np.asarray(daily_ret.values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("Daily returns must be numeric.")
    if not np.all(np.isfinite(mat)):
        raise ValueError("Daily returns must contain only finite values.")
    return daily_ret


def validate_estimator(method: str) -> str:
    """Normalise an estimator name or raise."""
    m = str(method or "").strip().lower()
    if m not in ESTIMATORS:
        raise ValueError(
            f"Unknown risk model {method!r}; expected one of {list(ESTIMATORS)}.")
    return m


def validate_ewma_lambda(value: float) -> float:
    """Decay factor must lie strictly inside (0, 1)."""
    try:
        lam = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"EWMA decay must be a number, got {value!r}.")
    if not np.isfinite(lam) or not 0.0 < lam < 1.0:
        raise ValueError(f"EWMA decay λ must satisfy 0 < λ < 1, got {value!r}.")
    return lam


# ---------------------------------------------------------------------------
# Mean (sample only — §4.5 shrinkage deliberately deferred)
# ---------------------------------------------------------------------------

def estimate_mean_returns(
    daily_ret: pd.DataFrame, trading_days: int = TRADING_DAYS
) -> pd.Series:
    """Annualised sample mean of daily simple returns: ``mean × 252``.

    Identical to ``allocation.annualised_mean_returns_simple``; kept as the
    single mean path so estimators never invent forecasting logic.
    """
    frame = _check_returns(daily_ret)
    return frame.mean() * float(trading_days)


# ---------------------------------------------------------------------------
# Covariance estimators (all annualised)
# ---------------------------------------------------------------------------

def sample_covariance(
    daily_ret: pd.DataFrame, trading_days: int = TRADING_DAYS
) -> pd.DataFrame:
    """Classic estimator — exactly ``daily_ret.cov() * 252`` (default)."""
    frame = _check_returns(daily_ret)
    return frame.cov() * float(trading_days)


def ewma_covariance(
    daily_ret: pd.DataFrame,
    lam: float = 0.94,
    trading_days: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Exponentially weighted covariance (correct normalisation).

    Raw weights ``(1-λ)·λ^{T-1-i}`` give the most recent observation the
    largest weight; they are normalised to sum to one, the mean is the
    weighted mean, and the covariance is the weighted outer-product sum
    annualised by 252. Positive weights keep the result PSD up to floating
    point noise (symmetrised on return).
    """
    frame = _check_returns(daily_ret)
    lam = validate_ewma_lambda(lam)
    cols = list(frame.columns)
    x = np.asarray(frame.values, dtype=float)
    n = x.shape[0]
    raw = np.array([(1.0 - lam) * (lam ** (n - 1 - i)) for i in range(n)])
    total = raw.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("EWMA weights degenerate for this history length.")
    w = raw / total
    mu = w @ x
    xc = x - mu
    cov_daily = (xc * w[:, None]).T @ xc
    cov_daily = 0.5 * (cov_daily + cov_daily.T)
    return pd.DataFrame(cov_daily * float(trading_days),
                        index=cols, columns=cols)


def _ledoit_wolf_core(xc: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage to the scaled identity, pure NumPy.

    Port of scikit-learn's ``ledoit_wolf_shrinkage`` computation
    (BSD-3-Clause; Gramfort/Varoquaux/Fritsch) — identical formulae, no
    sklearn import. ``xc`` is the demeaned ``(T, p)`` return matrix; the
    empirical covariance uses the MLE divisor ``T`` per LW theory (note:
    the ``sample`` estimator uses ``T-1`` via ``DataFrame.cov`` — the
    difference is documented, not hidden). Returns ``(shrunk_daily, δ)``
    with ``δ`` clipped to ``[0, 1]`` so output can never be wild.
    """
    n, p = xc.shape
    if p == 1:
        return np.atleast_2d(float((xc ** 2).mean())), 0.0
    x2 = xc ** 2
    emp_trace = x2.sum(axis=0) / n
    mu = float(emp_trace.sum() / p)
    beta_num = float((x2.T @ x2).sum())
    delta_num = float(((xc.T @ xc) ** 2).sum()) / n ** 2
    beta = (beta_num / n - delta_num) / (p * n)
    delta = (delta_num - 2.0 * mu * float(emp_trace.sum())
             + p * mu ** 2) / p
    beta = min(beta, delta)
    if delta <= 0 or beta == 0:
        shrinkage = 0.0
    else:
        shrinkage = float(np.clip(beta / delta, 0.0, 1.0))
    emp = (xc.T @ xc) / n
    shrunk = (1.0 - shrinkage) * emp
    shrunk.flat[:: p + 1] += shrinkage * mu
    return shrunk, shrinkage


def ledoit_wolf_shrinkage(daily_ret: pd.DataFrame) -> float:
    """Ledoit-Wolf shrinkage intensity δ ∈ [0, 1] for these returns."""
    frame = _check_returns(daily_ret)
    x = np.asarray(frame.values, dtype=float)
    _, shrinkage = _ledoit_wolf_core(x - x.mean(axis=0))
    return float(shrinkage)


def ledoit_wolf_covariance(
    daily_ret: pd.DataFrame, trading_days: int = TRADING_DAYS
) -> pd.DataFrame:
    """Ledoit-Wolf shrunk covariance, annualised (numpy LW core)."""
    frame = _check_returns(daily_ret)
    cols = list(frame.columns)
    x = np.asarray(frame.values, dtype=float)
    cov_daily, _ = _ledoit_wolf_core(x - x.mean(axis=0))
    cov_daily = 0.5 * (cov_daily + cov_daily.T)
    return pd.DataFrame(cov_daily * float(trading_days),
                        index=cols, columns=cols)


def estimate_covariance(
    daily_ret: pd.DataFrame,
    method: str = "sample",
    trading_days: int = TRADING_DAYS,
    ewma_lambda: float = 0.94,
) -> pd.DataFrame:
    """One entry point for all risk models (labels + order preserved)."""
    m = validate_estimator(method)
    if m == "sample":
        return sample_covariance(daily_ret, trading_days)
    if m == "ewma":
        return ewma_covariance(daily_ret, lam=ewma_lambda,
                               trading_days=trading_days)
    return ledoit_wolf_covariance(daily_ret, trading_days)


# ---------------------------------------------------------------------------
# PSD safety + diagnostics
# ---------------------------------------------------------------------------

def ensure_psd_cov(
    cov: pd.DataFrame | np.ndarray, tickers: list[str]
) -> tuple[pd.DataFrame, dict]:
    """Canonical PSD gate: minimal ridge for fp noise, else raise honestly.

    Delegates to ``allocation.validate_covariance`` so Milestone 4 never
    invents a second PSD policy. Returns ``(fixed_cov, info)`` where info
    carries ``was_adjusted`` and the pre-fix minimum eigenvalue.
    """
    from .allocation import validate_covariance

    arr = (cov.reindex(index=tickers, columns=tickers).values.astype(float)
           if isinstance(cov, pd.DataFrame)
           else np.asarray(cov, dtype=float))
    sym = 0.5 * (arr + arr.T)
    try:
        eig_min_before = float(np.min(np.linalg.eigvalsh(sym)))
    except np.linalg.LinAlgError:
        eig_min_before = float("nan")
    fixed, adjusted = validate_covariance(cov, tickers)
    return fixed, {"was_adjusted": bool(adjusted),
                   "min_eig_before": eig_min_before}


def covariance_diagnostics(cov: pd.DataFrame | np.ndarray) -> dict:
    """Conditioning report — never raises, never hides a singular matrix.

    Returns ``{n, eig_min, eig_max, cond, effective_rank, psd_ok, trace}``.
    ``cond`` is ``inf`` whenever the smallest eigenvalue is ~0 or negative
    (singular or indefinite — inspect ``psd_ok`` to tell them apart) and
    ``nan`` only for degenerate/non-finite input. Effective rank is the
    participation ratio ``(Σλ)²/Σλ²`` over non-negative eigenvalues
    (1 = concentrated, n = perfectly spread).
    """
    try:
        mat = (cov.values.astype(float) if isinstance(cov, pd.DataFrame)
               else np.asarray(cov, dtype=float))
    except (TypeError, ValueError):
        return {"n": 0, "eig_min": float("nan"), "eig_max": float("nan"),
                "cond": float("nan"), "effective_rank": float("nan"),
                "psd_ok": False, "trace": float("nan")}
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] == 0:
        return {"n": 0, "eig_min": float("nan"), "eig_max": float("nan"),
                "cond": float("nan"), "effective_rank": float("nan"),
                "psd_ok": False, "trace": float("nan")}
    n = int(mat.shape[0])
    if not np.all(np.isfinite(mat)):
        return {"n": n, "eig_min": float("nan"), "eig_max": float("nan"),
                "cond": float("nan"), "effective_rank": float("nan"),
                "psd_ok": False, "trace": float("nan")}
    sym = 0.5 * (mat + mat.T)
    try:
        eigs = np.linalg.eigvalsh(sym)
    except np.linalg.LinAlgError:
        return {"n": n, "eig_min": float("nan"), "eig_max": float("nan"),
                "cond": float("nan"), "effective_rank": float("nan"),
                "psd_ok": False, "trace": float(np.trace(sym))}
    lo, hi = float(eigs[0]), float(eigs[-1])
    psd_ok = bool(lo >= -1e-10)
    if hi <= 0 or not np.isfinite(hi):
        cond = float("nan")
    elif lo <= 1e-12:
        cond = float("inf")
    else:
        cond = float(hi / lo)
    pos = eigs[eigs > 0]
    if pos.size == 0:
        eff = float("nan")
    else:
        eff = float((pos.sum() ** 2) / (np.square(pos).sum()))
    return {"n": n, "eig_min": lo, "eig_max": hi, "cond": cond,
            "effective_rank": eff, "psd_ok": psd_ok,
            "trace": float(np.trace(sym))}


def correlation_stability(recent_rets: pd.DataFrame,
                          long_rets: pd.DataFrame) -> dict:
    """Recent-vs-long-run correlation drift (same columns required).

    Returns ``{frob, max_abs, mean_abs, n}`` over the correlation-matrix
    difference (diagonal excluded from max/mean). Pure descriptive stats —
    no score, no recommendation.
    """
    if recent_rets is None or long_rets is None:
        raise ValueError("Both return frames are required.")
    cols = list(recent_rets.columns)
    if list(long_rets.columns) != cols:
        raise ValueError("Return frames must share identical columns.")
    if len(recent_rets) < 3 or len(long_rets) < 3:
        raise ValueError("At least 3 observations per window are required.")
    c_r = np.asarray(recent_rets.corr().values, dtype=float)
    c_l = np.asarray(long_rets.corr().values, dtype=float)
    if not np.all(np.isfinite(c_r)) or not np.all(np.isfinite(c_l)):
        raise ValueError("Correlation is undefined for these windows.")
    diff = c_r - c_l
    mask = ~np.eye(len(cols), dtype=bool)
    off = np.abs(diff[mask])
    return {"frob": float(np.linalg.norm(diff, "fro")),
            "max_abs": float(off.max()) if off.size else 0.0,
            "mean_abs": float(off.mean()) if off.size else 0.0,
            "n": len(cols)}


# ---------------------------------------------------------------------------
# Estimation windows (strict date < end — no look-ahead)
# ---------------------------------------------------------------------------

def estimation_window_slice(
    prices: pd.DataFrame,
    end,
    window: str = "expanding",
    lookback_years: float = 3.0,
) -> pd.DataFrame:
    """Slice a price panel to the estimation window ending strictly at ``end``.

    Keeps rows with ``index < end`` only (never ``<=``); ``window`` is one of
    expanding / 1y / 2y / 3y / 5y. Raises on empty results.
    """
    if prices is None or prices.empty:
        raise ValueError("No price history supplied.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Price history needs a DatetimeIndex.")
    w = str(window or "expanding").strip().lower()
    if w not in ESTIMATION_WINDOWS:
        raise ValueError(
            f"Unknown estimation window {window!r}; expected one of "
            f"{list(ESTIMATION_WINDOWS)}.")
    try:
        end_ts = pd.Timestamp(end)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid window end {end!r}.")
    past = prices.loc[prices.index < end_ts]
    if w != "expanding":
        years = _WINDOW_YEARS[w] if w in _WINDOW_YEARS else float(lookback_years)
        cutoff = end_ts - pd.DateOffset(years=float(years))
        past = past.loc[past.index >= cutoff]
    if past.empty:
        raise ValueError("Estimation window holds no rows strictly before end.")
    return past

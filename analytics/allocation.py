"""
allocation.py
-------------
Institutional portfolio construction: Equal Risk Contribution (risk parity),
Maximum Diversification, and Black-Litterman — plus allocation diagnostics.

Return convention (simple arithmetic returns throughout this module)
--------------------------------------------------------------------
* ``simple_returns_from_prices``: ``prices.pct_change().dropna()``.
* Annual expected return = mean(daily simple returns) × 252.
* Annual covariance = daily covariance × 252.
* Black-Litterman inputs use these consistent annual units — never mix
  daily covariance with annual expected returns in one model.

This intentionally differs from modules that use log returns; the units
are documented at every boundary. Do not change other modules' conventions.

Ordering convention
-------------------
Canonical ticker order (see ``analytics.validation``). Labelled pandas
objects are preferred until the final numerical optimisation; every public
function reindexes/validates alignment explicitly.

Failure philosophy (matches ``analytics.optimisation``)
--------------------------------------------------------
Failed optimisations return ``success=False``, ``weights=None`` and a clear
message — never a fallback portfolio. Callers must check ``success`` first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .portfolio import (
    portfolio_expected_return,
    portfolio_std,
    risk_contributions,
    sharpe_ratio,
)
from .validation import canonical_tickers, validate_weight_bounds

TRADING_DAYS = 252

#: Tolerance for "approximately equal" ERC percentage risk contributions.
ERC_PRCTOL = 0.02

#: Lower bound guard for the diversification-ratio consistency check.
DR_TOL = 1e-6


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class AllocationResult:
    """Structured outcome of one allocation method.

    On success, ``weights`` holds labelled optimal weights and the numeric
    fields are finite. On failure, ``weights`` is None and numeric fields
    are NaN; ``message`` explains why. Never use the numeric fields when
    ``success`` is False.
    """
    method: str
    success: bool
    weights: Optional[pd.Series]
    expected_return: float
    volatility: float
    sharpe: float
    diversification_ratio: float
    risk_contributions: Optional[pd.Series]
    message: str


@dataclass
class ViewSpec:
    """Investor views as explicit matrices: P (K×N), Q (K,), confidence (K,)."""
    P: np.ndarray
    Q: np.ndarray
    confidence: np.ndarray
    labels: list[str]


@dataclass
class BlackLittermanResult:
    """Prior/posterior expected returns (annual, simple-return units)."""
    success: bool
    prior: Optional[pd.Series]
    posterior: Optional[pd.Series]
    n_views: int
    message: str


# ---------------------------------------------------------------------------
# Simple-return statistics
# ---------------------------------------------------------------------------

def simple_returns_from_prices(
    prices: pd.DataFrame, tickers: list[str]
) -> pd.DataFrame:
    """Daily **simple** returns from adjusted closes: ``pct_change().dropna()``.

    Columns are reindexed to canonical ticker order; rows with any missing
    observation are dropped (common trading dates, consistent with the
    backtest engine). Raises on empty frames, non-DatetimeIndex indexes, or
    non-positive/non-finite prices.
    """
    tickers = canonical_tickers(list(tickers))
    if prices is None or prices.empty:
        raise ValueError("No price history supplied.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Price history needs a DatetimeIndex.")
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Price data is missing tickers: {missing}")
    px = prices.reindex(columns=tickers).sort_index()
    bad = px.columns[((px <= 0) | (~np.isfinite(px))).any()].tolist()
    if bad:
        raise ValueError(f"Invalid (zero/negative/NaN) prices for: {', '.join(bad)}.")
    rets = px.pct_change().dropna(how="any")
    if rets.empty:
        raise ValueError("No overlapping return observations.")
    return rets


def annualised_mean_returns_simple(
    daily_ret: pd.DataFrame, trading_days: int = TRADING_DAYS
) -> pd.Series:
    """Annualised mean of daily **simple** returns: ``mean × 252``."""
    if daily_ret is None or daily_ret.empty:
        raise ValueError("No daily returns supplied.")
    return daily_ret.mean() * trading_days


def annualised_covariance_simple(
    daily_ret: pd.DataFrame, trading_days: int = TRADING_DAYS
) -> pd.DataFrame:
    """Annualised covariance of daily **simple** returns: ``cov × 252``."""
    if daily_ret is None or daily_ret.empty:
        raise ValueError("No daily returns supplied.")
    return daily_ret.cov() * trading_days


# ---------------------------------------------------------------------------
# Covariance validation
# ---------------------------------------------------------------------------

def validate_covariance(
    cov: pd.DataFrame | np.ndarray, tickers: list[str]
) -> tuple[pd.DataFrame, bool]:
    """Validate a covariance matrix; apply minimal ridge fix if warranted.

    Checks: square, labels match ``tickers`` (reindexed to canonical order),
    finite values, symmetry within 1e-8 relative tolerance, and positive
    semi-definiteness. Tiny negative eigenvalues from floating-point error
    get the minimal diagonal ridge that restores PSD; materially invalid
    matrices (large negative eigenvalues, asymmetry, non-finite) raise.

    Returns:
        Tuple ``(fixed_covariance, was_adjusted)`` as a labelled DataFrame.
    """
    tickers = canonical_tickers(list(tickers))
    n = len(tickers)
    if isinstance(cov, pd.DataFrame):
        if set(cov.index) != set(tickers) or set(cov.columns) != set(tickers):
            raise ValueError(
                "Covariance labels do not match tickers "
                f"(got rows={list(cov.index)}, cols={list(cov.columns)})."
            )
        mat = cov.reindex(index=tickers, columns=tickers).values.astype(float)
    else:
        mat = np.asarray(cov, dtype=float)
        if mat.shape != (n, n):
            raise ValueError(
                f"Covariance shape {mat.shape} does not match {n} tickers "
                f"(expected {(n, n)})."
            )
    if not np.all(np.isfinite(mat)):
        raise ValueError("Covariance matrix must contain only finite values.")
    scale = max(float(np.abs(mat).max()), 1e-12)
    asym = float(np.abs(mat - mat.T).max())
    if asym > 1e-8 * scale:
        raise ValueError(
            f"Covariance matrix is asymmetric (max asymmetry {asym:.3g}); "
            "refusing to symmetrise a materially broken matrix."
        )
    mat = (mat + mat.T) / 2.0
    eig = np.linalg.eigvalsh(mat)
    max_eig = max(float(eig.max()), 0.0)
    tol = 1e-8 * max(1.0, max_eig)
    min_eig = float(eig.min())
    if min_eig < -tol:
        raise ValueError(
            f"Covariance matrix is not positive semi-definite "
            f"(min eigenvalue {min_eig:.3g}); refusing to repair."
        )
    adjusted = False
    if min_eig < 0:
        # Minimal ridge: just enough to restore numerical PSD.
        mat = mat + (-min_eig + 1e-10) * np.eye(n)
        adjusted = True
    return pd.DataFrame(mat, index=tickers, columns=tickers), adjusted


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def risk_contributions_simple(
    weights: np.ndarray, cov: pd.DataFrame | np.ndarray, tickers: list[str]
) -> pd.Series:
    """Percentage risk contributions ``PRC_i = RC_i / σp`` (sums to 1).

    Reuses :func:`analytics.portfolio.risk_contributions` (identical
    mathematics); labelled here for alignment safety.
    """
    tickers = canonical_tickers(list(tickers))
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(tickers),):
        raise ValueError(
            f"Weights shape {w.shape} does not match {len(tickers)} tickers."
        )
    cov_arr = cov.values if isinstance(cov, pd.DataFrame) else np.asarray(
        cov, dtype=float)
    if cov_arr.shape != (len(tickers), len(tickers)):
        raise ValueError("Covariance shape does not match tickers.")
    return pd.Series(risk_contributions(w, cov_arr), index=tickers)


def diversification_ratio(
    weights: np.ndarray, cov: pd.DataFrame | np.ndarray
) -> float:
    """Diversification ratio ``DR = Σ w_i σ_i / √(w'Σw)``.

    Individual volatilities come from the covariance diagonal. Raises on
    non-positive portfolio volatility.
    """
    w = np.asarray(weights, dtype=float)
    cov_arr = cov.values if isinstance(cov, pd.DataFrame) else np.asarray(
        cov, dtype=float)
    if w.ndim != 1 or cov_arr.shape != (w.shape[0], w.shape[0]):
        raise ValueError("Weights/covariance dimensions are inconsistent.")
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(cov_arr)):
        raise ValueError("Weights and covariance must be finite.")
    asset_vol = np.sqrt(np.maximum(np.diag(cov_arr), 0.0))
    vol = float(np.sqrt(max(float(w @ cov_arr @ w), 0.0)))
    if vol <= 0:
        raise ValueError("Portfolio volatility is zero — ratio undefined.")
    return float((w @ asset_vol) / vol)


def effective_number_of_holdings(weights: np.ndarray) -> float:
    """Weight-concentration measure ``N_eff = 1 / Σ w_i²``.

    Equals N for equal weights and approaches 1 for concentrated
    portfolios. This is weight concentration — *not* "effective number of
    bets", which would require a risk-factor decomposition.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) == 0:
        raise ValueError("Weights must be a non-empty vector.")
    if not np.all(np.isfinite(w)):
        raise ValueError("Weights must be finite.")
    denom = float((w ** 2).sum())
    if denom <= 0:
        raise ValueError("Weights are all zero — concentration undefined.")
    return float(1.0 / denom)


def risk_contribution_dispersion(prc: pd.Series | np.ndarray) -> float:
    """Dispersion of percentage risk contributions: population std (ddof=0).

    Near zero means risk is spread evenly (the risk-parity ideal).
    """
    vals = np.asarray(prc.values if isinstance(prc, pd.Series) else prc,
                      dtype=float)
    if vals.size == 0:
        raise ValueError("No risk contributions supplied.")
    if not np.all(np.isfinite(vals)):
        raise ValueError("Risk contributions must be finite.")
    return float(vals.std(ddof=0))


def allocation_summary(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: pd.DataFrame | np.ndarray,
    rf: float,
    tickers: list[str],
) -> dict:
    """One-shot diagnostics for a weight vector (simple-return units).

    Returns dict with ``expected_return``, ``volatility``, ``sharpe``,
    ``diversification_ratio``, ``largest_position``, ``effective_holdings``,
    ``rc_dispersion`` and labelled ``risk_contributions``.
    """
    tickers = canonical_tickers(list(tickers))
    w = np.asarray(weights, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    if w.shape != (len(tickers),) or mu_arr.shape != (len(tickers),):
        raise ValueError("Weights/returns length does not match tickers.")
    if not np.isfinite(rf):
        raise ValueError(f"Risk-free rate must be finite, got {rf!r}.")
    cov_df, _ = validate_covariance(cov, tickers)
    prc = risk_contributions_simple(w, cov_df, tickers)
    return {
        "expected_return": portfolio_expected_return(w, mu_arr),
        "volatility": portfolio_std(w, cov_df.values),
        "sharpe": sharpe_ratio(portfolio_expected_return(w, mu_arr),
                               portfolio_std(w, cov_df.values), rf),
        "diversification_ratio": diversification_ratio(w, cov_df),
        "largest_position": float(np.max(w)),
        "effective_holdings": effective_number_of_holdings(w),
        "rc_dispersion": risk_contribution_dispersion(prc),
        "risk_contributions": prc,
    }


# ---------------------------------------------------------------------------
# Shared optimiser plumbing
# ---------------------------------------------------------------------------

def _alloc_failure(method: str, message: str) -> AllocationResult:
    """Explicit failure: no weights, NaN stats, clear message."""
    nan = float("nan")
    return AllocationResult(
        method=method, success=False, weights=None,
        expected_return=nan, volatility=nan, sharpe=nan,
        diversification_ratio=nan, risk_contributions=None, message=message,
    )


def _alloc_success(
    method: str,
    tickers: list[str],
    w: np.ndarray,
    mu: np.ndarray,
    cov: pd.DataFrame,
    rf: float,
    message: str = "Converged",
) -> AllocationResult:
    """Build a success result with full diagnostics."""
    summary = allocation_summary(w, mu, cov, rf, tickers)
    return AllocationResult(
        method=method, success=True,
        weights=pd.Series(np.asarray(w, dtype=float), index=tickers),
        expected_return=summary["expected_return"],
        volatility=summary["volatility"],
        sharpe=summary["sharpe"],
        diversification_ratio=summary["diversification_ratio"],
        risk_contributions=summary["risk_contributions"],
        message=message,
    )


def _slsqp_check(
    res, w: np.ndarray, n: int, min_weight: float, max_weight: float,
    label: str,
) -> AllocationResult | None:
    """Return a failure result if an SLSQP solve is unusable, else None."""
    if res is None or not getattr(res, "success", False) \
            or not np.all(np.isfinite(w)):
        msg = getattr(res, "message", "?")
        msg = msg if isinstance(msg, str) else str(msg)
        return _alloc_failure(label, f"{label} did not converge: {msg}")
    if abs(float(w.sum()) - 1.0) > 1e-6:
        return _alloc_failure(
            label, f"{label} weights do not sum to 1 (got {float(w.sum()):.6f}).")
    if np.any(w < min_weight - 1e-9) or np.any(w > max_weight + 1e-9):
        return _alloc_failure(label, f"{label} weights violate bounds.")
    return None


def _bounds(n: int, min_weight: float, max_weight: float) -> list[tuple]:
    return [(min_weight, max_weight)] * n


def _sum_to_one() -> list[dict]:
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]


# ---------------------------------------------------------------------------
# Equal Risk Contribution (risk parity)
# ---------------------------------------------------------------------------

def _erc_objective(w: np.ndarray, cov_arr: np.ndarray, n: int) -> float:
    """Sum of squared deviations of percentage risk contributions from 1/N.

    Deliberately *not* inverse-volatility weighting, which coincides with
    risk parity only when correlations are equal.
    """
    var = float(w @ cov_arr @ w)
    if not np.isfinite(var) or var <= 0:
        return 1e6
    vol = float(np.sqrt(var))
    mrc = cov_arr @ w
    prc = (w * mrc / vol) / vol
    if not np.all(np.isfinite(prc)):
        return 1e6
    return float(((prc - 1.0 / n) ** 2).sum())


def equal_risk_contribution(
    tickers: list[str],
    mu: np.ndarray,
    cov: pd.DataFrame | np.ndarray,
    rf: float = 0.05,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> AllocationResult:
    """Long-only Equal Risk Contribution portfolio.

    Minimises ``Σ(PRC_i − 1/N)²`` subject to budget + bounds, starting from
    equal weights. On success the percentage risk contributions are
    approximately equal (within ``ERC_PRCTOL``); otherwise a structured
    failure is returned instead of misleading weights.
    """
    method = "Equal Risk Contribution"
    tickers = canonical_tickers(list(tickers))
    n = len(tickers)
    try:
        validate_weight_bounds(n, min_weight, max_weight)
    except ValueError as e:
        return _alloc_failure(method, str(e))
    try:
        cov_df, _ = validate_covariance(cov, tickers)
    except ValueError as e:
        return _alloc_failure(method, f"Invalid covariance: {e}")
    mu_arr = np.asarray(mu, dtype=float)
    if mu_arr.shape != (n,) or not np.all(np.isfinite(mu_arr)):
        raise ValueError("Expected returns must be finite with length n.")
    if not np.isfinite(rf):
        raise ValueError(f"Risk-free rate must be finite, got {rf!r}.")
    cov_arr = cov_df.values
    w0 = np.ones(n) / n

    try:
        res = minimize(
            fun=lambda w: _erc_objective(w, cov_arr, n),
            x0=w0,
            method="SLSQP",
            bounds=_bounds(n, min_weight, max_weight),
            constraints=_sum_to_one(),
            options={"ftol": 1e-12, "maxiter": 1000},
        )
    except Exception as e:  # never leak solver exceptions
        return _alloc_failure(method, f"Risk-parity optimisation failed: {e}")

    w = np.asarray(res.x, dtype=float)
    bad = _slsqp_check(res, w, n, min_weight, max_weight, "Risk parity")
    if bad is not None:
        return bad
    prc = risk_contributions_simple(w, cov_df, tickers)
    dev = float(np.abs(prc.values - 1.0 / n).max())
    if not np.isfinite(dev) or dev > ERC_PRCTOL:
        return _alloc_failure(
            method,
            f"Risk parity did not equalise contributions "
            f"(max deviation {dev:.2%} > {ERC_PRCTOL:.0%} tolerance).")
    return _alloc_success(method, tickers, w, mu_arr, cov_df, rf)


# ---------------------------------------------------------------------------
# Maximum Diversification
# ---------------------------------------------------------------------------

def maximum_diversification(
    tickers: list[str],
    mu: np.ndarray,
    cov: pd.DataFrame | np.ndarray,
    rf: float = 0.05,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> AllocationResult:
    """Maximum Diversification portfolio: maximise ``Σwσ / √(w'Σw)``.

    Subject to budget + bounds from an equal-weight start. For ordinary
    long-only inputs DR ≥ 1 mathematically; a computed DR below ``1 − tol``
    signals numerical trouble and returns failure rather than a fake result.
    """
    method = "Maximum Diversification"
    tickers = canonical_tickers(list(tickers))
    n = len(tickers)
    try:
        validate_weight_bounds(n, min_weight, max_weight)
    except ValueError as e:
        return _alloc_failure(method, str(e))
    try:
        cov_df, _ = validate_covariance(cov, tickers)
    except ValueError as e:
        return _alloc_failure(method, f"Invalid covariance: {e}")
    mu_arr = np.asarray(mu, dtype=float)
    if mu_arr.shape != (n,) or not np.all(np.isfinite(mu_arr)):
        raise ValueError("Expected returns must be finite with length n.")
    if not np.isfinite(rf):
        raise ValueError(f"Risk-free rate must be finite, got {rf!r}.")
    cov_arr = cov_df.values
    asset_vol = np.sqrt(np.maximum(np.diag(cov_arr), 0.0))

    def neg_dr(w: np.ndarray) -> float:
        var = float(w @ cov_arr @ w)
        if not np.isfinite(var) or var <= 0:
            return 1e6
        return -float((w @ asset_vol) / np.sqrt(var))

    w0 = np.ones(n) / n
    try:
        res = minimize(
            fun=neg_dr,
            x0=w0,
            method="SLSQP",
            bounds=_bounds(n, min_weight, max_weight),
            constraints=_sum_to_one(),
            options={"ftol": 1e-12, "maxiter": 1000},
        )
    except Exception as e:
        return _alloc_failure(method, f"Max-diversification failed: {e}")

    w = np.asarray(res.x, dtype=float)
    bad = _slsqp_check(res, w, n, min_weight, max_weight,
                       "Max diversification")
    if bad is not None:
        return bad
    try:
        dr = diversification_ratio(w, cov_df)
    except ValueError as e:
        return _alloc_failure(method, f"Invalid diversification ratio: {e}")
    if not np.isfinite(dr):
        return _alloc_failure(method, "Diversification ratio is not finite.")
    if dr < 1.0 - DR_TOL:
        return _alloc_failure(
            method,
            f"Diversification ratio {dr:.4f} below 1 — numerically "
            "inconsistent for a long-only portfolio.")
    return _alloc_success(method, tickers, w, mu_arr, cov_df, rf)


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------

def validate_delta(delta: float) -> float:
    """Risk-aversion coefficient: finite and strictly positive."""
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError(
            f"Risk aversion δ must be positive and finite, got {delta!r}.")
    return float(delta)


def validate_tau(tau: float) -> float:
    """Prior-uncertainty scalar: finite and strictly positive."""
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError(
            f"Tau must be positive and finite, got {tau!r}.")
    return float(tau)


def validate_confidence(confidence: float) -> float:
    """View confidence strictly inside (0, 1) — never exactly 0 or 1."""
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError(
            f"View confidence must be strictly between 0 and 1 "
            f"(got {confidence!r}); use e.g. 0.7 for 70%."
        )
    return float(confidence)


def validate_reference_weights(
    ref_weights: np.ndarray, tickers: list[str]
) -> np.ndarray:
    """Reference-weight prior: long-only weights summing to 1.

    These reverse-engineer implied equilibrium returns; they are a proxy
    prior, not necessarily true market-cap equilibrium weights.
    """
    tickers = canonical_tickers(list(tickers))
    w = np.asarray(ref_weights, dtype=float)
    if w.shape != (len(tickers),):
        raise ValueError(
            f"Reference weights length {w.shape} does not match "
            f"{len(tickers)} tickers."
        )
    if not np.all(np.isfinite(w)):
        raise ValueError("Reference weights must be finite.")
    if np.any(w < -1e-9) or np.any(w > 1 + 1e-9):
        raise ValueError("Reference weights must be long-only (0 to 1).")
    if abs(float(w.sum()) - 1.0) > 1e-6:
        raise ValueError(
            f"Reference weights must sum to 1 (got {float(w.sum()):.6f}).")
    return w


def implied_equilibrium_returns(
    cov: pd.DataFrame | np.ndarray,
    ref_weights: np.ndarray,
    delta: float,
    tickers: list[str],
) -> pd.Series:
    """Implied prior expected returns ``π = δΣw_ref`` (annual units)."""
    tickers = canonical_tickers(list(tickers))
    cov_df, _ = validate_covariance(cov, tickers)
    w = validate_reference_weights(ref_weights, tickers)
    delta = validate_delta(delta)
    return pd.Series(cov_df.values @ w * delta, index=tickers, name="prior")


def build_view_matrices(
    views: list[dict], tickers: list[str]
) -> ViewSpec:
    """Build explicit P (K×N), Q (K,) and confidence (K,) from view dicts.

    Absolute view: ``{"type": "absolute", "ticker": T, "return": r,
    "confidence": c}`` → P row with 1 on T, ``Q = r``.
    Relative view: ``{"type": "relative", "long": L, "short": S,
    "return": r, "confidence": c}`` → +1 on L, −1 on S, ``Q = r`` (the
    expected return *spread*). Returns are decimal annual.
    An empty view list yields empty (0×N) matrices — allowed (no-view case).
    """
    tickers = canonical_tickers(list(tickers))
    n = len(tickers)
    pos = {t: i for i, t in enumerate(tickers)}
    rows, qs, confs, labels = [], [], [], []
    for i, v in enumerate(views or []):
        if not isinstance(v, dict):
            raise ValueError(f"View {i} must be a dict.")
        vtype = str(v.get("type", "")).strip().lower()
        row = np.zeros(n)
        try:
            ret = float(v["return"])
            conf = validate_confidence(float(v["confidence"]))
        except KeyError as e:
            raise ValueError(f"View {i} is missing {e}.")
        except (TypeError, ValueError) as e:
            raise ValueError(f"View {i} has invalid return/confidence: {e}")
        if not np.isfinite(ret):
            raise ValueError(f"View {i} return must be finite.")
        if vtype == "absolute":
            t = str(v.get("ticker", "")).strip().upper()
            if t not in pos:
                raise ValueError(
                    f"View {i}: unknown ticker {v.get('ticker')!r}.")
            row[pos[t]] = 1.0
            labels.append(f"{t} = {ret:.1%}")
        elif vtype == "relative":
            lo = str(v.get("long", "")).strip().upper()
            sh = str(v.get("short", "")).strip().upper()
            if lo not in pos or sh not in pos:
                raise ValueError(
                    f"View {i}: unknown tickers "
                    f"{v.get('long')!r}/{v.get('short')!r}.")
            if lo == sh:
                raise ValueError(f"View {i}: long and short must differ.")
            row[pos[lo]] = 1.0
            row[pos[sh]] = -1.0
            labels.append(f"{lo}−{sh} = {ret:+.1%}")
        else:
            raise ValueError(
                f"View {i}: type must be 'absolute' or 'relative', "
                f"got {v.get('type')!r}.")
        rows.append(row)
        qs.append(ret)
        confs.append(conf)
    P = np.array(rows, dtype=float).reshape(len(rows), n)
    return ViewSpec(P=P, Q=np.array(qs, dtype=float),
                    confidence=np.array(confs, dtype=float), labels=labels)


def view_uncertainty_matrix(
    P: np.ndarray, tau_cov: np.ndarray, confidence: np.ndarray
) -> np.ndarray:
    """View uncertainty ``Ω_ii = ((1−c_i)/c_i) × (PτΣP')_ii`` (diagonal).

    Higher confidence → smaller Ω → stronger posterior pull. Confidence is
    validated strictly inside (0, 1); this is a documented proportional
    convention, *not* Idzorek's method.
    """
    P = np.asarray(P, dtype=float)
    tau_cov = np.asarray(tau_cov, dtype=float)
    conf = np.asarray(confidence, dtype=float)
    if P.ndim != 2 or conf.ndim != 1 or P.shape[0] != conf.shape[0]:
        raise ValueError("P/confidence dimensions are inconsistent.")
    if tau_cov.shape != (P.shape[1], P.shape[1]):
        raise ValueError("tau·cov dimensions do not match P columns.")
    if P.shape[0] == 0:
        return np.zeros((0, 0))
    for c in conf:
        validate_confidence(float(c))
    prior_var = np.einsum("ki,ij,kj->k", P, tau_cov, P)
    if not np.all(np.isfinite(prior_var)) or np.any(prior_var < 0):
        raise ValueError("Prior view variance is invalid.")
    omega = ((1.0 - conf) / conf) * prior_var
    if not np.all(np.isfinite(omega)) or np.any(omega < 0):
        raise ValueError("View uncertainty is invalid.")
    return np.diag(omega)


def black_litterman_posterior(
    tickers: list[str],
    cov: pd.DataFrame | np.ndarray,
    ref_weights: np.ndarray,
    delta: float,
    tau: float,
    views: list[dict],
) -> BlackLittermanResult:
    """Standard Black-Litterman posterior via numerically stable solves.

    ``μ_BL = M⁻¹(Sπ + P'Ω⁻¹Q)`` with ``M = S + P'Ω⁻¹P``, ``S = (τΣ)⁻¹`` —
    computed with ``np.linalg.solve`` (no explicit inverses). With no views
    the posterior reduces sensibly to the prior.
    """
    tickers = canonical_tickers(list(tickers))
    n = len(tickers)
    try:
        delta = validate_delta(delta)
        tau = validate_tau(tau)
        cov_df, _ = validate_covariance(cov, tickers)
        w_ref = validate_reference_weights(ref_weights, tickers)
        spec = build_view_matrices(views, tickers)
    except ValueError as e:
        return BlackLittermanResult(
            success=False, prior=None, posterior=None,
            n_views=0, message=str(e))
    cov_arr = cov_df.values
    prior = pd.Series(cov_arr @ w_ref * delta, index=tickers, name="prior")
    if spec.P.shape[0] == 0:
        return BlackLittermanResult(
            success=True, prior=prior, posterior=prior.copy(), n_views=0,
            message="No views supplied — posterior equals the prior.")
    tau_cov = tau * cov_arr
    try:
        omega = view_uncertainty_matrix(spec.P, tau_cov, spec.confidence)
        oinv = np.diag(1.0 / np.diag(omega))
        Sinv_pi = np.linalg.solve(tau_cov, prior.values)
        Sinv = np.linalg.solve(tau_cov, np.eye(n))
        M = Sinv + spec.P.T @ oinv @ spec.P
        rhs = Sinv_pi + spec.P.T @ oinv @ spec.Q
        post = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError as e:
        return BlackLittermanResult(
            success=False, prior=prior, posterior=None,
            n_views=spec.P.shape[0],
            message=f"Posterior solve failed (ill-conditioned inputs): {e}")
    except (ValueError, FloatingPointError) as e:
        return BlackLittermanResult(
            success=False, prior=prior, posterior=None,
            n_views=spec.P.shape[0], message=str(e))
    if not np.all(np.isfinite(post)):
        return BlackLittermanResult(
            success=False, prior=prior, posterior=None,
            n_views=spec.P.shape[0],
            message="Posterior contains non-finite values.")
    return BlackLittermanResult(
        success=True, prior=prior,
        posterior=pd.Series(post, index=tickers, name="posterior"),
        n_views=spec.P.shape[0], message="Converged")


def black_litterman_allocation(
    tickers: list[str],
    posterior_mu: np.ndarray | pd.Series,
    cov: pd.DataFrame | np.ndarray,
    delta: float,
    rf: float = 0.05,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> AllocationResult:
    """Constrained mean-variance utility on BL posterior returns.

    Maximises ``w'μ_BL − (δ/2)·w'Σw`` subject to budget + bounds from an
    equal-weight start. Structured failure (never fake weights) on any
    solver/validation problem.
    """
    method = "Black-Litterman"
    tickers = canonical_tickers(list(tickers))
    n = len(tickers)
    try:
        validate_weight_bounds(n, min_weight, max_weight)
        delta = validate_delta(delta)
        cov_df, _ = validate_covariance(cov, tickers)
    except ValueError as e:
        return _alloc_failure(method, str(e))
    if isinstance(posterior_mu, pd.Series):
        if set(posterior_mu.index) != set(tickers):
            return _alloc_failure(method, "Posterior labels do not match tickers.")
        mu_arr = posterior_mu.reindex(tickers).values.astype(float)
    else:
        mu_arr = np.asarray(posterior_mu, dtype=float)
    if mu_arr.shape != (n,) or not np.all(np.isfinite(mu_arr)):
        return _alloc_failure(method, "Posterior returns must be finite (length n).")
    if not np.isfinite(rf):
        return _alloc_failure(method, f"Risk-free rate must be finite, got {rf!r}.")
    cov_arr = cov_df.values

    def neg_utility(w: np.ndarray) -> float:
        val = float(w @ mu_arr - (delta / 2.0) * (w @ cov_arr @ w))
        return -val if np.isfinite(val) else 1e6

    w0 = np.ones(n) / n
    try:
        res = minimize(
            fun=neg_utility,
            x0=w0,
            method="SLSQP",
            bounds=_bounds(n, min_weight, max_weight),
            constraints=_sum_to_one(),
            options={"ftol": 1e-12, "maxiter": 1000},
        )
    except Exception as e:
        return _alloc_failure(method, f"BL utility optimisation failed: {e}")

    w = np.asarray(res.x, dtype=float)
    bad = _slsqp_check(res, w, n, min_weight, max_weight, "Black-Litterman")
    if bad is not None:
        return bad
    return _alloc_success(method, tickers, w, mu_arr, cov_df, rf)


def estimate_risk_aversion_from_benchmark(
    bench_excess_ann: float, bench_var_ann: float
) -> float:
    """Estimate ``δ = (E[Rm] − Rf) / Var(Rm)`` in annual simple-return units.

    Rejects non-finite inputs, non-positive variance and non-positive
    estimates — callers must fall back clearly to a user-entered δ.
    """
    if not np.isfinite(bench_excess_ann) or not np.isfinite(bench_var_ann):
        raise ValueError("Benchmark excess return and variance must be finite.")
    if bench_var_ann <= 0:
        raise ValueError(
            f"Benchmark variance must be positive, got {bench_var_ann!r}.")
    delta = float(bench_excess_ann / bench_var_ann)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError(
            f"Benchmark-implied risk aversion is not positive "
            f"(got {delta!r}); use the manual δ instead.")
    return delta


# ---------------------------------------------------------------------------
# Estimation / test separation for out-of-sample backtests
# ---------------------------------------------------------------------------

def estimation_window(
    prices: pd.DataFrame,
    tickers: list[str],
    test_start,
    lookback_years: float,
    min_rows: int = 60,
) -> tuple[pd.DataFrame, dict]:
    """Estimation prices strictly before the test start (no look-ahead).

    Window = ``[test_start − lookback_years, test_start)`` on common trading
    dates (rows with any missing observation are dropped). The upper bound
    is exclusive: no observation dated ``>= test_start`` may enter
    covariance/expected-return/reference estimation, so the first
    test-period observation can never influence estimated weights.

    Returns ``(est_prices, info)`` with exact ``est_start``/``est_end`` dates
    and ``n_obs``. Raises on insufficient data.
    """
    tickers = canonical_tickers(list(tickers))
    if prices is None or prices.empty:
        raise ValueError("No price history supplied for estimation.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Price history needs a DatetimeIndex.")
    try:
        lookback_years = float(lookback_years)
    except (TypeError, ValueError):
        raise ValueError(f"Lookback must be a number of years, got {lookback_years!r}.")
    if not np.isfinite(lookback_years) or lookback_years <= 0:
        raise ValueError(f"Lookback must be positive years, got {lookback_years!r}.")
    try:
        T = pd.Timestamp(test_start)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid test start date: {e}")
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Price data is missing tickers: {missing}")
    px = prices.reindex(columns=tickers).sort_index()
    cutoff = T - pd.DateOffset(years=lookback_years)
    est = px.loc[(px.index >= cutoff) & (px.index < T)].dropna(how="any")
    if est.empty:
        raise ValueError(
            "No estimation data strictly before "
            f"{T.date()} for a {lookback_years:g}-year lookback."
        )
    bad = est.columns[((est <= 0) | (~np.isfinite(est))).any()].tolist()
    if bad:
        raise ValueError(f"Invalid prices for: {', '.join(bad)}.")
    if len(est) < min_rows:
        raise ValueError(
            f"Only {len(est)} estimation observations "
            f"({est.index[0].date()} to {est.index[-1].date()}); "
            f"at least {min_rows} are required."
        )
    info = {
        "est_start": est.index[0].date(),
        "est_end": est.index[-1].date(),
        "n_obs": int(len(est)),
        "test_start": T.date(),
        "lookback_years": float(lookback_years),
    }
    return est.copy(), info

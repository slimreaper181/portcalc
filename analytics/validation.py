"""
validation.py
-------------
Shared input validation and canonical ticker-order alignment helpers.

All analytics modules and the Streamlit UI use these so that:

* ticker ordering is canonical (sorted, upper-cased, de-duplicated) and every
  labelled structure (prices / returns / mu / cov) is explicitly reindexed to
  it before any positional NumPy computation happens;
* invalid optimisation / simulation / VaR inputs are rejected with clear
  ``ValueError`` messages instead of producing NaNs or solver crashes.

Nothing here performs network I/O or renders UI.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Tickers like AAPL, BRK.B, BF-B, ^IRX, BTC-USD are legitimate.
_TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^]{0,11}$")

MIN_HISTORY_ROWS = 30
MIN_HISTORICAL_VAR_WINDOWS = 20

MAX_SIMULATIONS = 50_000
MAX_SIMULATION_YEARS = 30
MAX_VAR_HORIZON_DAYS = 60


# ---------------------------------------------------------------------------
# Ticker normalisation
# ---------------------------------------------------------------------------

def normalise_ticker(raw: str) -> str:
    """Strip whitespace and upper-case a ticker symbol."""
    return str(raw).strip().upper()


def validate_ticker_symbol(raw: str) -> str:
    """Validate a single ticker string; return the normalised form.

    Raises:
        ValueError: if the ticker is empty or contains illegal characters.
    """
    sym = normalise_ticker(raw)
    if not sym:
        raise ValueError("Ticker symbol must not be empty.")
    if not _TICKER_RE.match(sym):
        raise ValueError(
            f"Invalid ticker symbol {raw!r}. Use 1-12 characters from "
            "A-Z, 0-9, '.', '-', '=' or '^' (e.g. AAPL, BRK.B, ^IRX)."
        )
    return sym


def canonical_tickers(raw_tickers: list[str]) -> list[str]:
    """Return the canonical (sorted, unique, normalised) ticker order.

    Raises:
        ValueError: if the list is empty or contains invalid symbols.
    """
    if not raw_tickers:
        raise ValueError("At least one ticker is required.")
    cleaned = [validate_ticker_symbol(t) for t in raw_tickers]
    return sorted(set(cleaned))


def find_duplicate_tickers(raw_tickers: list[str]) -> list[str]:
    """Return normalised tickers that appear more than once (for UI warnings)."""
    seen: set[str] = set()
    dupes: list[str] = []
    for t in raw_tickers:
        sym = normalise_ticker(t)
        if sym in seen and sym not in dupes:
            dupes.append(sym)
        seen.add(sym)
    return dupes


# ---------------------------------------------------------------------------
# Alignment (FIX 3)
# ---------------------------------------------------------------------------

def align_market_data(
    tickers: list[str],
    prices: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    mu: pd.Series | None = None,
    cov: pd.DataFrame | None = None,
) -> dict:
    """Reindex labelled market-data structures to one canonical ticker order.

    Args:
        tickers: Canonical ticker list (order is authoritative).
        prices:  Price DataFrame (columns = tickers, any order).
        returns: Return DataFrame (columns = tickers, any order).
        mu:      Expected-return Series indexed by ticker.
        cov:     Covariance DataFrame indexed/columned by ticker.

    Returns:
        Dict with the subset of ``{"prices", "returns", "mu", "cov"}`` that
        was supplied, each reindexed to ``tickers`` order.

    Raises:
        ValueError: if any structure is missing tickers or has wrong shape.
    """
    if not tickers:
        raise ValueError("Ticker list must not be empty.")
    if len(set(tickers)) != len(tickers):
        raise ValueError(f"Duplicate tickers in canonical order: {tickers}")
    out: dict = {}
    if prices is not None:
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise ValueError(f"Price data is missing tickers: {missing}")
        out["prices"] = prices.reindex(columns=tickers)
    if returns is not None:
        missing = [t for t in tickers if t not in returns.columns]
        if missing:
            raise ValueError(f"Return data is missing tickers: {missing}")
        out["returns"] = returns.reindex(columns=tickers)
    if mu is not None:
        missing = [t for t in tickers if t not in mu.index]
        if missing:
            raise ValueError(f"Expected-return vector is missing tickers: {missing}")
        out["mu"] = mu.reindex(tickers)
    if cov is not None:
        missing_idx = [t for t in tickers if t not in cov.index]
        missing_col = [t for t in tickers if t not in cov.columns]
        if missing_idx or missing_col:
            raise ValueError(
                f"Covariance matrix is missing tickers "
                f"(rows={missing_idx}, cols={missing_col})"
            )
        if cov.shape[0] != cov.shape[1]:
            raise ValueError(f"Covariance matrix must be square, got {cov.shape}")
        out["cov"] = cov.reindex(index=tickers, columns=tickers)
    return out


def validate_weights(weights: np.ndarray, tickers: list[str]) -> np.ndarray:
    """Validate a weight vector against the canonical ticker order.

    Returns the weights as a float ndarray. Raises ``ValueError`` on length
    mismatch, NaNs, or non-finite values.
    """
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(tickers),):
        raise ValueError(
            f"Weight vector length {w.shape} does not match "
            f"{len(tickers)} tickers {tickers}."
        )
    if not np.all(np.isfinite(w)):
        raise ValueError("Portfolio weights must all be finite (no NaN/inf).")
    return w


def validate_expected_return_cov(
    mu: np.ndarray, cov: np.ndarray, tickers: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Validate mu / cov dimensions against the ticker order."""
    mu_arr = np.asarray(mu, dtype=float)
    cov_arr = np.asarray(cov, dtype=float)
    n = len(tickers)
    if mu_arr.shape != (n,):
        raise ValueError(
            f"Expected-return vector shape {mu_arr.shape} does not match "
            f"{n} tickers."
        )
    if cov_arr.shape != (n, n):
        raise ValueError(
            f"Covariance matrix shape {cov_arr.shape} does not match "
            f"{n} tickers (expected {(n, n)})."
        )
    if not np.all(np.isfinite(mu_arr)):
        raise ValueError("Expected returns must all be finite (no NaN/inf).")
    if not np.all(np.isfinite(cov_arr)):
        raise ValueError("Covariance matrix must contain only finite values.")
    return mu_arr, cov_arr


def aligned_portfolio_returns(
    returns: pd.DataFrame, weights: np.ndarray, tickers: list[str]
) -> pd.Series:
    """Compute the daily portfolio log-return series with aligned ordering.

    Uses labelled reindexing so the result is invariant to the column order
    of ``returns``. The weighted sum of asset log returns is the standard
    (first-order) proxy for the portfolio log return.
    """
    w = validate_weights(weights, tickers)
    r = align_market_data(tickers, returns=returns)["returns"]
    return pd.Series(r.values @ w, index=r.index, name="portfolio")


# ---------------------------------------------------------------------------
# Price-history sanitation (FIX 10)
# ---------------------------------------------------------------------------

def sanitize_prices(
    prices: pd.DataFrame,
    tickers: list[str],
    min_rows: int = 2,
) -> pd.DataFrame:
    """Validate and clean a price DataFrame; reindex to canonical order.

    * raises if any ticker column is entirely missing/NaN;
    * drops rows with missing observations (keeps only complete days);
    * raises on empty frames, non-positive prices, or too-short history.

    Returns the cleaned frame with columns in ``tickers`` order.
    """
    if prices is None or prices.empty:
        raise ValueError(
            "No market data was returned. Check ticker symbols and "
            "network connectivity, then try again."
        )
    aligned = align_market_data(tickers, prices=prices)["prices"]
    empty_cols = [t for t in tickers if aligned[t].dropna().empty]
    if empty_cols:
        raise ValueError(
            f"No price history available for: {', '.join(empty_cols)}. "
            "Check the ticker symbols."
        )
    cleaned = aligned.dropna(how="any")
    if cleaned.empty:
        raise ValueError(
            "No overlapping price history across the selected tickers "
            "(their trading calendars do not intersect)."
        )
    if ((cleaned <= 0) | (~np.isfinite(cleaned))).any().any():
        bad = cleaned.columns[
            ((cleaned <= 0) | (~np.isfinite(cleaned))).any()
        ].tolist()
        raise ValueError(
            f"Invalid (zero/negative/NaN) prices detected for: {', '.join(bad)}."
        )
    if len(cleaned) < min_rows:
        raise ValueError(
            f"Only {len(cleaned)} usable price observations; "
            f"at least {min_rows} are required."
        )
    return cleaned


def check_history_length(n_obs: int, horizon_days: int) -> None:
    """Ensure enough observations exist for a rolling T-day historical VaR."""
    n_windows = n_obs - horizon_days + 1
    if n_windows < MIN_HISTORICAL_VAR_WINDOWS:
        raise ValueError(
            f"Insufficient history for {horizon_days}-day historical VaR: "
            f"{n_obs} daily observations give only {max(n_windows, 0)} "
            f"overlapping {horizon_days}-day windows "
            f"(need at least {MIN_HISTORICAL_VAR_WINDOWS}). "
            "Shorten the horizon or load a longer history period."
        )


# ---------------------------------------------------------------------------
# Scalar input validation (FIX 11)
# ---------------------------------------------------------------------------

def validate_shares(shares: float) -> float:
    """Validate a share count (must be finite and strictly positive)."""
    if not np.isfinite(shares) or shares <= 0:
        raise ValueError(f"Shares must be a positive number, got {shares!r}.")
    return float(shares)


def validate_risk_free_rate(rf: float) -> float:
    """Validate an annualised risk-free rate (decimal)."""
    if not np.isfinite(rf) or rf < -0.05 or rf > 0.25:
        raise ValueError(
            f"Risk-free rate {rf!r} looks invalid; expected a decimal "
            "roughly in [-5%, +25%] (e.g. 0.05 for 5%)."
        )
    return float(rf)


def validate_confidence(confidence: float) -> float:
    """Validate a VaR confidence level (exclusive 0..1)."""
    if not np.isfinite(confidence) or not 0.5 <= confidence < 1.0:
        raise ValueError(
            f"Confidence level must be in [0.50, 1.00), got {confidence!r}."
        )
    return float(confidence)


def validate_var_horizon(horizon_days: int) -> int:
    """Validate a VaR horizon in trading days."""
    if not isinstance(horizon_days, (int, np.integer)) or horizon_days < 1:
        raise ValueError(
            f"VaR horizon must be a positive integer number of days, "
            f"got {horizon_days!r}."
        )
    if horizon_days > MAX_VAR_HORIZON_DAYS:
        raise ValueError(
            f"VaR horizon of {horizon_days} days is too long "
            f"(maximum {MAX_VAR_HORIZON_DAYS})."
        )
    return int(horizon_days)


def validate_portfolio_value(value: float) -> float:
    """Validate a portfolio value in currency units (must be > 0)."""
    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"Portfolio value must be positive, got {value!r}."
        )
    return float(value)


def validate_simulation_count(n_sims: int) -> int:
    """Validate a Monte Carlo simulation count."""
    if not isinstance(n_sims, (int, np.integer)) or n_sims < 100:
        raise ValueError(
            f"Simulation count must be an integer >= 100, got {n_sims!r}."
        )
    if n_sims > MAX_SIMULATIONS:
        raise ValueError(
            f"Simulation count {n_sims:,} exceeds the safety cap of "
            f"{MAX_SIMULATIONS:,}. Reduce the count."
        )
    return int(n_sims)


def validate_horizon_years(years: float) -> float:
    """Validate a scenario horizon in years."""
    if not np.isfinite(years) or years <= 0:
        raise ValueError(f"Horizon must be positive years, got {years!r}.")
    if years > MAX_SIMULATION_YEARS:
        raise ValueError(
            f"Horizon of {years} years exceeds the cap of "
            f"{MAX_SIMULATION_YEARS} years."
        )
    return float(years)


def validate_monthly_contrib(monthly: float) -> float:
    """Validate a monthly contribution (must be finite and >= 0)."""
    if not np.isfinite(monthly) or monthly < 0:
        raise ValueError(
            f"Monthly contribution must be >= 0, got {monthly!r}."
        )
    return float(monthly)


# ---------------------------------------------------------------------------
# Optimisation constraint validation (FIX 6)
# ---------------------------------------------------------------------------

def validate_weight_bounds(n: int, min_weight: float, max_weight: float) -> None:
    """Validate per-asset optimisation weight bounds.

    Raises:
        ValueError: with a user-facing message if the constraints are
            impossible or malformed.
    """
    if n < 1:
        raise ValueError("At least one asset is required for optimisation.")
    for name, val in (("min_weight", min_weight), ("max_weight", max_weight)):
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite, got {val!r}.")
    if not 0 <= min_weight <= max_weight <= 1:
        raise ValueError(
            f"Weight bounds must satisfy 0 <= min ({min_weight:.2%}) <= max "
            f"({max_weight:.2%}) <= 100%."
        )
    if max_weight * n < 1 - 1e-12:
        raise ValueError(
            f"Infeasible bounds: with {n} assets capped at {max_weight:.0%} each, "
            f"weights can sum to at most {max_weight * n:.0%} (need 100%). "
            f"Raise the max weight to at least {1 / n:.1%}."
        )
    if min_weight * n > 1 + 1e-12:
        raise ValueError(
            f"Infeasible bounds: with {n} assets floored at {min_weight:.0%} each, "
            f"weights already sum to at least {min_weight * n:.0%} (need 100%). "
            f"Lower the min weight to at most {1 / n:.1%}."
        )

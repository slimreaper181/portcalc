"""
security.py
-----------
Single-holding analytics for the Security Detail view.

All mathematics reuse the project's existing engines:

* annualised return / volatility / Sharpe → ``analytics.performance``
* beta vs benchmark                        → ``analytics.performance.portfolio_beta``
* risk contribution                        → ``analytics.portfolio.risk_contributions``
* log-return conventions                   → ``analytics.returns``

New in this module: security↔portfolio correlation (with restrained,
non-causal interpretation bands) and date-based period returns
(1M/3M/6M/YTD/1Y) computed from Portfolio Calc's own price history.

Nothing here fetches market data or renders UI. Inputs are never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .performance import (
    align_return_series,
    annualised_return,
    annualised_volatility,
    portfolio_beta,
    sharpe_from_log,
)

# (lower bound, label) bands for correlation interpretation. Restrained,
# non-causal wording only.
_CORR_BANDS: tuple[tuple[float, str], ...] = (
    (0.70, "Strong positive co-movement with the portfolio."),
    (0.30, "Moderate positive co-movement with the portfolio."),
    (-0.30, "Low co-movement with the portfolio."),
    (-0.70, "Moderate negative co-movement with the portfolio."),
    (-1.01, "Strong negative co-movement with the portfolio."),
)


def describe_correlation(corr: float) -> str:
    """One-line, non-causal interpretation of a correlation coefficient."""
    if not np.isfinite(corr):
        raise ValueError(f"Correlation must be finite, got {corr!r}.")
    c = float(np.clip(corr, -1.0, 1.0))
    for lower, label in _CORR_BANDS:
        if c >= lower:
            return label
    return _CORR_BANDS[-1][1]  # unreachable, kept for safety


def security_beta(
    security_log_returns: pd.Series, benchmark_log_returns: pd.Series
) -> float:
    """Beta of a single holding vs a benchmark: ``Cov(Rs, Rb) / Var(Rb)``.

    Reuses :func:`analytics.performance.portfolio_beta` (identical
    mathematics) on properly aligned dates. Historical association only.
    """
    return portfolio_beta(security_log_returns, benchmark_log_returns)


def security_portfolio_correlation(
    security_log_returns: pd.Series, portfolio_log_returns: pd.Series
) -> float:
    """Pearson correlation between a holding and the portfolio return series.

    Series are inner-joined on dates with NaNs dropped, so mismatched
    calendars can never shift the result. Raises ``ValueError`` when fewer
    than 2 common observations remain or when either leg has no variation
    (correlation undefined for a constant series).
    """
    s, p = align_return_series(security_log_returns, portfolio_log_returns)
    if float(s.var(ddof=1)) < 1e-12 or float(p.var(ddof=1)) < 1e-12:
        raise ValueError(
            "Correlation is undefined — one return series is constant."
        )
    corr = float(pd.DataFrame({"s": s, "p": p}).corr().iloc[0, 1])
    if not np.isfinite(corr):
        raise ValueError("Could not compute correlation (invalid data).")
    return corr


def security_risk_contribution(
    weights: np.ndarray, cov_matrix: np.ndarray, index: int
) -> float:
    """Fractional risk contribution of one holding (existing model definition).

    ``RC_i = w_i (Σw)_i / σp``, normalised to sum to 1 across holdings —
    identical to :func:`analytics.portfolio.risk_contributions`, indexed to
    a single position. Returns 0.0 when portfolio volatility is zero.
    """
    from .portfolio import risk_contributions

    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    if w.ndim != 1 or cov.shape != (w.shape[0], w.shape[0]):
        raise ValueError("Weights/covariance dimensions are inconsistent.")
    if not 0 <= index < w.shape[0]:
        raise ValueError(f"Holding index {index} out of range.")
    return float(risk_contributions(w, cov)[index])


_PERIOD_MONTHS: dict[str, int | None] = {
    "1M": 1,
    "3M": 3,
    "6M": 6,
    "1Y": 12,
    "YTD": None,
}


def period_return(
    prices: pd.Series, period: str
) -> dict:
    """Simple return over a date-based window from a price series.

    ``period`` is one of ``1M`` / ``3M`` / ``6M`` / ``1Y`` / ``YTD``.
    Windows are date-based (calendar months/year boundary via ``DateOffset``),
    never a fixed observation count: the start is the first trading day on or
    after the calendar cutoff. Return is ``P_end / P_start − 1`` on actual
    prices. Returns ``{"return": float|None, "start": date|None,
    "end": date, "label": str}`` — ``return`` is ``None`` when fewer than 2
    trading days exist in the window.
    """
    if period not in _PERIOD_MONTHS:
        raise ValueError(
            f"Unknown period {period!r}; expected one of {sorted(_PERIOD_MONTHS)}."
        )
    px = pd.Series(prices, dtype=float).dropna()
    if px.empty:
        raise ValueError("No price observations for period return.")
    if not isinstance(px.index, pd.DatetimeIndex):
        raise ValueError("Prices need a DatetimeIndex for period returns.")
    if ((px <= 0) | (~np.isfinite(px))).any():
        raise ValueError("Prices contain zero/negative/NaN values.")

    end = px.index[-1]
    if period == "YTD":
        cutoff = pd.Timestamp(end.year, 1, 1)
    else:
        cutoff = end - pd.DateOffset(months=_PERIOD_MONTHS[period])
    window = px.loc[px.index >= cutoff]
    if len(window) < 2:
        return {"return": None, "start": None, "end": end.date(),
                "label": period}
    start_price = float(window.iloc[0])
    end_price = float(window.iloc[-1])
    return {
        "return": float(end_price / start_price - 1.0),
        "start": window.index[0].date(),
        "end": end.date(),
        "label": period,
    }


def period_returns(prices: pd.Series) -> dict[str, dict]:
    """All snapshot windows (``1M``/``3M``/``6M``/``YTD``/``1Y``) at once."""
    return {label: period_return(prices, label) for label in _PERIOD_MONTHS}


def security_summary(
    security_log_returns: pd.Series,
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    index: int,
    risk_free_annual: float,
    portfolio_log_returns: pd.Series | None = None,
    benchmark_log_returns: pd.Series | None = None,
) -> dict:
    """Bundled per-holding statistics reusing the existing engines.

    Returns dict with ``annualised_return``, ``annualised_volatility``,
    ``sharpe``, ``risk_contribution``, ``correlation_to_portfolio`` (None
    when no portfolio series supplied or correlation is undefined) and
    ``beta_vs_benchmark`` (None when no benchmark supplied or undefined).
    Never raises on undefined beta/correlation — those become ``None`` with
    a ``beta_error`` / ``correlation_error`` message. Raises ``ValueError``
    only for structurally invalid inputs (bad index, empty returns).
    """
    r = pd.Series(security_log_returns, dtype=float).dropna()
    if r.empty:
        raise ValueError("No return observations for security summary.")
    w = np.asarray(weights, dtype=float)
    if not 0 <= index < w.shape[0]:
        raise ValueError(f"Holding index {index} out of range.")

    out: dict = {
        "annualised_return": annualised_return(r),
        "annualised_volatility": annualised_volatility(r) if len(r) >= 2 else 0.0,
        "sharpe": sharpe_from_log(r, risk_free_annual) if len(r) >= 2 else 0.0,
        "risk_contribution": security_risk_contribution(weights, cov_matrix, index),
        "correlation_to_portfolio": None,
        "correlation_note": None,
        "beta_vs_benchmark": None,
        "beta_error": None,
        "correlation_error": None,
    }
    if portfolio_log_returns is not None:
        try:
            c = security_portfolio_correlation(r, portfolio_log_returns)
            out["correlation_to_portfolio"] = c
            out["correlation_note"] = describe_correlation(c)
        except ValueError as e:
            out["correlation_error"] = str(e)
    if benchmark_log_returns is not None:
        try:
            out["beta_vs_benchmark"] = security_beta(r, benchmark_log_returns)
        except ValueError as e:
            out["beta_error"] = str(e)
    return out

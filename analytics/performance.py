"""
performance.py
--------------
Benchmark-relative and downside performance analytics.

Mathematical conventions (project-wide, see analytics.returns)
---------------------------------------------------------------
* Inputs are daily **log** returns: ``r = ln(P_t / P_{t-1})``.
* Cumulative growth of $1 over log returns is ``exp(cumsum(r))`` — never
  ``(1 + r).cumprod()``, which would treat log returns as simple returns.
* A T-day log return is the **sum** of daily log returns; conversion to a
  simple (percentage) return is ``exp(r) - 1``.
* Annualisation uses 252 trading days: means scale by 252, volatilities
  (and tracking error / downside deviation) scale by ``sqrt(252)``.
* The annual risk-free rate converts to a daily rate by simple division
  (``rf / 252``), consistent with ``analytics.var.mean_to_daily``.
* Downside deviation is computed against a minimum acceptable return (MAR)
  of **0.0 daily** by default (documented at each use site); the Sortino
  numerator is the annualised return minus the annual risk-free rate.
* Historical CVaR reuses the genuine rolling T-day log-return methodology
  of ``analytics.var.historical_var`` (no square-root-of-time scaling):
  the tail mean is taken over simple returns ``exp(tail_logs) - 1``.
* Drawdowns are measured on a wealth index (growth of $1 or currency —
  the ratio is unit-free): ``dd = wealth / running_peak - 1``.

All functions are pure: they never mutate their inputs, align series on
common dates via inner joins, drop NaNs deliberately, and raise
``ValueError`` with actionable messages on unusable input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .validation import check_history_length
from .currency import currency_symbol


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

def log_returns_from_prices(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Daily **log** returns from a price Series (or single-column DataFrame)."""
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            raise ValueError(
                "Benchmark prices must be a Series or single-column DataFrame, "
                f"got {prices.shape[1]} columns."
            )
        prices = prices.iloc[:, 0]
    prices = prices.dropna()
    if len(prices) < 2:
        raise ValueError("Not enough benchmark prices to compute returns.")
    if ((prices <= 0) | (~np.isfinite(prices))).any():
        raise ValueError("Benchmark prices contain zero/negative/NaN values.")
    return np.log(prices / prices.shift(1)).dropna().rename("benchmark")


def align_return_series(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Align portfolio and benchmark log-return series on common dates.

    Inner-joins on the index and drops any row with a NaN on either side,
    so missing benchmark dates can never shift calculations. Neither input
    is mutated; aligned copies are returned.

    Raises:
        ValueError: if there is no overlap or fewer than 2 common
            observations remain.
    """
    if portfolio_returns is None or benchmark_returns is None:
        raise ValueError("Portfolio and benchmark return series are both required.")
    joint = pd.DataFrame({
        "portfolio": pd.Series(portfolio_returns, dtype=float),
        "benchmark": pd.Series(benchmark_returns, dtype=float),
    }).dropna(how="any")
    if joint.empty:
        raise ValueError(
            "No overlapping history between portfolio and benchmark "
            "(their date ranges do not intersect)."
        )
    if len(joint) < 2:
        raise ValueError(
            f"Only {len(joint)} overlapping observation(s); "
            "at least 2 are required for benchmark comparison."
        )
    return joint["portfolio"].copy(), joint["benchmark"].copy()


# ---------------------------------------------------------------------------
# Growth / annualised statistics
# ---------------------------------------------------------------------------

def growth_of_capital(log_returns: pd.Series, initial: float = 10_000.0) -> pd.Series:
    """Growth of an initial investment from daily **log** returns.

    ``V_t = initial * exp(cumsum(r))``. Raises on non-positive/NaN initial.
    """
    if not np.isfinite(initial) or initial <= 0:
        raise ValueError(f"Initial capital must be positive, got {initial!r}.")
    r = pd.Series(log_returns, dtype=float).dropna()
    if r.empty:
        raise ValueError("No return observations for growth calculation.")
    return (initial * np.exp(r.cumsum())).rename("growth")


def annualised_return(log_returns: pd.Series, trading_days: int = 252) -> float:
    """Annualised (log-space) mean return: ``mean(r) * 252``."""
    r = pd.Series(log_returns, dtype=float).dropna()
    if r.empty:
        raise ValueError("No return observations for annualised return.")
    return float(r.mean() * trading_days)


def annualised_volatility(log_returns: pd.Series, trading_days: int = 252) -> float:
    """Annualised volatility: ``std(r) * sqrt(252)`` (ddof=1)."""
    r = pd.Series(log_returns, dtype=float).dropna()
    if len(r) < 2:
        raise ValueError("At least 2 observations are required for volatility.")
    return float(r.std(ddof=1) * np.sqrt(trading_days))


def sharpe_from_log(
    log_returns: pd.Series, risk_free_annual: float, trading_days: int = 252
) -> float:
    """Annualised Sharpe from daily log returns: ``(Rp - Rf) / σp``.

    Returns 0.0 when volatility is zero (avoids division by zero).
    """
    rp = annualised_return(log_returns, trading_days)
    vol = annualised_volatility(log_returns, trading_days)
    if vol == 0 or not np.isfinite(vol):
        return 0.0
    return float((rp - risk_free_annual) / vol)


# ---------------------------------------------------------------------------
# Active-return metrics (§3)
# ---------------------------------------------------------------------------

def tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """Annualised tracking error: ``std(Rp - Rb) * sqrt(252)`` on common dates."""
    p, b = align_return_series(portfolio_returns, benchmark_returns)
    active = p - b
    if len(active) < 2:
        raise ValueError("Not enough overlapping data for tracking error.")
    return float(active.std(ddof=1) * np.sqrt(trading_days))


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """Annualised active return / tracking error on common dates.

    Returns 0.0 when tracking error is zero (identical series), avoiding
    division by zero.
    """
    p, b = align_return_series(portfolio_returns, benchmark_returns)
    excess_ann = float((p - b).mean() * trading_days)
    te = tracking_error(p, b, trading_days)
    if te == 0 or not np.isfinite(te):
        return 0.0
    return float(excess_ann / te)


# ---------------------------------------------------------------------------
# Beta / historical alpha (§4–5)
# ---------------------------------------------------------------------------

def portfolio_beta(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    """CAPM-style beta: ``Cov(Rp, Rb) / Var(Rb)`` on common dates.

    This is a historical association measure, not causality. Raises
    ``ValueError`` when benchmark variance is zero/invalid.
    """
    p, b = align_return_series(portfolio_returns, benchmark_returns)
    var_b = float(b.var(ddof=1))
    # NOTE: pandas .var() on economically-constant floats can return ~1e-38
    # instead of exactly 0, so a tolerance is used. Daily-return variance
    # below 1e-12 (std < 1e-6) means an effectively constant benchmark.
    if not np.isfinite(var_b) or var_b < 1e-12:
        raise ValueError(
            "Benchmark variance is zero — beta is undefined "
            "(benchmark returns are constant)."
        )
    cov = float(pd.DataFrame({"p": p, "b": b}).cov().iloc[0, 1])
    if not np.isfinite(cov):
        raise ValueError("Could not compute return covariance (NaNs present).")
    return float(cov / var_b)


def historical_alpha(
    portfolio_ann_return: float,
    benchmark_ann_return: float,
    beta: float,
    risk_free_annual: float,
) -> float:
    """Historical CAPM-style alpha: ``Rp - [Rf + β(Rb - Rf)]`` (annualised).

    Labelled historical and model-dependent: it describes the past under
    CAPM assumptions, not expected future outperformance.
    """
    for name, v in (
        ("portfolio return", portfolio_ann_return),
        ("benchmark return", benchmark_ann_return),
        ("beta", beta),
        ("risk-free rate", risk_free_annual),
    ):
        if not np.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v!r}.")
    return float(
        portfolio_ann_return
        - (risk_free_annual + beta * (benchmark_ann_return - risk_free_annual))
    )


# ---------------------------------------------------------------------------
# Drawdown engine (§6)
# ---------------------------------------------------------------------------

def drawdown_series(wealth: pd.Series) -> pd.Series:
    """Drawdown series for a wealth index: ``wealth / cummax(wealth) - 1``.

    Zero = previous peak; values fall below zero during drawdowns. If the
    series starts at its all-time high the leading drawdown is simply 0.
    """
    w = pd.Series(wealth, dtype=float).dropna()
    if w.empty:
        raise ValueError("No wealth observations for drawdown calculation.")
    if ((w <= 0) | (~np.isfinite(w))).any():
        raise ValueError("Wealth index must contain only positive finite values.")
    running_peak = w.cummax()
    return (w / running_peak - 1).rename("drawdown")


def drawdown_stats(wealth: pd.Series) -> dict:
    """Full drawdown statistics for a wealth index.

    Returns dict with:
      - current_drawdown: drawdown at the last observation (<= 0)
      - max_drawdown: most negative drawdown (<= 0)
      - peak_date / trough_date: labels of max-drawdown peak and trough
      - recovery_date: first label after the trough back at/above peak
        wealth, or ``pd.NaT`` when not yet recovered
      - recovered: bool
      - duration_days: calendar days from peak to recovery (or to the last
        observation when not yet recovered); 0 when max_drawdown == 0
    """
    w = pd.Series(wealth, dtype=float).dropna()
    if w.empty:
        raise ValueError("No wealth observations for drawdown calculation.")
    if ((w <= 0) | (~np.isfinite(w))).any():
        raise ValueError("Wealth index must contain only positive finite values.")
    dd = drawdown_series(w)
    current = float(dd.iloc[-1])
    max_dd = float(dd.min())
    trough_date = dd.idxmin()
    # Peak = highest wealth on or before the trough.
    peak_date = w.loc[:trough_date].idxmax()
    peak_value = float(w.loc[peak_date])
    if max_dd == 0:
        return {
            "current_drawdown": 0.0,
            "max_drawdown": 0.0,
            "peak_date": peak_date,
            "trough_date": peak_date,
            "recovery_date": peak_date,
            "recovered": True,
            "duration_days": 0,
        }
    after = w.loc[trough_date:].iloc[1:]
    recovered_idx = after[after >= peak_value]
    if recovered_idx.empty:
        recovery_date = pd.NaT
        recovered = False
        end_date = w.index[-1]
    else:
        recovery_date = recovered_idx.index[0]
        recovered = True
        end_date = recovery_date
    try:
        duration_days = int((pd.Timestamp(end_date) - pd.Timestamp(peak_date)).days)
    except Exception:
        duration_days = 0
    return {
        "current_drawdown": current,
        "max_drawdown": max_dd,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "recovered": recovered,
        "duration_days": max(duration_days, 0),
    }


def underwater_episodes(wealth: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """Identify the largest non-overlapping historical drawdown episodes.

    Episodes are extracted sequentially (peak → trough → recovery), so they
    are disjoint by construction — overlapping observations from the same
    drawdown are never double-counted. Unrecovered trailing episodes report
    ``Recovery`` as ``NaT`` with duration measured to the last observation.

    Returns a DataFrame with columns
    [Peak, Trough, Recovery, Drawdown %, Duration (days)], sorted by depth.
    """
    w = pd.Series(wealth, dtype=float).dropna()
    if w.empty:
        raise ValueError("No wealth observations for underwater analysis.")
    if ((w <= 0) | (~np.isfinite(w))).any():
        raise ValueError("Wealth index must contain only positive finite values.")
    if len(w) < 2:
        return pd.DataFrame(
            columns=["Peak", "Trough", "Recovery", "Drawdown %", "Duration (days)"]
        )

    episodes: list[dict] = []
    vals = w.values
    idx = w.index
    n = len(w)
    i = 0
    while i < n - 1:
        # Scan forward from anchor i: re-anchor the peak on new highs seen
        # before any decline, then track trough until full recovery.
        peak_pos = i
        peak_val = vals[i]
        trough_pos = -1
        recovered_pos: int | None = None
        j = i + 1
        while j < n:
            if vals[j] >= peak_val:
                if trough_pos != -1:
                    recovered_pos = j
                    break
                peak_pos = j  # higher high before any drawdown: re-anchor
                peak_val = vals[j]
            elif trough_pos == -1 or vals[j] < vals[trough_pos]:
                trough_pos = j
            j += 1
        if trough_pos == -1:
            break  # remainder never declines: no further episodes
        depth = float(vals[trough_pos] / peak_val - 1)
        end_pos = recovered_pos if recovered_pos is not None else n - 1
        try:
            duration = int((pd.Timestamp(idx[end_pos]) - pd.Timestamp(idx[peak_pos])).days)
        except Exception:
            duration = end_pos - peak_pos
        episodes.append({
            "Peak": idx[peak_pos],
            "Trough": idx[trough_pos],
            "Recovery": idx[recovered_pos] if recovered_pos is not None else pd.NaT,
            "Drawdown %": depth,
            "Duration (days)": max(duration, 0),
        })
        if recovered_pos is None:
            break  # trailing unrecovered episode consumes the remainder
        i = recovered_pos

    episodes = sorted(episodes, key=lambda e: e["Drawdown %"])[: max(int(top_n), 0)]
    rows = [
        {k: e[k] for k in ["Peak", "Trough", "Recovery", "Drawdown %", "Duration (days)"]}
        for e in episodes
    ]
    return pd.DataFrame(rows, columns=["Peak", "Trough", "Recovery", "Drawdown %", "Duration (days)"])


# ---------------------------------------------------------------------------
# Downside deviation / Sortino / Calmar (§8–10)
# ---------------------------------------------------------------------------

def downside_deviation(
    log_returns: pd.Series,
    target: float = 0.0,
    trading_days: int = 252,
) -> float:
    """Annualised downside deviation vs a minimum acceptable return (MAR).

    ``DD = sqrt(mean(min(0, r - target)^2)) * sqrt(252)``.

    Convention: ``target`` defaults to **0.0 daily** — only negative daily
    log returns contribute. Pass an explicit target (e.g. ``rf / 252``) to
    use a different MAR.
    """
    r = pd.Series(log_returns, dtype=float).dropna()
    if r.empty:
        raise ValueError("No return observations for downside deviation.")
    if not np.isfinite(target):
        raise ValueError(f"Downside target must be finite, got {target!r}.")
    shortfalls = np.minimum(0.0, r.values - target)
    return float(np.sqrt(np.mean(shortfalls ** 2)) * np.sqrt(trading_days))


def sortino_ratio(
    annualised_return_value: float,
    risk_free_annual: float,
    downside_deviation_ann: float,
) -> float:
    """Sortino = (Rp_ann − Rf) / annualised downside deviation.

    Returns 0.0 when downside deviation is zero/non-finite (avoids division
    by zero); documented at the call site.
    """
    if not np.isfinite(annualised_return_value) or not np.isfinite(risk_free_annual):
        raise ValueError("Return and risk-free rate must be finite for Sortino.")
    if downside_deviation_ann == 0 or not np.isfinite(downside_deviation_ann):
        return 0.0
    return float((annualised_return_value - risk_free_annual) / downside_deviation_ann)


def calmar_ratio(annualised_return_value: float, max_drawdown: float) -> float:
    """Calmar = annualised return / |maximum drawdown|.

    Returns 0.0 when maximum drawdown is zero/non-finite (a flat series has
    no drawdown to scale by); documented at the call site.
    """
    if not np.isfinite(annualised_return_value):
        raise ValueError("Annualised return must be finite for Calmar.")
    if max_drawdown == 0 or not np.isfinite(max_drawdown):
        return 0.0
    return float(annualised_return_value / abs(max_drawdown))


# ---------------------------------------------------------------------------
# Historical Expected Shortfall / CVaR (§11)
# ---------------------------------------------------------------------------

def historical_cvar(
    portfolio_log_returns: pd.Series,
    portfolio_value: float,
    confidence: float = 0.95,
    horizon_days: int = 1,
    min_windows: int = 20,
) -> dict:
    """Historical Expected Shortfall (CVaR) from genuine rolling T-day returns.

    Method: build overlapping T-day **log** returns (rolling sums, same as
    ``historical_var`` — no square-root-of-time scaling), take the VaR cutoff
    (the ``(1 − confidence)`` percentile), select realised returns at or
    worse than the cutoff, and average their **simple** losses
    (``exp(r) − 1``).

    Returns dict with:
      - cvar_pct: average tail loss as a negative simple return (e.g. −0.071)
      - cvar_currency: positive currency loss (e.g. £7,120 on £100k)
      - var_pct / var_currency: the corresponding Historical VaR cutoff
      - n_windows / n_tail: overlapping windows used and tail count

    Raises:
        ValueError: on invalid inputs or insufficient history.
    """
    from .validation import (
        validate_confidence,
        validate_portfolio_value,
        validate_var_horizon,
    )
    import analytics.validation as _v

    confidence = validate_confidence(confidence)
    horizon_days = validate_var_horizon(horizon_days)
    portfolio_value = validate_portfolio_value(portfolio_value)

    if portfolio_log_returns is None or len(portfolio_log_returns) == 0:
        raise ValueError("No historical returns available for Historical CVaR.")
    rets = pd.Series(np.asarray(portfolio_log_returns, dtype=float)).dropna()
    if len(rets) == 0 or not np.all(np.isfinite(rets)):
        raise ValueError("Historical returns contain no usable observations.")

    check_history_length(len(rets), horizon_days)
    required = max(int(min_windows), _v.MIN_HISTORICAL_VAR_WINDOWS)
    n_windows = len(rets) - horizon_days + 1
    if n_windows < required:
        raise ValueError(
            f"Insufficient history for {horizon_days}-day historical CVaR: "
            f"{len(rets)} daily observations give only {n_windows} overlapping "
            f"{horizon_days}-day windows (need at least {required}). "
            "Shorten the horizon or load a longer history period."
        )

    if horizon_days == 1:
        t_day_log = rets
    else:
        t_day_log = rets.rolling(window=horizon_days).sum().dropna()

    percentile = (1 - confidence) * 100
    cutoff_log = float(np.percentile(t_day_log, percentile))
    tail_logs = t_day_log[t_day_log <= cutoff_log]
    if tail_logs.empty:
        # Degenerate (e.g. constant series): fall back to the worst observation.
        tail_logs = t_day_log.nsmallest(1)
    tail_simple = np.exp(tail_logs.values) - 1
    cvar_pct = float(tail_simple.mean())
    var_pct = float(np.exp(cutoff_log) - 1)
    return {
        "cvar_pct": cvar_pct,
        "cvar_currency": float(-cvar_pct * portfolio_value),
        "var_pct": var_pct,
        "var_currency": float(-var_pct * portfolio_value),
        "n_windows": int(len(t_day_log)),
        "n_tail": int(len(tail_logs)),
    }


# ---------------------------------------------------------------------------
# Rolling analytics (§13–15)
# ---------------------------------------------------------------------------

def _require_window(log_returns: pd.Series, window: int) -> pd.Series:
    r = pd.Series(log_returns, dtype=float).dropna()
    window = int(window)
    if window < 2:
        raise ValueError(f"Rolling window must be >= 2 days, got {window!r}.")
    if len(r) < window:
        raise ValueError(
            f"Only {len(r)} observations for a {window}-day rolling window — "
            "need at least as many observations as the window."
        )
    return r


def rolling_annualised_return(
    log_returns: pd.Series, window: int, trading_days: int = 252
) -> pd.Series:
    """Rolling annualised return from log returns.

    ``exp(rolling_sum(r, w) * 252 / w) − 1`` with ``min_periods=window``, so
    no annualised value is produced from a partial window.
    """
    r = _require_window(log_returns, window)
    roll_sum = r.rolling(window=window, min_periods=window).sum()
    return (np.exp(roll_sum * trading_days / window) - 1).rename(f"roll_ret_{window}d")


def rolling_volatility(
    log_returns: pd.Series, window: int, trading_days: int = 252
) -> pd.Series:
    """Rolling annualised volatility: ``rolling_std(w) * sqrt(252)``."""
    r = _require_window(log_returns, window)
    return (r.rolling(window=window, min_periods=window).std(ddof=1)
            * np.sqrt(trading_days)).rename(f"roll_vol_{window}d")


def rolling_sharpe(
    log_returns: pd.Series,
    risk_free_annual: float,
    window: int,
    trading_days: int = 252,
) -> pd.Series:
    """Rolling annualised Sharpe: ``(mean(r)*252 − Rf) / (std(r)*sqrt(252))``.

    The daily risk-free rate is ``Rf / 252`` (consistent with
    ``mean_to_daily``). Windows with zero volatility yield 0.0, never
    NaN/inf from division by zero.
    """
    r = _require_window(log_returns, window)
    if not np.isfinite(risk_free_annual):
        raise ValueError("Risk-free rate must be finite for rolling Sharpe.")
    roll_mean = r.rolling(window=window, min_periods=window).mean() * trading_days
    roll_vol = r.rolling(window=window, min_periods=window).std(ddof=1) * np.sqrt(trading_days)
    excess = roll_mean - risk_free_annual
    out = pd.Series(np.where(roll_vol > 0, excess / roll_vol, 0.0), index=r.index)
    out[roll_vol.isna()] = np.nan  # keep leading partial windows as NaN
    return out.rename(f"roll_sharpe_{window}d")


# ---------------------------------------------------------------------------
# Monthly aggregation / best-worst periods (§18)
# ---------------------------------------------------------------------------

def monthly_returns(log_returns: pd.Series) -> pd.Series:
    """Aggregate daily **log** returns to monthly **simple** returns.

    Monthly log return = sum of daily logs within the month (not an
    average); converted with ``exp(sum) − 1``. Indexed by month-end date.
    """
    r = pd.Series(log_returns, dtype=float).dropna()
    if r.empty:
        raise ValueError("No return observations for monthly aggregation.")
    if not isinstance(r.index, pd.DatetimeIndex):
        raise ValueError("Returns need a DatetimeIndex for monthly aggregation.")
    monthly_log = r.resample("ME").sum()
    out = (np.exp(monthly_log) - 1).rename("monthly_return")
    out.index = out.index.to_period("M").to_timestamp("M")
    return out


def best_worst_periods(log_returns: pd.Series) -> dict:
    """Best/worst day and best/worst month (simple returns with dates).

    Returns dict with keys ``best_day``, ``worst_day`` (each a
    ``(date, simple_return)`` tuple) and ``best_month``, ``worst_month``
    (each a ``(YYYY-MM string, simple_return)`` tuple).
    """
    r = pd.Series(log_returns, dtype=float).dropna()
    if r.empty:
        raise ValueError("No return observations for best/worst periods.")
    if not isinstance(r.index, pd.DatetimeIndex):
        raise ValueError("Returns need a DatetimeIndex for best/worst periods.")
    daily_simple = np.exp(r) - 1
    best_d = daily_simple.idxmax()
    worst_d = daily_simple.idxmin()
    m = monthly_returns(r)
    best_m = m.idxmax()
    worst_m = m.idxmin()
    return {
        "best_day": (best_d, float(daily_simple.loc[best_d])),
        "worst_day": (worst_d, float(daily_simple.loc[worst_d])),
        "best_month": (best_m.strftime("%Y-%m"), float(m.loc[best_m])),
        "worst_month": (worst_m.strftime("%Y-%m"), float(m.loc[worst_m])),
    }


# ---------------------------------------------------------------------------
# One-shot summary
# ---------------------------------------------------------------------------

def summarise_performance(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    risk_free_annual: float,
    trading_days: int = 252,
) -> dict:
    """Assemble portfolio, benchmark and relative performance metrics.

    Returns dict with keys ``portfolio`` (ann_return, ann_vol, sharpe,
    sortino, calmar, max_drawdown, downside_deviation), ``benchmark`` (same
    shape, or None) and ``relative`` (beta, historical_alpha,
    tracking_error, information_ratio — each None when no benchmark).
    Never mutates inputs.
    """
    p = pd.Series(portfolio_returns, dtype=float).dropna()
    if p.empty:
        raise ValueError("No portfolio returns for performance summary.")

    def _block(rets: pd.Series) -> dict:
        ann = annualised_return(rets, trading_days)
        vol = annualised_volatility(rets, trading_days) if len(rets) >= 2 else 0.0
        dd = downside_deviation(rets, trading_days=trading_days)
        wealth = growth_of_capital(rets, initial=1.0)
        mdd = drawdown_stats(wealth)["max_drawdown"]
        return {
            "ann_return": ann,
            "ann_vol": vol,
            "sharpe": sharpe_from_log(rets, risk_free_annual, trading_days),
            "sortino": sortino_ratio(ann, risk_free_annual, dd),
            "calmar": calmar_ratio(ann, mdd),
            "max_drawdown": mdd,
            "downside_deviation": dd,
        }

    out: dict = {"portfolio": _block(p), "benchmark": None, "relative": {
        "beta": None, "historical_alpha": None,
        "tracking_error": None, "information_ratio": None,
    }}
    if benchmark_returns is not None:
        b = pd.Series(benchmark_returns, dtype=float).dropna()
        if not b.empty:
            p_al, b_al = align_return_series(p, b)
            out["benchmark"] = _block(b_al)
            beta = portfolio_beta(p_al, b_al)
            out["relative"] = {
                "beta": beta,
                "historical_alpha": historical_alpha(
                    annualised_return(p_al, trading_days),
                    annualised_return(b_al, trading_days),
                    beta, risk_free_annual,
                ),
                "tracking_error": tracking_error(p_al, b_al, trading_days),
                "information_ratio": information_ratio(p_al, b_al, trading_days),
            }
    return out


# ---------------------------------------------------------------------------
# Plotly visualisations (dark terminal theme)
# ---------------------------------------------------------------------------

_DARK_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Mono, monospace", color="#c9d1d9"),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    margin=dict(l=60, r=30, t=60, b=60),
)


def plot_growth_comparison(
    portfolio_growth: pd.Series,
    benchmark_growth: pd.Series | None = None,
    initial: float = 10_000.0,
    benchmark_label: str = "Benchmark",
    currency: str = "USD",
) -> go.Figure:
    """Growth-of-capital chart: portfolio (and benchmark) in currency units."""
    sym = currency_symbol(currency)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_growth.index, y=portfolio_growth.values,
        mode="lines", name="Portfolio",
        line=dict(color="#58a6ff", width=2.5),
    ))
    if benchmark_growth is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_growth.index, y=benchmark_growth.values,
            mode="lines", name=benchmark_label,
            line=dict(color="#f0883e", width=2),
        ))
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text=f"Growth of {sym}{initial:,.0f} - Portfolio vs {benchmark_label}",
                   font=dict(size=16)),
        xaxis_title="Date", yaxis_title=f"Portfolio Value ({currency})",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def plot_drawdown(
    portfolio_dd: pd.Series,
    benchmark_dd: pd.Series | None = None,
    benchmark_label: str = "Benchmark",
    max_dd: float | None = None,
) -> go.Figure:
    """Drawdown chart (0 = previous peak, values fall below zero)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_dd.index, y=portfolio_dd.values,
        mode="lines", name="Portfolio",
        fill="tozeroy", fillcolor="rgba(248, 81, 73, 0.15)",
        line=dict(color="#f85149", width=1.8),
    ))
    if benchmark_dd is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_dd.index, y=benchmark_dd.values,
            mode="lines", name=benchmark_label,
            line=dict(color="#8b949e", width=1.2, dash="dot"),
        ))
    if max_dd is not None and max_dd < 0:
        trough_date = portfolio_dd.idxmin()
        fig.add_annotation(
            x=trough_date, y=max_dd, text=f"Max DD {max_dd:.1%}",
            showarrow=True, arrowcolor="#f85149",
            font=dict(color="#f85149"),
        )
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text="Drawdown (0 = previous peak)", font=dict(size=16)),
        xaxis_title="Date", yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig

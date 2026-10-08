"""
backtest.py
-----------
Deterministic historical portfolio backtesting engine.

What it answers
---------------
How would this portfolio have performed historically under fixed rules:
true buy-and-hold, periodic rebalancing to static targets, equal weight,
and a passive benchmark — with transaction costs accounted for.

Historical price convention
---------------------------
Inputs are **adjusted closing prices** (yfinance ``auto_adjust=True``, as
used by ``data.market_data.fetch_price_history``). Adjusted closes reflect
stock splits and dividend adjustments to the extent Yahoo's adjusted data
provides them; this is *not* claimed to be exact cash-dividend
reinvestment. No Alpaca/live prices enter this module — it accepts only
explicit historical price DataFrames (see the isolation test).

Core conventions (project-wide, see analytics.returns)
-------------------------------------------------------
* Portfolio return series stored on results are daily **log** returns
  derived from the simulated wealth path (never constant-weight
  analytical returns for buy-and-hold, whose weights drift).
* Annualisation uses 252 trading days; CAGR uses actual calendar elapsed
  time and is kept separate from annualised mean return.
* Rebalance convention: on the first available trading day **on or after**
  each scheduled calendar date (month/quarter/half-year/year start),
  executed at that day's close using only prices known then.
* Transaction cost: ``|trade_notional| * bps / 10_000`` per traded leg,
  summed per event; untouched holdings incur no cost. Costs are deducted
  pro-rata so ``value_before - costs == value_after`` exactly.
* Turnover per event: ``sum(|trade|) / value_before``.
* Fractional shares are assumed (deterministic, no cash drag); any
  residual is fully deployed, so no separate cash balance exists.

All functions are pure: inputs are never mutated, date alignment is by
common trading dates, and invalid input raises ``ValueError``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .performance import (
    annualised_return,
    annualised_volatility,
    best_worst_periods,
    calmar_ratio,
    downside_deviation,
    drawdown_series,
    drawdown_stats,
    sharpe_from_log,
    sortino_ratio,
)
from .validation import align_market_data, canonical_tickers

TRADING_DAYS = 252
MIN_BACKTEST_ROWS = 30
MAX_COST_BPS = 1000.0

REBALANCE_FREQUENCIES = ("never", "monthly", "quarterly", "semi-annual", "annual")

# Calendar anchors for each periodic frequency (period *starts*).
_FREQUENCY_RULE = {
    "monthly": "MS",
    "quarterly": "QS",
    "semi-annual": "6MS",
    "annual": "YS",
}

_DARK_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Mono, monospace", color="#c9d1d9"),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    margin=dict(l=60, r=30, t=60, b=60),
)


@dataclass
class BacktestResult:
    """Structured outcome of one backtested strategy."""

    name: str
    values: pd.Series              # daily portfolio wealth (currency)
    log_returns: pd.Series         # daily log returns derived from values
    drawdowns: pd.Series           # drawdown series of values
    weights: pd.DataFrame | None   # daily holdings weights (drift), None for benchmark
    turnover: pd.Series            # per-event turnover, 0.0 elsewhere (daily index)
    costs: pd.Series               # deducted transaction costs (daily index)
    rebalance_events: pd.DataFrame  # one row per rebalance (see §17)
    trades: pd.DataFrame | None    # per-leg trade detail, None for benchmark
    metrics: dict                  # summary statistics (see backtest_summary)
    is_benchmark: bool = False
    # Audit trail (static inputs, never future-derived):
    target_weights: np.ndarray | None = None
    frequency: str = "never"
    cost_bps: float = 0.0
    initial_capital: float = 0.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_backtest_window(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate a backtest date range (start strictly before end)."""
    try:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid backtest dates: {e}")
    if not (np.isfinite(start_ts.value) and np.isfinite(end_ts.value)):
        raise ValueError("Backtest dates must be valid calendar dates.")
    if start_ts >= end_ts:
        raise ValueError(
            f"Backtest start ({start_ts.date()}) must be before end ({end_ts.date()})."
        )
    return start_ts, end_ts


def validate_backtest_capital(initial: float) -> float:
    """Validate initial capital (finite, strictly positive)."""
    if not np.isfinite(initial) or initial <= 0:
        raise ValueError(f"Initial capital must be positive, got {initial!r}.")
    return float(initial)


def validate_transaction_cost_bps(cost_bps: float) -> float:
    """Validate a transaction cost in basis points (0–1000)."""
    if not np.isfinite(cost_bps) or cost_bps < 0 or cost_bps > MAX_COST_BPS:
        raise ValueError(
            f"Transaction cost must be between 0 and {MAX_COST_BPS:.0f} bps, "
            f"got {cost_bps!r}."
        )
    return float(cost_bps)


def validate_target_weights(weights: np.ndarray, tickers: list[str]) -> np.ndarray:
    """Validate a static long-only target weight vector (sums to 1)."""
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(tickers),):
        raise ValueError(
            f"Target weights length {w.shape} does not match "
            f"{len(tickers)} tickers."
        )
    if not np.all(np.isfinite(w)):
        raise ValueError("Target weights must all be finite (no NaN/inf).")
    if np.any(w < -1e-9) or np.any(w > 1 + 1e-9):
        raise ValueError(
            "Target weights must be long-only (each between 0 and 1); "
            "short positions are not supported."
        )
    if abs(float(w.sum()) - 1.0) > 1e-6:
        raise ValueError(
            f"Target weights must sum to 1 (got {float(w.sum()):.6f})."
        )
    return w


def validate_frequency(frequency: str) -> str:
    """Validate a rebalancing frequency label."""
    freq = str(frequency).strip().lower()
    if freq not in REBALANCE_FREQUENCIES:
        raise ValueError(
            f"Unknown rebalancing frequency {frequency!r}; expected one of "
            f"{list(REBALANCE_FREQUENCIES)}."
        )
    return freq


def detect_mixed_markets(tickers: list[str]) -> str | None:
    """Warn when holdings span different market suffixes (no FX system).

    Groups tickers by Yahoo suffix (``""``, ``"L"``, …). More than one group
    means prices may be in different currencies, which this engine does not
    normalise. Returns a warning string, or ``None`` when uniform.
    """
    groups: dict[str, list[str]] = {}
    for t in tickers:
        suffix = t.rsplit(".", 1)[1] if "." in t else ""
        groups.setdefault(suffix, []).append(t)
    if len(groups) > 1:
        detail = "; ".join(f"{s or 'no-suffix'}: {', '.join(ts)}"
                           for s, ts in sorted(groups.items()))
        return (
            "Holdings span multiple market suffixes "
            f"({detail}). This engine has no FX normalisation — results "
            "are not currency-adjusted."
        )
    return None


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_backtest_data(
    prices: pd.DataFrame,
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_rows: int = MIN_BACKTEST_ROWS,
) -> tuple[pd.DataFrame, list[str]]:
    """Slice and align historical prices to common valid dates.

    * reindexes columns to canonical ticker order (order-independent);
    * restricts to ``[start, end]``;
    * keeps only rows where **every** asset has a valid price (common dates);
    * rejects non-positive / non-finite prices.

    Returns ``(window, notes)`` where ``notes`` lists non-fatal coverage
    caveats (e.g. an asset whose history starts after the requested start —
    the engine uses only real data, never invented history).

    Raises:
        ValueError: on empty overlap, too-short history, or bad prices.
    """
    start_ts, end_ts = validate_backtest_window(start, end)
    tickers = canonical_tickers(list(tickers))
    if prices is None or prices.empty:
        raise ValueError("No price history supplied for backtesting.")
    aligned = align_market_data(tickers, prices=prices)["prices"]
    if not isinstance(aligned.index, pd.DatetimeIndex):
        try:
            aligned.index = pd.DatetimeIndex(aligned.index)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Price history needs a DatetimeIndex: {e}")
    aligned = aligned.sort_index()

    notes: list[str] = []
    for t in tickers:
        first_valid = aligned[t].dropna()
        if first_valid.empty:
            raise ValueError(f"No price history available for {t}.")
        first_date = first_valid.index[0].date()
        if first_date > start_ts.date():
            notes.append(
                f"{t} history starts {first_date} (after requested start "
                f"{start_ts.date()}); backtest uses only its real data."
            )

    window = aligned.loc[start_ts:end_ts].dropna(how="any")
    if window.empty:
        raise ValueError(
            "No overlapping price history in the selected date range. "
            "Widen the range or drop recently-listed assets."
        )
    if ((window <= 0) | (~np.isfinite(window))).any().any():
        bad = window.columns[
            ((window <= 0) | (~np.isfinite(window))).any()
        ].tolist()
        raise ValueError(
            f"Invalid (zero/negative/NaN) prices for: {', '.join(bad)}."
        )
    if len(window) < 2:
        raise ValueError("Need at least 2 common trading days to backtest.")
    if len(window) < min_rows:
        raise ValueError(
            f"Only {len(window)} common trading days in range "
            f"({window.index[0].date()} to {window.index[-1].date()}); "
            f"at least {min_rows} are required for a meaningful backtest."
        )
    # Return a copy: downstream must never mutate caller data.
    return window.copy(), notes


# ---------------------------------------------------------------------------
# Rebalance scheduling
# ---------------------------------------------------------------------------

def generate_rebalance_dates(
    index: pd.DatetimeIndex, frequency: str
) -> pd.DatetimeIndex:
    """Scheduled rebalance dates for a trading-day index.

    Convention: the first available trading day **on or after** each
    calendar period start (month/quarter/half-year/year). The series start
    itself is the initial allocation, never a rebalance event. Non-trading
    calendar dates (weekends/holidays) roll forward to the next trading day;
    duplicate hits collapse to one event.

    Raises:
        ValueError: on unknown frequency or a non-datetime index.
    """
    frequency = validate_frequency(frequency)
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Rebalance scheduling needs a DatetimeIndex.")
    if frequency == "never" or len(index) < 2:
        return pd.DatetimeIndex([])
    rule = _FREQUENCY_RULE[frequency]
    scheduled = pd.date_range(start=index[0], end=index[-1], freq=rule)
    positions = np.unique(index.searchsorted(scheduled))
    # Drop position 0 (initial allocation, not a rebalance) and guard the
    # trailing edge (a schedule mark past the last bar is not tradeable).
    positions = positions[(positions > 0) & (positions < len(index))]
    return index[positions]


# ---------------------------------------------------------------------------
# Costs & turnover primitives
# ---------------------------------------------------------------------------

def apply_transaction_costs(trade_values: np.ndarray, cost_bps: float) -> float:
    """Total transaction cost for one rebalance event.

    ``sum(|trade_i|) * bps / 10_000`` — only traded notional incurs cost.
    """
    cost_bps = validate_transaction_cost_bps(cost_bps)
    trades = np.asarray(trade_values, dtype=float)
    if not np.all(np.isfinite(trades)):
        raise ValueError("Trade values must be finite.")
    return float(np.abs(trades).sum() * cost_bps / 10_000.0)


def calculate_turnover(trade_values: np.ndarray, portfolio_value_before: float) -> float:
    """Event turnover: ``sum(|trade|) / value_before`` (fraction, >= 0)."""
    if not np.isfinite(portfolio_value_before) or portfolio_value_before <= 0:
        raise ValueError(
            f"Portfolio value before rebalance must be positive, "
            f"got {portfolio_value_before!r}."
        )
    trades = np.asarray(trade_values, dtype=float)
    if not np.all(np.isfinite(trades)):
        raise ValueError("Trade values must be finite.")
    return float(np.abs(trades).sum() / portfolio_value_before)


# ---------------------------------------------------------------------------
# Core share-based simulation engine
# ---------------------------------------------------------------------------

def _run_share_simulation(
    prices: pd.DataFrame,
    target_weights: np.ndarray,
    initial: float,
    rebalance_dates: pd.DatetimeIndex,
    cost_bps: float,
    name: str,
) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Simulate holdings in share space; see module docstring for accounting.

    Returns ``(values, weights_df, turnover_daily, costs_daily, events, trades)``.
    ``values[d]`` at a rebalance date is the post-cost value, so the series
    is continuous and ``value_before - cost == value_after`` holds exactly.
    """
    tickers = list(prices.columns)
    w = validate_target_weights(target_weights, tickers)
    initial = validate_backtest_capital(initial)
    cost_bps = validate_transaction_cost_bps(cost_bps)
    P = prices.copy()
    dates = P.index
    n = len(dates)

    reb_set = set(pd.DatetimeIndex(rebalance_dates))
    # Segment boundaries: t0, each rebalance date, final date.
    cuts = [0] + sorted(dates.get_loc(d) for d in reb_set if d in dates) + [n - 1]
    cuts = sorted(set(cuts))

    shares = initial * w / P.iloc[0].values
    value_parts: list[pd.Series] = []
    weight_parts: list[pd.DataFrame] = []
    turnover_daily = pd.Series(0.0, index=dates)
    costs_daily = pd.Series(0.0, index=dates)
    events: list[dict] = []
    trade_rows: list[dict] = []

    for seg_i in range(len(cuts) - 1):
        lo, hi = cuts[seg_i], cuts[seg_i + 1]
        seg = P.iloc[lo:hi + 1]
        holdings = seg * shares
        seg_values = holdings.sum(axis=1)
        d = dates[hi]
        is_rebalance = d in reb_set and hi < n - 1
        if is_rebalance:
            value_before = float(seg_values.iloc[-1])
            current = holdings.iloc[-1].values
            target = value_before * w
            trade = target - current
            cost = apply_transaction_costs(trade, cost_bps)
            turnover = calculate_turnover(trade, value_before)
            value_after = value_before - cost
            seg_values.iloc[-1] = value_after
            turnover_daily.loc[d] = turnover
            costs_daily.loc[d] = cost
            events.append({
                "Date": d,
                "Portfolio Value Before": value_before,
                "Turnover": turnover,
                "Transaction Cost": cost,
                "Portfolio Value After": value_after,
            })
            before_w = current / value_before
            leg_costs = np.abs(trade) * cost_bps / 10_000.0
            for j, t in enumerate(tickers):
                trade_rows.append({
                    "Date": d,
                    "Ticker": t,
                    "Before Weight": float(before_w[j]),
                    "Target Weight": float(w[j]),
                    "Trade Value": float(trade[j]),
                    "Transaction Cost": float(leg_costs[j]),
                })
            # Restore exact targets pro-rata so accounting reconciles.
            scale = value_after / value_before if value_before > 0 else 0.0
            shares = (target * scale) / P.iloc[hi].values
        value_parts.append(seg_values)
        # Drift weights for every simulated day. At a rebalance date the
        # stored row is the post-rebalance allocation (exactly the targets);
        # otherwise weights drift naturally with prices.
        w_frame = holdings.div(seg_values.replace(0, np.nan), axis=0).fillna(0.0)
        if is_rebalance:
            w_frame.iloc[-1] = w
        weight_parts.append(w_frame)

    values = pd.concat(value_parts).rename("value")
    weights_df = pd.concat(weight_parts)
    # Guard against any duplicated boundary from adjacent cuts (none by
    # construction, but keep the series strictly unique).
    values = values[~values.index.duplicated(keep="first")]
    weights_df = weights_df[~weights_df.index.duplicated(keep="first")]

    events_df = pd.DataFrame(
        events,
        columns=["Date", "Portfolio Value Before", "Turnover",
                 "Transaction Cost", "Portfolio Value After"],
    )
    trades_df = pd.DataFrame(
        trade_rows,
        columns=["Date", "Ticker", "Before Weight", "Target Weight",
                 "Trade Value", "Transaction Cost"],
    )
    return values, weights_df, turnover_daily, costs_daily, events_df, trades_df


def _finalise_result(
    name: str,
    values: pd.Series,
    weights: pd.DataFrame | None,
    turnover_daily: pd.Series,
    costs_daily: pd.Series,
    events: pd.DataFrame,
    trades: pd.DataFrame | None,
    initial: float,
    risk_free_annual: float,
    target_weights: np.ndarray | None,
    frequency: str,
    cost_bps: float,
    is_benchmark: bool = False,
) -> BacktestResult:
    """Assemble a BacktestResult with derived series and metrics."""
    values = pd.Series(values, dtype=float).dropna()
    if len(values) < 2:
        raise ValueError(f"Backtest '{name}' produced too little data.")
    log_rets = np.log(values / values.shift(1)).dropna().rename("strategy")
    metrics = backtest_summary(
        values, initial, risk_free_annual,
        float(costs_daily.sum()),
        int(len(events)),
        turnover_daily[turnover_daily > 0],
    )
    return BacktestResult(
        name=name,
        values=values.rename("value"),
        log_returns=log_rets,
        drawdowns=drawdown_series(values),
        weights=weights,
        turnover=turnover_daily.rename("turnover"),
        costs=costs_daily.rename("costs"),
        rebalance_events=events,
        trades=trades,
        metrics=metrics,
        is_benchmark=is_benchmark,
        target_weights=None if target_weights is None else np.asarray(
            target_weights, dtype=float),
        frequency=frequency,
        cost_bps=cost_bps,
        initial_capital=initial,
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def backtest_buy_and_hold(
    prices: pd.DataFrame,
    target_weights: np.ndarray,
    initial: float,
    risk_free_annual: float = 0.05,
    name: str = "Buy & Hold",
) -> BacktestResult:
    """True buy-and-hold: shares fixed at inception, weights drift naturally.

    ``shares_i = initial * w_i / P_i(t0)``; ``V(t) = sum(shares_i * P_i(t))``.
    This is *not* constant daily weights (that would be daily rebalancing).
    Fractional shares are assumed. Costs are zero (no trades after t0).
    """
    tickers = list(prices.columns)
    w = validate_target_weights(target_weights, tickers)
    initial = validate_backtest_capital(initial)
    dates = prices.index
    shares = initial * w / prices.iloc[0].values
    holdings = prices * shares
    values = holdings.sum(axis=1).rename("value")
    weights_df = holdings.div(values.replace(0, np.nan), axis=0).fillna(0.0)
    zeros = pd.Series(0.0, index=dates)
    events = pd.DataFrame(columns=["Date", "Portfolio Value Before", "Turnover",
                                   "Transaction Cost", "Portfolio Value After"])
    trades = pd.DataFrame(columns=["Date", "Ticker", "Before Weight", "Target Weight",
                                    "Trade Value", "Transaction Cost"])
    return _finalise_result(
        name, values, weights_df, zeros.rename("turnover"), zeros.rename("costs"),
        events, trades, initial, risk_free_annual, w, "never", 0.0,
    )


def backtest_rebalanced(
    prices: pd.DataFrame,
    target_weights: np.ndarray,
    initial: float,
    frequency: str = "quarterly",
    cost_bps: float = 0.0,
    risk_free_annual: float = 0.05,
    name: str | None = None,
) -> BacktestResult:
    """Periodic rebalancing to a static target vector (share-space engine).

    At each scheduled date (first trading day on/after the calendar mark):
    value the holdings, trade back to ``target_weights``, deduct
    ``|trade| * bps / 10_000`` pro-rata. ``frequency="never"`` behaves
    exactly like :func:`backtest_buy_and_hold` with zero costs.
    """
    frequency = validate_frequency(frequency)
    tickers = list(prices.columns)
    w = validate_target_weights(target_weights, tickers)
    if frequency == "never":
        return backtest_buy_and_hold(
            prices, w, initial, risk_free_annual,
            name=name or "Buy & Hold",
        )
    dates = generate_rebalance_dates(prices.index, frequency)
    values, weights_df, turnover, costs, events, trades = _run_share_simulation(
        prices, w, initial, dates, cost_bps,
        name or f"Rebalanced ({frequency})",
    )
    return _finalise_result(
        name or f"Rebalanced ({frequency})", values, weights_df, turnover,
        costs, events, trades, initial, risk_free_annual, w, frequency,
        cost_bps,
    )


def backtest_benchmark(
    benchmark_prices: pd.Series,
    index: pd.DatetimeIndex,
    initial: float,
    risk_free_annual: float = 0.05,
    name: str = "Benchmark",
) -> BacktestResult:
    """Passive benchmark leg: ``units = initial / P(t0)``, ``V = units * P``.

    Aligned to the backtest ``index`` (inner join; a bounded forward-fill of
    up to 5 days covers exchange-holiday misalignment, longer gaps fall
    back to common valid dates only). Single-security, never rebalanced;
    costs display as N/A downstream (``is_benchmark=True``).
    """
    initial = validate_backtest_capital(initial)
    if benchmark_prices is None or len(benchmark_prices) == 0:
        raise ValueError("No benchmark prices supplied.")
    b = pd.Series(benchmark_prices, dtype=float)
    if not isinstance(b.index, pd.DatetimeIndex):
        raise ValueError("Benchmark prices need a DatetimeIndex.")
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        raise ValueError("Backtest index must contain at least 2 dates.")
    in_range = b.loc[index[0]:index[-1]].dropna()
    if len(in_range) < 2:
        raise ValueError(
            "Benchmark has insufficient overlap with the backtest dates "
            "(need at least 2 benchmark observations in range; a single "
            "price is never forward-filled across the window)."
        )
    aligned = b.reindex(index).ffill(limit=5).dropna()
    if len(aligned) < 2:
        raise ValueError(
            "Benchmark has insufficient overlap with the backtest dates."
        )
    if ((aligned <= 0) | (~np.isfinite(aligned))).any():
        raise ValueError("Benchmark prices contain zero/negative/NaN values.")
    units = initial / float(aligned.iloc[0])
    values = (aligned * units).rename("value")
    zeros = pd.Series(0.0, index=values.index)
    events = pd.DataFrame(columns=["Date", "Portfolio Value Before", "Turnover",
                                   "Transaction Cost", "Portfolio Value After"])
    return _finalise_result(
        name, values, None, zeros.rename("turnover"), zeros.rename("costs"),
        events, None, initial, risk_free_annual, None, "never", 0.0,
        is_benchmark=True,
    )


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

def backtest_summary(
    values: pd.Series,
    initial: float,
    risk_free_annual: float,
    total_costs: float = 0.0,
    n_rebalances: int = 0,
    event_turnover: pd.Series | None = None,
    trading_days: int = TRADING_DAYS,
) -> dict:
    """Performance metrics for a simulated wealth path.

    * ``total_return``: ``V_end / V_start − 1`` (simple).
    * ``cagr``: ``(V_end / V_start) ** (1 / years) − 1`` on actual calendar
      elapsed time — kept separate from the log-space ``annualised_return``.
    * Volatility/Sharpe/Sortino/Calmar reuse ``analytics.performance`` on
      daily log returns derived from the wealth path.
    """
    v = pd.Series(values, dtype=float).dropna()
    if len(v) < 2:
        raise ValueError("Need at least 2 wealth observations for summary.")
    initial = validate_backtest_capital(initial)
    if not np.isfinite(risk_free_annual):
        raise ValueError("Risk-free rate must be finite.")
    first, last = float(v.iloc[0]), float(v.iloc[-1])
    if first <= 0 or not np.isfinite(first):
        raise ValueError("Backtest wealth must start positive and finite.")
    if not np.isfinite(last) or last < 0:
        raise ValueError("Backtest wealth must stay finite and non-negative.")
    days = (v.index[-1] - v.index[0]).days if isinstance(
        v.index, pd.DatetimeIndex) else len(v) - 1
    years = max(float(days) / 365.25, 1.0 / 365.25)
    total_return = last / first - 1.0
    cagr = (last / first) ** (1.0 / years) - 1.0
    log_rets = np.log(v / v.shift(1)).dropna()
    ann_ret = annualised_return(log_rets, trading_days)
    # Degenerate (fewer than 2 return observations): report zero risk rather
    # than raising — the wealth path itself is still valid and auditable.
    if len(log_rets) >= 2:
        ann_vol = annualised_volatility(log_rets, trading_days)
        shr = sharpe_from_log(log_rets, risk_free_annual, trading_days)
    else:
        ann_vol, shr = 0.0, 0.0
    dd = downside_deviation(log_rets, trading_days=trading_days)
    dd_stats = drawdown_stats(v)
    bw = best_worst_periods(log_rets) if isinstance(
        log_rets.index, pd.DatetimeIndex) else None
    turnover_vals = (pd.Series(event_turnover, dtype=float).dropna()
                     if event_turnover is not None
                     else pd.Series(dtype=float))
    turnover_vals = turnover_vals[turnover_vals > 0]
    return {
        "final_value": last,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "years": float(years),
        "n_days": int(len(v)),
        "annualised_return": ann_ret,
        "annualised_volatility": ann_vol,
        "sharpe": shr,
        "sortino": sortino_ratio(ann_ret, risk_free_annual, dd),
        "calmar": calmar_ratio(ann_ret, dd_stats["max_drawdown"]),
        "downside_deviation": dd,
        "max_drawdown": dd_stats["max_drawdown"],
        "current_drawdown": dd_stats["current_drawdown"],
        "peak_date": dd_stats["peak_date"],
        "trough_date": dd_stats["trough_date"],
        "recovery_date": dd_stats["recovery_date"],
        "recovered": dd_stats["recovered"],
        "duration_days": dd_stats["duration_days"],
        "best_day": bw["best_day"] if bw else (None, float("nan")),
        "worst_day": bw["worst_day"] if bw else (None, float("nan")),
        "best_month": bw["best_month"] if bw else (None, float("nan")),
        "worst_month": bw["worst_month"] if bw else (None, float("nan")),
        "total_costs": float(total_costs),
        "n_rebalances": int(n_rebalances),
        "avg_turnover": float(turnover_vals.mean()) if len(turnover_vals) else 0.0,
        "total_turnover": float(turnover_vals.sum()) if len(turnover_vals) else 0.0,
    }


def compare_backtests(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Strategy comparison table (metrics × strategies) for UI display.

    Row order is fixed; values are raw floats (the UI formats them).
    Benchmark ``Total Costs`` is genuinely 0.0 here — the UI renders it
    as ``N/A`` via each result's ``is_benchmark`` flag.
    """
    if not results:
        raise ValueError("No backtest results to compare.")
    rows = [
        ("Final Value", "final_value"),
        ("Total Return", "total_return"),
        ("CAGR", "cagr"),
        ("Annualised Return", "annualised_return"),
        ("Volatility", "annualised_volatility"),
        ("Sharpe", "sharpe"),
        ("Sortino", "sortino"),
        ("Calmar", "calmar"),
        ("Max Drawdown", "max_drawdown"),
        ("Best Day", lambda m: m["best_day"][1]),
        ("Worst Day", lambda m: m["worst_day"][1]),
        ("Total Costs", "total_costs"),
        ("Rebalances", "n_rebalances"),
    ]
    data: dict[str, list] = {}
    for name, res in results.items():
        col = []
        for _, key in rows:
            col.append(key(res.metrics) if callable(key) else res.metrics[key])
        data[name] = col
    return pd.DataFrame(data, index=[label for label, _ in rows])


# ---------------------------------------------------------------------------
# Plots (dark terminal theme)
# ---------------------------------------------------------------------------

def plot_backtest_growth(
    results: dict[str, BacktestResult], initial: float
) -> go.Figure:
    """Historical growth chart: actual wealth paths, one line per strategy."""
    palette = ["#58a6ff", "#3fb950", "#f0883e", "#bc8cff", "#f85149", "#79c0ff"]
    fig = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        fig.add_trace(go.Scatter(
            x=res.values.index, y=res.values.values,
            mode="lines", name=name,
            line=dict(color=palette[i % len(palette)], width=2.2),
        ))
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text=f"Historical Growth of ${initial:,.0f} — Backtest",
                   font=dict(size=16)),
        xaxis_title="Date", yaxis_title="Portfolio Value ($)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def plot_weight_drift(weights: pd.DataFrame, title: str = "Weight Drift") -> go.Figure:
    """Stacked-area chart of holdings weights over time (sums to 1)."""
    if weights is None or weights.empty:
        raise ValueError("No weight history to plot.")
    palette = ["#58a6ff", "#3fb950", "#f0883e", "#bc8cff", "#f85149",
               "#79c0ff", "#ffa657", "#7ee787"]
    fig = go.Figure()
    for i, col in enumerate(weights.columns):
        fig.add_trace(go.Scatter(
            x=weights.index, y=weights[col].values,
            mode="lines", name=str(col), stackgroup="one",
            line=dict(color=palette[i % len(palette)], width=0.5),
        ))
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Date", yaxis_title="Weight",
        yaxis_tickformat=".0%",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig

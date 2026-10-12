"""
stress.py
---------
Historical stress analysis on actual data (pure).

Runs the CURRENT portfolio composition (fixed weights, buy-and-hold
shares) through a historical window — a current-composition hypothetical,
NOT transaction-aware history (see :mod:`analytics.walkforward` and the
ledger for those). Windows are predefined (COVID shock, 2022 rate shock)
or custom; all inputs must be covered by real data (bounded, disclosed).

Methodology labels travel with every result so the UI cannot blur
current-composition stress with actual historical performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .attribution import security_contributions, winners_losers
from .backtest import backtest_benchmark, backtest_buy_and_hold, prepare_backtest_data
from .performance import annualised_volatility, drawdown_stats
from .validation import canonical_tickers

STRESS_WINDOWS = {
    "COVID shock (Feb–Mar 2020)": ("2020-02-19", "2020-03-23"),
    "2022 rate shock (Jan–Oct 2022)": ("2022-01-03", "2022-10-12"),
}

METHOD_NOTE = (
    "Current-composition hypothetical: today's fixed mix replayed through "
    "history. Not transaction-aware performance; survivorship/selection "
    "bias may apply."
)


def resolve_window(name: str, custom_start=None,
                   custom_end=None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve a predefined or custom stress window (start < end)."""
    if name == "Custom period":
        try:
            start, end = pd.Timestamp(custom_start), pd.Timestamp(custom_end)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid custom stress dates: {e}")
        if start >= end:
            raise ValueError("Custom stress start must precede its end.")
        return start, end
    if name not in STRESS_WINDOWS:
        raise ValueError(f"Unknown stress window {name!r}; expected one of "
                         f"{list(STRESS_WINDOWS) + ['Custom period']}.")
    start, end = STRESS_WINDOWS[name]
    return pd.Timestamp(start), pd.Timestamp(end)


def run_stress(base_prices: pd.DataFrame, tickers: list[str],
               weights, window: str = "COVID shock (Feb–Mar 2020)",
               custom_start=None, custom_end=None,
               benchmark_base: pd.Series | None = None,
               benchmark_name: str = "Benchmark",
               initial: float = 10000.0, risk_free_annual: float = 0.05,
               min_days: int = 10) -> dict:
    """Replay fixed weights through [start, end] on real data.

    Returns dict with wealth series, benchmark leg (or None), metrics
    (total return, max drawdown, worst day, volatility), contributors,
    actual covered window and methodology notes. Raises ``ValueError``
    when coverage is insufficient (never invented bars).
    """
    tickers = canonical_tickers(list(tickers))
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(tickers),) or not np.all(np.isfinite(w)):
        raise ValueError("Weights must be finite with length n.")
    if abs(float(w.sum()) - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.")
    if float(initial) <= 0 or not np.isfinite(float(initial)):
        raise ValueError("Initial capital must be positive.")
    start_ts, end_ts = resolve_window(window, custom_start, custom_end)
    if start_ts >= end_ts:
        raise ValueError("Stress start must precede its end.")
    window, notes = prepare_backtest_data(
        base_prices, tickers, start_ts, end_ts, min_rows=min_days)
    actual = (window.index[0].date(), window.index[-1].date())
    if actual[0] > start_ts.date() or actual[1] < end_ts.date():
        notes = list(notes) + [
            f"Stress window clipped to available data: {actual[0]} to "
            f"{actual[1]} (requested {start_ts.date()} to {end_ts.date()})."]
    res = backtest_buy_and_hold(window, w, float(initial),
                                risk_free_annual,
                                name="Stress (current mix)")
    log_rets = np.log(res.values / res.values.shift(1)).dropna()
    dd = drawdown_stats(res.values)
    contrib = security_contributions(
        {t: float(wi) for t, wi in zip(tickers, w)},
        {t: float(window[t].iloc[0]) for t in tickers},
        {t: float(window[t].iloc[-1]) for t in tickers}, tickers)
    bench = None
    if benchmark_base is not None:
        try:
            bench = backtest_benchmark(
                benchmark_base, window.index, float(initial),
                risk_free_annual, name=benchmark_name)
        except ValueError as e:
            notes = list(notes) + [f"Benchmark skipped: {e}"]
    vol = float(annualised_volatility(log_rets)) if len(log_rets) >= 2 else 0.0
    worst = (log_rets.idxmin().date(), float(log_rets.min())) \
        if len(log_rets) else (None, float("nan"))
    return {"values": res.values, "benchmark": bench,
            "total_return": float(res.values.iloc[-1] / res.values.iloc[0] - 1),
            "max_drawdown": dd["max_drawdown"], "worst_day": worst,
            "volatility": vol, "contributions": contrib,
            "winners_losers": winners_losers(contrib),
            "actual_start": actual[0], "actual_end": actual[1],
            "n_days": len(window), "notes": notes,
            "method": METHOD_NOTE}

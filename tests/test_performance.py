"""Tests for analytics.performance (benchmark, drawdown, downside, CVaR, rolling)."""

import numpy as np
import pandas as pd
import pytest

from analytics.performance import (
    align_return_series,
    annualised_return,
    annualised_volatility,
    best_worst_periods,
    calmar_ratio,
    downside_deviation,
    drawdown_series,
    drawdown_stats,
    growth_of_capital,
    historical_alpha,
    historical_cvar,
    information_ratio,
    log_returns_from_prices,
    monthly_returns,
    portfolio_beta,
    rolling_annualised_return,
    rolling_sharpe,
    rolling_volatility,
    sharpe_from_log,
    sortino_ratio,
    summarise_performance,
    tracking_error,
    underwater_episodes,
)

IDX = pd.date_range("2024-01-01", periods=10, freq="B")
P = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.005, 0.008, -0.012, 0.01],
              index=IDX, name="portfolio")


def test_growth_and_annualised():
    g = growth_of_capital(P, initial=10_000.0)
    assert g.iloc[-1] == pytest.approx(10_000.0 * np.exp(P.sum()))
    assert annualised_return(P) == pytest.approx(P.mean() * 252)
    assert annualised_volatility(P) == pytest.approx(P.std(ddof=1) * np.sqrt(252))
    with pytest.raises(ValueError):
        growth_of_capital(P, initial=0.0)


def test_alignment_missing_dates_nan_nomutation():
    b_full = pd.Series(np.linspace(0.001, 0.01, 10), index=IDX)
    b = b_full.drop(IDX[3]).copy()  # missing date
    b.iloc[0] = np.nan  # NaN handled deliberately
    p_before, b_before = P.copy(), b.copy()
    pa, ba = align_return_series(P, b)
    assert len(pa) == len(ba) == 8  # 10 - 1 missing - 1 NaN
    assert pa.index.equals(ba.index)
    pd.testing.assert_series_equal(P, p_before)  # inputs unmutated
    pd.testing.assert_series_equal(b, b_before)
    # Column order of portfolio frame must not matter (Series here by design).
    with pytest.raises(ValueError):
        align_return_series(P.iloc[:1], b_full.iloc[:1])


def test_tracking_error_and_information_ratio():
    b = pd.Series([0.005, -0.01, 0.01, 0.0, -0.005, 0.01, 0.0, 0.004, -0.006, 0.005],
                  index=IDX)
    active = (P - b)
    assert tracking_error(P, b) == pytest.approx(active.std(ddof=1) * np.sqrt(252))
    assert information_ratio(P, b) == pytest.approx(
        active.mean() * 252 / (active.std(ddof=1) * np.sqrt(252)))
    # Zero tracking error -> safe 0.0.
    assert tracking_error(P, P.copy()) == 0.0
    assert information_ratio(P, P.copy()) == 0.0


def test_beta_and_alpha():
    # Portfolio exactly 2x benchmark -> beta 2.
    b = pd.Series([0.01, -0.005, 0.015, 0.002, -0.008, 0.012, -0.003, 0.004,
                   -0.006, 0.007], index=IDX)
    p = (b * 2).rename("portfolio")
    assert portfolio_beta(p, b) == pytest.approx(2.0)
    rp, rb = p.mean() * 252, b.mean() * 252
    assert historical_alpha(rp, rb, 2.0, 0.03) == pytest.approx(
        rp - (0.03 + 2.0 * (rb - 0.03)))
    with pytest.raises(ValueError, match="variance is zero"):
        portfolio_beta(P, pd.Series(np.full(10, 0.001), index=IDX))


def test_drawdown_stats_recovered():
    wealth = pd.Series([1.0, 1.1, 1.05, 0.9, 0.95, 1.12], index=pd.date_range(
        "2024-01-01", periods=6, freq="D"))
    dd = drawdown_series(wealth)
    assert dd.iloc[0] == 0.0 and (dd <= 0).all()
    s = drawdown_stats(wealth)
    assert s["max_drawdown"] == pytest.approx(0.9 / 1.1 - 1)
    assert s["peak_date"] == wealth.index[1]
    assert s["trough_date"] == wealth.index[3]
    assert s["recovery_date"] == wealth.index[5]
    assert s["recovered"] is True
    assert s["duration_days"] == 4
    assert s["current_drawdown"] == pytest.approx(0.0)


def test_drawdown_not_recovered_and_starts_at_high():
    wealth = pd.Series([1.0, 1.1, 0.9, 0.95], index=pd.date_range(
        "2024-01-01", periods=4, freq="D"))
    s = drawdown_stats(wealth)
    assert s["recovered"] is False
    assert pd.isna(s["recovery_date"])
    assert s["duration_days"] == 2  # peak idx1 -> last idx3
    # Series starting at its high: max drawdown simply 0-ish or measured after.
    flat = pd.Series([1.0, 1.0, 1.0], index=pd.date_range("2024-01-01", periods=3))
    sf = drawdown_stats(flat)
    assert sf["max_drawdown"] == 0.0 and sf["recovered"] is True
    assert sf["duration_days"] == 0


def test_underwater_episodes():
    wealth = pd.Series(
        [1.0, 1.1, 1.05, 0.9, 0.95, 1.12, 1.12, 0.8, 0.9, 1.15],
        index=pd.date_range("2024-01-01", periods=10, freq="D"))
    uw = underwater_episodes(wealth, top_n=5)
    assert list(uw.columns) == ["Peak", "Trough", "Recovery", "Drawdown %", "Duration (days)"]
    assert len(uw) == 2  # two disjoint episodes
    assert uw["Drawdown %"].iloc[0] == pytest.approx(0.8 / 1.12 - 1)
    assert uw["Drawdown %"].is_monotonic_increasing  # deepest first
    # No overlapping observations counted twice: troughs distinct.
    assert uw["Trough"].nunique() == len(uw)


def test_downside_sortino_calmar():
    r = pd.Series([0.01, -0.02, 0.015, -0.005])
    dd = downside_deviation(r)
    expected = np.sqrt(np.mean(np.array([0.0, 0.0004, 0.0, 0.000025]))) * np.sqrt(252)
    assert dd == pytest.approx(expected)
    assert sortino_ratio(0.10, 0.03, dd) == pytest.approx(0.07 / dd)
    assert sortino_ratio(0.10, 0.03, 0.0) == 0.0  # zero downside safe
    assert calmar_ratio(0.10, -0.20) == pytest.approx(0.5)
    assert calmar_ratio(0.10, 0.0) == 0.0  # zero drawdown safe
    assert sharpe_from_log(pd.Series([0.001, 0.001, 0.001]), 0.0) == 0.0


def test_cvar_manual_and_multiday():
    rng = np.random.default_rng(4)
    rets = pd.Series(rng.normal(0.0005, 0.01, size=300))
    out = historical_cvar(rets, 100_000.0, 0.95, 1)
    cutoff = np.percentile(rets, 5)
    tail = rets[rets <= cutoff]
    assert out["var_pct"] == pytest.approx(np.exp(cutoff) - 1)
    assert out["cvar_pct"] == pytest.approx((np.exp(tail.values) - 1).mean())
    assert out["cvar_currency"] == pytest.approx(-out["cvar_pct"] * 100_000.0)
    assert out["cvar_currency"] >= out["var_currency"]  # CVaR worse than VaR
    # Multi-day uses rolling sums, not sqrt scaling.
    out10 = historical_cvar(rets, 100_000.0, 0.95, 10)
    rolling = rets.rolling(10).sum().dropna()
    cutoff10 = np.percentile(rolling, 5)
    tail10 = rolling[rolling <= cutoff10]
    assert out10["cvar_pct"] == pytest.approx((np.exp(tail10.values) - 1).mean())
    assert out10["n_windows"] == len(rolling)
    with pytest.raises(ValueError, match="Insufficient history"):
        historical_cvar(rets.iloc[:25], 100_000.0, 0.95, 10)


def test_rolling_metrics():
    r = pd.Series(np.linspace(-0.005, 0.005, 100),
                  index=pd.date_range("2024-01-01", periods=100, freq="B"))
    w = 21
    rv = rolling_volatility(r, w)
    assert rv.iloc[w - 2] is np.nan or np.isnan(rv.iloc[w - 2])
    assert rv.iloc[w - 1] == pytest.approx(r.iloc[:w].std(ddof=1) * np.sqrt(252))
    rr = rolling_annualised_return(r, w)
    assert rr.iloc[w - 1] == pytest.approx(np.exp(r.iloc[:w].sum() * 252 / w) - 1)
    rs = rolling_sharpe(r, 0.03, w)
    m, s = r.iloc[:w].mean() * 252, r.iloc[:w].std(ddof=1) * np.sqrt(252)
    assert rs.iloc[w - 1] == pytest.approx((m - 0.03) / s)
    # Constant series: vol 0 -> Sharpe 0, never NaN/inf.
    c = pd.Series(np.full(50, 0.001),
                  index=pd.date_range("2024-01-01", periods=50, freq="B"))
    assert (rolling_sharpe(c, 0.03, 21).dropna() == 0.0).all()
    with pytest.raises(ValueError):
        rolling_volatility(r.iloc[:10], 21)


def test_monthly_aggregation_and_best_worst():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02"])
    # Jan: logs sum to ln(1.1) -> +10%; Feb: ln(0.95) -> -5%.
    r = pd.Series([np.log(1.06), np.log(1.1 / 1.06), np.log(0.97), np.log(0.95 / 0.97)],
                  index=idx)
    m = monthly_returns(r)
    assert m.loc["2024-01-31"] == pytest.approx(0.10)
    assert m.loc["2024-02-29"] == pytest.approx(-0.05)
    bw = best_worst_periods(r)
    assert bw["best_month"][0] == "2024-01" and bw["worst_month"][0] == "2024-02"
    assert bw["best_day"][1] == pytest.approx(np.exp(r).max() - 1)
    assert bw["worst_day"][1] == pytest.approx(np.exp(r).min() - 1)


def test_log_returns_from_prices_and_summary():
    px = pd.Series([100.0, 101.0, 100.5, 102.0], index=pd.date_range("2024-01-01", periods=4))
    lr = log_returns_from_prices(px)
    assert lr.iloc[0] == pytest.approx(np.log(101 / 100))
    with pytest.raises(ValueError):
        log_returns_from_prices(pd.Series([100.0], index=pd.date_range("2024-01-01", periods=1)))
    s = summarise_performance(P, P.copy(), 0.03)
    assert s["portfolio"]["sharpe"] == pytest.approx(
        sharpe_from_log(P, 0.03))
    assert s["relative"]["beta"] == pytest.approx(1.0)
    assert s["relative"]["tracking_error"] == 0.0
    s2 = summarise_performance(P, None, 0.03)
    assert s2["benchmark"] is None and s2["relative"]["beta"] is None

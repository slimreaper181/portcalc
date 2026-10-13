"""M6 — walk-forward robustness lab: grid, cache, no-look-ahead, ranking."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from analytics.robustness import (
    RobustnessConfig,
    breakeven_cost,
    cost_sensitivity_frame,
    expand_grid,
    grid_size,
    grid_to_csv,
    robustness_scorecard,
    run_robustness_grid,
    sharpe_heatmap_frame,
)

TICKERS = ["A", "B"]


def _panel(n=1400, seed=5):
    idx = pd.bdate_range("2016-01-01", periods=n)
    g = np.random.default_rng(seed)
    return pd.DataFrame({
        "A": 100.0 * np.exp(np.cumsum(g.normal(0.0005, 0.010, n))),
        "B": 100.0 * np.exp(np.cumsum(g.normal(0.0003, 0.014, n))),
    }, index=idx)


def _cfg(**kw):
    base = dict(methods=("min_variance",), lookbacks=(1.0,),
                frequencies=("quarterly",), estimators=("sample",),
                costs_bps=(10.0,), test_start=date(2020, 1, 2),
                test_end=date(2020, 12, 31))
    base.update(kw)
    return RobustnessConfig(**base)


def test_grid_size_known_before_run_and_capped():
    cfg = _cfg(lookbacks=(1.0, 3.0), frequencies=("monthly", "quarterly"),
               estimators=("sample", "ewma"), costs_bps=(0.0, 10.0),
               methods=("min_variance", "erc"))
    assert grid_size(cfg, TICKERS) == 2 * 2 * 2 * 2 * 2
    assert len(expand_grid(cfg, TICKERS)) == 32
    with pytest.raises(ValueError):
        grid_size(RobustnessConfig(
            methods=("min_variance", "max_sharpe", "erc",
                     "max_diversification"),
            lookbacks=(1.0, 2.0, 3.0, 5.0),
            frequencies=("monthly", "quarterly", "annual"),
            estimators=("sample", "ewma", "ledoit_wolf"),
            costs_bps=(0.0, 5.0, 10.0, 25.0, 50.0),
            test_start=date(2020, 1, 2), test_end=date(2020, 12, 31),
            max_cells=10), TICKERS)


def test_small_grid_runs_and_ranks():
    px = _panel()
    cfg = _cfg(methods=("min_variance", "erc"), costs_bps=(0.0, 10.0))
    res = run_robustness_grid(px, TICKERS, cfg, 10000.0)
    assert len(res.cells) == 4
    assert all(c.success for c in res.cells)
    frame = res.frame()
    assert list(frame.columns)[:5] == ["Method", "Lookback (y)", "Frequency",
                                      "Risk model", "Cost (bps)"]
    assert {"CAGR", "Vol", "Sharpe", "Max DD", "Avg Turnover", "Costs",
            "Final Value", "Failed", "Status"} <= set(frame.columns)
    ranked = frame.sort_values("Sharpe", ascending=False)
    assert ranked["Sharpe"].is_monotonic_decreasing
    sc = res.scorecard
    assert set(sc["Method"]) == {"Minimum Variance", "ERC"}
    row = sc.iloc[0]
    assert row["Worst Sharpe"] <= row["Median Sharpe"] <= row["Best Sharpe"]


def test_cache_skips_identical_configs():
    px = _panel()
    cfg = _cfg()
    cache: dict = {}
    calls = []
    r1 = run_robustness_grid(px, TICKERS, cfg, 10000.0, cache=cache,
                             progress_cb=lambda i, n: calls.append((i, n)))
    assert r1.n_cached == 0 and calls[-1] == (1, 1)
    r2 = run_robustness_grid(px, TICKERS, cfg, 10000.0, cache=cache)
    assert r2.n_cached == 1 and all(c.cached for c in r2.cells)
    assert r2.cells[0].sharpe == r1.cells[0].sharpe


def test_grid_no_look_ahead_truncation():
    full = _panel(n=1400)
    # Drop everything after the test window: post-window data must not
    # change any cell (grid layer adds no future-data path).
    cut = full.loc[full.index <= pd.Timestamp("2020-12-31")].copy()
    cfg = _cfg()
    r_full = run_robustness_grid(full, TICKERS, cfg, 10000.0)
    r_cut = run_robustness_grid(cut, TICKERS, cfg, 10000.0)
    assert r_full.cells[0].success and r_cut.cells[0].success
    assert r_full.cells[0].final_value == pytest.approx(
        r_cut.cells[0].final_value)


def test_heatmap_cost_breakeven_csv():
    px = _panel()
    cfg = _cfg(methods=("min_variance",), lookbacks=(1.0, 2.0),
               frequencies=("quarterly", "annual"),
               costs_bps=(0.0, 10.0, 50.0))
    res = run_robustness_grid(px, TICKERS, cfg, 10000.0)
    hm = sharpe_heatmap_frame(res.cells, "min_variance")
    assert hm.shape == (2, 2) and np.all(np.isfinite(hm.values))
    cs = cost_sensitivity_frame(res.cells, "min_variance")
    assert list(cs["Cost (bps)"]) == [0.0, 10.0, 50.0]
    be = breakeven_cost([(0.0, 1.0), (10.0, 0.5), (50.0, -0.2)], 0.2)
    assert be["bracket"] == (10.0, 50.0) and be["breakeven_bps"] is not None
    be2 = breakeven_cost([(0.0, 1.0), (10.0, 0.9)], 0.2)
    assert be2["breakeven_bps"] is None
    text = grid_to_csv(res)
    assert text.splitlines()[0].startswith("Method,Lookback")
    assert len(text.splitlines()) == 1 + len(res.cells)

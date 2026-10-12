"""Live/historical isolation: absurd live quotes must not move history."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _panel():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(77)
    return pd.DataFrame({
        "AAPL": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 400))),
        "MSFT": 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 400))),
    }, index=idx)


def test_live_quote_change_moves_nav_not_history():
    from analytics.allocation import (
        annualised_covariance_simple, annualised_mean_returns_simple,
        equal_risk_contribution, simple_returns_from_prices)
    from analytics.backtest import backtest_rebalanced
    from data.live_market import Quote, build_mark_snapshot, select_mark
    from datetime import datetime, timezone
    px = _panel()
    tickers = ["AAPL", "MSFT"]

    def _hist():
        rets = simple_returns_from_prices(px, tickers)
        mu = annualised_mean_returns_simple(rets)
        cov = annualised_covariance_simple(rets)
        erc = equal_risk_contribution(tickers, mu.values, cov, 0.02)
        bt = backtest_rebalanced(px, np.array([0.5, 0.5]), 10000.0,
                                 "quarterly", 10.0)
        return mu, cov, erc.weights.values.copy(), bt.values.copy()

    mu0, cov0, w0, v0 = _hist()
    # Absurd live quotes: 100x marks (the isolation firewall under test).
    now = datetime(2026, 10, 6, 15, 30, tzinfo=timezone.utc)
    quotes = {t: Quote(symbol=t, last_trade=px[t].iloc[-1] * 100.0,
                       last_trade_timestamp=now, source="Alpaca", feed="iex",
                       previous_close=px[t].iloc[-2], received_at=now)
              for t in tickers}
    snap = build_mark_snapshot(quotes, now)
    assert snap["AAPL"].mark == pytest.approx(px["AAPL"].iloc[-1] * 100.0)
    mu1, cov1, w1, v1 = _hist()  # historical path rerun identically
    pd.testing.assert_series_equal(mu0, mu1)
    pd.testing.assert_frame_equal(cov0, cov1)
    assert np.array_equal(w0, w1)
    pd.testing.assert_series_equal(v0, v1)


def test_no_live_or_display_imports_in_historical_engines():
    checked = []
    for mod in ["allocation", "backtest", "walkforward", "factors",
                "optimisation", "performance", "positions", "currency",
                "attribution", "history", "ledger", "rebalance_plan",
                "risk_extra", "stress", "performance_cashflows",
                "validation", "portfolio", "returns", "var", "scenario",
                "security"]:
        tree = ast.parse((ROOT / "analytics" / f"{mod}.py").read_text(
            encoding="utf-8"))
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
        bad = mods & {"alpaca", "live_market", "tradingview", "streamlit",
                      "yfinance", "requests"}
        assert not bad, (mod, bad)
        checked.append(mod)
    assert len(checked) == 21
    # data/alpaca itself never imports yfinance history mutation paths
    tree = ast.parse((ROOT / "data" / "live_market.py").read_text(
        encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(m.startswith("analytics.backtest")
                   or m.startswith("analytics.walkforward") for m in imported)

"""Tests for analytics.rebalance_plan (targets, trades, cash, costs)."""

import numpy as np
import pandas as pd
import pytest

from analytics.rebalance_plan import (
    OrderIntent,
    plan_diagnostics,
    plan_trades,
    resolve_planner_targets,
    validate_target_weights,
)

TICKERS = ["A", "B", "C"]
MU = np.array([0.08, 0.06, 0.10])
COV = pd.DataFrame(np.array([[0.04, 0.01, 0.005],
                             [0.01, 0.03, 0.002],
                             [0.005, 0.002, 0.05]]),
                   index=TICKERS, columns=TICKERS)
PX = {"A": 100.0, "B": 50.0, "C": 25.0}


def test_target_validation():
    with pytest.raises(ValueError):
        validate_target_weights([0.5, 0.5], TICKERS)
    with pytest.raises(ValueError):
        validate_target_weights([0.5, 0.3, 0.3], TICKERS)
    with pytest.raises(ValueError):
        validate_target_weights([-0.1, 0.6, 0.5], TICKERS)
    with pytest.raises(ValueError):
        resolve_planner_targets("nope", TICKERS, MU, COV, 0.02, 0.0, 1.0)
    name, w = resolve_planner_targets("equal_weight", TICKERS, MU, COV,
                                      0.02, 0.0, 1.0)
    assert name == "Equal Weight"
    assert w.values == pytest.approx(np.ones(3) / 3)
    for method in ("min_variance", "max_sharpe", "erc",
                   "max_diversification"):
        name, w = resolve_planner_targets(method, TICKERS, MU, COV, 0.02,
                                          0.0, 1.0)
        assert abs(w.sum() - 1.0) < 1e-6, method
    with pytest.raises(ValueError):
        resolve_planner_targets("black_litterman", TICKERS, MU, COV, 0.02,
                                0.0, 1.0)


def test_plan_trades_notionals_shares_cash():
    cw = np.array([0.6, 0.3, 0.1])
    tw = np.array([1 / 3] * 3)
    plan = plan_trades(cw, tw, TICKERS, 9000.0, PX, cash_available=2000.0,
                       min_cash_buffer=500.0, cost_bps=10.0)
    row = plan.trades.set_index("Ticker")
    assert row.loc["A", "Trade Notional"] == pytest.approx(
        (1 / 3 - 0.6) * 9000.0)
    assert row.loc["A", "Approx Shares"] == pytest.approx(
        row.loc["A", "Trade Notional"] / 100.0)
    assert row.loc["A", "Action"] == "SELL"
    assert row.loc["C", "Action"] == "BUY"
    assert plan.gross_turnover == pytest.approx(
        abs(plan.trades["Trade Notional"]).sum() / 9000.0)
    assert plan.estimated_cost == pytest.approx(
        abs(plan.trades["Trade Notional"]).sum() * 10.0 / 10000.0)
    # Rotation self-funds (2400+300 sells vs 300 buys... net 0); the 4800
    # traded notional at 10bps (4.8) draws deployable cash.
    assert plan.cash_deployed == pytest.approx(4.8)
    assert plan.cash_remaining == pytest.approx(2000.0 - 4.8)
    assert plan.cash_remaining >= 500.0 - 1e-9
    assert all(isinstance(i, OrderIntent) for i in plan.intents)
    assert plan.intents[0].side in ("BUY", "SELL", "HOLD")


def test_cash_shortfall_reported_not_assumed():
    cw = np.array([1.0, 0.0, 0.0])
    tw = np.array([0.0, 0.5, 0.5])
    plan = plan_trades(cw, tw, TICKERS, 10000.0, PX, cash_available=100.0,
                       cost_bps=0.0)
    # Net buys 10000 funded by sells 10000 → no cash needed, no shortfall.
    assert plan.cash_shortfall == pytest.approx(0.0)
    plan2 = plan_trades(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]),
                        TICKERS, 10000.0, PX, cash_available=0.0,
                        cost_bps=0.0)
    # Rotation is self-funding through sells.
    assert plan2.cash_shortfall == pytest.approx(0.0)


def test_no_trade_band_holds_small_legs():
    cw = np.array([0.34, 0.33, 0.33])
    tw = np.array([1 / 3] * 3)
    plan = plan_trades(cw, tw, TICKERS, 9000.0, PX, min_trade_value=500.0,
                       cost_bps=10.0)
    assert plan.trades["Action"].tolist() == ["HOLD"] * 3
    assert plan.estimated_cost == pytest.approx(0.0)
    assert plan.gross_turnover == pytest.approx(0.0)
    assert plan.skipped_small == TICKERS


def test_cash_target_and_buffer():
    cw = np.array([0.5, 0.5, 0.0])
    tw = np.array([0.4, 0.4, 0.2])
    plan = plan_trades(cw, tw, TICKERS, 10000.0, PX, cash_available=3000.0,
                       min_cash_buffer=500.0, cash_target_weight=0.05,
                       cost_bps=0.0)
    # Pure rotation is self-funding (sells cover buys); 0bps → no cost draw.
    assert plan.cash_deployed == pytest.approx(0.0)
    assert plan.cash_remaining == pytest.approx(3000.0)
    assert plan.cash_shortfall == pytest.approx(0.0)
    # With costs, the estimate itself draws deployable cash.
    plan2 = plan_trades(cw, tw, TICKERS, 10000.0, PX, cash_available=3000.0,
                        min_cash_buffer=500.0, cash_target_weight=0.05,
                        cost_bps=10.0)
    assert plan2.cash_deployed == pytest.approx(4000 * 10.0 / 10000.0)
    assert plan2.cash_remaining == pytest.approx(
        3000.0 - 4000 * 10.0 / 10000.0)


def test_plan_diagnostics_before_after():
    d = plan_diagnostics(np.array([0.6, 0.3, 0.1]), np.array([1 / 3] * 3),
                         TICKERS, MU, COV, 0.02)
    assert set(d) == {"current", "target"}
    assert d["current"]["volatility"] > 0
    assert abs(d["target"]["risk_contributions"].sum() - 1.0) < 1e-9
    with pytest.raises(ValueError):
        plan_trades([0.5, 0.5], [0.5, 0.5], TICKERS, 1000.0, PX)

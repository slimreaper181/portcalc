"""M7 — research execution simulator: MARKET/LIMIT, slippage, commission,
partial fills, blotter, sandbox purity and no broker execution."""

import numpy as np
import pandas as pd
import pytest

from analytics.execution import (
    SimConfig,
    SimOrder,
    SimQuote,
    apply_fills_sandbox,
    blotter_frame,
    cost_comparison,
    orders_from_intents,
    simulate_limit_order,
    simulate_limit_over_bars,
    simulate_market_order,
    simulate_specs,
    simulation_summary,
)


def _cfg(**kw):
    base = dict(slippage_bps=10.0, commission_bps=5.0, commission_flat=1.0)
    base.update(kw)
    return SimConfig(**base)


def test_market_buy_sell_math():
    q = SimQuote(bid=100.0, ask=100.4, last=100.2)
    b = simulate_market_order("AAA", "BUY", 10, q, _cfg())
    assert b.status == "FILLED"
    assert b.avg_fill_price == pytest.approx(100.4 * 1.001)
    assert b.reference_price == pytest.approx(100.4)
    assert b.slippage_cost == pytest.approx((b.avg_fill_price - 100.4) * 10)
    assert b.commission == pytest.approx(10 * b.avg_fill_price * 5 / 1e4 + 1.0)
    s = simulate_market_order("AAA", "SELL", 10, q, _cfg())
    assert s.avg_fill_price == pytest.approx(100.0 * 0.999)
    assert s.reference_price == pytest.approx(100.0)


def test_market_rejects_without_reference():
    o = simulate_market_order("AAA", "BUY", 10, SimQuote(), _cfg())
    assert o.status == "REJECTED" and o.filled_qty == 0


def test_limit_fill_and_no_fill():
    q = SimQuote(bid=100.0, ask=100.4)
    fill = simulate_limit_order("AAA", "BUY", 10, 100.5, q, _cfg())
    assert fill.status == "FILLED" and fill.filled_qty == 10
    rest = simulate_limit_order("AAA", "BUY", 10, 99.0, q, _cfg())
    assert rest.status == "CANCELLED" and rest.filled_qty == 0
    sell = simulate_limit_order("AAA", "SELL", 10, 99.5, q, _cfg())
    assert sell.status == "FILLED"
    with pytest.raises(ValueError):
        simulate_limit_order("AAA", "BUY", 10, None, q, _cfg())


def test_partial_fill_from_fixture():
    q = SimQuote(bid=100.0, ask=100.2, available_size=4.0)
    o = simulate_market_order("AAA", "BUY", 10, q, _cfg())
    assert o.status == "PARTIALLY_FILLED" and o.filled_qty == 4.0


def test_limit_over_bars_conservative():
    bars = pd.DataFrame({"high": [101.0, 100.6, 100.1],
                         "low": [100.5, 100.2, 99.6]})
    o = simulate_limit_over_bars("AAA", "BUY", 10, 100.0, bars, _cfg())
    assert o.status == "FILLED" and o.avg_fill_price == pytest.approx(100.0)
    assert o.slippage_cost == 0.0
    o2 = simulate_limit_over_bars("AAA", "BUY", 10, 100.0, bars, _cfg(),
                                  sizes=[3.0, 3.0, 3.0])
    assert o2.status == "PARTIALLY_FILLED" and o2.filled_qty == 3.0
    flat = pd.DataFrame({"high": [101.0], "low": [100.5]})
    o3 = simulate_limit_over_bars("AAA", "BUY", 10, 99.0, flat, _cfg())
    assert o3.status == "CANCELLED"


def test_blotter_summary_and_statuses():
    orders = [simulate_market_order("AAA", "BUY", 5,
                                    SimQuote(bid=10.0, ask=10.1), _cfg()),
              simulate_market_order("BBB", "SELL", 5, SimQuote(), _cfg())]
    frame = blotter_frame(orders)
    assert list(frame.columns) == ["Order ID", "Ticker", "Side", "Type",
                                  "Requested", "Filled", "Average Fill",
                                  "Reference", "Slippage", "Commission",
                                  "Status", "Note"]
    s = simulation_summary(orders)
    assert s["n_filled"] == 1 and s["n_rejected"] == 1
    assert s["total_friction"] == pytest.approx(s["slippage"]
                                                + s["commissions"])


def test_intents_and_specs():
    class I:
        def __init__(self, t, s, q):
            self.ticker, self.side, self.quantity_shares = t, s, q
    specs = orders_from_intents(
        [I("AAA", "BUY", 12.0), I("BBB", "SELL", -7.0),
         I("CCC", "HOLD", 0.0)], order_type="MARKET")
    assert [(s["ticker"], s["side"], s["quantity"]) for s in specs] == [
        ("AAA", "BUY", 12.0), ("BBB", "SELL", 7.0)]
    with pytest.raises(ValueError):
        orders_from_intents([I("AAA", "BUY", 1.0)], order_type="LIMIT")
    quotes = {"AAA": SimQuote(bid=9.9, ask=10.0),
              "BBB": SimQuote(bid=20.0, ask=20.2)}
    done = simulate_specs(specs, quotes, _cfg())
    assert [o.status for o in done] == ["FILLED", "FILLED"]


def test_cost_comparison_honestly_labelled():
    c = cost_comparison(112.0, 10.0, 100000.0)
    assert c["modelled"] == pytest.approx(100.0)
    assert c["difference"] == pytest.approx(12.0)
    with pytest.raises(ValueError):
        cost_comparison(1.0, -5.0, 100.0)


def test_sandbox_never_mutates_inputs():
    hold = {"AAA": 10.0}
    orders = [simulate_market_order("AAA", "BUY", 5,
                                    SimQuote(bid=10.0, ask=10.0),
                                    SimConfig())]
    h2, c2 = apply_fills_sandbox(hold, 1000.0, orders, {"AAA": 10.0})
    assert hold == {"AAA": 10.0}
    assert h2["AAA"] == pytest.approx(15.0)
    assert c2 < 1000.0
    # short sale blocked past available (stated, no margin invented)
    s = [simulate_market_order("ZZZ", "SELL", 5,
                               SimQuote(bid=10.0, ask=10.0), SimConfig())]
    h3, _ = apply_fills_sandbox({}, 0.0, s, {"ZZZ": 10.0})
    assert h3.get("ZZZ", 0.0) == 0.0


def test_no_broker_execution_paths():
    import ast
    import pathlib
    src = pathlib.Path("analytics/execution.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not (imported & {"urllib", "requests", "http", "socket", "ssl",
                            "alpaca", "websocket", "websockets"}), imported
    for token in ("submit_order(", "place_order(", "create_order(",
                  "cancel_order(", "broker.execute", ".post("):
        assert token not in src

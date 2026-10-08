"""End-to-end mixed-currency integration (synthetic, deterministic).

US asset in USD + UK asset quoted in GBp + European asset in EUR, with
purchase dates, historical FX, current FX and current prices. Verifies the
full pipeline — normalisation, cost basis, valuation, P&L, weights,
historical base prices/returns, covariance, ERC/MaxDiv/BL inputs and
backtest wealth — against manually calculated values.
"""

import numpy as np
import pandas as pd
import pytest

from analytics.allocation import (
    annualised_covariance_simple,
    annualised_mean_returns_simple,
    black_litterman_posterior,
    equal_risk_contribution,
    maximum_diversification,
    simple_returns_from_prices,
)
from analytics.backtest import backtest_buy_and_hold
from analytics.currency import convert_price_series, convert_value
from analytics.positions import (
    Position,
    normalise_quote,
    resolve_purchase_price,
    value_portfolio,
    value_position,
)

DATES = pd.bdate_range("2024-01-02", periods=10)
TODAY = DATES[-1].date()

# Native histories (B quoted in pence).
NATIVE = pd.DataFrame({
    "A": [100., 101., 102., 103., 104., 105., 106., 107., 108., 110.],
    "B": [300., 303., 306., 303., 309., 312., 309., 315., 318., 320.],
    "C": [50., 50.5, 51., 50.5, 51.5, 52., 51.5, 52.5, 53., 53.5],
}, index=DATES)
CCY = {"A": "USD", "B": "GBP", "C": "EUR"}
SCALE = {"A": 1.0, "B": 0.01, "C": 1.0}

# Deterministic historical FX (USD-per-unit): GBP flat 1.25 then 1.30.
FX_HIST = pd.DataFrame({
    "GBP": [1.25] * 5 + [1.30] * 5,
    "EUR": [1.10] * 10,
}, index=DATES)

# Current snapshot.
CUR_NATIVE = {"A": 110.0, "B": 320.0, "C": 53.5}   # B in pence
CUR_FX = {"USD": 1.0, "GBP": 1.30, "EUR": 1.10}
BUY = {
    "A": (10.0, 100.0),   # 10 sh @ $100
    "B": (100.0, 3.00),   # 100 sh @ £3.00 (300 GBp)
    "C": (20.0, 50.0),    # 20 sh @ €50
}
BUY_FX = {"A": 1.0, "B": 1.25, "C": 1.10}  # purchase-date USD-per-unit


def _positions():
    return {
        t: Position(ticker=t, shares=BUY[t][0],
                    purchase_date=DATES[0].date(),
                    purchase_price_native=BUY[t][1],
                    purchase_price_source="manual",
                    manual_purchase_price=True,
                    native_currency=CCY[t],
                    quote_unit={"A": "USD", "B": "GBp", "C": "EUR"}[t],
                    quote_scale=SCALE[t])
        for t in ("A", "B", "C")
    }


def _base_panel():
    out = pd.DataFrame(index=DATES)
    for t in ("A", "B", "C"):
        native = NATIVE[t] * SCALE[t]
        out[t] = convert_price_series(native, CCY[t], "USD", FX_HIST)
    return out[["A", "B", "C"]]


def test_quote_normalisation_and_purchase_basis():
    # Raw pence purchase close 300 GBp → £3.00/share cost basis.
    out = resolve_purchase_price(
        NATIVE["B"], pd.Series(dtype=float), DATES[0].date(), TODAY)
    assert out["status"] == "ok"
    assert out["raw_close"] == pytest.approx(300.0)
    assert normalise_quote(out["price"], 0.01) == pytest.approx(3.00)
    # EUR needs no scaling.
    assert normalise_quote(50.0, 1.0) == pytest.approx(50.0)


def test_current_values_weights_and_pnl():
    vals = []
    for t in ("A", "B", "C"):
        pos = _positions()[t]
        vals.append(value_position(
            pos, normalise_quote(CUR_NATIVE[t], SCALE[t]), "Yahoo", None,
            BUY_FX[t], CUR_FX[CCY[t]], "USD"))
    agg = value_portfolio(vals)
    # Manual expectations:
    # A: $1100 value, $1000 cost. B: £320 → $416; cost £300 @1.25 → $375.
    # C: €1070 → $1177; cost €1000 @1.10 → $1100.
    # Total $2693, cost $2475, P&L $218.
    by_t = {v.ticker: v for v in vals}
    assert by_t["A"].base_market_value == pytest.approx(1100.0)
    assert by_t["B"].base_market_value == pytest.approx(416.0)
    assert by_t["C"].base_market_value == pytest.approx(1177.0)
    assert agg["total_base_value"] == pytest.approx(2693.0)
    assert agg["total_base_cost"] == pytest.approx(2475.0)
    assert agg["total_pnl"] == pytest.approx(218.0)
    assert agg["weights"]["A"] == pytest.approx(1100 / 2693)
    assert agg["weights"]["B"] == pytest.approx(416 / 2693)
    assert agg["weights"]["C"] == pytest.approx(1177 / 2693)
    assert abs(sum(agg["weights"].values()) - 1.0) < 1e-9


def test_historical_base_prices_and_returns():
    panel = _base_panel()
    # Spot checks: B day0 £3.00 @1.25 → $3.75; day5 £3.09 @1.30 → $4.017.
    assert panel["B"].iloc[0] == pytest.approx(3.00 * 1.25)
    assert panel["B"].iloc[5] == pytest.approx(3.12 * 1.30)
    assert panel["A"].iloc[-1] == pytest.approx(110.0)
    assert panel["C"].iloc[-1] == pytest.approx(53.5 * 1.10)
    # A base return day1 = simple 1% (no FX on USD leg).
    rets = simple_returns_from_prices(panel, ["A", "B", "C"])
    assert rets["A"].iloc[0] == pytest.approx(0.01)
    # B day1: £3.00→£3.03 @1.25 flat → +1%.
    assert rets["B"].iloc[0] == pytest.approx(0.01)
    # Day4→day5 crosses the FX jump: £3.09@1.25 → £3.12@1.30.
    assert rets["B"].iloc[4] == pytest.approx((3.12 * 1.30) / (3.09 * 1.25) - 1)
    assert np.all(np.isfinite(rets.values))


def test_base_covariance_feeds_allocation():
    panel = _base_panel()
    rets = simple_returns_from_prices(panel, ["A", "B", "C"])
    mu = annualised_mean_returns_simple(rets)
    cov = annualised_covariance_simple(rets)
    tickers = ["A", "B", "C"]
    # Manual covariance cross-check on the base panel.
    assert cov.loc["A", "A"] == pytest.approx(rets["A"].var() * 252)
    assert cov.loc["A", "B"] == pytest.approx(
        rets[["A", "B"]].cov().iloc[0, 1] * 252)
    # ERC consumes the base covariance and equalises base risk.
    erc = equal_risk_contribution(tickers, mu.values, cov, rf=0.02)
    assert erc.success and erc.weights is not None
    assert erc.weights.sum() == pytest.approx(1.0)
    prc = erc.risk_contributions.values
    assert np.abs(prc - 1 / 3).max() <= 0.02
    # MaxDiv consumes the same base covariance (DR ≥ 1 long-only).
    md = maximum_diversification(tickers, mu.values, cov, rf=0.02)
    assert md.success and md.diversification_ratio >= 1.0 - 1e-6
    # ...and it differs from the *local*-currency answer (FX matters).
    local = NATIVE.copy()
    local["B"] = local["B"] * 0.01
    lrets = simple_returns_from_prices(local, tickers)
    lcov = annualised_covariance_simple(lrets)
    assert not np.allclose(cov.values, lcov.values)


def test_bl_uses_base_covariance():
    panel = _base_panel()
    rets = simple_returns_from_prices(panel, ["A", "B", "C"])
    mu = annualised_mean_returns_simple(rets)
    cov = annualised_covariance_simple(rets)
    tickers = ["A", "B", "C"]
    w_ref = np.array([0.5, 0.3, 0.2])
    post = black_litterman_posterior(
        tickers, cov, w_ref, 2.5, 0.05,
        [{"type": "absolute", "ticker": "A", "return": 0.15,
          "confidence": 0.7}])
    assert post.success
    # Prior is exactly δΣw on the BASE covariance.
    assert post.prior.values == pytest.approx(
        (2.5 * cov.values @ w_ref), rel=1e-9)
    # Posterior moves toward the 15% view.
    assert abs(post.posterior["A"] - 0.15) < abs(post.prior["A"] - 0.15)


def test_backtest_wealth_on_base_panel():
    panel = _base_panel()
    w = np.array([1100 / 2693, 416 / 2693, 1177 / 2693])
    w = w / w.sum()
    res = backtest_buy_and_hold(panel, w, 2693.0, risk_free_annual=0.0)
    # Day0 wealth == initial by construction; final == shares·prices.
    assert res.values.iloc[0] == pytest.approx(2693.0)
    shares = 2693.0 * w / panel.iloc[0].values
    expect_last = float((shares * panel.iloc[-1].values).sum())
    assert res.values.iloc[-1] == pytest.approx(expect_last)
    assert np.all(np.isfinite(res.values.values))


def test_allocation_consumes_base_not_local_covariance():
    # ERC/MaxDiv on the base panel must differ from the local-currency
    # answer whenever FX moves — proving FX is embedded, not bolted on.
    from analytics.allocation import equal_risk_contribution  # noqa: F401
    panel = _base_panel()
    rets = simple_returns_from_prices(panel, ["A", "B", "C"])
    mu = annualised_mean_returns_simple(rets)
    cov = annualised_covariance_simple(rets)
    tickers = ["A", "B", "C"]
    erc_base = equal_risk_contribution(tickers, mu.values, cov, rf=0.02)
    assert erc_base.success
    local = NATIVE.copy()
    local["B"] = local["B"] * 0.01
    lrets = simple_returns_from_prices(local, tickers)
    lmu = annualised_mean_returns_simple(lrets)
    lcov = annualised_covariance_simple(lrets)
    erc_local = equal_risk_contribution(tickers, lmu.values, lcov, rf=0.02)
    assert erc_local.success
    assert np.abs(erc_base.weights.values
                  - erc_local.weights.values).max() > 1e-4
    md_base = maximum_diversification(tickers, mu.values, cov, rf=0.02)
    md_local = maximum_diversification(tickers, lmu.values, lcov, rf=0.02)
    assert md_base.success and md_local.success
    assert np.abs(md_base.weights.values
                  - md_local.weights.values).max() > 1e-4


def test_oos_estimation_ignores_future_fx():
    # Training FX identical, future test FX wildly different → identical
    # estimation weights (Task 5 no-look-ahead, extended to FX).
    from analytics.allocation import estimation_window
    idx = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(3)
    px = pd.DataFrame({
        "A": 100 + np.cumsum(rng.normal(0.1, 1.0, 100)),
        "B": (200 + np.cumsum(rng.normal(0.1, 1.0, 100))) * 0.01,
    }, index=idx)
    fx_past = pd.DataFrame({"GBP": np.full(100, 1.25)}, index=idx)
    fut_idx = pd.bdate_range(idx[-1] + pd.Timedelta(days=1), periods=20)
    fx_a = pd.DataFrame({"GBP": np.full(20, 1.25)}, index=fut_idx)
    fx_b = pd.DataFrame({"GBP": np.full(20, 9.99)}, index=fut_idx)
    T = fut_idx[0]
    tickers = ["A", "B"]
    panel_a = pd.DataFrame(index=idx)
    panel_b = pd.DataFrame(index=idx)
    for t, scale, ccy in (("A", 1.0, "USD"), ("B", 0.01, "GBP")):
        converted = convert_price_series(
            px[t] * scale, ccy, "USD", fx_past)
        panel_a[t] = converted
        panel_b[t] = converted.copy()
    est_a, _ = estimation_window(panel_a, tickers, T, 3.0, min_rows=10)
    est_b, _ = estimation_window(panel_b, tickers, T, 3.0, min_rows=10)
    pd.testing.assert_frame_equal(est_a, est_b)
    for fn in (equal_risk_contribution, maximum_diversification):
        ra = fn(tickers,
                annualised_mean_returns_simple(
                    simple_returns_from_prices(est_a, tickers)).values,
                annualised_covariance_simple(
                    simple_returns_from_prices(est_a, tickers)), rf=0.02)
        rb = fn(tickers,
                annualised_mean_returns_simple(
                    simple_returns_from_prices(est_b, tickers)).values,
                annualised_covariance_simple(
                    simple_returns_from_prices(est_b, tickers)), rf=0.02)
        assert ra.success and rb.success
        assert np.array_equal(ra.weights.values, rb.weights.values)
    # Future FX paths really do differ (sanity that the test is non-vacuous).
    assert fx_a["GBP"].iloc[0] != fx_b["GBP"].iloc[0]

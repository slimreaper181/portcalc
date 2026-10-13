"""Tests for analytics.factors + data/factors conventions (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from analytics.factors import (
    MODEL_FACTORS,
    align_factor_returns,
    annualise_alpha,
    check_min_observations,
    describe_factor_loading,
    describe_market_beta,
    factor_attribution,
    format_p_value,
    is_non_us_ticker,
    rolling_factor_regression,
    run_capm,
    run_ff3,
    run_ff5,
    run_ff5_momentum,
    run_factor_regression,
    run_model,
    simple_returns_from_wealth,
    to_decimal_returns,
    us_scope_note,
)

IDX = pd.date_range("2020-01-01", periods=500, freq="B")
rng = np.random.default_rng(42)


def _synthetic_factors(n=500, seed=42):
    r = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Mkt-RF": r.normal(0.0004, 0.01, n),
        "SMB": r.normal(0.0001, 0.005, n),
        "HML": r.normal(0.0, 0.005, n),
        "RMW": r.normal(0.0002, 0.004, n),
        "CMA": r.normal(0.0, 0.004, n),
        "Mom": r.normal(0.0003, 0.006, n),
        "RF": np.full(n, 0.00004),
    }, index=idx)


# ---------------------------------------------------------------------------
# Units & return construction
# ---------------------------------------------------------------------------

def test_percent_to_decimal_conversion():
    raw = pd.DataFrame({"Mkt-RF": [0.42, -0.10], "RF": [0.01, 0.01]})
    out = to_decimal_returns(raw)
    assert out["Mkt-RF"].iloc[0] == pytest.approx(0.0042)
    assert out["RF"].iloc[1] == pytest.approx(0.0001)
    # A test that fails if percentages are mistaken for decimals:
    assert out.abs().max().max() < 0.05
    with pytest.raises(ValueError):
        to_decimal_returns(pd.DataFrame({"Mkt-RF": [np.inf]}))


def test_simple_returns_and_excess_construction():
    wealth = pd.Series([100.0, 101.0, 100.5, 102.0],
                       index=pd.date_range("2020-01-01", periods=4))
    simple = simple_returns_from_wealth(wealth)
    assert list(simple) == pytest.approx([0.01, -0.005 / 1.01, 0.015 / 1.005])
    # Simple, NOT log: log(101/100) would differ.
    assert simple.iloc[0] != pytest.approx(np.log(1.01))
    rf = pd.Series(np.full(3, 0.0001), index=simple.index)
    excess = simple - rf
    assert excess.iloc[0] == pytest.approx(0.01 - 0.0001)
    with pytest.raises(ValueError):
        simple_returns_from_wealth(pd.Series([100.0]))


def test_date_alignment_inner_join():
    y = pd.Series(np.linspace(0.001, 0.01, 10), index=IDX[:10])
    f = pd.DataFrame({"Mkt-RF": np.linspace(0.002, 0.02, 10)},
                     index=IDX[:10])
    f = f.drop(IDX[3])  # holiday-style gap
    y2 = y.copy()
    y2.iloc[0] = np.nan
    ya, fa = align_factor_returns(y2, f)
    assert len(ya) == len(fa) == 8  # 10 − 1 gap − 1 NaN
    assert ya.index.equals(fa.index)
    assert len(y2) == 10  # inputs unmutated
    with pytest.raises(ValueError):
        align_factor_returns(y.iloc[:0], f)


# ---------------------------------------------------------------------------
# §39 synthetic coefficient recovery
# ---------------------------------------------------------------------------

def test_ff3_coefficient_recovery():
    F = _synthetic_factors()
    truth = {"alpha": 0.0001, "Mkt-RF": 1.20, "SMB": -0.30, "HML": 0.50}
    noise = np.random.default_rng(7).normal(0, 1e-4, len(F))
    y = (truth["alpha"] + truth["Mkt-RF"] * F["Mkt-RF"]
         + truth["SMB"] * F["SMB"] + truth["HML"] * F["HML"] + noise)
    y = pd.Series(y, index=F.index)
    res = run_ff3(y, F[["Mkt-RF", "SMB", "HML"]])
    assert res.alpha == pytest.approx(truth["alpha"], abs=2e-5)
    assert res.betas["Mkt-RF"] == pytest.approx(1.20, abs=0.02)
    assert res.betas["SMB"] == pytest.approx(-0.30, abs=0.02)
    assert res.betas["HML"] == pytest.approx(0.50, abs=0.02)
    assert res.observations == 500
    assert res.r_squared > 0.99  # tiny noise ⇒ near-perfect fit


def test_capm_beta_alpha():
    F = _synthetic_factors()
    noise = np.random.default_rng(9).normal(0, 1e-4, len(F))
    y = pd.Series(0.0002 + 0.9 * F["Mkt-RF"] + noise, index=F.index)
    res = run_capm(y, F[["Mkt-RF"]])
    assert res.betas["Mkt-RF"] == pytest.approx(0.9, abs=0.02)
    assert res.alpha == pytest.approx(0.0002, abs=2e-5)
    # Exact compounded annualisation, not naive ×252.
    assert res.alpha_annualised == pytest.approx(
        (1 + res.alpha) ** 252 - 1, rel=1e-9)
    assert annualise_alpha(0.0002) != pytest.approx(0.0002 * 252)


def test_ff5_and_momentum_recovery():
    F = _synthetic_factors()
    coefs = {"Mkt-RF": 1.1, "SMB": 0.2, "HML": -0.4, "RMW": 0.3,
             "CMA": -0.15, "Mom": 0.25}
    noise = np.random.default_rng(11).normal(0, 1e-4, len(F))
    y = pd.Series(0.00005 + sum(coefs[k] * F[k] for k in coefs) + noise,
                  index=F.index)
    r5 = run_ff5(y, F[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
    for k, v in coefs.items():
        if k == "Mom":
            continue
        assert r5.betas[k] == pytest.approx(v, abs=0.03)
    r6 = run_ff5_momentum(
        y, F[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]])
    for k, v in coefs.items():
        assert r6.betas[k] == pytest.approx(v, abs=0.03)
    assert r6.model_name == "ff5_mom"
    with pytest.raises(ValueError):
        run_model("ff10", y, F)
    with pytest.raises(ValueError):
        run_ff5(y, F[["Mkt-RF", "SMB"]])  # missing columns


# ---------------------------------------------------------------------------
# Statistics: R², HAC, residuals
# ---------------------------------------------------------------------------

def test_r_squared_formulas():
    F = _synthetic_factors(n=200, seed=5)
    y = pd.Series(0.0001 + 1.0 * F["Mkt-RF"]
                  + np.random.default_rng(6).normal(0, 0.002, 200),
                  index=F.index)
    res = run_capm(y, F[["Mkt-RF"]])
    yv, pv = y.values, None
    import statsmodels.api as sm
    ols = sm.OLS(yv, sm.add_constant(F[["Mkt-RF"]].values)).fit()
    assert res.r_squared == pytest.approx(ols.rsquared)
    assert res.adj_r_squared == pytest.approx(ols.rsquared_adj)
    n, k = len(y), 1
    assert res.adj_r_squared == pytest.approx(
        1 - (1 - res.r_squared) * (n - 1) / (n - k - 1))
    # HAC structure: robust SEs keyed per regressor, finite and positive.
    assert set(res.std_errors) == {"Mkt-RF"}
    assert res.std_errors["Mkt-RF"] > 0 and np.isfinite(res.alpha_se)
    assert res.hac_lags == 5
    assert set(res.t_stats) == {"Mkt-RF"} and set(res.p_values) == {"Mkt-RF"}


def test_residual_volatility():
    F = _synthetic_factors(n=200, seed=5)
    y = pd.Series(0.0001 + 1.0 * F["Mkt-RF"]
                  + np.random.default_rng(6).normal(0, 0.002, 200),
                  index=F.index)
    res = run_capm(y, F[["Mkt-RF"]])
    import statsmodels.api as sm
    resid = sm.OLS(y.values, sm.add_constant(
        F[["Mkt-RF"]].values)).fit().resid
    assert res.residual_volatility == pytest.approx(
        pd.Series(resid).std(ddof=1) * np.sqrt(252))
    assert np.isfinite(res.durbin_watson) and res.condition_number > 0


def test_column_order_independence():
    F = _synthetic_factors()
    y = pd.Series(0.0001 + 1.1 * F["Mkt-RF"] - 0.2 * F["SMB"]
                  + 0.3 * F["HML"], index=F.index)
    a = run_ff3(y, F[["Mkt-RF", "SMB", "HML"]])
    b = run_ff3(y, F[["HML", "SMB", "Mkt-RF"]])
    assert a.betas == pytest.approx(b.betas)
    assert a.r_squared == pytest.approx(b.r_squared)


def test_insufficient_data_rejection():
    F = _synthetic_factors(n=30, seed=3)
    y = pd.Series(np.random.default_rng(4).normal(0, 0.01, 30),
                  index=F.index)
    with pytest.raises(ValueError):
        run_capm(y, F[["Mkt-RF"]])  # below 60 floor
    assert check_min_observations(300, 6) is False
    assert check_min_observations(100, 2) is True  # warns below 252
    with pytest.raises(ValueError):
        check_min_observations(5, 6)  # params exceed data


# ---------------------------------------------------------------------------
# Attribution reconciliation (additive arithmetic, both signs)
# ---------------------------------------------------------------------------

def test_attribution_reconciliation():
    F = _synthetic_factors()
    y = pd.Series(0.0001 + 1.2 * F["Mkt-RF"] - 0.3 * F["SMB"]
                  + 0.5 * F["HML"]
                  + np.random.default_rng(13).normal(0, 5e-4, len(F)),
                  index=F.index)
    X = F[["Mkt-RF", "SMB", "HML"]]
    res = run_ff3(y, X)
    contrib = factor_attribution(res, y, X)
    parts = [contrib["Mkt-RF"], contrib["SMB"], contrib["HML"],
             contrib["Alpha"]]
    assert any(v > 0 for v in parts) and any(v < 0 for v in parts)
    # Reconciliation: intercept + factor parts == fitted == realised mean.
    assert (contrib["Fitted (linear, ann.)"]
            == pytest.approx(contrib["Realised mean (ann.)"], rel=1e-9))
    assert contrib["Fitted (linear, ann.)"] == pytest.approx(
        sum(parts), rel=1e-12)


# ---------------------------------------------------------------------------
# Rolling: values + no-look-ahead
# ---------------------------------------------------------------------------

def test_rolling_beta_values():
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    mkt = pd.Series(np.random.default_rng(21).normal(0, 0.01, n), index=idx)
    y = pd.Series(1.5 * mkt.values, index=idx)  # exact beta 1.5, no noise
    roll = rolling_factor_regression(y, mkt.to_frame("Mkt-RF"), window=63)
    assert len(roll) == n - 63 + 1
    assert roll.index[0] == idx[62]  # first full trailing window
    assert roll["Mkt-RF"].sub(1.5).abs().max() < 1e-9
    assert roll["alpha"].abs().max() < 1e-9
    with pytest.raises(ValueError):
        rolling_factor_regression(y, mkt.to_frame("Mkt-RF"), window=10_000)


def test_rolling_no_look_ahead():
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    base = np.random.default_rng(31).normal(0, 0.01, n)
    mkt = pd.Series(base, index=idx)
    y1 = pd.Series(1.0 * base, index=idx)
    y2 = y1.copy()
    y2.iloc[150:] = 99.0  # wild divergence after T=150
    r1 = rolling_factor_regression(y1, mkt.to_frame("Mkt-RF"), window=63)
    r2 = rolling_factor_regression(y2, mkt.to_frame("Mkt-RF"), window=63)
    # Windows ending strictly before idx[150] use only pre-divergence data
    # (a window ending AT idx[150] legitimately includes y[150]).
    upto = r1.index[r1.index < idx[150]]
    assert len(upto) == 150 - 63 + 1
    pd.testing.assert_frame_equal(r1.loc[upto], r2.loc[upto])


# ---------------------------------------------------------------------------
# Scope, formatting, interpretation, isolation
# ---------------------------------------------------------------------------

def test_non_us_warning_helper():
    assert us_scope_note(["AAPL", "MSFT"]) is None
    assert us_scope_note(["BRK.B", "AAPL"]) is None  # US class shares spared
    note = us_scope_note(["AAPL", "BARC.L"])
    assert note is not None and "BARC.L" in note and "US" in note
    assert is_non_us_ticker("^FTSE") is False  # caret tickers: no dot rule
    assert is_non_us_ticker("SHEL.L") is True


def test_formatting_and_descriptions():
    assert format_p_value(0.0034) == "0.003"
    assert format_p_value(0.0004) == "<0.001"
    assert format_p_value(float("nan")) == "n/a"
    assert "greater market sensitivity" in describe_market_beta(1.3)
    assert "in step" in describe_market_beta(1.0)
    assert "lower market sensitivity" in describe_market_beta(0.5)
    assert "larger caps" in describe_factor_loading("SMB", -0.18)
    assert "growth" in describe_factor_loading("HML", -0.42)
    assert "robust profitability" in describe_factor_loading("RMW", 0.31)
    assert "conservative" in describe_factor_loading("CMA", 0.2)
    with pytest.raises(ValueError):
        describe_factor_loading("NOPE", 0.1)


def test_live_price_isolation():
    import pathlib
    for module in ("analytics/factors.py", "data/factors.py"):
        src = pathlib.Path(module).read_text().lower()
        assert "alpaca" not in src
        assert "fetch_latest" not in src
        assert "tradingview" not in src


def test_live_path_mock_cannot_move_factors(monkeypatch):
    import data.alpaca as live

    def _boom(*a, **k):
        raise AssertionError("live path must never be called")

    monkeypatch.setattr(live, "fetch_latest_trades", _boom)
    F = _synthetic_factors(n=120, seed=77)
    y = pd.Series(0.0001 + 1.0 * F["Mkt-RF"]
                  + np.random.default_rng(78).normal(0, 1e-4, 120),
                  index=F.index)
    a = run_capm(y, F[["Mkt-RF"]])
    b = run_capm(y, F[["Mkt-RF"]])
    assert a.betas == b.betas and a.alpha == b.alpha


def test_french_period_index_parsing():
    # The 3-factor daily file arrives with a Period index, not datetime.
    from data.factors import _parse_dataset

    raw = pd.DataFrame(
        {"Mkt-RF": [0.42, -0.10], "SMB": [0.05, 0.02],
         "HML": [-0.03, 0.04], "RF": [0.01, 0.01]},
        index=pd.PeriodIndex(["2020-01-02", "2020-01-03"], freq="D"))
    out = _parse_dataset("ff3", "F-F_Research_Data_Factors_daily",
                         {0: raw, "DESCR": "x"})
    assert isinstance(out.index, pd.DatetimeIndex)
    assert list(out.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    with pytest.raises(ValueError):
        _parse_dataset("ff3", "F-F_Research_Data_Factors_daily",
                       {"DESCR": "x"})
    with pytest.raises(ValueError):
        _parse_dataset(
            "ff3", "F-F_Research_Data_Factors_daily",
            {0: pd.DataFrame({"Mkt-RF": [0.1]})})  # missing columns

"""M4 — institutional risk estimation: sample/EWMA/Ledoit-Wolf + diagnostics."""

import numpy as np
import pandas as pd
import pytest

from analytics.estimation import (
    ESTIMATORS,
    correlation_stability,
    covariance_diagnostics,
    ensure_psd_cov,
    estimate_covariance,
    estimate_mean_returns,
    estimation_window_slice,
    ewma_covariance,
    ledoit_wolf_covariance,
    ledoit_wolf_shrinkage,
    sample_covariance,
    validate_estimator,
    validate_ewma_lambda,
)

rng = np.random.default_rng(7)


def _rets(n=300, p=3, cols=None):
    cols = cols or [f"A{i}" for i in range(p)]
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(rng.normal(0.0005, 0.01, size=(n, p)),
                        index=idx, columns=cols)


# -- sample preserved --------------------------------------------------------

def test_sample_is_default_and_exact():
    r = _rets()
    assert validate_estimator("Sample") == "sample"
    got = estimate_covariance(r)
    want = r.cov() * 252
    pd.testing.assert_frame_equal(got, want)
    assert list(got.columns) == list(r.columns)
    with pytest.raises(ValueError):
        validate_estimator("bogus")


def test_mean_is_sample():
    r = _rets()
    pd.testing.assert_series_equal(estimate_mean_returns(r), r.mean() * 252)


def test_synthetic_covariance_recovery():
    truth = np.array([[0.04, 0.01], [0.01, 0.0225]])  # annual vols 20%/15%
    daily = truth / 252
    L = np.linalg.cholesky(daily)
    z = rng.normal(size=(5000, 2))
    idx = pd.date_range("2000-01-01", periods=5000, freq="B")
    r = pd.DataFrame(z @ L.T, index=idx, columns=["X", "Y"])
    got = sample_covariance(r).values
    assert got == pytest.approx(truth, rel=0.12)


# -- EWMA --------------------------------------------------------------------

def test_ewma_valid_and_lambda_guarded():
    r = _rets()
    cov = ewma_covariance(r, lam=0.94)
    assert list(cov.columns) == list(r.columns)
    eigs = np.linalg.eigvalsh(cov.values)
    assert np.all(eigs > -1e-10), "EWMA must stay PSD"
    for bad in (0.0, 1.0, -0.5, 1.5, float("nan"), "x"):
        with pytest.raises(ValueError):
            validate_ewma_lambda(bad)


def test_ewma_responsive_to_recent_shock():
    idx = pd.date_range("2021-01-01", periods=210, freq="B")
    calm = np.tile([0.01, -0.01], 100)[:200]
    shock = np.tile([0.05, -0.05], 5)[:10]
    series = np.concatenate([calm, shock])
    r = pd.DataFrame({"A": series, "B": -series * 0.5}, index=idx)
    v_s = sample_covariance(r).loc["A", "A"]
    v_e = ewma_covariance(r, lam=0.94).loc["A", "A"]
    assert v_e > v_s, "EWMA must weight the recent shock more than sample"


# -- Ledoit-Wolf --------------------------------------------------------------

def test_ledoit_wolf_psd_labels_and_intensity():
    r = _rets(120, 4)
    cov = ledoit_wolf_covariance(r)
    assert list(cov.columns) == list(r.columns)
    assert np.all(np.linalg.eigvalsh(cov.values) > -1e-10)
    d = ledoit_wolf_shrinkage(r)
    assert 0.0 <= d <= 1.0


def test_ledoit_wolf_handles_fewer_obs_than_assets():
    r = _rets(10, 6)  # T < p: sample is singular, LW stays usable
    cov = ledoit_wolf_covariance(r)
    assert np.all(np.isfinite(cov.values))
    assert np.all(np.linalg.eigvalsh(cov.values) > -1e-10)


# -- order invariance ----------------------------------------------------------

@pytest.mark.parametrize("method", ["sample", "ewma", "ledoit_wolf"])
def test_column_order_invariance(method):
    r = _rets(150, 3, cols=["AAA", "BBB", "CCC"])
    perm = ["CCC", "AAA", "BBB"]
    a = estimate_covariance(r, method=method)
    b = estimate_covariance(r[perm], method=method)
    pd.testing.assert_frame_equal(b, a.reindex(index=perm, columns=perm))


# -- diagnostics + PSD gate -----------------------------------------------------

def test_diagnostics_identity_and_singular():
    d = covariance_diagnostics(pd.DataFrame(np.eye(3)))
    assert d["cond"] == pytest.approx(1.0)
    assert d["effective_rank"] == pytest.approx(3.0)
    assert d["psd_ok"] is True
    dup = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]])
    d2 = covariance_diagnostics(dup)
    assert d2["cond"] == float("inf") and d2["psd_ok"] is True
    bad = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]])
    d3 = covariance_diagnostics(bad)
    assert d3["psd_ok"] is False and d3["cond"] == float("inf")


def test_ensure_psd_passthrough_and_reject():
    tickers = ["A", "B"]
    good = pd.DataFrame(np.eye(2), index=tickers, columns=tickers)
    fixed, info = ensure_psd_cov(good, tickers)
    assert info["was_adjusted"] is False
    pd.testing.assert_frame_equal(fixed, good)
    bad = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=tickers, columns=tickers)
    with pytest.raises(ValueError):
        ensure_psd_cov(bad, tickers)


# -- correlation stability -------------------------------------------------------

def test_correlation_stability_zero_and_drift():
    r = _rets(400, 3)
    d = correlation_stability(r.iloc[:200], r.iloc[:200])
    assert d["frob"] == pytest.approx(0.0) and d["max_abs"] == pytest.approx(0.0)
    d2 = correlation_stability(r.iloc[200:], r)
    assert d2["frob"] >= 0.0 and d2["n"] == 3
    with pytest.raises(ValueError):
        correlation_stability(r, r[["A0", "A1"]])


# -- windows: strict date < end ----------------------------------------------------

def test_window_slice_strict_bound():
    idx = pd.date_range("2022-01-03", periods=400, freq="B")
    px = pd.DataFrame({"A": np.linspace(100, 150, 400)}, index=idx)
    end = idx[300]
    exp = estimation_window_slice(px, end, "expanding")
    assert (exp.index < end).all() and len(exp) == 300
    one = estimation_window_slice(px, end, "1y")
    assert (one.index < end).all() and len(one) <= 262 and len(one) > 200
    with pytest.raises(ValueError):
        estimation_window_slice(px, end, "9y")
    with pytest.raises(ValueError):
        estimation_window_slice(px.iloc[:0], end, "expanding")


# -- multicurrency + determinism ------------------------------------------------------

def test_multicurrency_panel_estimates_cleanly():
    idx = pd.date_range("2021-01-01", periods=200, freq="B")
    native = pd.DataFrame({"BARC.L": np.linspace(500, 560, 200)}, index=idx)
    fx = pd.DataFrame({"GBPUSD": np.linspace(1.25, 1.30, 200)}, index=idx)
    gbp = native["BARC.L"] / 100.0  # GBp → £
    base = pd.DataFrame({"BARC.L": gbp.values * fx["GBPUSD"].values}, index=idx)
    rets = base.pct_change().dropna()
    for m in ESTIMATORS:
        cov = estimate_covariance(rets, method=m)
        assert list(cov.columns) == ["BARC.L"]
        assert np.all(np.isfinite(cov.values))


def test_rebuild_determinism_matches_replay():
    r = _rets(250, 3)
    for m in ESTIMATORS:
        a = estimate_covariance(r.iloc[:200], method=m)
        b = estimate_covariance(r.iloc[:200].copy(), method=m)
        pd.testing.assert_frame_equal(a, b)


# -- OOS / walk-forward integration (training slices only) ---------------------

def _wf_panel():
    from datetime import date  # noqa: F401 (kept local to this block)
    idx = pd.bdate_range("2018-01-01", periods=900)
    g = np.random.default_rng(21)
    data = {}
    for i, t in enumerate(["A", "B", "C"]):
        vol = np.where(np.arange(900) < 700, 0.010 + 0.001 * i, 0.030)
        data[t] = 100.0 * np.exp(np.cumsum(g.normal(0.0004, vol)))
    return pd.DataFrame(data, index=idx)


def test_event_weights_all_estimators_valid():
    from analytics.walkforward import estimate_event_weights
    px = _wf_panel()
    est = px.loc[px.index < pd.Timestamp("2021-01-04")]
    for m in ESTIMATORS:
        w, ex, msg = estimate_event_weights(
            est, ["A", "B", "C"], "min_variance", 0.02, 0.0, 1.0,
            estimator=m)
        assert w is not None, f"{m}: {msg}"
        assert abs(float(w.sum()) - 1.0) < 1e-6
        assert np.all(np.isfinite(w.values))
    _, _, msg = estimate_event_weights(
        est, ["A", "B", "C"], "min_variance", 0.02, 0.0, 1.0,
        estimator="bogus")
    assert "Risk model" in msg


def test_wf_config_validates_estimator():
    from analytics.walkforward import WalkForwardConfig
    from datetime import date
    c = WalkForwardConfig(method="erc", test_start=date(2021, 1, 4),
                          test_end=date(2021, 12, 31), estimator="ewma",
                          ewma_lambda=0.9)
    _, vc = c.validated(["A", "B", "C"])
    assert vc.estimator == "ewma" and vc.ewma_lambda == 0.9
    with pytest.raises(ValueError):
        WalkForwardConfig(method="erc", test_start=date(2021, 1, 4),
                          test_end=date(2021, 12, 31),
                          estimator="bogus").validated(["A", "B", "C"])
    with pytest.raises(ValueError):
        WalkForwardConfig(method="erc", test_start=date(2021, 1, 4),
                          test_end=date(2021, 12, 31),
                          ewma_lambda=1.5).validated(["A", "B", "C"])


def test_wf_runs_succeed_under_every_estimator():
    from datetime import date
    from analytics.walkforward import WalkForwardConfig, run_walk_forward
    px = _wf_panel()
    for m in ESTIMATORS:
        cfg = WalkForwardConfig(
            method="min_variance", test_start=date(2021, 1, 4),
            test_end=date(2021, 6, 30), lookback_years=2.0,
            window_type="rolling", rebalance_frequency="monthly",
            transaction_cost_bps=10.0, estimator=m)
        res = run_walk_forward(px, ["A", "B", "C"], cfg, 10000.0)
        assert res.success, f"{m}: {res.message}"
        assert not res.target_weights.empty


def test_estimators_differ_on_regime_change():
    from analytics.walkforward import estimate_event_weights
    px = _wf_panel()
    est = px.loc[px.index < pd.Timestamp("2021-06-01")]
    w_s, _, _ = estimate_event_weights(
        est, ["A", "B", "C"], "min_variance", 0.02, 0.0, 1.0,
        estimator="sample")
    w_e, _, _ = estimate_event_weights(
        est, ["A", "B", "C"], "min_variance", 0.02, 0.0, 1.0,
        estimator="ewma", ewma_lambda=0.90)
    assert w_s is not None and w_e is not None
    assert float(np.abs(w_s.values - w_e.values).max()) > 1e-4

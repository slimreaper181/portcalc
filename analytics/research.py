"""
research.py
-----------
Persistent research workspace engine (Task 10, Milestone 8) — pure logic.

Saved analyses (types: scenario / optimisation / backtest / oos /
walkforward / robustness / stress) persist named configurations plus key
result summaries in the Milestone-1 ``saved_analyses`` table (never giant
simulation arrays). This module defines the JSON-safe config/result
schemas, generic comparison (compatible types only), recursive config
differences, and per-type re-run against caller-supplied current panels —
so Saved Result vs Recomputed Result are always distinguished and the real
ledger is never touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .validation import canonical_tickers

#: Stored analysis types (match storage.repositories._ANALYSIS_TYPES).
ANALYSIS_TYPES = ("scenario", "optimisation", "backtest", "oos",
                  "walkforward", "robustness", "stress")

ANALYSIS_LABELS = {
    "scenario": "Scenario",
    "optimisation": "Optimisation",
    "backtest": "Static Backtest",
    "oos": "OOS",
    "walkforward": "Walk-Forward",
    "robustness": "WF Robustness",
    "stress": "Stress Test",
}


def validate_analysis_type(atype: str) -> str:
    """Normalise a saved-analysis type or raise."""
    a = str(atype or "").strip().lower()
    if a not in ANALYSIS_TYPES:
        raise ValueError(f"Unknown analysis type {atype!r}.")
    return a


def check_json_safe(payload: dict, label: str) -> dict:
    """Require a JSON-safe dict (round-trip verified)."""
    import json

    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a dict.")
    try:
        return json.loads(json.dumps(payload, sort_keys=True))
    except (TypeError, ValueError) as e:
        raise ValueError(f"{label} must be JSON-safe: {e}")


def parse_tags(raw: str) -> list[str]:
    """Comma-separated tags → cleaned list (order-preserving, de-duplicated)."""
    seen: list[str] = []
    for part in str(raw or "").split(","):
        tag = part.strip()
        if tag and tag not in seen:
            seen.append(tag)
    return seen


# ---------------------------------------------------------------------------
# Config differences (recursive, path-labelled)
# ---------------------------------------------------------------------------

def _fmt_scalar(value) -> str:
    if isinstance(value, float):
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.6g}"
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "…"


def flatten_diff(a, b, prefix: str = "") -> list[tuple[str, str, str]]:
    """Recursive (path, a, b) differences between JSON-safe structures."""
    rows: list[tuple[str, str, str]] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in a:
                rows.append((path, "—", _fmt_scalar(b[key])))
            elif key not in b:
                rows.append((path, _fmt_scalar(a[key]), "—"))
            else:
                rows.extend(flatten_diff(a[key], b[key], path))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            rows.append((prefix or "(list)",
                         f"{len(a)} items", f"{len(b)} items"))
        for i, (x, y) in enumerate(zip(a, b)):
            rows.extend(flatten_diff(x, y, f"{prefix}[{i}]"))
    elif a != b:
        rows.append((prefix or "(value)", _fmt_scalar(a), _fmt_scalar(b)))
    return rows


def config_diff(config_a: dict, config_b: dict) -> pd.DataFrame:
    """Exactly what differs between two saved configurations."""
    rows = flatten_diff(dict(config_a or {}), dict(config_b or {}))
    return pd.DataFrame(rows, columns=["Setting", "A", "B"])


# ---------------------------------------------------------------------------
# Result comparison (compatible types only)
# ---------------------------------------------------------------------------

def check_compatible(type_a: str, type_b: str) -> str:
    """Analyses compare only within one type (raise otherwise)."""
    a, b = validate_analysis_type(type_a), validate_analysis_type(type_b)
    if a != b:
        raise ValueError(
            f"Incompatible analyses ({ANALYSIS_LABELS[a]} vs "
            f"{ANALYSIS_LABELS[b]}): compare within one type.")
    return a


def _numeric_metrics(result: dict) -> dict[str, float]:
    metrics = (result or {}).get("metrics", result or {})
    out: dict[str, float] = {}
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(f):
                out[str(k)] = f
    return out


def results_comparison(result_a: dict, result_b: dict) -> pd.DataFrame:
    """Side-by-side numeric metrics with deltas (union of keys)."""
    ma, mb = _numeric_metrics(result_a), _numeric_metrics(result_b)
    rows = []
    for key in sorted(set(ma) | set(mb)):
        va, vb = ma.get(key, float("nan")), mb.get(key, float("nan"))
        delta = (vb - va if np.isfinite(va) and np.isfinite(vb)
                 else float("nan"))
        rows.append({"Metric": key, "A": va, "B": vb, "Δ (B−A)": delta})
    return pd.DataFrame(rows, columns=["Metric", "A", "B", "Δ (B−A)"])


def summarise_result(result: dict) -> pd.DataFrame:
    """Saved-result summary rows for the library detail view."""
    result = dict(result or {})
    rows = []
    for k, v in _numeric_metrics(result).items():
        rows.append({"Metric": k, "Value": v})
    weights = result.get("weights")
    if isinstance(weights, dict):
        for t, w in weights.items():
            try:
                rows.append({"Metric": f"weight {t}", "Value": float(w)})
            except (TypeError, ValueError):
                continue
    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ---------------------------------------------------------------------------
# Re-run (per-type recompute on caller-supplied current panels)
# ---------------------------------------------------------------------------

def _panel_from(data: dict, key: str = "prices") -> pd.DataFrame:
    panel = (data or {}).get(key)
    if panel is None or not isinstance(panel, pd.DataFrame) or panel.empty:
        raise ValueError(f"Re-run needs a current price panel ('{key}').")
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise ValueError("Re-run panel needs a DatetimeIndex.")
    return panel


def _tickers_of(config: dict) -> list[str]:
    tickers = canonical_tickers(list(config.get("tickers") or []))
    if not tickers:
        raise ValueError("Re-run needs saved tickers.")
    return tickers


def _rerun_optimisation(config: dict, data: dict) -> dict:
    from .allocation import (
        equal_risk_contribution,
        maximum_diversification,
        simple_returns_from_prices,
    )
    from .estimation import (
        estimate_covariance,
        estimate_mean_returns,
        estimation_window_slice,
    )
    from .optimisation import max_sharpe, min_variance

    tickers = _tickers_of(config)
    prices = _panel_from(data).reindex(columns=tickers).dropna(how="any")
    if prices.empty:
        raise ValueError("No current price coverage for the saved universe.")
    window = str(config.get("window", "expanding"))
    est = estimation_window_slice(
        prices, prices.index[-1] + pd.Timedelta(days=1), window)
    rets = simple_returns_from_prices(est, tickers)
    mu = estimate_mean_returns(rets)
    cov = estimate_covariance(rets, method=config.get("risk_model", "sample"),
                              ewma_lambda=float(config.get("ewma_lambda", 0.94)))
    rf = float(config.get("rf", 0.05))
    out: dict = {"methods": {}, "risk_model": config.get("risk_model", "sample"),
                 "n_obs": len(rets)}
    for method in (config.get("methods") or ["min_variance"]):
        if method == "min_variance":
            r = min_variance(mu.values, cov.values, rf)
        elif method == "max_sharpe":
            r = max_sharpe(mu.values, cov.values, rf)
        elif method == "erc":
            r = equal_risk_contribution(tickers, mu.values, cov, rf)
        elif method == "max_diversification":
            r = maximum_diversification(tickers, mu.values, cov, rf)
        else:
            continue
        if r is None or not r.success or r.weights is None:
            out["methods"][method] = {"success": False,
                                      "message": getattr(r, "message", "?")}
            continue
        w = (r.weights.values.astype(float) if isinstance(r.weights, pd.Series)
             else np.asarray(r.weights, dtype=float))
        out["methods"][method] = {
            "success": True, "weights": {t: float(x) for t, x in
                                         zip(tickers, w)},
            "metrics": {"expected_return": float(r.expected_return),
                        "volatility": float(r.volatility),
                        "sharpe": float(r.sharpe)}}
    out["metrics"] = {"n_methods": len(out["methods"])}
    return out


def _rerun_walkforward(config: dict, data: dict) -> dict:
    from datetime import date as _date

    from .walkforward import WalkForwardConfig, run_walk_forward

    tickers = _tickers_of(config)
    panel = _panel_from(data, "base_panel")
    out: dict = {"methods": {}}
    for method in (config.get("methods") or ["erc"]):
        cfg = WalkForwardConfig(
            method=method, test_start=config["test_start"],
            test_end=config["test_end"],
            lookback_years=float(config.get("lookback_years", 3.0)),
            window_type=str(config.get("window_type", "rolling")),
            rebalance_frequency=str(config.get("frequency", "quarterly")),
            transaction_cost_bps=float(config.get("costs_bps", 10.0)),
            base_currency=str(config.get("base_ccy", "USD")),
            min_weight=float(config.get("min_w", 0.0)),
            max_weight=float(config.get("max_w", 1.0)),
            risk_free_annual=float(config.get("rf", 0.05)),
            estimator=str(config.get("estimator", "sample")),
            ewma_lambda=float(config.get("ewma_lambda", 0.94)))
        res = run_walk_forward(panel, tickers, cfg,
                               float(config.get("capital", 10000.0)))
        if not res.success:
            out["methods"][method] = {"success": False, "message": res.message}
            continue
        m = res.metrics
        out["methods"][method] = {
            "success": True,
            "metrics": {k: float(m[k]) for k in
                        ("final_value", "cagr", "sharpe",
                         "annualised_volatility", "max_drawdown",
                         "total_costs") if np.isfinite(float(m.get(k, np.nan)))},
            "n_failed": int(m.get("n_failed", 0))}
    ok = [v for v in out["methods"].values() if v.get("success")]
    out["metrics"] = {"n_methods": len(ok)}
    return out


def _rerun_robustness(config: dict, data: dict) -> dict:
    from .robustness import RobustnessConfig, run_robustness_grid

    tickers = _tickers_of(config)
    panel = _panel_from(data, "base_panel")
    cfg = RobustnessConfig(
        methods=tuple(config.get("methods", ("min_variance",))),
        lookbacks=tuple(float(x) for x in config.get("lookbacks", (3.0,))),
        frequencies=tuple(config.get("frequencies", ("quarterly",))),
        estimators=tuple(config.get("estimators", ("sample",))),
        costs_bps=tuple(float(c) for c in config.get("costs_bps", (10.0,))),
        window_type=str(config.get("window_type", "rolling")),
        test_start=config["test_start"], test_end=config["test_end"],
        base_currency=str(config.get("base_ccy", "USD")),
        risk_free_annual=float(config.get("rf", 0.05)))
    res = run_robustness_grid(panel, tickers, cfg,
                              float(config.get("capital", 10000.0)))
    return {"metrics": {"n_cells": len(res.cells),
                        "n_ok": sum(1 for c in res.cells if c.success)},
            "scorecard": res.scorecard.to_dict(orient="records")}


def _rerun_oos(config: dict, data: dict) -> dict:
    from .allocation import (
        equal_risk_contribution,
        maximum_diversification,
        simple_returns_from_prices,
    )
    from .backtest import backtest_rebalanced, prepare_backtest_data
    from .estimation import estimate_covariance, estimate_mean_returns
    from .optimisation import max_sharpe, min_variance

    tickers = _tickers_of(config)
    panel = _panel_from(data, "base_panel")
    t0 = pd.Timestamp(config["test_start"])
    lb = float(config.get("lookback_years", 3.0))
    est = panel.loc[(panel.index >= t0 - pd.DateOffset(years=lb))
                    & (panel.index < t0)].dropna(how="any")
    if est.empty:
        raise ValueError("No estimation data strictly before test start.")
    rets = simple_returns_from_prices(est, tickers)
    mu = estimate_mean_returns(rets)
    cov = estimate_covariance(rets, method=config.get("estimator", "sample"))
    rf = float(config.get("rf", 0.05))
    weights: dict[str, np.ndarray] = {}
    for method in (config.get("methods") or ["min_variance"]):
        if method == "equal_weight":
            weights[method] = np.ones(len(tickers)) / len(tickers)
        elif method == "min_variance":
            r = min_variance(mu.values, cov.values, rf)
            if r.success and r.weights is not None:
                weights[method] = np.asarray(r.weights, dtype=float)
        elif method == "max_sharpe":
            r = max_sharpe(mu.values, cov.values, rf)
            if r.success and r.weights is not None:
                weights[method] = np.asarray(r.weights, dtype=float)
        elif method == "erc":
            r = equal_risk_contribution(tickers, mu.values, cov, rf)
            if r.success and r.weights is not None:
                weights[method] = r.weights.values.astype(float)
        elif method == "max_diversification":
            r = maximum_diversification(tickers, mu.values, cov, rf)
            if r.success and r.weights is not None:
                weights[method] = r.weights.values.astype(float)
    window, _ = prepare_backtest_data(
        panel, tickers, t0, pd.Timestamp(config["test_end"]))
    out: dict = {"methods": {}}
    for method, w in weights.items():
        res = backtest_rebalanced(
            window, w, float(config.get("capital", 10000.0)),
            str(config.get("frequency", "quarterly")),
            float(config.get("costs_bps", 10.0)), rf, name=method)
        out["methods"][method] = {
            "success": True,
            "metrics": {k: float(res.metrics[k]) for k in
                        ("final_value", "cagr", "sharpe",
                         "annualised_volatility", "max_drawdown")
                        if np.isfinite(float(res.metrics.get(k, np.nan)))}}
    out["metrics"] = {"n_methods": len(weights)}
    return out


def _rerun_backtest(config: dict, data: dict) -> dict:
    from .backtest import backtest_rebalanced, prepare_backtest_data

    tickers = _tickers_of(config)
    panel = _panel_from(data, "base_panel")
    window, _ = prepare_backtest_data(
        panel, tickers, pd.Timestamp(config["start"]),
        pd.Timestamp(config["end"]))
    w = np.asarray(config["weights"], dtype=float)
    if w.shape != (len(tickers),):
        raise ValueError("Saved weights do not match the saved universe.")
    res = backtest_rebalanced(
        window, w, float(config.get("capital", 10000.0)),
        str(config.get("frequency", "quarterly")),
        float(config.get("costs_bps", 10.0)),
        float(config.get("rf", 0.05)), name="rerun")
    return {"metrics": {k: float(res.metrics[k]) for k in
                        ("final_value", "cagr", "sharpe",
                         "annualised_volatility", "max_drawdown",
                         "total_costs")
                        if np.isfinite(float(res.metrics.get(k, np.nan)))},
            "weights": {t: float(x) for t, x in zip(tickers, w)}}


def _rerun_stress(config: dict, data: dict) -> dict:
    from .stress import run_stress

    tickers = _tickers_of(config)
    panel = _panel_from(data, "base_panel")
    w = np.asarray(config["weights"], dtype=float)
    res = run_stress(panel, tickers, w, window=str(config.get("window", "")),
                     custom_start=config.get("custom_start"),
                     custom_end=config.get("custom_end"),
                     initial=float(config.get("capital", 10000.0)),
                     risk_free_annual=float(config.get("rf", 0.05)))
    m = res.get("metrics", {})
    return {"metrics": {k: float(m[k]) for k in
                        ("total_return", "max_drawdown", "worst_day",
                         "volatility")
                        if k in m and np.isfinite(float(m[k]))}}


def _rerun_scenario(config: dict, data: dict) -> dict:
    # Log-return basis, exactly like the Scenario view (ctx mu/cov come
    # from returns.daily_returns → annualised log stats for the GBM sim).
    from .returns import (
        annualised_cov_matrix,
        annualised_mean_returns,
        daily_returns,
    )
    from .scenario import (
        calculate_outcome_distribution,
        run_predefined_scenario,
        simulate_portfolio_paths,
    )

    tickers = _tickers_of(config)
    prices = _panel_from(data).reindex(columns=tickers).dropna(how="any")
    rets = daily_returns(prices).dropna(how="any")
    if rets.empty:
        raise ValueError("No return observations for a scenario re-run.")
    mu = annualised_mean_returns(rets)
    cov = annualised_cov_matrix(rets)
    mu_s, cov_s, _ = run_predefined_scenario(
        mu.values, cov.values, str(config.get("scenario", "Normal")),
        float(config.get("return_shock", 0.0)),
        float(config.get("vol_multiplier", 1.0)))
    w = np.asarray(config["weights"], dtype=float)
    capital = float(config.get("capital", 10000.0))
    paths = simulate_portfolio_paths(
        w, mu_s, cov_s, capital,
        years=float(config.get("horizon_years", 5.0)),
        n_sims=int(config.get("n_sims", 1000)),
        seed=int(config.get("seed", 7)))
    dist = calculate_outcome_distribution(paths, capital)
    loss = dist.get("prob_loss", float("nan"))
    return {"metrics": {
        "median_final": float(dist.get("median_value", float("nan"))),
        "p5_final": float(dist.get("worst_5pct", float("nan"))),
        "p95_final": float(dist.get("best_5pct", float("nan"))),
        "prob_profit": float(1.0 - loss) if np.isfinite(loss)
        else float("nan")}}


_RERUNNERS = {
    "optimisation": _rerun_optimisation,
    "walkforward": _rerun_walkforward,
    "robustness": _rerun_robustness,
    "oos": _rerun_oos,
    "backtest": _rerun_backtest,
    "stress": _rerun_stress,
    "scenario": _rerun_scenario,
}


def rerun_analysis(analysis_type: str, config: dict,
                   data: dict) -> dict:
    """Recompute an analysis on current panels (never mutates stored rows).

    Returns a fresh result dict in the same schema as the saved one —
    callers display Saved vs Recomputed and only overwrite on explicit
    user confirmation.
    """
    atype = validate_analysis_type(analysis_type)
    check_json_safe(dict(config or {}), "config")
    try:
        return _RERUNNERS[atype](dict(config or {}), dict(data or {}))
    except KeyError as e:
        raise ValueError(f"Saved config is missing {e} for a re-run.")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Re-run failed: {e}")

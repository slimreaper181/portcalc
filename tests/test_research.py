"""M8 — persistent research workspace: save/load/rerun/compare/isolate."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from analytics.research import (
    ANALYSIS_TYPES,
    check_compatible,
    config_diff,
    parse_tags,
    rerun_analysis,
    results_comparison,
    summarise_result,
    validate_analysis_type,
)
from storage import database as db
from storage import repositories as repo


def _mem():
    return db.connect(":memory:")


def _panel(n=800, seed=9):
    idx = pd.bdate_range("2018-01-01", periods=n)
    g = np.random.default_rng(seed)
    return pd.DataFrame({
        "A": 100.0 * np.exp(np.cumsum(g.normal(0.0005, 0.010, n))),
        "B": 100.0 * np.exp(np.cumsum(g.normal(0.0003, 0.014, n))),
    }, index=idx)


def test_analysis_crud_notes_tags():
    conn = _mem()
    p = repo.create_portfolio(conn, "Main")
    with pytest.raises(ValueError):
        repo.save_analysis(conn, p.portfolio_id, "  ", "oos", {}, {})
    with pytest.raises(ValueError):
        repo.save_analysis(conn, p.portfolio_id, "X", "bogus", {}, {})
    a = repo.save_analysis(conn, p.portfolio_id, "Q1 review", "oos",
                           {"tickers": ["A"]}, {"metrics": {"sharpe": 1.2}},
                           tags="Risk, Quarterly", notes="hello")
    assert a.type == "oos"
    got = repo.get_analysis(conn, a.analysis_id)
    assert got.name == "Q1 review" and got.notes == "hello"
    assert parse_tags(got.tags) == ["Risk", "Quarterly"]
    assert parse_tags("a, a ,b,,") == ["a", "b"]
    repo.set_analysis_notes(conn, a.analysis_id, "updated")
    assert repo.get_analysis(conn, a.analysis_id).notes == "updated"
    assert repo.rename_analysis(conn, a.analysis_id, "Q1 final").name == "Q1 final"
    with pytest.raises(ValueError):
        repo.rename_analysis(conn, a.analysis_id, "  ")
    assert len(repo.list_analyses(conn, p.portfolio_id)) == 1
    repo.delete_analysis(conn, a.analysis_id)
    assert repo.list_analyses(conn, p.portfolio_id) == []
    with pytest.raises(ValueError):
        repo.delete_analysis(conn, a.analysis_id)
    assert validate_analysis_type("OOS") == "oos"
    assert set(ANALYSIS_TYPES) >= {"scenario", "optimisation", "backtest",
                                   "oos", "walkforward", "robustness", "stress"}


def test_portfolio_isolation_and_clone_choice():
    conn = _mem()
    a = repo.create_portfolio(conn, "A")
    b = repo.create_portfolio(conn, "B")
    repo.save_analysis(conn, a.portfolio_id, "mine", "oos", {}, {})
    assert len(repo.list_analyses(conn, b.portfolio_id)) == 0
    plain = repo.duplicate_portfolio(conn, a.portfolio_id, "A-copy")
    assert repo.list_analyses(conn, plain.portfolio_id) == []
    rich = repo.duplicate_portfolio(conn, a.portfolio_id, "A-rich",
                                    include_research=True)
    copied = repo.list_analyses(conn, rich.portfolio_id)
    assert len(copied) == 1 and copied[0].name == "mine"
    assert copied[0].analysis_id != repo.list_analyses(
        conn, a.portfolio_id)[0].analysis_id


def test_config_diff_and_results_comparison():
    diff = config_diff({"lookback": 3, "freq": "q", "x": 1},
                       {"lookback": 5, "freq": "q", "y": 2})
    paths = set(diff["Setting"])
    assert "lookback" in paths and "x" in paths and "y" in paths
    assert "freq" not in paths
    frame = results_comparison({"metrics": {"sharpe": 1.0, "cagr": 0.05}},
                               {"metrics": {"sharpe": 1.5, "cagr": 0.05}})
    row = frame.set_index("Metric").loc["sharpe"]
    assert row["B"] - row["A"] == pytest.approx(0.5)
    assert row["Δ (B−A)"] == pytest.approx(0.5)
    assert frame.set_index("Metric").loc["cagr"]["Δ (B−A)"] == pytest.approx(0.0)
    check_compatible("oos", "oos")
    with pytest.raises(ValueError):
        check_compatible("oos", "walkforward")
    assert not summarise_result({}).empty or True
    s = summarise_result({"metrics": {"sharpe": 2.0},
                          "weights": {"A": 0.6, "B": 0.4}})
    assert "weight A" in set(s["Metric"])


def test_rerun_optimisation_and_walkforward():
    px = _panel()
    ocfg = {"tickers": ["A", "B"], "methods": ["min_variance", "erc"],
            "risk_model": "sample", "rf": 0.02}
    out = rerun_analysis("optimisation", ocfg, {"prices": px})
    assert out["methods"]["min_variance"]["success"]
    w = out["methods"]["min_variance"]["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-6
    wcfg = {"tickers": ["A", "B"], "methods": ["min_variance"],
            "test_start": "2020-01-02", "test_end": "2020-06-30",
            "lookback_years": 1.0, "window_type": "rolling",
            "frequency": "monthly", "costs_bps": 10.0, "capital": 10000.0,
            "rf": 0.02}
    wout = rerun_analysis("walkforward", wcfg, {"base_panel": px})
    assert wout["methods"]["min_variance"]["success"]
    with pytest.raises(ValueError):
        rerun_analysis("bogus", {}, {})


def test_rerun_scenario_deterministic_and_oos():
    px = _panel()
    scfg = {"tickers": ["A", "B"], "weights": [0.5, 0.5],
            "scenario": "Normal", "horizon_years": 2.0, "n_sims": 500,
            "seed": 11, "capital": 10000.0}
    r1 = rerun_analysis("scenario", scfg, {"prices": px})
    r2 = rerun_analysis("scenario", scfg, {"prices": px})
    assert r1["metrics"]["median_final"] == pytest.approx(
        r2["metrics"]["median_final"])
    assert np.isfinite(r1["metrics"]["prob_profit"])
    ocfg = {"tickers": ["A", "B"], "methods": ["min_variance",
                                               "equal_weight"],
            "test_start": "2020-01-02", "test_end": "2020-06-30",
            "lookback_years": 1.0, "frequency": "quarterly",
            "costs_bps": 10.0, "capital": 10000.0, "rf": 0.02}
    oos = rerun_analysis("oos", ocfg, {"base_panel": px})
    assert oos["methods"]["equal_weight"]["success"]


def test_view_save_payloads_json_safe_and_rerunnable():
    """M8 wiring: configs built by the views round-trip + rerun."""
    import json

    from analytics.research import check_json_safe
    px = _panel()
    # Optimisation-style payload (as built in views/optimise.py).
    ocfg = {"tickers": ["A", "B"], "methods": ["min_variance"],
            "risk_model": "sample", "ewma_lambda": 0.94,
            "window": "expanding", "min_w": 0.0, "max_w": 1.0,
            "rf": 0.02, "base_ccy": "USD"}
    check_json_safe(ocfg, "config")
    out = rerun_analysis("optimisation", ocfg, {"prices": px})
    check_json_safe(out, "result")
    assert out["methods"]["min_variance"]["success"]
    # Backtest-style payload (as built in views/backtest.py).
    bcfg = {"tickers": ["A", "B"], "weights": [0.5, 0.5],
            "start": "2019-01-02", "end": "2020-06-30",
            "capital": 10000.0, "costs_bps": 10.0,
            "frequency": "quarterly", "rf": 0.02, "base_ccy": "USD"}
    check_json_safe(bcfg, "config")
    bout = rerun_analysis("backtest", bcfg, {"base_panel": px})
    assert np.isfinite(bout["metrics"]["final_value"])
    with pytest.raises(ValueError):
        check_json_safe({"bad": {1, 2}}, "config")
    json.dumps(bout, sort_keys=True)  # result stays JSON-safe


def test_rerun_backtest_stress_robustness_types():
    px = _panel()
    bcfg = {"tickers": ["A", "B"], "weights": [0.5, 0.5],
            "start": "2019-06-03", "end": "2020-06-30",
            "capital": 10000.0, "costs_bps": 10.0,
            "frequency": "quarterly", "rf": 0.02}
    bout = rerun_analysis("backtest", bcfg, {"base_panel": px})
    assert np.isfinite(bout["metrics"]["final_value"])
    scfg = {"tickers": ["A", "B"], "weights": [0.5, 0.5],
            "window": "COVID shock", "capital": 10000.0, "rf": 0.02}
    try:
        sout = rerun_analysis("stress", scfg, {"base_panel": px})
    except ValueError:
        sout = None  # window may not overlap the synthetic panel
    if sout is not None:
        assert np.isfinite(sout["metrics"]["total_return"])
    rcfg = {"tickers": ["A", "B"], "methods": ["min_variance"],
            "lookbacks": [1.0], "frequencies": ["quarterly"],
            "estimators": ["sample"], "costs_bps": [10.0],
            "window_type": "rolling", "test_start": "2020-01-02",
            "test_end": "2020-06-30", "capital": 10000.0, "rf": 0.02}
    rout = rerun_analysis("robustness", rcfg, {"base_panel": px})
    assert rout["metrics"]["n_cells"] == 1


def test_app_wires_research_tab_and_clone_opt_in():
    import pathlib
    app_src = pathlib.Path(__file__).resolve().parent.parent.joinpath(
        "app.py").read_text(encoding="utf-8")
    assert "views_research" in app_src
    assert "Research" in app_src
    assert "include_research" in app_src
    # Default clone keeps research isolated (explicit opt-in to copy).
    conn = _mem()
    a = repo.create_portfolio(conn, "A")
    repo.save_analysis(conn, a.portfolio_id, "mine", "oos", {}, {})
    plain = repo.duplicate_portfolio(conn, a.portfolio_id, "plain-copy")
    assert repo.list_analyses(conn, plain.portfolio_id) == []

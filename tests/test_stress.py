"""Tests for analytics.stress (historical windows, contributors)."""

import numpy as np
import pandas as pd
import pytest
from datetime import date

from analytics.stress import STRESS_WINDOWS, resolve_window, run_stress


def _panel():
    idx = pd.bdate_range("2019-01-01", periods=700)
    rng = np.random.default_rng(12)
    return pd.DataFrame({
        "A": 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, 700))),
        "B": 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.010, 700))),
    }, index=idx)


def test_known_windows_resolve():
    assert set(STRESS_WINDOWS) == {"COVID shock (Feb–Mar 2020)",
                                   "2022 rate shock (Jan–Oct 2022)"}
    s, e = resolve_window("COVID shock (Feb–Mar 2020)")
    assert (str(s.date()), str(e.date())) == ("2020-02-19", "2020-03-23")
    s, e = resolve_window("Custom period", date(2021, 1, 4), date(2021, 2, 1))
    assert (str(s.date()), str(e.date())) == ("2021-01-04", "2021-02-01")
    with pytest.raises(ValueError):
        resolve_window("Nope")
    with pytest.raises(ValueError):
        resolve_window("Custom period", date(2021, 2, 1), date(2021, 1, 4))


def test_stress_result_reconciles():
    px = _panel()
    out = run_stress(px, ["A", "B"], np.array([0.6, 0.4]),
                     window="Custom period",
                     custom_start=date(2020, 1, 2),
                     custom_end=date(2020, 6, 30), initial=10000.0)
    assert out["actual_start"] == date(2020, 1, 2)
    assert out["n_days"] == len(out["values"])
    assert out["contributions"]["contribution"].sum() == pytest.approx(
        out["total_return"], rel=1e-9)
    assert out["max_drawdown"] <= 0
    assert "current-composition" in out["method"].lower()
    assert out["benchmark"] is None


def test_stress_with_benchmark_and_clipping():
    px = _panel()
    bench = px["A"] * 1.0
    out = run_stress(px, ["A", "B"], np.array([0.5, 0.5]),
                     window="COVID shock (Feb–Mar 2020)",
                     benchmark_base=bench, benchmark_name="MKT")
    assert out["benchmark"] is not None
    assert out["benchmark"].values.iloc[0] == pytest.approx(10000.0)
    # Request starting before data but ending inside it → clipped with
    # disclosure, not invented bars. Fully outside data still raises.
    out2 = run_stress(px, ["A", "B"], np.array([0.5, 0.5]),
                      window="Custom period", custom_start=date(2018, 6, 1),
                      custom_end=date(2019, 6, 1))
    assert any("clipped" in n for n in out2["notes"])
    assert out2["actual_start"] == date(2019, 1, 1)
    with pytest.raises(ValueError, match="No overlapping"):
        run_stress(px, ["A", "B"], np.array([0.5, 0.5]),
                   window="Custom period", custom_start=date(2018, 1, 1),
                   custom_end=date(2018, 6, 1))
    with pytest.raises(ValueError):
        run_stress(px[["A"]], ["A", "B"], np.array([0.5, 0.5]),
                   window="Custom period", custom_start=date(2020, 1, 2),
                   custom_end=date(2020, 2, 1))

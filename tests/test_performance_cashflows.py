"""Tests for analytics.performance_cashflows (TWR, XIRR, flow benchmark)."""

from datetime import date

import pytest

from analytics.performance_cashflows import (
    benchmark_with_flows,
    time_weighted_return,
    xirr,
)


def test_twr_no_flows_is_simple_return():
    vals = [(date(2026, 1, 1), 10000.0), (date(2026, 7, 1), 11000.0)]
    assert time_weighted_return(vals, [])["twr"] == pytest.approx(0.10)


def test_twr_deposit_not_counted_as_return():
    # +50% then deposit doubling capital, flat after: TWR stays 50%.
    vals = [(date(2026, 1, 1), 10000.0), (date(2026, 4, 1), 15000.0),
            (date(2026, 7, 1), 30000.0)]
    flows = [(date(2026, 4, 1), 15000.0)]
    assert time_weighted_return(vals, flows)["twr"] == pytest.approx(0.50)


def test_twr_links_sub_periods():
    vals = [(date(2026, 1, 1), 100.0), (date(2026, 2, 1), 120.0),
            (date(2026, 3, 1), 90.0)]
    assert time_weighted_return(vals, [])["twr"] == pytest.approx(1.2 * 0.75 - 1)


def test_twr_rejects_bad_inputs():
    with pytest.raises(ValueError):
        time_weighted_return([(date(2026, 1, 1), 100.0)], [])
    with pytest.raises(ValueError):
        time_weighted_return(
            [(date(2026, 1, 1), 100.0), (date(2026, 2, 1), -5.0)], [])


def test_xirr_known_example():
    # Invest 1000, worth 1200 after 365 days; engine annualises on /365.25.
    flows = [(date(2025, 1, 1), -1000.0), (date(2026, 1, 1), 1200.0)]
    expect = (1200.0 / 1000.0) ** (365.25 / 365.0) - 1.0
    assert xirr(flows)["xirr"] == pytest.approx(expect, rel=1e-6)


def _npv(flows, r):
    t0 = min(d for d, _ in flows).toordinal()
    return sum(a / ((1.0 + r) ** ((d.toordinal() - t0) / 365.25))
               for d, a in flows)


def test_xirr_multiple_flows_against_closed_form():
    # -1000 then -1000 a year apart, 2500 after one more year.
    flows = [(date(2024, 1, 1), -1000.0), (date(2025, 1, 1), -1000.0),
             (date(2026, 1, 1), 2500.0)]
    r = xirr(flows)["xirr"]
    assert 0.10 < r < 0.20  # sane bracket for ~13-16% annualised
    assert _npv(flows, r) == pytest.approx(0.0, abs=1e-4)  # exact root


def test_xirr_structured_failure_single_signed():
    with pytest.raises(ValueError, match="both negative and positive"):
        xirr([(date(2025, 1, 1), -100.0), (date(2026, 1, 1), -50.0)])
    with pytest.raises(ValueError):
        xirr([(date(2025, 1, 1), -100.0)])


def test_benchmark_with_flows_matches_deposits():
    px = {date(2026, 1, 1): 100.0, date(2026, 2, 1): 110.0,
          date(2026, 3, 1): 121.0}
    out = benchmark_with_flows(px, [(date(2026, 2, 1), 1100.0)], 10000.0)
    by_day = dict(out)
    assert by_day[date(2026, 1, 1)] == pytest.approx(10000.0)
    # 100 units + 10 units bought 2/1 → 110 units × 121 on 3/1.
    assert by_day[date(2026, 3, 1)] == pytest.approx(110 * 121.0)
    assert by_day[date(2026, 2, 1)] == pytest.approx(110 * 110.0)
    with pytest.raises(ValueError):
        benchmark_with_flows({}, [], 100.0)

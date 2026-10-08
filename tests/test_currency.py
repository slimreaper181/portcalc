"""Tests for analytics.currency (pure FX/conversion math, no network)."""

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from analytics.currency import (
    align_fx_to_index,
    check_weights_sum,
    convert_price_series,
    convert_value,
    currency_symbol,
    decompose_pnl,
    freshness_report,
    fx_rate_between,
    fx_shock_portfolio,
    quote_age_days,
    rate_on_date,
    validate_base_currency,
    validate_return_series,
)

USD_PER = {"USD": 1.0, "GBP": 1.20, "EUR": 1.10}


def test_base_currency_and_symbols():
    assert validate_base_currency("gbp") == "GBP"
    with pytest.raises(ValueError):
        validate_base_currency("JPY")
    assert currency_symbol("USD") == "$"
    assert currency_symbol("GBP") == "£"
    assert currency_symbol("EUR") == "€"


def test_identities_need_no_data():
    assert fx_rate_between("USD", "USD", {}) == 1.0
    assert fx_rate_between("GBP", "GBP", {}) == 1.0
    assert fx_rate_between("EUR", "EUR", {}) == 1.0
    assert convert_value(100.0, "USD", "USD", {}) == pytest.approx(100.0)


def test_routing_and_reciprocity():
    # GBP->USD at 1.20; USD->GBP reciprocal; EUR->GBP via triangulation.
    assert fx_rate_between("GBP", "USD", USD_PER) == pytest.approx(1.20)
    assert fx_rate_between("USD", "GBP", USD_PER) == pytest.approx(1 / 1.20)
    assert fx_rate_between("EUR", "USD", USD_PER) == pytest.approx(1.10)
    assert fx_rate_between("GBP", "EUR", USD_PER) == pytest.approx(1.20 / 1.10)
    assert fx_rate_between("EUR", "GBP", USD_PER) == pytest.approx(1.10 / 1.20)
    # Reciprocity property: r(a->b) * r(b->a) == 1.
    for a in ("USD", "GBP", "EUR"):
        for b in ("USD", "GBP", "EUR"):
            assert fx_rate_between(a, b, USD_PER) * fx_rate_between(
                b, a, USD_PER) == pytest.approx(1.0)
    # Triangulation property: GBP->EUR == GBP->USD->EUR.
    assert fx_rate_between("GBP", "EUR", USD_PER) == pytest.approx(
        fx_rate_between("GBP", "USD", USD_PER)
        * fx_rate_between("USD", "EUR", USD_PER))
    with pytest.raises(ValueError):
        fx_rate_between("GBP", "USD", {"USD": 1.0})  # missing leg, never 1
    with pytest.raises(ValueError):
        fx_rate_between("GBP", "USD", {"GBP": -1.0, "USD": 1.0})
    with pytest.raises(ValueError):
        convert_value(float("nan"), "GBP", "USD", USD_PER)


def test_convert_value_example():
    # £100 at 1.20 → $120.
    assert convert_value(100.0, "GBP", "USD", USD_PER) == pytest.approx(120.0)
    assert convert_value(120.0, "USD", "GBP", USD_PER) == pytest.approx(100.0)


def test_fx_return_is_investment_return():
    # Native price flat at 100; conversion 1.00 -> 1.10 gives +10% base return.
    idx = pd.bdate_range("2024-01-02", periods=4)
    native = pd.Series([100.0] * 4, index=idx)
    fx = pd.DataFrame({"GBP": [1.00, 1.10, 1.10, 1.10],
                       "USD": [1.0] * 4}, index=idx)
    base = convert_price_series(native, "GBP", "USD", fx)
    assert list(base) == pytest.approx([100.0, 110.0, 110.0, 110.0])
    rets = base.pct_change().dropna()
    assert rets.iloc[0] == pytest.approx(0.10)


def test_fx_alignment_bounded_ffill():
    idx = pd.bdate_range("2024-01-02", periods=6)
    # FX missing one middle day (holiday misalignment) -> carried forward.
    fx = pd.Series([1.20, np.nan, 1.22, 1.23, 1.24, 1.25], index=idx)
    out = align_fx_to_index(fx, idx, limit=5)
    assert out.iloc[1] == pytest.approx(1.20)
    # Leading gap can never be backfilled (would use future data).
    fx2 = pd.Series([np.nan, np.nan, 1.22, 1.23, 1.24, 1.25], index=idx)
    with pytest.raises(ValueError):
        align_fx_to_index(fx2, idx, limit=5)
    # Long outage beyond the bound raises instead of inventing data.
    fx3 = pd.Series([1.20] + [np.nan] * 5, index=idx)
    with pytest.raises(ValueError):
        align_fx_to_index(fx3, idx, limit=2)


def test_return_series_validation():
    idx = pd.bdate_range("2024-01-02", periods=5)
    good = pd.Series([0.01, -0.02, 0.0, 0.03, -0.01], index=idx)
    out = validate_return_series(good, "r")
    assert len(out) == 5
    bad_inf = good.copy()
    bad_inf.iloc[2] = np.inf
    with pytest.raises(ValueError):
        validate_return_series(bad_inf, "r")
    dup = pd.Series([0.01, 0.02], index=[idx[0], idx[0]])
    with pytest.raises(ValueError):
        validate_return_series(dup, "r")
    rev = good.iloc[::-1]
    with pytest.raises(ValueError):
        validate_return_series(rev, "r")
    with pytest.raises(ValueError):
        validate_return_series(good.iloc[:1], "r", min_obs=2)
    with pytest.raises(ValueError):
        validate_return_series([0.1], "r")


def test_pnl_decomposition_exact():
    # £100 cost, +10% local, +8.333...% FX: exact split, no approximation.
    d = decompose_pnl(100.0, 0.10, 0.08333333333333333)
    assert d["total"] == pytest.approx(100.0 * (1.10 * 1.0833333333333333 - 1))
    assert d["local_effect"] + d["fx_effect"] + d["interaction"] == \
        pytest.approx(d["total"])
    assert d["local_effect"] == pytest.approx(10.0)
    with pytest.raises(ValueError):
        decompose_pnl(float("nan"), 0.1, 0.1)


def test_fx_shock_static():
    # £800 of GBP exposure, $1200 USD, base GBP: +5% USD shock.
    out = fx_shock_portfolio({"GBP": 800.0, "USD": 1200.0},
                             {"USD": 0.05}, "GBP")
    assert out["base_total"] == pytest.approx(2000.0)
    assert out["shocked_total"] == pytest.approx(2060.0)
    assert out["pct_change"] == pytest.approx(0.03)
    assert out["per_currency"] == {"USD": pytest.approx(60.0)}
    with pytest.raises(ValueError):
        fx_shock_portfolio({"GBP": 800.0}, {"GBP": 0.05}, "GBP")
    with pytest.raises(ValueError):
        fx_shock_portfolio({"GBP": 800.0}, {"JPY": 0.05}, "GBP")


def test_freshness_and_weights():
    now = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
    assert quote_age_days(None, now) is None
    age = quote_age_days(datetime(2026, 1, 9, 12, 0, tzinfo=timezone.utc), now)
    assert age == pytest.approx(1.0)
    assert freshness_report({"A": 0.1}, 0.1) == []
    warns = freshness_report({"A": None}, None)
    assert len(warns) == 2  # missing quote + missing FX
    warns = freshness_report({"A": 0.1, "B": 10.0}, 0.1)
    assert any("Mixed" in w for w in warns)
    assert check_weights_sum({"A": 0.6, "B": 0.4}) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        check_weights_sum({"A": 0.6, "B": 0.3})
    with pytest.raises(ValueError):
        check_weights_sum({"A": float("nan")})


def test_rate_on_date_no_lookahead():
    from datetime import date
    idx = pd.bdate_range("2024-01-02", periods=5)
    fx = pd.DataFrame({"GBP": [1.20, 1.21, 1.22, 1.23, 1.24]}, index=idx)
    assert rate_on_date(fx, "GBP", "USD", date(2024, 1, 5)) == pytest.approx(1.23)
    # Weekend uses Friday's rate (past only), pre-history is unknown.
    assert rate_on_date(fx, "GBP", "USD", date(2024, 1, 7)) == pytest.approx(1.23)
    assert rate_on_date(fx, "GBP", "USD", date(2023, 6, 1)) is None
    assert rate_on_date(fx, "USD", "EUR", date(2024, 1, 5)) is None
    assert rate_on_date(pd.DataFrame(), "GBP", "USD", date(2024, 1, 5)) is None

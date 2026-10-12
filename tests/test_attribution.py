"""Tests for analytics.attribution (contributions, FX identity, income)."""

import pytest

from analytics.attribution import (
    attribution_summary,
    fx_contribution,
    fx_decomposition,
    security_contributions,
    winners_losers,
)


def test_contributions_reconcile_to_portfolio_return():
    w = {"A": 0.5, "B": 0.3, "C": 0.2}
    start = {"A": 100.0, "B": 50.0, "C": 25.0}
    end = {"A": 110.0, "B": 45.0, "C": 30.0}
    frame = security_contributions(w, start, end, ["A", "B", "C"])
    assert frame.loc["A", "contribution"] == pytest.approx(0.5 * 0.10)
    assert frame.loc["B", "contribution"] == pytest.approx(0.3 * -0.10)
    assert frame["contribution"].sum() == pytest.approx(
        0.05 - 0.03 + 0.04)
    with pytest.raises(ValueError):
        security_contributions(w, start, {"A": 1.0}, ["A", "B", "C"])


def test_winners_losers_by_contribution_not_return():
    # B's raw return (200%) dwarfs A's (50%), but A's weight makes its
    # contribution (0.45) beat B's (0.20): ranking uses contribution.
    frame = security_contributions(
        {"A": 0.9, "B": 0.1}, {"A": 100.0, "B": 100.0},
        {"A": 150.0, "B": 300.0}, ["A", "B"])
    wl = winners_losers(frame, n=1)
    assert list(wl["top"].index) == ["A"]
    assert wl["top"].iloc[0] == pytest.approx(0.45)
    assert list(wl["bottom"].index) == ["B"]
    with pytest.raises(ValueError):
        winners_losers(frame, n=0)


def test_fx_identity_exact():
    parts = fx_decomposition(300.0, 330.0, 1.25, 1.30)
    assert parts["local"] == pytest.approx(0.10)
    assert parts["fx"] == pytest.approx(0.04)
    assert (1 + parts["local"]) * (1 + parts["fx"]) - 1 == pytest.approx(
        parts["base"])
    assert parts["base"] == pytest.approx(3.30 * 1.30 / (3.00 * 1.25) - 1)
    legs = fx_contribution(0.5, parts["local"], parts["fx"])
    assert legs["local_leg"] + legs["fx_leg"] == pytest.approx(
        0.5 * parts["base"])
    with pytest.raises(ValueError):
        fx_decomposition(0.0, 1.0, 1.0, 1.0)


def test_summary_with_fx_and_income():
    out = attribution_summary(
        {"A": 0.6, "B": 0.4},
        {"A": 100.0, "B": 3.75}, {"A": 110.0, "B": 4.16},
        ["A", "B"],
        native_panels={"B": (3.00, 3.20)},
        fx_panels={"B": (1.25, 1.30)},
        income_base=25.0, portfolio_start_value=10000.0)
    assert out["portfolio_return"] == pytest.approx(
        0.6 * 0.10 + 0.4 * (4.16 / 3.75 - 1))
    assert "B" in out["fx"]
    assert out["income"] == pytest.approx(25.0)
    assert out["income_contribution"] == pytest.approx(0.0025)
    assert list(out["winners_losers"]["top"].index)[0] == "A"
    with pytest.raises(ValueError):
        attribution_summary({"A": 0.6, "B": 0.4}, {"A": 100.0},
                            {"A": 1.0}, ["A", "B"])

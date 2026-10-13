"""
attribution.py
--------------
Return attribution over a selected period (pure, documented methodology).

Methodology (simple returns, beginning weights):
* Security contribution: ``w_start[i] × R_simple[i]`` over the window, where
  ``w_start`` are beginning-of-period weights and ``R_simple`` the asset's
  simple return (end/start − 1). Contributions sum to the buy-and-hold
  portfolio simple return up to cash-flow effects (no external flows inside
  the window — callers must split windows at deposits/withdrawals).
* FX decomposition (exact identity, per foreign holding): with ``R_local``
  the native simple return and ``R_fx`` the FX simple return (both over the
  window), ``R_base = (1 + R_local)(1 + R_fx) − 1`` exactly. The FX
  contribution leg is ``(1 + R_local) × R_fx × w_start`` and the local leg
  is ``R_local × w_start``; interaction is folded into the FX leg and the
  identity is exact by construction (tested).
* Income: ledger-recorded dividends/interest enter as a separate income
  contribution (amount ÷ beginning portfolio value). Adjusted-price
  historical returns are never mixed with recorded income (no double
  counting — callers choose one return basis and label it).

All inputs explicit; invalid data raises ``ValueError``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _check_window(prices_start: dict, prices_end: dict, tickers: list):
    for t in tickers:
        for label, leg in (("start", prices_start), ("end", prices_end)):
            v = leg.get(t)
            if v is None or not math.isfinite(float(v)) or float(v) <= 0:
                raise ValueError(
                    f"Invalid {label} price for {t}: {v!r}.")


def security_contributions(weights_start: dict | pd.Series,
                           prices_start: dict, prices_end: dict,
                           tickers: list[str]) -> pd.DataFrame:
    """Per-security contribution to buy-and-hold simple return.

    Returns DataFrame indexed by ticker with ``weight_start``,
    ``return_simple`` and ``contribution`` (weight × return). The
    contributions sum to the portfolio simple return (up to rounding).
    """
    tickers = list(tickers)
    if isinstance(weights_start, pd.Series):
        weights_start = weights_start.to_dict()
    _check_window(prices_start, prices_end, tickers)
    rows = []
    for t in tickers:
        try:
            w = float(weights_start[t])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Missing beginning weight for {t}.") from e
        if not math.isfinite(w):
            raise ValueError(f"Non-finite beginning weight for {t}.")
        r = float(prices_end[t]) / float(prices_start[t]) - 1.0
        rows.append({"ticker": t, "weight_start": w,
                     "return_simple": r, "contribution": w * r})
    frame = pd.DataFrame(rows).set_index("ticker")
    return frame


def winners_losers(contributions: pd.DataFrame, n: int = 3) -> dict:
    """Top contributors / detractors by contribution (not raw return)."""
    if contributions is None or contributions.empty \
            or "contribution" not in contributions.columns:
        raise ValueError("No contributions supplied.")
    if n < 1:
        raise ValueError("n must be >= 1.")
    ordered = contributions["contribution"].sort_values(ascending=False)
    return {"top": ordered.head(n), "bottom": ordered.tail(n)}


def fx_decomposition(native_start: float, native_end: float,
                     fx_start: float, fx_end: float) -> dict:
    """Exact local/FX split of one foreign holding's base return.

    ``R_base = (1 + R_local)(1 + R_fx) − 1`` with ``R_fx = fx_end/fx_start
    − 1``. Returns ``local``, ``fx`` (both simple fractions) and
    ``base``; ``(1 + local)(1 + fx) − 1 == base`` exactly.
    """
    for label, v in (("native_start", native_start),
                     ("native_end", native_end),
                     ("fx_start", fx_start), ("fx_end", fx_end)):
        if v is None or not math.isfinite(float(v)) or float(v) <= 0:
            raise ValueError(f"Invalid {label}: {v!r}.")
    r_local = float(native_end) / float(native_start) - 1.0
    r_fx = float(fx_end) / float(fx_start) - 1.0
    r_base = (1.0 + r_local) * (1.0 + r_fx) - 1.0
    return {"local": r_local, "fx": r_fx, "base": r_base}


def fx_contribution(weight_start: float, r_local: float, r_fx: float) -> dict:
    """Portfolio-level legs: local + FX (interaction folded into FX leg).

    ``local_leg = w × R_local``; ``fx_leg = w × ((1 + R_local) × R_fx)``;
    their sum is exactly ``w × R_base``.
    """
    for label, v in (("weight", weight_start), ("local", r_local),
                     ("fx", r_fx)):
        if v is None or not math.isfinite(float(v)):
            raise ValueError(f"Non-finite {label}: {v!r}.")
    local_leg = float(weight_start) * float(r_local)
    fx_leg = float(weight_start) * ((1.0 + float(r_local)) * float(r_fx))
    return {"local_leg": local_leg, "fx_leg": fx_leg,
            "total": local_leg + fx_leg}


def attribution_summary(weights_start, prices_start: dict, prices_end: dict,
                        tickers: list[str],
                        native_panels: dict | None = None,
                        fx_panels: dict | None = None,
                        income_base: float = 0.0,
                        portfolio_start_value: float | None = None) -> dict:
    """One-shot attribution: contributions, winners/losers, FX, income.

    Args:
        weights_start/prices_start/prices_end/tickers: as in
            :func:`security_contributions` (base-currency prices).
        native_panels/fx_panels: optional ticker → (start, end) tuples for
            the FX decomposition of foreign holdings.
        income_base: ledger-recorded base-currency income over the window.
        portfolio_start_value: beginning portfolio value (for the income
            contribution ratio; None skips it).

    Returns dict with ``contributions`` (DataFrame), ``portfolio_return``,
    ``winners_losers``, ``fx`` (ticker → legs), ``income`` and
    ``income_contribution`` (|None).
    """
    tickers = list(tickers)
    contrib = security_contributions(weights_start, prices_start,
                                     prices_end, tickers)
    port_ret = float(contrib["contribution"].sum())
    fx = {}
    if native_panels or fx_panels:
        native_panels = native_panels or {}
        fx_panels = fx_panels or {}
        for t in tickers:
            if t in native_panels and t in fx_panels:
                (ns, ne), (fs, fe) = native_panels[t], fx_panels[t]
                parts = fx_decomposition(ns, ne, fs, fe)
                legs = fx_contribution(
                    float(contrib.loc[t, "weight_start"]),
                    parts["local"], parts["fx"])
                fx[t] = {**parts, **legs}
    income = float(income_base or 0.0)
    if not math.isfinite(income) or income < 0:
        raise ValueError("Income must be finite and >= 0.")
    income_contrib = None
    if portfolio_start_value is not None:
        if not math.isfinite(float(portfolio_start_value)) \
                or float(portfolio_start_value) <= 0:
            raise ValueError("Portfolio start value must be positive.")
        income_contrib = income / float(portfolio_start_value)
    return {"contributions": contrib, "portfolio_return": port_ret,
            "winners_losers": winners_losers(contrib),
            "fx": fx, "income": income,
            "income_contribution": income_contrib}

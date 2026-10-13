"""
rebalance_plan.py
-----------------
Research-only rebalancing planner (no execution — Task 9 never sends orders).

Given current holdings/marks and a target allocation, computes per-ticker
trades (notional, shares, BUY/SELL), cash-aware funding (deployable cash,
minimum buffer, optional cash target), minimum-trade thresholds, explicit
cost estimates on the existing cost model, and before/after diagnostics.

Target sources reuse the tested engines (MinVar/MaxSharpe/ERC/MaxDiv/BL on
current simple-return statistics, Equal Weight, or custom weights).

Conventions:
* Fractional shares assumed (matching the backtest engine); whole-share
  preview rounds down share counts and reports the residual.
* Costs use :func:`analytics.backtest.apply_transaction_costs` on FINAL
  notionals (same model as historical simulation).
* Cash-aware: sells fund buys first; remaining net buys plus the
  explicit cost estimate draw deployable cash (cash − buffer). Net need
  beyond deployable cash is reported as a shortfall — never silently
  assumed fundable.
* Thresholds: legs below ``min_trade_value`` or ``min_weight_change`` are
  held (zeroed) before costing.
* ``OrderIntent`` objects describe what a later paper-trading task could
  execute; nothing here calls any broker endpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .allocation import (
    allocation_summary,
    annualised_covariance_simple,
    annualised_mean_returns_simple,
    black_litterman_allocation,
    black_litterman_posterior,
    equal_risk_contribution,
    maximum_diversification,
    simple_returns_from_prices,
    validate_covariance,
)
from .backtest import apply_transaction_costs, calculate_turnover
from .optimisation import max_sharpe, min_variance
from .validation import canonical_tickers, validate_weight_bounds

PLANNER_METHODS = ("equal_weight", "min_variance", "max_sharpe", "erc",
                   "max_diversification", "black_litterman", "custom")


@dataclass
class OrderIntent:
    """Paper-trading-ready description of one proposed trade (research only)."""

    ticker: str
    side: str              # BUY | SELL | HOLD
    quantity_shares: float
    estimated_notional: float
    target_weight: float
    current_weight: float


@dataclass
class RebalancePlan:
    """Complete plan outcome (structured; failures raise, never faked)."""

    target_name: str
    target_weights: pd.Series
    trades: pd.DataFrame       # per-ticker detail (see plan_trades)
    intents: list = field(default_factory=list)   # OrderIntent
    gross_turnover: float = 0.0
    estimated_cost: float = 0.0
    cash_deployed: float = 0.0
    cash_remaining: float = 0.0
    cash_shortfall: float = 0.0
    min_buffer: float = 0.0
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    skipped_small: list = field(default_factory=list)


def math_is_finite_positive(x) -> bool:
    """Positive finite scalar."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0


def validate_target_weights(weights, tickers: list[str]) -> np.ndarray:
    """Custom targets: finite, long-only, matching tickers, summing to 1."""
    tickers = canonical_tickers(list(tickers))
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(tickers),):
        raise ValueError(
            f"Target weights length {w.shape} does not match "
            f"{len(tickers)} tickers.")
    if not np.all(np.isfinite(w)):
        raise ValueError("Target weights must be finite.")
    if np.any(w < -1e-9) or np.any(w > 1 + 1e-9):
        raise ValueError("Planner supports long-only targets (0 to 1).")
    if abs(float(w.sum()) - 1.0) > 1e-6:
        raise ValueError(
            f"Target weights must sum to 1 (got {float(w.sum()):.6f}).")
    return w


def resolve_planner_targets(method: str, tickers: list[str],
                            mu: np.ndarray, cov,
                            rf: float, min_w: float, max_w: float,
                            custom: np.ndarray | None = None,
                            bl_posterior=None, bl_delta: float = 2.5
                            ) -> tuple[str, pd.Series]:
    """Target weights for a planner method (engines reused, never rederived).

    ``bl_posterior`` is required for ``black_litterman`` (posterior expected
    returns); other methods ignore it. Returns (display_name, weights).
    """
    tickers = canonical_tickers(list(tickers))
    n = len(tickers)
    if method not in PLANNER_METHODS:
        raise ValueError(f"Unknown planner method {method!r}.")
    if method == "custom":
        if custom is None:
            raise ValueError("Custom targets need a weight vector.")
        return "Custom", pd.Series(validate_target_weights(custom, tickers),
                                   index=tickers)
    if method == "equal_weight":
        return "Equal Weight", pd.Series(np.ones(n) / n, index=tickers)
    mu_arr = np.asarray(mu, dtype=float)
    if mu_arr.shape != (n,) or not np.all(np.isfinite(mu_arr)):
        raise ValueError("Expected returns must be finite with length n.")
    cov_df, _ = validate_covariance(cov, tickers)
    if method == "min_variance":
        res = min_variance(mu_arr, cov_df.values, rf, min_w, max_w)
    elif method == "max_sharpe":
        res = max_sharpe(mu_arr, cov_df.values, rf, min_w, max_w)
    elif method == "erc":
        res = equal_risk_contribution(tickers, mu_arr, cov_df, rf,
                                      min_w, max_w)
    elif method == "max_diversification":
        res = maximum_diversification(tickers, mu_arr, cov_df, rf,
                                      min_w, max_w)
    elif method == "black_litterman":
        if bl_posterior is None:
            raise ValueError("Black-Litterman needs posterior returns.")
        from analytics.allocation import validate_delta
        res = black_litterman_allocation(
            tickers, bl_posterior, cov_df, validate_delta(bl_delta),
            rf, min_w, max_w)
    else:  # pragma: no cover - guarded above
        raise ValueError(f"Unknown planner method {method!r}.")
    if not res.success or res.weights is None:
        raise ValueError(f"Target estimation failed: {res.message}")
    label = {"min_variance": "Minimum Variance", "max_sharpe": "Maximum Sharpe",
             "erc": "Equal Risk Contribution",
             "max_diversification": "Maximum Diversification",
             "black_litterman": "Black-Litterman"}.get(
                 method, getattr(res, "method", method))
    return label, pd.Series(np.asarray(res.weights, dtype=float),
                            index=tickers)


def plan_trades(current_weights, target_weights, tickers: list[str],
                portfolio_value: float, current_prices: dict,
                cash_available: float = 0.0, min_cash_buffer: float = 0.0,
                min_trade_value: float = 0.0, min_weight_change: float = 0.0,
                cost_bps: float = 10.0, cash_target_weight: float = 0.0
                ) -> RebalancePlan:
    """Build the cash-aware trade plan (pure).

    Args:
        current_weights/target_weights: canonical-order vectors (targets
            validated long-only; current drifted weights allowed as-is).
        portfolio_value: current securities value (ex-cash) in base currency.
        current_prices: ticker → current native/base price per share for
            share counts (must be positive finite).
        cash_available: deployable cash in base currency.
        min_cash_buffer: cash retained even when deploying (<= available).
        min_trade_value / min_weight_change: legs below either are held.
        cost_bps: explicit cost assumption (existing model).
        cash_target_weight: deliberate post-trade cash share of (securities
            + available cash); 0 disables (keeps optimiser labels clear).

    Returns :class:`RebalancePlan` with notionals, shares, intents, costs,
    cash residual/shortfall and before/after diagnostics hooks (metrics
    filled by the caller via :func:`plan_diagnostics` when mu/cov known).
    """
    tickers = canonical_tickers(list(tickers))
    n = len(tickers)
    cw = np.asarray(current_weights, dtype=float)
    tw = validate_target_weights(target_weights, tickers)
    if cw.shape != (n,) or not np.all(np.isfinite(cw)):
        raise ValueError("Current weights must be finite with length n.")
    if not math_is_finite_positive(portfolio_value):
        raise ValueError("Portfolio value must be positive.")
    prices = []
    for t in tickers:
        try:
            px = float(current_prices[t])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Missing current price for {t}.") from e
        if not math_is_finite_positive(px):
            raise ValueError(f"Invalid current price for {t}: {px!r}.")
        prices.append(px)
    prices = np.asarray(prices)
    for label, v in (("cash_available", cash_available),
                     ("min_cash_buffer", min_cash_buffer),
                     ("min_trade_value", min_trade_value),
                     ("min_weight_change", min_weight_change),
                     ("cost_bps", cost_bps),
                     ("cash_target_weight", cash_target_weight)):
        if v is None or not np.isfinite(float(v)) or float(v) < 0:
            raise ValueError(f"{label} must be finite and >= 0.")
    cash_available = float(cash_available)
    min_cash_buffer = min(float(min_cash_buffer), cash_available)
    if float(cash_target_weight) > 1:
        raise ValueError("Cash target weight cannot exceed 1.")

    delta_w = tw - cw
    trade_notional = delta_w * portfolio_value
    # Threshold band: tiny legs are held (no churn).
    hold = (np.abs(trade_notional) < float(min_trade_value)) | \
           (np.abs(delta_w) < float(min_weight_change))
    skipped = [tickers[i] for i in range(n) if hold[i]]
    trade_notional = np.where(hold, 0.0, trade_notional)
    shares = trade_notional / prices
    side = np.array(["BUY" if v > 0 else ("SELL" if v < 0 else "HOLD")
                     for v in trade_notional])

    deployable = max(cash_available - min_cash_buffer, 0.0)
    # Sells fund buys first; net buys draw deployable cash.
    net_buy = float(trade_notional[trade_notional > 0].sum())
    net_sell = float(-trade_notional[trade_notional < 0].sum())
    deploy_needed = max(net_buy - net_sell, 0.0)
    # Optional deliberate cash target (share of securities + cash): retain
    # at least max(buffer, target) and deploy only the rest.
    total_considered = portfolio_value + cash_available
    retain = max(min_cash_buffer,
                 float(cash_target_weight) * total_considered)
    deployable = max(min(cash_available - retain, deployable), 0.0)
    # Explicit costs are paid from cash alongside net buys (first-order
    # planning estimate on displayed notionals; the historical simulator
    # refines this with the exact fixed-point solver).
    cost = apply_transaction_costs(trade_notional, float(cost_bps))
    need = deploy_needed + cost
    funded = min(need, deployable)
    shortfall = need - funded
    cash_remaining = cash_available - funded
    turnover = (float(np.abs(trade_notional).sum() / portfolio_value)
                if portfolio_value > 0 else 0.0)

    rows = []
    intents = []
    for i, t in enumerate(tickers):
        rows.append({
            "Ticker": t,
            "Current Weight": float(cw[i]),
            "Target Weight": float(tw[i]),
            "Difference": float(tw[i] - cw[i]),
            "Current Value": float(cw[i] * portfolio_value),
            "Target Value": float(tw[i] * portfolio_value),
            "Trade Notional": float(trade_notional[i]),
            "Approx Shares": float(shares[i]),
            "Action": str(side[i]),
        })
        intents.append(OrderIntent(
            ticker=t, side=str(side[i]),
            quantity_shares=float(shares[i]),
            estimated_notional=float(trade_notional[i]),
            target_weight=float(tw[i]), current_weight=float(cw[i])))
    return RebalancePlan(
        target_name="", target_weights=pd.Series(tw, index=tickers),
        trades=pd.DataFrame(rows),
        intents=intents, gross_turnover=turnover, estimated_cost=cost,
        cash_deployed=funded, cash_remaining=cash_remaining,
        cash_shortfall=shortfall, min_buffer=min_cash_buffer,
        skipped_small=skipped)


def plan_diagnostics(current_weights, target_weights, tickers: list[str],
                     mu: np.ndarray, cov, rf: float) -> dict:
    """Before/after comparison (expected return, vol, Sharpe, concentration).

    Uses :func:`allocation_summary` for both legs (identical maths to the
    Optimise tab's comparison).
    """
    tickers = canonical_tickers(list(tickers))
    cw = np.asarray(current_weights, dtype=float)
    tw = validate_target_weights(target_weights, tickers)
    mu_arr = np.asarray(mu, dtype=float)
    cov_df, _ = validate_covariance(cov, tickers)
    return {"current": allocation_summary(cw, mu_arr, cov_df, rf, tickers),
            "target": allocation_summary(tw, mu_arr, cov_df, rf, tickers)}

"""
constraints.py
--------------
Real-world optimisation constraints (Task 10, Milestone 5).

One reusable constraint model shared by every optimiser (classical MinVar /
MaxSharpe / target return-vol plus ERC / risk-budget / MaxDiv / BL):

* per-asset minimum/maximum weights (override the global box),
* turnover limit relative to current weights — one-way convention
  ``Σ|w − w₀| / 2 ≤ limit`` used consistently everywhere,
* transaction-cost-aware planning penalty ``(bps/10⁴) × turnover``
  added to the objective (an optimiser planning knob — the exact
  fixed-point backtest cost accounting is untouched),
* quote-currency exposure caps on Base-valued risky weights,
* cash target (risky weights + cash = 1; statistics stay cash-aware),
* custom risk budgets for the ERC family (validated to sum to 1).

Turnover convention, feasibility diagnostics (no silent infeasibility) and
before/after stability helpers live here so UI, engines and tests share one
definition. This module imports only NumPy/pandas plus
``analytics.validation`` — engines import it lazily, so there is no cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .validation import canonical_tickers, validate_weight_bounds

#: Tolerance for budget/bounds feasibility comparisons.
_FEAS_TOL = 1e-9


# ---------------------------------------------------------------------------
# Turnover (single convention)
# ---------------------------------------------------------------------------

def turnover_fraction(weights: np.ndarray,
                      reference: np.ndarray) -> float:
    """One-way turnover ``Σ|w − w₀| / 2`` (fraction of portfolio)."""
    w = np.asarray(weights, dtype=float)
    w0 = np.asarray(reference, dtype=float)
    if w.shape != w0.shape:
        raise ValueError(
            f"Weights {w.shape} and reference {w0.shape} must match.")
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(w0)):
        raise ValueError("Weights and reference must be finite.")
    return float(np.abs(w - w0).sum() / 2.0)


def max_single_name_shift(weights: np.ndarray,
                           reference: np.ndarray) -> float:
    """Largest single-name absolute weight change."""
    w = np.asarray(weights, dtype=float)
    w0 = np.asarray(reference, dtype=float)
    if w.shape != w0.shape:
        raise ValueError("Weights and reference must match.")
    return float(np.abs(w - w0).max(initial=0.0))


# ---------------------------------------------------------------------------
# Constraint model
# ---------------------------------------------------------------------------

@dataclass
class OptimizationConstraints:
    """Reusable real-world constraints (all optional; empty = classic)."""
    per_asset_min: dict | None = None
    per_asset_max: dict | None = None
    turnover_limit: float | None = None
    turnover_reference: np.ndarray | list | None = None
    transaction_cost_bps: float = 0.0
    cash_target: float = 0.0
    asset_currencies: list | None = None
    currency_caps: dict | None = None
    risk_budgets: dict | pd.Series | None = None

    def has_any(self) -> bool:
        """True when at least one non-default constraint is set."""
        return bool(
            self.per_asset_min or self.per_asset_max
            or self.turnover_limit is not None
            or float(self.transaction_cost_bps or 0.0) > 0
            or float(self.cash_target or 0.0) > 0
            or self.currency_caps or self.risk_budgets is not None)


def budget_total(constraints: OptimizationConstraints | None) -> float:
    """Risky-asset budget: ``1 − cash_target`` (1 when unconstrained)."""
    if constraints is None:
        return 1.0
    try:
        cash = float(constraints.cash_target or 0.0)
    except (TypeError, ValueError):
        raise ValueError("Cash target must be a number.")
    if not np.isfinite(cash) or not 0.0 <= cash < 1.0:
        raise ValueError(
            f"Cash target must satisfy 0 ≤ c < 1, got {cash!r}.")
    return 1.0 - cash


def resolve_bounds(n: int, tickers: list[str],
                   constraints: OptimizationConstraints | None,
                   min_weight: float = 0.0,
                   max_weight: float = 1.0) -> list[tuple]:
    """Per-asset SLSQP bounds: global box overridden per ticker."""
    tickers = canonical_tickers(list(tickers))
    validate_weight_bounds(n, min_weight, max_weight)
    lo = [float(min_weight)] * n
    hi = [float(max_weight)] * n
    if constraints is not None:
        for key, arr, side in (("per_asset_min", lo, "minimum"),
                               ("per_asset_max", hi, "maximum")):
            mapping = getattr(constraints, key, None) or {}
            for t, v in dict(mapping).items():
                try:
                    sym = canonical_tickers([str(t)])[0]
                except ValueError:
                    raise ValueError(f"Unknown ticker in {key}: {t!r}.")
                if sym not in tickers:
                    raise ValueError(
                        f"Ticker {sym} in {key} is not in the universe.")
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Per-asset {side} for {sym} must be a number.")
                if not np.isfinite(val):
                    raise ValueError(
                        f"Per-asset {side} for {sym} must be finite.")
                arr[tickers.index(sym)] = val
    for i, t in enumerate(tickers):
        if lo[i] - hi[i] > _FEAS_TOL:
            raise ValueError(
                f"Infeasible per-asset bounds for {t}: min {lo[i]:.2%} > "
                f"max {hi[i]:.2%}.")
    return list(zip(lo, hi))


def validate_risk_budgets(budgets: dict | pd.Series | None,
                          tickers: list[str]) -> pd.Series | None:
    """Custom ERC risk budgets: labelled, non-negative, summing to 1."""
    if budgets is None:
        return None
    tickers = canonical_tickers(list(tickers))
    b = (pd.Series(dict(budgets), dtype=float)
         if not isinstance(budgets, pd.Series)
         else budgets.astype(float))
    if set(b.index) != set(tickers):
        raise ValueError("Risk budgets must cover exactly the universe "
                         f"(got {sorted(map(str, b.index))}).")
    b = b.reindex(tickers)
    if not np.all(np.isfinite(b.values)):
        raise ValueError("Risk budgets must be finite.")
    if np.any(b.values < -1e-12):
        raise ValueError("Risk budgets cannot be negative.")
    if abs(float(b.values.sum()) - 1.0) > 1e-6:
        raise ValueError(f"Risk budgets must sum to 1 "
                         f"(got {float(b.values.sum()):.6f}).")
    return b


def risk_budget_objective(w: np.ndarray, cov_arr: np.ndarray,
                          budgets: np.ndarray) -> float:
    """``Σ(PRC_i − b_i)²`` — ERC is the special case ``b_i = 1/N``."""
    var = float(w @ cov_arr @ w)
    if not np.isfinite(var) or var <= 0:
        return 1e6
    vol = float(np.sqrt(var))
    mrc = cov_arr @ w
    prc = (w * mrc / vol) / vol
    if not np.all(np.isfinite(prc)):
        return 1e6
    return float(((prc - budgets) ** 2).sum())


def _turnover_reference(constraints: OptimizationConstraints | None,
                        n: int) -> np.ndarray | None:
    if constraints is None or constraints.turnover_limit is None:
        return None
    ref = constraints.turnover_reference
    if ref is None:
        raise ValueError("A turnover limit needs current (reference) weights.")
    arr = np.asarray(ref, dtype=float)
    if arr.shape != (n,) or not np.all(np.isfinite(arr)):
        raise ValueError("Reference weights must be finite with length n.")
    return arr


def solve_with_turnover(fun, n: int, bounds: list[tuple],
                        base_cons: list[dict], ref: np.ndarray,
                        limit: float, x0: np.ndarray):
    """SLSQP solve with an exact one-way turnover cap, L1-kink-free.

    The non-smooth ``Σ|w − w₀|/2 ≤ limit`` is reformulated with auxiliary
    variables ``u ≥ ±(w − w₀)`` and ``Σu/2 ≤ limit`` — all smooth and
    linear, so SLSQP converges reliably even from starts sitting exactly on
    the kink. Returns the raw ``scipy`` result (weights are ``res.x[:n]``).
    """
    from scipy.optimize import minimize

    ref = np.asarray(ref, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    u0 = np.abs(x0 - ref) + 1e-8
    x_init = np.concatenate([x0, u0])
    wide = bounds + [(0.0, None)] * n

    def _w(x):
        return np.asarray(x[:n], dtype=float)

    def _u(x):
        return np.asarray(x[n:], dtype=float)

    cons: list[dict] = []
    for c in base_cons or []:
        if c.get("type") == "eq":
            cons.append({"type": "eq",
                         "fun": lambda x, f=c["fun"]: float(f(_w(x)))})
        else:
            cons.append({"type": "ineq",
                         "fun": lambda x, f=c["fun"]: float(f(_w(x)))})
    for i in range(n):
        cons.append({"type": "ineq",
                     "fun": lambda x, i=i: float(x[n + i] - (x[i] - ref[i]))})
        cons.append({"type": "ineq",
                     "fun": lambda x, i=i: float(x[n + i] - (ref[i] - x[i]))})
    cons.append({"type": "ineq",
                 "fun": lambda x: float(limit - float(_u(x).sum()) / 2.0)})

    def _fun(x):
        return float(fun(_w(x)))

    return minimize(fun=_fun, x0=x_init, method="SLSQP", bounds=wide,
                    constraints=cons,
                    options={"ftol": 1e-12, "maxiter": 2000})


def _take_weights(res, n: int) -> np.ndarray | None:
    """Extract the weight vector from a direct or turnover-aux solve."""
    try:
        x = np.asarray(res.x, dtype=float)
    except (AttributeError, TypeError, ValueError):
        return None
    if x.shape != (n,) and x.shape == (2 * n,):
        x = x[:n]
    if x.shape != (n,) or not np.all(np.isfinite(x)):
        return None
    return x


def turnover_reference(constraints: OptimizationConstraints | None,
                       n: int) -> np.ndarray | None:
    """Validated turnover reference weights (None when no limit is set)."""
    return _turnover_reference(constraints, n)


def min_achievable_turnover(
    tickers: list[str],
    constraints: OptimizationConstraints,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> float:
    """Smallest one-way turnover attainable under bounds/total/currency.

    Solved directly through the kink-free auxiliary formulation, so the
    feasibility explainer reports a genuine minimum, never a guess.
    """
    tickers = canonical_tickers(list(tickers))
    n = len(tickers)
    bounds = resolve_bounds(n, tickers, constraints, min_weight, max_weight)
    ref = _turnover_reference(constraints, n)
    if ref is None:
        return 0.0
    _, base_cons = slsqp_fragments(
        tickers, constraints, min_weight, max_weight, include_turnover=False)
    x0 = np.clip(ref, [b[0] for b in bounds], [b[1] for b in bounds])
    try:
        res = solve_with_turnover(
            lambda w: turnover_fraction(w, ref), n, bounds, base_cons,
            ref, 1.0, x0)
    except Exception as e:
        raise ValueError(f"Turnover feasibility solve failed: {e}")
    w = _take_weights(res, n)
    if w is None or not getattr(res, "success", False):
        raise ValueError("Could not find any feasible portfolio under the "
                         "configured bounds (turnover check inconclusive).")
    return turnover_fraction(w, ref)


def check_constraints(
    tickers: list[str],
    constraints: OptimizationConstraints | None,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> list[str]:
    """Feasibility explainer: list of human-readable problems (empty = OK)."""
    tickers = canonical_tickers(list(tickers))
    n = len(tickers)
    problems: list[str] = []
    if constraints is None or not constraints.has_any():
        return problems
    try:
        bounds = resolve_bounds(n, tickers, constraints,
                                min_weight, max_weight)
    except ValueError as e:
        return [str(e)]
    try:
        total = budget_total(constraints)
    except ValueError as e:
        return [str(e)]
    lo_sum = sum(b[0] for b in bounds)
    hi_sum = sum(b[1] for b in bounds)
    if lo_sum - total > _FEAS_TOL:
        problems.append(
            f"Sum of minimum weights ({lo_sum:.2%}) exceeds the "
            f"risky budget ({total:.2%}) — loosen minima or raise cash.")
    if total - hi_sum > _FEAS_TOL:
        problems.append(
            f"Sum of maximum weights ({hi_sum:.2%}) cannot reach the "
            f"risky budget ({total:.2%}) — raise maxima or lower cash.")
    if constraints.turnover_limit is not None:
        try:
            lim = float(constraints.turnover_limit)
        except (TypeError, ValueError):
            problems.append("Turnover limit must be a number.")
            lim = float("nan")
        if not np.isfinite(lim) or not 0.0 <= lim <= 1.0:
            problems.append(
                f"Turnover limit must satisfy 0 ≤ t ≤ 1, "
                f"got {constraints.turnover_limit!r}.")
        else:
            try:
                _turnover_reference(constraints, n)
            except ValueError as e:
                problems.append(str(e))
            else:
                try:
                    floor = min_achievable_turnover(
                        tickers, constraints, min_weight, max_weight)
                except ValueError as e:
                    problems.append(str(e))
                else:
                    if floor - lim > 1e-6:
                        problems.append(
                            f"Even the closest feasible portfolio needs "
                            f"{floor:.2%} turnover — above the {lim:.2%} "
                            f"limit. Loosen bounds or raise the limit.")
    try:
        cost = float(constraints.transaction_cost_bps or 0.0)
    except (TypeError, ValueError):
        problems.append("Transaction-cost penalty must be a number.")
    else:
        if not np.isfinite(cost) or cost < 0:
            problems.append("Transaction-cost penalty must be ≥ 0 bps.")
    if constraints.currency_caps:
        if constraints.asset_currencies is None:
            problems.append("Currency caps need per-asset quote currencies.")
        else:
            ac = list(constraints.asset_currencies)
            if len(ac) != n:
                problems.append("Per-asset currencies must match the universe.")
            else:
                for c, cap in dict(constraints.currency_caps).items():
                    try:
                        capv = float(cap)
                    except (TypeError, ValueError):
                        problems.append(
                            f"Currency cap for {c} must be a number.")
                        continue
                    if not np.isfinite(capv) or capv < 0 or capv > 1:
                        problems.append(
                            f"Currency cap for {c} must satisfy 0 ≤ cap ≤ 1.")
                if all(a is not None for a in ac):
                    caps = {str(k).upper(): float(v) for k, v in
                            dict(constraints.currency_caps).items()}
                    for c, capv in caps.items():
                        members = [i for i, a in enumerate(ac)
                                   if str(a).upper() == c]
                        if not members:
                            problems.append(
                                f"Currency cap for {c} matches no asset.")
                            continue
                        max_group = sum(bounds[i][1] for i in members)
                        if float(capv) - max_group > _FEAS_TOL:
                            continue  # cap looser than bounds — harmless
    if constraints.risk_budgets is not None:
        try:
            validate_risk_budgets(constraints.risk_budgets, tickers)
        except ValueError as e:
            problems.append(str(e))
    return problems


def slsqp_fragments(
    tickers: list[str],
    constraints: OptimizationConstraints | None,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    include_turnover: bool = True,
) -> tuple[list[tuple], list[dict]]:
    """SLSQP ``(bounds, constraints)`` honouring the constraint model."""
    tickers = canonical_tickers(list(tickers))
    n = len(tickers)
    bounds = resolve_bounds(n, tickers, constraints,
                            min_weight, max_weight)
    total = budget_total(constraints)
    cons: list[dict] = [
        {"type": "eq", "fun": lambda w, t=total: float(np.sum(w)) - t}]
    if constraints is not None and include_turnover \
            and constraints.turnover_limit is not None:
        ref = _turnover_reference(constraints, n)
        lim = float(constraints.turnover_limit)
        cons.append({"type": "ineq",
                     "fun": lambda w, r=ref, l=lim:
                     float(l - turnover_fraction(w, r))})
    if constraints is not None and constraints.currency_caps:
        ac = list(constraints.asset_currencies or [])
        if len(ac) == n:
            for c, cap in dict(constraints.currency_caps).items():
                members = [i for i, a in enumerate(ac)
                           if a is not None and str(a).upper()
                           == str(c).upper()]
                if not members:
                    continue
                capv = float(cap)

                def _cap_fun(w, idx=tuple(members), c=capv):
                    return float(c - float(np.sum(w[list(idx)])))

                cons.append({"type": "ineq", "fun": _cap_fun})
    return bounds, cons


def penalised_objective(base_fn, constraints: OptimizationConstraints | None,
                        n: int):
    """Wrap an objective with the turnover cost-aversion penalty."""
    if constraints is None:
        return base_fn
    try:
        bps = float(constraints.transaction_cost_bps or 0.0)
    except (TypeError, ValueError):
        bps = 0.0
    if not np.isfinite(bps) or bps <= 0:
        return base_fn
    ref = _turnover_reference(constraints, n) \
        if constraints.turnover_limit is not None else None
    if ref is None and bps > 0:
        # Penalty without a hard limit still needs the reference mix.
        alt = getattr(constraints, "turnover_reference", None)
        if alt is None:
            return base_fn
        ref = np.asarray(alt, dtype=float)
        if ref.shape != (n,) or not np.all(np.isfinite(ref)):
            return base_fn

    def _wrapped(w):
        w = np.asarray(w, dtype=float)
        val = float(base_fn(w))
        if not np.isfinite(val):
            return val
        return val + (bps / 1e4) * turnover_fraction(w, ref)

    return _wrapped

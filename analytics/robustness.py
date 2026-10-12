"""
robustness.py
-------------
Walk-forward robustness laboratory (Task 10, Milestone 6).

Orchestration around the existing walk-forward engine (which is NOT
replaced): a configuration grid over lookbacks × rebalance frequencies ×
risk models × transaction-cost assumptions is expanded, executed cell by
cell with ``run_walk_forward``, and summarised descriptively.

* Every cell is one honest WF run — strict ``date < D`` estimation comes
  from the engine itself; the grid layer adds no data path that could leak
  the future (grid-level regression test truncates the panel).
* Identical configurations reuse a caller-supplied result cache
  (``{(method, lookback, frequency, estimator, cost, window, ...): result}``).
* No composite scores and no investment recommendations: the scorecard
  reports medians, worst/best values and dispersions only.
* Cost break-even is reported as the grid bracket where a strategy's Sharpe
  crosses its static counterpart (interpolated point marked approximate).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .validation import canonical_tickers
from .walkforward import (
    WALK_FORWARD_LABELS,
    WALK_FORWARD_METHODS,
    WalkForwardConfig,
    run_walk_forward,
)

#: Grid methods (BL excluded by default — view complexity; still runnable).
GRID_METHODS = ("min_variance", "max_sharpe", "erc", "max_diversification")

#: Default manageable matrix (§6.2).
DEFAULT_LOOKBACKS = (1.0, 3.0, 5.0)
DEFAULT_FREQUENCIES = ("monthly", "quarterly", "annual")
DEFAULT_ESTIMATORS = ("sample", "ewma", "ledoit_wolf")
DEFAULT_COSTS_BPS = (0.0, 5.0, 10.0, 25.0, 50.0)

#: Hard cap against accidental massive runs (§6.13).
MAX_GRID_CELLS = 500


@dataclass
class RobustnessConfig:
    """One robustness-laboratory specification."""

    methods: tuple = GRID_METHODS
    lookbacks: tuple = DEFAULT_LOOKBACKS
    frequencies: tuple = DEFAULT_FREQUENCIES
    estimators: tuple = ("sample",)
    costs_bps: tuple = (10.0,)
    window_type: str = "rolling"
    test_start: object = None
    test_end: object = None
    base_currency: str = "USD"
    min_weight: float = 0.0
    max_weight: float = 1.0
    risk_free_annual: float = 0.05
    ewma_lambda: float = 0.94
    max_cells: int = MAX_GRID_CELLS

    def validated(self, tickers: list[str]) -> "RobustnessConfig":
        tickers = canonical_tickers(list(tickers))
        methods = tuple(m for m in (self.methods or ()) if m)
        for m in methods:
            if m not in WALK_FORWARD_METHODS:
                raise ValueError(f"Unknown grid method {m!r}.")
        if not methods:
            raise ValueError("Select at least one grid method.")
        lookbacks = tuple(float(x) for x in (self.lookbacks or ()))
        if not lookbacks or any(not np.isfinite(x) or x <= 0
                                for x in lookbacks):
            raise ValueError("Lookbacks must be positive years.")
        freqs = tuple(str(f).strip().lower() for f in (self.frequencies or ()))
        if not freqs:
            raise ValueError("Select at least one rebalance frequency.")
        from .backtest import validate_frequency
        for f in freqs:
            if validate_frequency(f) == "never":
                raise ValueError(f"Grid needs a periodic frequency, got {f!r}.")
        from .estimation import validate_estimator
        estimators = tuple(validate_estimator(e)
                           for e in (self.estimators or ("sample",)))
        if not estimators:
            raise ValueError("Select at least one risk model.")
        costs = tuple(float(c) for c in (self.costs_bps or (0.0,)))
        from .backtest import validate_transaction_cost_bps
        for c in costs:
            validate_transaction_cost_bps(c)
        wt = str(self.window_type or "rolling").strip().lower()
        if wt not in ("rolling", "expanding"):
            raise ValueError(f"Unknown window type {self.window_type!r}.")
        try:
            max_cells = int(self.max_cells)
        except (TypeError, ValueError):
            raise ValueError("max_cells must be an integer.")
        if max_cells < 1:
            raise ValueError("max_cells must be at least 1.")
        n = (len(methods) * len(lookbacks) * len(freqs) * len(estimators)
             * len(costs))
        if n > max_cells:
            raise ValueError(
                f"Grid would run {n} configurations (cap {max_cells}): "
                "narrow methods, lookbacks, frequencies, estimators or costs.")
        self.methods, self.lookbacks, self.frequencies = methods, lookbacks, freqs
        self.estimators, self.costs_bps = estimators, costs
        self.window_type = wt
        self.max_cells = max_cells
        return self


@dataclass
class RobustnessCell:
    """One executed grid cell (JSON-safe summary + config)."""

    method: str
    lookback_years: float
    frequency: str
    estimator: str
    cost_bps: float
    success: bool = False
    message: str = ""
    cagr: float = float("nan")
    volatility: float = float("nan")
    sharpe: float = float("nan")
    max_drawdown: float = float("nan")
    avg_turnover: float = float("nan")
    total_costs: float = float("nan")
    final_value: float = float("nan")
    n_failed: int = 0
    cached: bool = False


@dataclass
class RobustnessResult:
    """Full grid outcome (cells + descriptive scorecard)."""

    cells: list = field(default_factory=list)
    scorecard: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame())
    n_cached: int = 0

    def frame(self) -> pd.DataFrame:
        """Result matrix (§6.4) as a DataFrame (one row per configuration)."""
        rows = []
        for c in self.cells:
            rows.append({
                "Method": WALK_FORWARD_LABELS.get(c.method, c.method),
                "Lookback (y)": c.lookback_years,
                "Frequency": c.frequency,
                "Risk model": c.estimator,
                "Cost (bps)": c.cost_bps,
                "CAGR": c.cagr, "Vol": c.volatility, "Sharpe": c.sharpe,
                "Max DD": c.max_drawdown, "Avg Turnover": c.avg_turnover,
                "Costs": c.total_costs, "Final Value": c.final_value,
                "Failed": c.n_failed,
                "Status": "ok" if c.success else f"failed: {c.message[:80]}",
            })
        cols = ["Method", "Lookback (y)", "Frequency", "Risk model",
                "Cost (bps)", "CAGR", "Vol", "Sharpe", "Max DD",
                "Avg Turnover", "Costs", "Final Value", "Failed", "Status"]
        return pd.DataFrame(rows, columns=cols)


def expand_grid(config: RobustnessConfig,
                tickers: list[str]) -> list[dict]:
    """List every configuration dict (count known before execution)."""
    config.validated(list(tickers))
    specs = []
    for method in config.methods:
        for lookback in config.lookbacks:
            for freq in config.frequencies:
                for est in config.estimators:
                    for cost in config.costs_bps:
                        specs.append({
                            "method": method, "lookback_years": lookback,
                            "frequency": freq, "estimator": est,
                            "cost_bps": cost,
                            "window_type": config.window_type,
                            "test_start": config.test_start,
                            "test_end": config.test_end,
                            "base_currency": config.base_currency,
                            "min_weight": config.min_weight,
                            "max_weight": config.max_weight,
                            "risk_free_annual": config.risk_free_annual,
                            "ewma_lambda": config.ewma_lambda,
                        })
    return specs


def grid_size(config: RobustnessConfig, tickers: list[str]) -> int:
    """Number of configurations (cheap — call before running)."""
    return len(expand_grid(config, tickers))


def _cell_key(spec: dict, tickers: list[str], capital: float) -> tuple:
    return (tuple(tickers), str(spec["test_start"]), str(spec["test_end"]),
            spec["method"], float(spec["lookback_years"]), spec["frequency"],
            spec["estimator"], float(spec["cost_bps"]), spec["window_type"],
            float(capital), float(spec["risk_free_annual"]),
            float(spec["min_weight"]), float(spec["max_weight"]),
            float(spec["ewma_lambda"]), str(spec["base_currency"]))


def run_robustness_grid(
    base_prices: pd.DataFrame,
    tickers: list[str],
    config: RobustnessConfig,
    initial_capital: float,
    cache: dict | None = None,
    progress_cb=None,
) -> RobustnessResult:
    """Execute the grid cell by cell (existing engine only).

    ``cache`` maps cell keys to cells: identical configurations are not
    recomputed (their cells are returned with ``cached=True``). ``cache``
    is updated in place. ``progress_cb(i, n)`` is called after each cell.
    """
    from .backtest import validate_backtest_capital

    tickers = canonical_tickers(list(tickers))
    config.validated(tickers)
    capital = validate_backtest_capital(initial_capital)
    cache = cache if cache is not None else {}
    specs = expand_grid(config, tickers)
    n = len(specs)
    cells: list[RobustnessCell] = []
    n_cached = 0
    for i, spec in enumerate(specs):
        key = _cell_key(spec, tickers, capital)
        if key in cache:
            hit = cache[key]
            hit.cached = True
            cells.append(hit)
            n_cached += 1
        else:
            cell = _run_cell(base_prices, tickers, spec, capital)
            cache[key] = cell
            cells.append(cell)
        if progress_cb is not None:
            try:
                progress_cb(i + 1, n)
            except Exception:
                pass
    return RobustnessResult(cells=cells,
                            scorecard=robustness_scorecard(cells),
                            n_cached=n_cached)


def _run_cell(base_prices: pd.DataFrame, tickers: list[str], spec: dict,
              capital: float) -> RobustnessCell:
    base = RobustnessCell(method=spec["method"],
                          lookback_years=float(spec["lookback_years"]),
                          frequency=spec["frequency"],
                          estimator=spec["estimator"],
                          cost_bps=float(spec["cost_bps"]))
    try:
        cfg = WalkForwardConfig(
            method=spec["method"], test_start=spec["test_start"],
            test_end=spec["test_end"],
            lookback_years=float(spec["lookback_years"]),
            window_type=spec["window_type"],
            rebalance_frequency=spec["frequency"],
            transaction_cost_bps=float(spec["cost_bps"]),
            base_currency=spec["base_currency"],
            min_weight=float(spec["min_weight"]),
            max_weight=float(spec["max_weight"]),
            risk_free_annual=float(spec["risk_free_annual"]),
            estimator=spec["estimator"],
            ewma_lambda=float(spec["ewma_lambda"]))
    except ValueError as e:
        base.success, base.message = False, str(e)
        return base
    try:
        res = run_walk_forward(base_prices, tickers, cfg, capital)
    except ValueError as e:
        base.success, base.message = False, str(e)
        return base
    if not res.success:
        base.success, base.message = False, res.message
        return base
    m = res.metrics
    base.success = True
    base.message = "Completed"
    base.cagr = float(m.get("cagr", float("nan")))
    base.volatility = float(m.get("annualised_volatility", float("nan")))
    base.sharpe = float(m.get("sharpe", float("nan")))
    base.max_drawdown = float(m.get("max_drawdown", float("nan")))
    base.avg_turnover = float(m.get("avg_turnover", float("nan")))
    base.total_costs = float(m.get("total_costs", float("nan")))
    base.final_value = float(m.get("final_value", float("nan")))
    base.n_failed = int(m.get("n_failed", 0))
    return base


def robustness_scorecard(cells: list[RobustnessCell]) -> pd.DataFrame:
    """Descriptive robustness statistics per method (§6.6 — no composite).

    Median / worst / best Sharpe, Sharpe dispersion (population std),
    median turnover and worst drawdown across a method's successful cells.
    """
    rows = []
    for method in sorted({c.method for c in cells}):
        ok = [c for c in cells if c.method == method and c.success]
        sharpes = np.array([c.sharpe for c in ok
                            if np.isfinite(c.sharpe)], dtype=float)
        turns = np.array([c.avg_turnover for c in ok
                          if np.isfinite(c.avg_turnover)], dtype=float)
        dds = np.array([c.max_drawdown for c in ok
                        if np.isfinite(c.max_drawdown)], dtype=float)
        rows.append({
            "Method": WALK_FORWARD_LABELS.get(method, method),
            "Configs": len(ok),
            "Median Sharpe": float(np.median(sharpes)) if len(sharpes) else np.nan,
            "Worst Sharpe": float(sharpes.min()) if len(sharpes) else np.nan,
            "Best Sharpe": float(sharpes.max()) if len(sharpes) else np.nan,
            "Sharpe Dispersion": float(sharpes.std(ddof=0)) if len(sharpes) else np.nan,
            "Median Turnover": float(np.median(turns)) if len(turns) else np.nan,
            "Worst Drawdown": float(dds.min()) if len(dds) else np.nan,
        })
    cols = ["Method", "Configs", "Median Sharpe", "Worst Sharpe",
            "Best Sharpe", "Sharpe Dispersion", "Median Turnover",
            "Worst Drawdown"]
    return pd.DataFrame(rows, columns=cols)


def sharpe_heatmap_frame(cells: list[RobustnessCell],
                         method: str) -> pd.DataFrame:
    """Mean Sharpe by lookback × frequency for one method (§6.5)."""
    sub = [c for c in cells if c.method == method and c.success
           and np.isfinite(c.sharpe)]
    if not sub:
        return pd.DataFrame()
    df = pd.DataFrame([{"Lookback": c.lookback_years,
                        "Frequency": c.frequency, "Sharpe": c.sharpe}
                       for c in sub])
    return df.pivot_table(index="Lookback", columns="Frequency",
                          values="Sharpe", aggfunc="mean")


def cost_sensitivity_frame(cells: list[RobustnessCell],
                           method: str) -> pd.DataFrame:
    """Mean Sharpe per cost level for one method (§6.8)."""
    sub = [c for c in cells if c.method == method and c.success
           and np.isfinite(c.sharpe)]
    if not sub:
        return pd.DataFrame(columns=["Cost (bps)", "Mean Sharpe", "Configs"])
    df = pd.DataFrame([{"Cost (bps)": c.cost_bps, "Sharpe": c.sharpe}
                       for c in sub])
    g = df.groupby("Cost (bps)")["Sharpe"].agg(["mean", "count"]).reset_index()
    return g.rename(columns={"mean": "Mean Sharpe",
                             "count": "Configs"}).sort_values("Cost (bps)")


def breakeven_cost(cost_sharpes: list[tuple[float, float]],
                   reference_sharpe: float) -> dict:
    """Cost level where an edge over ``reference_sharpe`` disappears (§6.9).

    Linear interpolation between the bracketing grid points; reported with
    the bracket (no overstated precision). ``None`` cost when the curves
    never cross inside the grid.
    """
    pts = sorted((float(c), float(s)) for c, s in cost_sharpes
                 if np.isfinite(c) and np.isfinite(s))
    if len(pts) < 2 or not np.isfinite(reference_sharpe):
        return {"breakeven_bps": None, "bracket": None,
                "note": "insufficient grid points"}
    prev = None
    for cost, sharpe in pts:
        if prev is not None:
            c0, s0 = prev
            if (s0 - reference_sharpe) == 0:
                return {"breakeven_bps": round(c0, 1), "bracket": (c0, c0),
                        "note": "approximate (grid interpolation)"}
            if (s0 - reference_sharpe) * (sharpe - reference_sharpe) < 0:
                frac = ((s0 - reference_sharpe)
                        / ((s0 - reference_sharpe)
                           - (sharpe - reference_sharpe)))
                est = c0 + frac * (cost - c0)
                return {"breakeven_bps": round(float(est), 1),
                        "bracket": (c0, cost),
                        "note": "approximate (grid interpolation)"}
        prev = (cost, sharpe)
    return {"breakeven_bps": None, "bracket": None,
            "note": "no crossing inside the cost grid"}


def grid_to_csv(result: RobustnessResult) -> str:
    """CSV of the robustness matrix (§6.15)."""
    return result.frame().to_csv(index=False)

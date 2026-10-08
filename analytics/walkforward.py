"""
walkforward.py
--------------
True walk-forward (rolling/expanding) portfolio optimisation.

What it answers
---------------
"If I had actually run this portfolio construction process through time,
using only information available at each point, how would it have
performed?" — estimate, invest, move forward, re-estimate, trade, repeat.
This differs from the static out-of-sample test (estimate once, freeze,
backtest), which never re-estimates during the test.

CRITICAL no-look-ahead rule
----------------------------
For a rebalance executed on historical trading date ``D``, EVERY
observation used to calculate the target weights satisfies ``date < D``
(strictly — never ``<=``). The execution day's closing price may execute
the already-determined target portfolio but must NOT estimate it::

    history through previous eligible date
      → estimate weights
      → D arrives
      → execute at D close

Correctness notes
-----------------
* Inputs are **base-currency price panels** (Task 6 pipeline: quoted price
  × quote_scale → native → × date-matched historical FX → base). Currency
  conversion is pointwise-causal (bounded forward-fill looks only
  backward), so slicing the panel to ``date < D`` can never embed future FX.
  Estimation always runs on base-currency returns/covariance — never on
  local-currency series converted afterwards.
* Initial allocation for test start ``T`` is estimated on data strictly
  before ``T`` (never on ``[T, E0]`` even when the first tradeable date
  ``E0`` falls after ``T``), then established at ``E0`` with the existing
  backtest entry convention (fully invested, no entry cost — same as the
  static engine; documented, not silently different).
* Between rebalances shares stay fixed; wealth is ``Σ shares × price`` and
  weights drift. At each successful event the engine values holdings,
  trades to the previously-computed targets and solves the cost/target
  circularity exactly with the shared fixed-point solver, so
  ``value_before − cost == value_after`` and
  ``cost == rate × Σ|final executed trades|`` hold by construction.
* Failed optimisations never substitute equal weights. Initial failure →
  structured failure with no wealth path. Later failure → keep holdings,
  skip the rebalance (zero turnover/cost), record the reason, continue.
* The asset universe is fixed for a run (no survivorship repair); see
  ``SURVIVORSHIP_WARNING``. Parameters chosen today may carry hindsight;
  see ``HINDSIGHT_WARNING``.

All functions are pure (inputs never mutated); ordering follows the
canonical ticker order; invalid input raises ``ValueError``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .allocation import (
    annualised_covariance_simple,
    annualised_mean_returns_simple,
    black_litterman_allocation,
    black_litterman_posterior,
    diversification_ratio,
    equal_risk_contribution,
    estimate_risk_aversion_from_benchmark,
    maximum_diversification,
    risk_contributions_simple,
    simple_returns_from_prices,
    validate_covariance,
    validate_delta,
    validate_reference_weights,
    validate_tau,
    estimation_window,
)
from .backtest import (
    REBALANCE_FREQUENCIES,
    apply_transaction_costs,
    backtest_summary,
    calculate_turnover,
    generate_rebalance_dates,
    prepare_backtest_data,
    solve_rebalance_cost_fraction,
    validate_backtest_capital,
    validate_backtest_window,
    validate_frequency,
    validate_transaction_cost_bps,
)
from .optimisation import max_sharpe, min_variance
from .performance import drawdown_series
from .validation import canonical_tickers, validate_weight_bounds

TRADING_DAYS = 252

#: Optimisation methods available walk-forward (plus equal_weight baseline).
WALK_FORWARD_METHODS = (
    "min_variance",
    "max_sharpe",
    "erc",
    "max_diversification",
    "black_litterman",
    "equal_weight",
)

#: Display labels (UI + result names).
WALK_FORWARD_LABELS = {
    "min_variance": "Minimum Variance",
    "max_sharpe": "Maximum Sharpe",
    "erc": "ERC",
    "max_diversification": "Maximum Diversification",
    "black_litterman": "Black-Litterman",
    "equal_weight": "Equal Weight",
}

#: Estimation window shapes.
WINDOW_TYPES = ("rolling", "expanding")

#: Supported trailing lookbacks in calendar years (rolling mode).
LOOKBACK_CHOICES = (1, 2, 3, 5)

#: Rebalance frequencies shared with the static engine (never "never").
WALK_FORWARD_FREQUENCIES = ("monthly", "quarterly", "semi-annual", "annual")

SURVIVORSHIP_WARNING = (
    "The selected universe is fixed using the securities supplied by the "
    "user and may therefore contain survivorship/selection bias. Assets "
    "are never added or removed during a walk-forward run."
)

HINDSIGHT_WARNING = (
    "Lookback, rebalance frequency, transaction costs, method and bounds "
    "are chosen today and may themselves reflect hindsight; walk-forward "
    "results reduce — but do not eliminate — historical bias."
)

BL_HINDSIGHT_WARNING = (
    "Walk-forward Black-Litterman results assume the supplied views were "
    "known at each historical rebalance. If the views were formed using "
    "later information, the analysis contains human hindsight."
)

RF_CONSTANT_NOTE = (
    "The risk-free rate is held constant across historical windows; "
    "historical rate regimes are not reconstructed."
)


# ---------------------------------------------------------------------------
# Structured objects
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    """One walk-forward run specification (all dates inclusive calendar)."""

    method: str = "erc"
    test_start: object = None
    test_end: object = None
    lookback_years: float = 3.0
    window_type: str = "rolling"
    rebalance_frequency: str = "quarterly"
    transaction_cost_bps: float = 10.0
    base_currency: str = "USD"
    min_weight: float = 0.0
    max_weight: float = 1.0
    risk_free_annual: float = 0.05
    min_est_rows: int = 60
    # Black-Litterman only (views fixed through the test — see warning):
    bl_views: list = field(default_factory=list)
    bl_ref: str = "equal"  # "equal" | "retrospective_current"
    bl_ref_weights: object = None  # required when bl_ref is retrospective
    bl_delta: float = 2.5
    bl_delta_from_benchmark: bool = False
    bl_tau: float = 0.05

    def validated(self, tickers: list[str]) -> tuple[list[str], "WalkForwardConfig"]:
        """Validate in place; returns (canonical tickers, self)."""
        tickers = canonical_tickers(list(tickers))
        if self.method not in WALK_FORWARD_METHODS:
            raise ValueError(
                f"Unknown walk-forward method {self.method!r}; expected one of "
                f"{list(WALK_FORWARD_METHODS)}."
            )
        try:
            start_ts, end_ts = validate_backtest_window(
                self.test_start, self.test_end)
        except ValueError as e:
            raise ValueError(f"Invalid walk-forward test window: {e}")
        self.test_start, self.test_end = start_ts, end_ts
        try:
            lookback = float(self.lookback_years)
        except (TypeError, ValueError):
            raise ValueError(
                f"Lookback must be a number of years, got {self.lookback_years!r}.")
        if not np.isfinite(lookback) or lookback <= 0:
            raise ValueError(f"Lookback must be positive years, got {lookback!r}.")
        self.lookback_years = lookback
        wt = str(self.window_type).strip().lower()
        if wt not in WINDOW_TYPES:
            raise ValueError(
                f"Unknown window type {self.window_type!r}; expected one of "
                f"{list(WINDOW_TYPES)}."
            )
        self.window_type = wt
        freq = validate_frequency(self.rebalance_frequency)
        if freq == "never" or freq not in WALK_FORWARD_FREQUENCIES:
            raise ValueError(
                f"Walk-forward needs a periodic frequency, got "
                f"{self.rebalance_frequency!r}; expected one of "
                f"{list(WALK_FORWARD_FREQUENCIES)}."
            )
        self.rebalance_frequency = freq
        self.transaction_cost_bps = validate_transaction_cost_bps(
            self.transaction_cost_bps)
        validate_weight_bounds(len(tickers), self.min_weight, self.max_weight)
        if not np.isfinite(self.risk_free_annual):
            raise ValueError("Risk-free rate must be finite.")
        try:
            min_rows = int(self.min_est_rows)
        except (TypeError, ValueError):
            raise ValueError(
                f"Minimum estimation rows must be an integer, "
                f"got {self.min_est_rows!r}.")
        if min_rows < 2:
            raise ValueError("Minimum estimation rows must be at least 2.")
        self.min_est_rows = min_rows
        if self.method == "black_litterman":
            validate_delta(self.bl_delta)
            validate_tau(self.bl_tau)
            if str(self.bl_ref).strip().lower() not in (
                    "equal", "retrospective_current"):
                raise ValueError(
                    "BL reference must be 'equal' (genuine OOS) or "
                    "'retrospective_current' (labelled retrospective).")
            self.bl_ref = str(self.bl_ref).strip().lower()
            if self.bl_ref == "retrospective_current":
                if self.bl_ref_weights is None:
                    raise ValueError(
                        "Retrospective BL reference needs explicit weights.")
                validate_reference_weights(
                    np.asarray(self.bl_ref_weights, dtype=float), tickers)
        return tickers, self


@dataclass
class WalkForwardResult:
    """Structured outcome of one walk-forward strategy run."""

    name: str
    success: bool
    message: str
    values: pd.Series
    log_returns: pd.Series
    drawdowns: pd.Series
    target_weights: pd.DataFrame       # successful events only (date × ticker)
    realised_weights: pd.DataFrame | None  # daily drift weights
    events: pd.DataFrame               # §40 columns incl. Reason + ex-ante
    failures: pd.DataFrame             # failed/skipped events (date/method/reason)
    trades: pd.DataFrame | None
    turnover: pd.Series                # daily (0.0 off-event)
    costs: pd.Series                   # daily (0.0 off-event)
    metrics: dict
    notes: list = field(default_factory=list)
    is_benchmark: bool = False
    method: str = ""
    config: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scheduling + estimation windows (strict date < execution)
# ---------------------------------------------------------------------------

def walk_forward_schedule(
    index: pd.DatetimeIndex, frequency: str
) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    """Initial establishment date + rebalance execution dates.

    The first index date is the initial allocation (never a rebalance);
    rebalances follow :func:`generate_rebalance_dates` (first trading day
    on/after each calendar mark, last bar excluded).
    """
    frequency = validate_frequency(frequency)
    if frequency not in WALK_FORWARD_FREQUENCIES:
        raise ValueError(
            f"Walk-forward needs a periodic frequency, got {frequency!r}."
        )
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        raise ValueError("Walk-forward scheduling needs at least 2 trading days.")
    return index[0], generate_rebalance_dates(index, frequency)


def estimation_slice(
    panel: pd.DataFrame,
    tickers: list[str],
    cutoff,
    lookback_years: float | None,
    window_type: str,
    min_rows: int = 60,
) -> tuple[pd.DataFrame, dict]:
    """Estimation prices strictly before ``cutoff`` (never ``>=``).

    Rolling (``lookback_years`` required) delegates to the shared
    :func:`estimation_window` (``[cutoff − lookback, cutoff)`` on common
    trading dates). Expanding (``lookback_years`` ignored) uses all common
    observations from the panel start through ``cutoff`` (exclusive) with
    the same validity checks. Raises on insufficient/invalid data.
    """
    tickers = canonical_tickers(list(tickers))
    try:
        cut = pd.Timestamp(cutoff)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid estimation cutoff date: {e}")
    wt = str(window_type).strip().lower()
    if wt not in WINDOW_TYPES:
        raise ValueError(f"Unknown window type {window_type!r}.")
    if panel is None or panel.empty:
        raise ValueError("No price history supplied for estimation.")
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise ValueError("Price history needs a DatetimeIndex.")
    if wt == "rolling":
        if lookback_years is None:
            raise ValueError("Rolling estimation needs a lookback in years.")
        return estimation_window(
            panel, tickers, cut, float(lookback_years), min_rows=min_rows)
    # Expanding: same contract as estimation_window, explicit slice.
    missing = [t for t in tickers if t not in panel.columns]
    if missing:
        raise ValueError(f"Price data is missing tickers: {missing}")
    est = panel.reindex(columns=tickers).sort_index()
    est = est.loc[est.index < cut].dropna(how="any")
    if est.empty:
        raise ValueError(
            f"No estimation data strictly before {cut.date()} "
            "(expanding window)."
        )
    bad = est.columns[((est <= 0) | (~np.isfinite(est))).any()].tolist()
    if bad:
        raise ValueError(f"Invalid prices for: {', '.join(bad)}.")
    if len(est) < min_rows:
        raise ValueError(
            f"Only {len(est)} estimation observations "
            f"({est.index[0].date()} to {est.index[-1].date()}); "
            f"at least {min_rows} are required."
        )
    info = {
        "est_start": est.index[0].date(),
        "est_end": est.index[-1].date(),
        "n_obs": int(len(est)),
        "cutoff": cut.date(),
        "lookback_years": None,
    }
    return est.copy(), info


def earliest_feasible_test_start(
    panel: pd.DataFrame,
    tickers: list[str],
    lookback_years: float,
    window_type: str = "rolling",
    min_rows: int = 60,
):
    """Earliest test start with sufficient strictly-prior estimation data.

    Rolling: first common observation + lookback (calendar). Expanding: the
    date of the ``min_rows``-th common observation. Used to explain history
    deficits instead of silently shortening the lookback (§27).
    """
    tickers = canonical_tickers(list(tickers))
    if panel is None or panel.empty:
        raise ValueError("No price history supplied.")
    common = panel.reindex(columns=tickers).sort_index().dropna(how="any")
    if common.empty:
        raise ValueError("No common price history for the selected universe.")
    wt = str(window_type).strip().lower()
    if wt == "expanding":
        if len(common) < min_rows:
            raise ValueError(
                f"Only {len(common)} common observations; at least {min_rows} "
                "are required before any test start."
            )
        return common.index[min_rows - 1].date()
    try:
        lookback = float(lookback_years)
    except (TypeError, ValueError):
        raise ValueError(f"Lookback must be a number of years, got {lookback_years!r}.")
    if not np.isfinite(lookback) or lookback <= 0:
        raise ValueError(f"Lookback must be positive years, got {lookback!r}.")
    return (common.index[0] + pd.DateOffset(years=lookback)).date()


# ---------------------------------------------------------------------------
# Per-event estimation (base-currency returns → optimiser)
# ---------------------------------------------------------------------------

def _estimation_stats(est: pd.DataFrame, tickers: list[str]):
    """Simple-return mean/covariance in annual units + validation."""
    rets = simple_returns_from_prices(est, tickers)
    mu = annualised_mean_returns_simple(rets)
    cov = annualised_covariance_simple(rets)
    cov_df, _ = validate_covariance(cov, tickers)
    return rets, mu, cov_df


def estimate_event_weights(
    est: pd.DataFrame,
    tickers: list[str],
    method: str,
    risk_free_annual: float,
    min_weight: float,
    max_weight: float,
    bl_views: list | None = None,
    bl_ref_weights: np.ndarray | None = None,
    bl_delta: float = 2.5,
    bl_tau: float = 0.05,
) -> tuple[pd.Series | None, dict, str]:
    """Target weights from one estimation window (no look-ahead by caller).

    Returns ``(weights, ex_ante, message)``; ``weights`` is None on any
    optimiser failure (never a fallback). ``ex_ante`` carries estimated
    return/volatility/sharpe/diversification-ratio where the method
    naturally provides them (NaN otherwise — never invented).
    """
    tickers = canonical_tickers(list(tickers))
    nan = float("nan")
    blank = {"est_return": nan, "est_volatility": nan, "est_sharpe": nan,
             "diversification_ratio": nan}
    if method not in WALK_FORWARD_METHODS:
        return None, blank, f"Unknown method {method!r}."
    try:
        validate_weight_bounds(len(tickers), min_weight, max_weight)
    except ValueError as e:
        return None, blank, str(e)
    n = len(tickers)
    if method == "equal_weight":
        w = np.ones(n) / n
        if np.any(w < min_weight - 1e-9) or np.any(w > max_weight + 1e-9):
            return None, blank, (
                "Equal weights violate the configured weight bounds.")
        return pd.Series(w, index=tickers), dict(blank), "Equal weights."
    try:
        _rets, mu, cov_df = _estimation_stats(est, tickers)
    except ValueError as e:
        return None, blank, f"Estimation statistics failed: {e}"
    mu_arr = mu.values
    if method == "min_variance":
        res = min_variance(mu_arr, cov_df.values, risk_free_annual,
                           min_weight, max_weight)
        if not res.success or res.weights is None:
            return None, blank, f"Minimum Variance failed: {res.message}"
        ex = {"est_return": float(res.expected_return),
              "est_volatility": float(res.volatility),
              "est_sharpe": float(res.sharpe),
              "diversification_ratio": nan}
        return pd.Series(np.asarray(res.weights, dtype=float), index=tickers), ex, res.message
    if method == "max_sharpe":
        res = max_sharpe(mu_arr, cov_df.values, risk_free_annual,
                         min_weight, max_weight)
        if not res.success or res.weights is None:
            return None, blank, f"Maximum Sharpe failed: {res.message}"
        ex = {"est_return": float(res.expected_return),
              "est_volatility": float(res.volatility),
              "est_sharpe": float(res.sharpe),
              "diversification_ratio": nan}
        return pd.Series(np.asarray(res.weights, dtype=float), index=tickers), ex, res.message
    if method == "erc":
        res = equal_risk_contribution(tickers, mu_arr, cov_df, risk_free_annual,
                                      min_weight, max_weight)
        if not res.success or res.weights is None:
            return None, blank, f"ERC failed: {res.message}"
        ex = {"est_return": float(res.expected_return),
              "est_volatility": float(res.volatility),
              "est_sharpe": float(res.sharpe),
              "diversification_ratio": float(res.diversification_ratio)}
        return res.weights.reindex(tickers), ex, res.message
    if method == "max_diversification":
        res = maximum_diversification(tickers, mu_arr, cov_df, risk_free_annual,
                                      min_weight, max_weight)
        if not res.success or res.weights is None:
            return None, blank, f"Maximum Diversification failed: {res.message}"
        ex = {"est_return": float(res.expected_return),
              "est_volatility": float(res.volatility),
              "est_sharpe": float(res.sharpe),
              "diversification_ratio": float(res.diversification_ratio)}
        return res.weights.reindex(tickers), ex, res.message
    # Black-Litterman: fixed views + reference through time; covariance
    # (hence prior/posterior/weights) evolves with the window.
    try:
        ref_w = (np.ones(n) / n if bl_ref_weights is None
                 else validate_reference_weights(
                     np.asarray(bl_ref_weights, dtype=float), tickers))
        post = black_litterman_posterior(tickers, cov_df, ref_w, bl_delta,
                                         bl_tau, bl_views or [])
    except ValueError as e:
        return None, blank, f"Black-Litterman inputs invalid: {e}"
    if not post.success or post.posterior is None:
        return None, blank, f"Black-Litterman posterior failed: {post.message}"
    alloc = black_litterman_allocation(
        tickers, post.posterior, cov_df, bl_delta, risk_free_annual,
        min_weight, max_weight)
    if not alloc.success or alloc.weights is None:
        return None, blank, f"Black-Litterman allocation failed: {alloc.message}"
    ex = {"est_return": float(alloc.expected_return),
          "est_volatility": float(alloc.volatility),
          "est_sharpe": float(alloc.sharpe),
          "diversification_ratio": (float(alloc.diversification_ratio)
                                    if np.isfinite(alloc.diversification_ratio)
                                    else nan)}
    return alloc.weights.reindex(tickers), ex, alloc.message


# ---------------------------------------------------------------------------
# Walk-forward run
# ---------------------------------------------------------------------------

def _empty_result(name: str, method: str, message: str, config: dict,
                  notes: list) -> WalkForwardResult:
    """Structured failure with no wealth path (initial estimation failed)."""
    empty_idx = pd.DatetimeIndex([])
    cols = ["Execution Date", "Training Start", "Training End", "Method",
            "Status", "Turnover", "Transaction Cost", "Portfolio Value Before",
            "Portfolio Value After", "Reason", "Est Return", "Est Volatility",
            "Est Sharpe", "Diversification Ratio"]
    return WalkForwardResult(
        name=name, success=False, message=message,
        values=pd.Series(dtype=float, name="value"),
        log_returns=pd.Series(dtype=float, name="strategy"),
        drawdowns=pd.Series(dtype=float),
        target_weights=pd.DataFrame(),
        realised_weights=None,
        events=pd.DataFrame(columns=cols),
        failures=pd.DataFrame(
            columns=["Execution Date", "Method", "Reason"]),
        trades=None,
        turnover=pd.Series(dtype=float, name="turnover"),
        costs=pd.Series(dtype=float, name="costs"),
        metrics={},
        notes=list(notes),
        is_benchmark=False, method=method, config=dict(config),
    )


def run_walk_forward(
    base_prices: pd.DataFrame,
    tickers: list[str],
    config: WalkForwardConfig,
    initial_capital: float,
    benchmark_base: pd.Series | None = None,
    name: str | None = None,
    estimator=None,
) -> WalkForwardResult:
    """Execute one walk-forward strategy over a base-currency panel.

    Args:
        base_prices: base-currency price panel (full history incl. the
            pre-test estimation era; Task 6 pipeline output).
        tickers: asset universe (fixed for the run).
        config: validated :class:`WalkForwardConfig`.
        initial_capital: starting wealth.
        benchmark_base: optional base-currency benchmark series, used ONLY
            for per-window BL δ estimation (never for weight estimation).
        name: result label (defaults to the method label).
        estimator: optional ``fn(est_panel, exec_date) →
            (weights|None, ex_ante, message)`` overriding the method
            dispatch (testing seam; production passes None).

    Every estimation slice satisfies ``date < execution date``; the initial
    slice satisfies ``date < test_start`` even when the first tradeable date
    falls later. Execution uses the execution-day close on already-computed
    targets only.
    """
    tickers, config = config.validated(list(tickers))
    initial = validate_backtest_capital(initial_capital)
    label = WALK_FORWARD_LABELS[config.method]
    name = name or f"Walk-Forward {label}"
    cfg_dict = {
        "method": config.method, "test_start": str(config.test_start.date()),
        "test_end": str(config.test_end.date()),
        "lookback_years": config.lookback_years,
        "window_type": config.window_type,
        "rebalance_frequency": config.rebalance_frequency,
        "transaction_cost_bps": config.transaction_cost_bps,
        "base_currency": config.base_currency,
        "min_weight": config.min_weight, "max_weight": config.max_weight,
        "risk_free_annual": config.risk_free_annual,
        "min_est_rows": config.min_est_rows,
    }
    notes: list = []
    if config.method == "black_litterman":
        notes.append(BL_HINDSIGHT_WARNING)
        if config.bl_ref == "retrospective_current":
            notes.append(
                "BL reference uses today's Current Portfolio weights — "
                "labelled RETROSPECTIVE, excluded from strict OOS claims.")
    notes.append(RF_CONSTANT_NOTE)

    if base_prices is None or base_prices.empty:
        return _empty_result(name, config.method,
                             "No price history supplied.", cfg_dict, notes)
    full = base_prices.reindex(columns=tickers).sort_index()
    if not isinstance(full.index, pd.DatetimeIndex):
        return _empty_result(name, config.method,
                             "Price history needs a DatetimeIndex.",
                             cfg_dict, notes)

    # Pre-test history gate (§27): never silently shorten the lookback.
    try:
        if config.window_type == "rolling":
            earliest = earliest_feasible_test_start(
                full, tickers, config.lookback_years, "rolling",
                config.min_est_rows)
            if config.test_start.date() < earliest:
                return _empty_result(
                    name, config.method,
                    f"Insufficient pre-test history for a "
                    f"{config.lookback_years:g}-year lookback: earliest valid "
                    f"test start is {earliest}.", cfg_dict, notes)
        else:
            earliest = earliest_feasible_test_start(
                full, tickers, config.lookback_years, "expanding",
                config.min_est_rows)
            if config.test_start.date() < earliest:
                return _empty_result(
                    name, config.method,
                    f"Insufficient pre-test history: at least "
                    f"{config.min_est_rows} common observations are required "
                    f"before the test start (earliest valid: {earliest}).",
                    cfg_dict, notes)
    except ValueError as e:
        return _empty_result(name, config.method, f"History check: {e}",
                             cfg_dict, notes)

    try:
        window, prep_notes = prepare_backtest_data(
            full, tickers, config.test_start, config.test_end)
    except ValueError as e:
        return _empty_result(name, config.method, f"Test window: {e}",
                             cfg_dict, notes)
    notes.extend(prep_notes)

    try:
        e0, reb_dates = walk_forward_schedule(window.index,
                                              config.rebalance_frequency)
    except ValueError as e:
        return _empty_result(name, config.method, f"Scheduling: {e}",
                             cfg_dict, notes)
    exec_dates = [e0] + list(reb_dates)
    n = len(window.index)
    reb_set = set(reb_dates)

    if estimator is None:
        estimator = _default_estimator(
            config, tickers, benchmark_base, notes)

    # --- estimation pass (all targets fixed before any trading) ---
    targets: dict = {}
    event_meta: dict = {}
    for i, d in enumerate(exec_dates):
        cutoff = config.test_start if i == 0 else d
        try:
            est, info = estimation_slice(
                full, tickers, cutoff,
                config.lookback_years if config.window_type == "rolling" else None,
                config.window_type, config.min_est_rows)
        except ValueError as e:
            event_meta[d] = (None, str(e), {}, None)
            continue
        w, ex_ante, msg = estimator(est, d)
        if w is None:
            event_meta[d] = (None, msg, ex_ante, info)
            continue
        wv = np.asarray(pd.Series(w).reindex(tickers), dtype=float)
        if wv.shape != (len(tickers),) or not np.all(np.isfinite(wv)):
            event_meta[d] = (None, "Estimator returned invalid weights.",
                             ex_ante, info)
            continue
        targets[d] = pd.Series(wv, index=tickers)
        event_meta[d] = (pd.Series(wv, index=tickers), msg, ex_ante, info)

    if e0 not in targets:
        reason = event_meta.get(e0, (None, "No estimation.", {}, None))[1]
        fail = pd.DataFrame(
            [{"Execution Date": e0, "Method": label, "Reason": reason}])
        res = _empty_result(
            name, config.method,
            f"Initial optimisation failed ({reason}); no wealth path "
            "generated — refusing to invent weights.", cfg_dict, notes)
        res.failures = fail
        return res

    # --- share-space execution (mirrors the static engine's conventions) ---
    P = window.copy()
    dates = P.index
    cuts = [0] + sorted(dates.get_loc(d) for d in reb_set if d in dates) + [n - 1]
    cuts = sorted(set(cuts))

    w0 = targets[e0].values
    shares = initial * w0 / P.iloc[0].values
    value_parts: list[pd.Series] = []
    weight_parts: list[pd.DataFrame] = []
    turnover_daily = pd.Series(0.0, index=dates)
    costs_daily = pd.Series(0.0, index=dates)
    event_rows: list[dict] = []
    fail_rows: list[dict] = []
    trade_rows: list[dict] = []

    def _record(d, status, turnover, cost, vb, va, reason, ex_ante, info):
        event_rows.append({
            "Execution Date": d,
            "Training Start": (info.get("est_start") if info else None),
            "Training End": (info.get("est_end") if info else None),
            "Method": label,
            "Status": status,
            "Turnover": float(turnover),
            "Transaction Cost": float(cost),
            "Portfolio Value Before": float(vb),
            "Portfolio Value After": float(va),
            "Reason": reason,
            "Est Return": float(ex_ante.get("est_return", float("nan"))),
            "Est Volatility": float(ex_ante.get("est_volatility", float("nan"))),
            "Est Sharpe": float(ex_ante.get("est_sharpe", float("nan"))),
            "Diversification Ratio": float(
                ex_ante.get("diversification_ratio", float("nan"))),
        })

    # Initial establishment (existing entry convention: fully invested,
    # no entry cost — same as the static backtest engine).
    _w, _msg, _ex, _info = event_meta[e0]
    _record(e0, "established", 0.0, 0.0, initial, initial,
            f"Initial allocation ({_msg})", _ex, _info)

    for seg_i in range(len(cuts) - 1):
        lo, hi = cuts[seg_i], cuts[seg_i + 1]
        seg = P.iloc[lo:hi + 1]
        holdings = seg * shares
        seg_values = holdings.sum(axis=1)
        d = dates[hi]
        is_event = d in reb_set and hi < n - 1
        if is_event:
            value_before = float(seg_values.iloc[-1])
            if d in targets:
                w = targets[d].values
                current = holdings.iloc[-1].values
                current_w = current / value_before
                cost_frac = solve_rebalance_cost_fraction(
                    current_w, w, config.transaction_cost_bps)
                target_alloc = value_before * w * (1.0 - cost_frac)
                trade = target_alloc - current
                cost = apply_transaction_costs(trade, config.transaction_cost_bps)
                turnover = calculate_turnover(trade, value_before)
                value_after = value_before - cost
                seg_values.iloc[-1] = value_after
                turnover_daily.loc[d] = turnover
                costs_daily.loc[d] = cost
                _w2, _msg2, _ex2, _info2 = event_meta[d]
                _record(d, "success", turnover, cost, value_before,
                        value_after, _msg2, _ex2, _info2)
                leg_costs = np.abs(trade) * config.transaction_cost_bps / 10_000.0
                for j, t in enumerate(tickers):
                    trade_rows.append({
                        "Date": d, "Ticker": t,
                        "Before Weight": float(current_w[j]),
                        "Target Weight": float(w[j]),
                        "Trade Value": float(trade[j]),
                        "Transaction Cost": float(leg_costs[j]),
                    })
                shares = target_alloc / P.iloc[hi].values
                post_w = w
            else:
                _w2, _msg2, _ex2, _info2 = event_meta[d]
                _record(d, "failed/skipped", 0.0, 0.0, value_before,
                        value_before,
                        f"Held existing positions ({_msg2})", _ex2, _info2)
                fail_rows.append({"Execution Date": d, "Method": label,
                                  "Reason": _msg2})
                post_w = holdings.iloc[-1].values / value_before
        value_parts.append(seg_values)
        w_frame = holdings.div(
            seg_values.replace(0, np.nan), axis=0).fillna(0.0)
        if is_event and d in targets:
            w_frame.iloc[-1] = targets[d].values
        weight_parts.append(w_frame)

    values = pd.concat(value_parts).rename("value")
    values = values[~values.index.duplicated(keep="first")]
    realised = pd.concat(weight_parts)
    realised = realised[~realised.index.duplicated(keep="first")]

    events = pd.DataFrame(event_rows)
    failures = pd.DataFrame(fail_rows,
                            columns=["Execution Date", "Method", "Reason"])
    target_df = pd.DataFrame(
        {d: targets[d] for d in exec_dates if d in targets}).T
    target_df.index.name = "Execution Date"
    trades = pd.DataFrame(trade_rows, columns=[
        "Date", "Ticker", "Before Weight", "Target Weight",
        "Trade Value", "Transaction Cost"])

    turnover_pos = turnover_daily[turnover_daily > 0]
    metrics = backtest_summary(
        values, initial, config.risk_free_annual,
        float(costs_daily.sum()), int((events["Status"] == "success").sum()),
        turnover_pos)
    n_sched = int((events["Status"] != "established").sum())
    n_ok = int((events["Status"] == "success").sum())
    n_fail = int((events["Status"] == "failed/skipped").sum())
    metrics.update({
        "n_scheduled": n_sched,
        "n_successful": n_ok,
        "n_failed": n_fail,
        "median_turnover": float(turnover_pos.median()) if len(turnover_pos) else 0.0,
        "max_turnover": float(turnover_pos.max()) if len(turnover_pos) else 0.0,
        "cumulative_turnover": float(turnover_pos.sum()),
        "weight_stability_l1_mean": _target_stability(target_df, "mean"),
        "weight_stability_l1_max": _target_stability(target_df, "max"),
    })

    log_rets = np.log(values / values.shift(1)).dropna().rename("strategy")
    return WalkForwardResult(
        name=name, success=True, message="Completed",
        values=values, log_returns=log_rets,
        drawdowns=drawdown_series(values),
        target_weights=target_df, realised_weights=realised,
        events=events, failures=failures,
        trades=trades if not trades.empty else pd.DataFrame(columns=[
            "Date", "Ticker", "Before Weight", "Target Weight",
            "Trade Value", "Transaction Cost"]),
        turnover=turnover_daily.rename("turnover"),
        costs=costs_daily.rename("costs"),
        metrics=metrics, notes=notes, is_benchmark=False,
        method=config.method, config=cfg_dict,
    )


def _target_stability(target_df: pd.DataFrame, kind: str) -> float:
    """Mean/max L1 distance between consecutive target vectors."""
    if target_df is None or len(target_df) < 2:
        return 0.0
    dists = np.abs(np.diff(target_df.values.astype(float), axis=0)).sum(axis=1)
    dists = dists[np.isfinite(dists)]
    if len(dists) == 0:
        return 0.0
    return float(dists.max() if kind == "max" else dists.mean())


def _default_estimator(config: WalkForwardConfig, tickers: list[str],
                       benchmark_base: pd.Series | None, notes: list):
    """Build the per-event estimator for a configured method.

    The estimator closes over fixed views/reference/delta-mode; only the
    estimation panel (strictly pre-execution) varies per event, so
    posteriors and weights evolve purely from evolving data.
    """
    tickers = canonical_tickers(list(tickers))
    n = len(tickers)
    if config.bl_ref == "retrospective_current":
        ref_fixed = validate_reference_weights(
            np.asarray(config.bl_ref_weights, dtype=float), tickers)
    else:
        ref_fixed = np.ones(n) / n

    def _estimate(est: pd.DataFrame, exec_date) -> tuple:
        if config.method == "black_litterman":
            delta = config.bl_delta
            if config.bl_delta_from_benchmark:
                try:
                    delta = _window_delta(
                        benchmark_base, est.index[0], exec_date,
                        config.risk_free_annual)
                except ValueError:
                    delta = config.bl_delta
            return estimate_event_weights(
                est, tickers, "black_litterman", config.risk_free_annual,
                config.min_weight, config.max_weight,
                bl_views=list(config.bl_views or []),
                bl_ref_weights=ref_fixed, bl_delta=delta,
                bl_tau=config.bl_tau)
        return estimate_event_weights(
            est, tickers, config.method, config.risk_free_annual,
            config.min_weight, config.max_weight)

    return _estimate


def _window_delta(benchmark_base: pd.Series | None, est_start,
                  exec_date, risk_free_annual: float) -> float:
    """Benchmark-implied δ from training-window observations only (< D)."""
    if benchmark_base is None or len(benchmark_base) == 0:
        raise ValueError("no benchmark series for δ estimation")
    b = pd.Series(benchmark_base, dtype=float)
    if not isinstance(b.index, pd.DatetimeIndex):
        raise ValueError("benchmark needs a DatetimeIndex")
    win = b.loc[(b.index >= pd.Timestamp(est_start))
                & (b.index < pd.Timestamp(exec_date))].dropna()
    if len(win) < 2:
        raise ValueError("too few benchmark points in the training window")
    r = win.pct_change().dropna()
    if len(r) < 2:
        raise ValueError("too few benchmark returns in the training window")
    return estimate_risk_aversion_from_benchmark(
        float(r.mean() * TRADING_DAYS - risk_free_annual),
        float(r.var() * TRADING_DAYS))


# ---------------------------------------------------------------------------
# Diagnostics helpers (UI + tests)
# ---------------------------------------------------------------------------

def ex_ante_realised_frame(result: WalkForwardResult) -> pd.DataFrame:
    """Per-event estimated-vs-status table (never claims prediction)."""
    if result is None or result.events is None or result.events.empty:
        return pd.DataFrame(columns=[
            "Execution Date", "Status", "Est Return", "Est Volatility",
            "Est Sharpe"])
    cols = ["Execution Date", "Status", "Est Return", "Est Volatility",
            "Est Sharpe"]
    return result.events[[c for c in cols if c in result.events.columns]].copy()


def turnover_diagnostics(result: WalkForwardResult) -> dict:
    """Institutional turnover/cost diagnostics from a completed run."""
    if result is None or not result.success:
        return {"n_scheduled": 0, "n_successful": 0, "n_failed": 0,
                "avg_turnover": 0.0, "median_turnover": 0.0,
                "max_turnover": 0.0, "cumulative_turnover": 0.0,
                "total_costs": 0.0}
    m = result.metrics
    return {
        "n_scheduled": int(m.get("n_scheduled", 0)),
        "n_successful": int(m.get("n_successful", 0)),
        "n_failed": int(m.get("n_failed", 0)),
        "avg_turnover": float(m.get("avg_turnover", 0.0)),
        "median_turnover": float(m.get("median_turnover", 0.0)),
        "max_turnover": float(m.get("max_turnover", 0.0)),
        "cumulative_turnover": float(m.get("cumulative_turnover", 0.0)),
        "total_costs": float(m.get("total_costs", 0.0)),
    }

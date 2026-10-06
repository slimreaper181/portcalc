"""
scenario.py
-----------
Forward-looking scenario and Monte Carlo simulation engine.

Provides:
  - simulate_portfolio_paths    : GBM simulation (memory-safe, chunked)
  - run_predefined_scenario     : Apply stress scenario shocks
  - calculate_outcome_distribution : Summary stats from simulation paths
  - summarise_future_metrics    : Percentile time-series DataFrame
  - explain_scenario_results    : Natural language explainability
  - plot_simulation_paths       : Plotly chart of sample paths
  - plot_final_distribution     : Plotly histogram of final values
  - plot_confidence_bands       : Plotly confidence interval bands

Return-type conventions
-----------------------
* Simulations evolve in **log-return** space
  (``V_{t+1} = V_t * exp(r_t)`` with ``r_t`` a portfolio log return) which is
  the correct GBM discretisation.
* ``calculate_outcome_distribution`` / ``summarise_future_metrics`` compare
  final wealth against **total contributed capital**
  (initial investment + all monthly contributions), not the initial value
  alone, so probability-of-loss and profit figures are correct when
  contributions are enabled.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .validation import (
    validate_horizon_years,
    validate_monthly_contrib,
    validate_portfolio_value,
    validate_simulation_count,
)

# ---------------------------------------------------------------------------
# Predefined scenario definitions
# ---------------------------------------------------------------------------

SCENARIO_DEFINITIONS = {
    "Normal": {
        "label": "Normal Market",
        "return_shock": 0.0,
        "vol_multiplier": 1.0,
        "description": "Expected market conditions based on historical data.",
    },
    "Market Crash": {
        "label": "Market Crash",
        "return_shock": -0.20,
        "vol_multiplier": 1.8,
        "description": "Sudden 20% drop in returns with elevated volatility (1.8x).",
    },
    "Bull Market": {
        "label": "Bull Market",
        "return_shock": 0.40,
        "vol_multiplier": 0.85,
        "description": "Strong upward momentum (+40% return boost) with slightly lower vol.",
    },
    "High Volatility": {
        "label": "High Volatility Regime",
        "return_shock": 0.0,
        "vol_multiplier": 2.0,
        "description": "Turbulent market: returns unchanged but volatility doubled.",
    },
}

# Simulations are generated in chunks of this many paths so that peak memory
# stays bounded (~chunk * steps floats) no matter how many sims are requested.
_SIM_CHUNK_SIZE = 2_000
# Hard cap on stored path entries (n_sims * (steps + 1)); above this the
# caller gets a clear error instead of an out-of-memory crash.
_MAX_PATH_ENTRIES = 60_000_000


def _month_step(trading_days: int) -> int:
    """Number of trading days treated as one calendar month (~21 for 252)."""
    return max(int(round(trading_days / 12)), 1)


def count_deposits(steps: int, trading_days: int = 252) -> int:
    """Number of monthly deposits made over ``steps`` trading days.

    Deposits land at the end of each approximate calendar month
    (steps ``month_step, 2*month_step, ...``), so this is
    ``steps // month_step``.
    """
    return int(steps // _month_step(trading_days))


def total_contributed_capital(
    initial_value: float,
    monthly_contrib: float,
    years: float,
    trading_days: int = 252,
) -> tuple[float, float, float]:
    """Return ``(initial, contributions, total)`` capital figures.

    * ``initial`` — the starting investment.
    * ``contributions`` — ``monthly_contrib * n_deposits`` where
      ``n_deposits`` is the number of month-ends inside the horizon.
    * ``total`` — initial + contributions (the break-even reference).
    """
    steps = int(round(float(years) * trading_days))
    n_dep = count_deposits(steps, trading_days)
    contributions = float(monthly_contrib) * n_dep
    return float(initial_value), contributions, float(initial_value) + contributions


# ---------------------------------------------------------------------------
# Core simulation (memory-safe)
# ---------------------------------------------------------------------------

def simulate_portfolio_paths(
    weights: np.ndarray,
    mu_annual: np.ndarray,
    cov_annual: np.ndarray,
    initial_value: float,
    years: float = 5.0,
    n_sims: int = 10_000,
    monthly_contrib: float = 0.0,
    trading_days: int = 252,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate forward portfolio value paths using Geometric Brownian Motion.

    Memory safety: instead of allocating an
    ``(n_sims, steps, n_assets)`` tensor of asset returns (which reaches
    multiple GB for 20k sims x 10y x 15 assets), the fixed-mix portfolio is
    collapsed to a single univariate process first — the weighted sum of
    jointly-normal asset log returns is exactly normal with mean
    ``w . mu_daily`` and variance ``w' Σ_daily w`` — and paths are generated
    in chunks of ``_SIM_CHUNK_SIZE`` simulations. Peak working memory is
    therefore bounded by ``chunk * steps`` floats regardless of ``n_sims``.

    Monthly contributions (if any) are added at the end of each approximate
    calendar month (every ``round(trading_days/12)`` trading days).

    Args:
        weights:         Portfolio weight vector (must sum to 1; never mutated).
        mu_annual:       Annualised mean (log) returns per asset (never mutated).
        cov_annual:      Annualised covariance matrix (never mutated).
        initial_value:   Starting portfolio value in currency units (> 0).
        years:           Simulation horizon in years (> 0).
        n_sims:          Number of Monte Carlo paths (>= 100, capped).
        monthly_contrib: Monthly cash contribution in currency units (>= 0),
                         added at each month-end step.
        trading_days:    Trading days per year (default 252).
        seed:            Random seed for reproducibility (deterministic: the
                         same seed and inputs always give identical paths).

    Returns:
        ndarray of shape (n_sims, steps + 1) with portfolio values at each step.
        Step 0 is always initial_value. Inputs are never mutated.
    """
    w = np.asarray(weights, dtype=float)
    mu_a = np.asarray(mu_annual, dtype=float)
    cov_a = np.asarray(cov_annual, dtype=float)

    n_sims = validate_simulation_count(int(n_sims))
    years = validate_horizon_years(years)
    initial_value = validate_portfolio_value(initial_value)
    monthly_contrib = validate_monthly_contrib(monthly_contrib)
    trading_days = int(trading_days)
    if trading_days < 1:
        raise ValueError(f"trading_days must be >= 1, got {trading_days!r}.")

    n = w.shape[0]
    if mu_a.shape != (n,) or cov_a.shape != (n, n):
        raise ValueError(
            f"Weights shape {w.shape}, mu shape {mu_a.shape} and cov shape "
            f"{cov_a.shape} are inconsistent."
        )
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(mu_a)):
        raise ValueError("Weights and expected returns must be finite.")
    if not np.all(np.isfinite(cov_a)):
        raise ValueError("Covariance matrix must contain only finite values.")

    steps = int(round(years * trading_days))
    if steps < 1:
        raise ValueError("Horizon is too short: it contains zero trading days.")
    if n_sims * (steps + 1) > _MAX_PATH_ENTRIES:
        raise ValueError(
            f"Requested simulation ({n_sims:,} paths x {steps:,} steps) would need "
            f"~{n_sims * (steps + 1) * 8 / 1e9:.1f} GB just to store. "
            "Reduce the number of simulations or the horizon."
        )

    # Daily portfolio-level parameters (exact collapse of the fixed-mix
    # multivariate process to one univariate normal per day).
    mu_daily = mu_a / trading_days
    cov_daily = cov_a / trading_days
    port_mu = float(w @ mu_daily)
    port_var = float(w @ cov_daily @ w)
    if not np.isfinite(port_mu):
        raise ValueError("Portfolio mean return is not finite.")
    if not np.isfinite(port_var) or port_var < 0:
        port_var = 0.0  # tiny negative from rounding on singular matrices
    port_std = float(np.sqrt(port_var))

    rng = np.random.default_rng(seed)
    m_step = _month_step(trading_days)
    # Pre-compute which post-growth indices receive a deposit.
    deposit_idx = set(range(m_step, steps + 1, m_step)) if monthly_contrib > 0 else set()

    paths = np.empty((n_sims, steps + 1), dtype=float)
    paths[:, 0] = initial_value

    chunk = min(_SIM_CHUNK_SIZE, n_sims)
    for start in range(0, n_sims, chunk):
        stop = min(start + chunk, n_sims)
        c = stop - start
        # Univariate daily portfolio LOG returns for this chunk.
        daily_r = rng.normal(loc=port_mu, scale=port_std, size=(c, steps))
        block = paths[start:stop]
        for t in range(steps):
            # GBM update: V_{t+1} = V_t * exp(r_t).
            block[:, t + 1] = block[:, t] * np.exp(daily_r[:, t])
            if (t + 1) in deposit_idx:
                block[:, t + 1] += monthly_contrib

    return paths


# ---------------------------------------------------------------------------
# Scenario shock application
# ---------------------------------------------------------------------------

def run_predefined_scenario(
    mu_annual: np.ndarray,
    cov_annual: np.ndarray,
    scenario: str = "Normal",
    return_shock: float = 0.0,
    vol_multiplier: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Apply a predefined or custom scenario shock to mu and cov.

    Args:
        mu_annual:      Base annualised mean returns (never mutated; copied).
        cov_annual:     Base annualised covariance matrix (never mutated; copied).
        scenario:       One of 'Normal', 'Market Crash', 'Bull Market',
                        'High Volatility', or 'Custom'. Unknown names fall
                        back to 'Normal'.
        return_shock:   Return shock to apply (used only for 'Custom').
        vol_multiplier: Volatility multiplier (used only for 'Custom', must be > 0).

    Returns:
        Tuple of (mu_shocked, cov_shocked, scenario_info_dict). Inputs are
        never mutated; shocked arrays are fresh copies.
    """
    mu_base = np.array(mu_annual, dtype=float, copy=True)
    cov_base = np.array(cov_annual, dtype=float, copy=True)

    if scenario == "Custom":
        if not np.isfinite(return_shock):
            raise ValueError(f"Custom return shock must be finite, got {return_shock!r}.")
        if not np.isfinite(vol_multiplier) or vol_multiplier <= 0:
            raise ValueError(
                f"Custom volatility multiplier must be positive, got {vol_multiplier!r}."
            )
        info = {
            "label": "Custom Scenario",
            "return_shock": float(return_shock),
            "vol_multiplier": float(vol_multiplier),
            "description": (
                f"User-defined scenario: return shock {return_shock:+.1%}, "
                f"vol multiplier {vol_multiplier:.2f}x."
            ),
        }
    else:
        info = dict(SCENARIO_DEFINITIONS.get(scenario, SCENARIO_DEFINITIONS["Normal"]))
        return_shock = info["return_shock"]
        vol_multiplier = info["vol_multiplier"]

    mu_shocked = mu_base + return_shock
    cov_shocked = cov_base * (vol_multiplier ** 2)

    return mu_shocked, cov_shocked, info


# ---------------------------------------------------------------------------
# Outcome statistics (contribution-aware)
# ---------------------------------------------------------------------------

def _max_drawdown_vectorised(paths: np.ndarray) -> np.ndarray:
    """
    Compute maximum drawdown for each simulation path.

    Args:
        paths: ndarray of shape (n_sims, steps).

    Returns:
        Array of shape (n_sims,) with max drawdown as a positive fraction.
    """
    # Running maximum along time axis
    running_max = np.maximum.accumulate(paths, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(running_max > 0, (running_max - paths) / running_max, 0.0)
    return drawdowns.max(axis=1)


def calculate_outcome_distribution(
    paths: np.ndarray,
    initial_value: float,
    monthly_contrib: float = 0.0,
    years: float = 0.0,
    trading_days: int = 252,
) -> dict:
    """
    Compute summary statistics from simulation paths.

    Profit/loss and probability-of-loss are measured against **total
    contributed capital** (initial + all monthly deposits), so enabling
    contributions correctly raises the break-even bar.

    Args:
        paths:           ndarray of shape (n_sims, steps + 1) (never mutated).
        initial_value:   Starting portfolio value.
        monthly_contrib: Monthly contribution used in the simulation.
        years:           Horizon in years (used with ``trading_days`` to count
                         deposits when contributions are enabled).
        trading_days:    Trading days per year used in the simulation.

    Returns:
        Dictionary with keys:
          - expected_value: mean final portfolio value
          - median_value:   median final portfolio value
          - worst_5pct:     5th percentile final value
          - best_5pct:      95th percentile final value
          - prob_loss:      probability of ending below TOTAL contributed capital
          - mean_max_drawdown: average max drawdown across paths
          - final_values:   array of all final values (a copy)
          - initial_capital: starting investment
          - contributions:  total monthly deposits paid in
          - total_contributed: initial + contributions (break-even reference)
          - expected_profit_loss: expected final value minus total contributed
          - return_on_invested: expected profit/loss divided by total contributed
    """
    if paths is None or paths.ndim != 2 or paths.shape[0] == 0:
        raise ValueError("Paths must be a non-empty 2-D array.")
    steps = paths.shape[1] - 1
    if steps < 1:
        raise ValueError("Paths must contain at least one simulation step.")
    final_values = np.array(paths[:, -1], dtype=float, copy=True)
    if not np.all(np.isfinite(final_values)):
        raise ValueError("Simulation produced non-finite final values.")

    monthly_contrib = float(monthly_contrib or 0.0)
    if monthly_contrib < 0:
        raise ValueError("monthly_contrib must be >= 0.")
    if monthly_contrib > 0 and years and years > 0:
        _init, contributions, total = total_contributed_capital(
            initial_value, monthly_contrib, years, trading_days
        )
    else:
        # Derive deposit count from the realised path length so metrics stay
        # exact even if the caller omits `years`.
        n_dep = count_deposits(steps, trading_days) if monthly_contrib > 0 else 0
        contributions = monthly_contrib * n_dep
        total = float(initial_value) + contributions

    expected = float(np.mean(final_values))
    profit_loss = expected - total

    return {
        "expected_value": expected,
        "median_value": float(np.median(final_values)),
        "worst_5pct": float(np.percentile(final_values, 5)),
        "best_5pct": float(np.percentile(final_values, 95)),
        "prob_loss": float(np.mean(final_values < total)),
        "mean_max_drawdown": float(_max_drawdown_vectorised(paths).mean()),
        "final_values": final_values,
        "initial_capital": float(initial_value),
        "contributions": float(contributions),
        "total_contributed": float(total),
        "expected_profit_loss": float(profit_loss),
        "return_on_invested": float(profit_loss / total) if total > 0 else 0.0,
    }


def summarise_future_metrics(
    paths: np.ndarray,
    initial_value: float,
    years: float,
    trading_days: int = 252,
    monthly_contrib: float = 0.0,
) -> tuple[dict, pd.DataFrame]:
    """
    Summarise simulation results and build a percentile time-series DataFrame.

    Args:
        paths:           ndarray of shape (n_sims, steps + 1) (never mutated).
        initial_value:   Starting portfolio value.
        years:           Simulation horizon in years.
        trading_days:    Trading days per year.
        monthly_contrib: Monthly contribution used in the simulation (needed
                         so loss probabilities use total contributed capital).

    Returns:
        Tuple of (metrics_dict, pct_df) where:
          - metrics_dict: output of calculate_outcome_distribution
          - pct_df: DataFrame indexed by year fraction with columns
                    [p5, p25, p50, p75, p95]
    """
    metrics = calculate_outcome_distribution(
        paths, initial_value,
        monthly_contrib=monthly_contrib, years=years, trading_days=trading_days,
    )

    steps = paths.shape[1]
    time_index = np.linspace(0, years, steps)

    pct_df = pd.DataFrame({
        "year": time_index,
        "p5": np.percentile(paths, 5, axis=0),
        "p25": np.percentile(paths, 25, axis=0),
        "p50": np.percentile(paths, 50, axis=0),
        "p75": np.percentile(paths, 75, axis=0),
        "p95": np.percentile(paths, 95, axis=0),
    }).set_index("year")

    return metrics, pct_df


# ---------------------------------------------------------------------------
# Natural language explainability (contribution-aware)
# ---------------------------------------------------------------------------

def explain_scenario_results(
    metrics: dict,
    scenario_info: dict,
    initial_value: float,
    monthly_contrib: float = 0.0,
    years: float = 0.0,
) -> str:
    """
    Generate a dynamic natural language summary of scenario simulation results.

    Gains, losses and the probability of loss are expressed relative to
    **total contributed capital** (initial + monthly deposits) whenever
    contributions are present.

    Args:
        metrics:       Output from calculate_outcome_distribution / summarise_future_metrics.
        scenario_info: Scenario metadata dict (label, return_shock, description, etc.).
        initial_value: Starting portfolio value.
        monthly_contrib: Monthly contribution assumed (for the narrative).
        years:         Horizon in years (for the narrative).

    Returns:
        Multi-sentence explanatory string.
    """
    label = scenario_info.get("label", "this scenario")
    return_shock = scenario_info.get("return_shock", 0.0)
    vol_mult = scenario_info.get("vol_multiplier", 1.0)

    expected = metrics["expected_value"]
    worst = metrics["worst_5pct"]
    best = metrics["best_5pct"]
    prob_loss = metrics["prob_loss"]
    max_dd = metrics.get("mean_max_drawdown", 0.0)

    # Break-even reference: prefer the metrics' own total if available.
    total = float(metrics.get("total_contributed", initial_value))
    contributions = float(metrics.get("contributions", 0.0))
    if total <= 0:
        total = float(initial_value)

    expected_gain_pct = (expected - total) / total * 100 if total else 0.0
    worst_pct = (worst - total) / total * 100 if total else 0.0
    best_pct = (best - total) / total * 100 if total else 0.0

    lines = []

    # Scenario context
    if return_shock < 0:
        lines.append(
            f"Under a **{label}** scenario, returns are shocked by "
            f"{return_shock:.0%} with volatility at {vol_mult:.1f}x the historical level."
        )
    elif return_shock > 0:
        lines.append(
            f"Under a **{label}** scenario, returns receive a boost of "
            f"{return_shock:.0%} with volatility at {vol_mult:.1f}x the historical level."
        )
    else:
        lines.append(
            f"Under a **{label}** scenario, returns follow historical expectations "
            f"with volatility at {vol_mult:.1f}x the baseline."
        )

    # Contribution context
    if contributions > 0:
        lines.append(
            f"You invest **${initial_value:,.0f}** upfront plus "
            f"**${monthly_contrib:,.0f}/month**, i.e. **${total:,.0f}** total capital."
        )

    # Expected outcome (relative to total contributed capital)
    direction = "grow" if expected_gain_pct >= 0 else "decline"
    base_word = "total invested capital" if contributions > 0 else "initial investment"
    lines.append(
        f"Your portfolio is expected to {direction} by **{abs(expected_gain_pct):.1f}%** on average, "
        f"reaching **${expected:,.0f}** from **${total:,.0f}** {base_word}."
    )

    # Worst / best case
    lines.append(
        f"In the worst 5% of outcomes, the portfolio would reach **${worst:,.0f}** "
        f"({worst_pct:+.1f}% vs invested capital). In the best 5%, it would reach "
        f"**${best:,.0f}** ({best_pct:+.1f}%)."
    )

    # Probability of loss
    if prob_loss > 0.50:
        lines.append(
            f"There is a **{prob_loss:.0%} probability** of finishing below the total "
            f"invested capital — this scenario carries significant downside risk."
        )
    elif prob_loss > 0.20:
        lines.append(
            f"There is a **{prob_loss:.0%} probability** of not recovering the total "
            f"invested capital — moderate downside risk is present."
        )
    elif prob_loss > 0:
        lines.append(
            f"The probability of ending below invested capital is low at **{prob_loss:.0%}**."
        )
    else:
        lines.append("In all simulated paths, the portfolio ends above the invested capital.")

    # Max drawdown
    if max_dd > 0.01:
        lines.append(
            f"The average maximum drawdown across simulations is **{max_dd:.1%}**, "
            f"representing the typical peak-to-trough decline experienced along the way."
        )

    return " ".join(lines)


# ---------------------------------------------------------------------------
# Plotly visualisations
# ---------------------------------------------------------------------------

_DARK_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Mono, monospace", color="#c9d1d9"),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    margin=dict(l=60, r=30, t=60, b=60),
)


def plot_simulation_paths(
    paths: np.ndarray,
    pct_df: pd.DataFrame,
    initial_value: float,
    scenario_label: str = "Normal Market",
    n_sample: int = 75,
    seed: int = 0,
    total_contributed: float | None = None,
) -> go.Figure:
    """
    Plot a sample of simulation paths with the median highlighted.

    Args:
        paths:          ndarray (n_sims, steps + 1).
        pct_df:         Percentile DataFrame from summarise_future_metrics.
        initial_value:  Starting value for reference line.
        scenario_label: Chart title suffix.
        n_sample:       Number of individual paths to display.
        seed:           Random seed for path sampling.
        total_contributed: Optional break-even reference (initial + deposits);
            drawn as a second dashed line when it differs from initial_value.

    Returns:
        Plotly Figure.
    """
    rng = np.random.default_rng(seed)
    n_sims = paths.shape[0]
    idx = rng.choice(n_sims, size=min(n_sample, n_sims), replace=False)

    years = pct_df.index.values
    fig = go.Figure()

    # Sample paths
    for i in idx:
        fig.add_trace(go.Scatter(
            x=years,
            y=paths[i],
            mode="lines",
            line=dict(color="rgba(88, 166, 255, 0.12)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Median path
    fig.add_trace(go.Scatter(
        x=years,
        y=pct_df["p50"].values,
        mode="lines",
        name="Median",
        line=dict(color="#f0883e", width=2.5),
    ))

    # Initial value reference
    fig.add_hline(
        y=initial_value,
        line=dict(color="#58a6ff", dash="dash", width=1.5),
        annotation_text="Initial Value",
        annotation_position="top left",
        annotation_font=dict(color="#58a6ff", size=11),
    )

    if total_contributed is not None and abs(total_contributed - initial_value) > 1e-9:
        fig.add_hline(
            y=total_contributed,
            line=dict(color="#3fb950", dash="dot", width=1.5),
            annotation_text="Total Invested",
            annotation_position="bottom left",
            annotation_font=dict(color="#3fb950", size=11),
        )

    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text=f"Simulation Paths — {scenario_label}", font=dict(size=16)),
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def plot_final_distribution(
    paths: np.ndarray,
    metrics: dict,
    initial_value: float,
    scenario_label: str = "Normal Market",
) -> go.Figure:
    """
    Histogram of final portfolio values with mean, median, and worst 5% marked.

    When contributions were used, the break-even (total invested) marker is
    drawn instead of / in addition to the initial-value marker.

    Args:
        paths:          ndarray (n_sims, steps + 1).
        metrics:        Output from calculate_outcome_distribution.
        initial_value:  Starting value.
        scenario_label: Chart title suffix.

    Returns:
        Plotly Figure.
    """
    final_values = paths[:, -1]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=final_values,
        nbinsx=60,
        name="Final Values",
        marker_color="rgba(88, 166, 255, 0.6)",
        marker_line=dict(color="rgba(88,166,255,0.9)", width=0.5),
    ))

    total = float(metrics.get("total_contributed", initial_value))
    # Vertical markers
    vlines = [
        (metrics["expected_value"], "#3fb950", "Mean"),
        (metrics["median_value"], "#f0883e", "Median"),
        (metrics["worst_5pct"], "#f85149", "Worst 5%"),
        (total, "#8b949e", "Invested" if abs(total - initial_value) > 1e-9 else "Initial"),
    ]

    y_max_approx = len(final_values) / 10  # rough upper bound for annotation

    for val, colour, name in vlines:
        fig.add_vline(
            x=val,
            line=dict(color=colour, dash="dash", width=1.5),
        )
        fig.add_annotation(
            x=val,
            y=y_max_approx,
            text=name,
            showarrow=False,
            font=dict(color=colour, size=11),
            xanchor="left",
            yanchor="bottom",
        )

    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text=f"Final Portfolio Distribution — {scenario_label}", font=dict(size=16)),
        xaxis_title="Final Portfolio Value ($)",
        yaxis_title="Frequency",
        showlegend=False,
    )
    return fig


def plot_confidence_bands(
    pct_df: pd.DataFrame,
    initial_value: float,
    scenario_label: str = "Normal Market",
) -> go.Figure:
    """
    Plot 5th / 25th / 50th / 75th / 95th percentile confidence bands over time.

    Args:
        pct_df:         Percentile DataFrame from summarise_future_metrics.
        initial_value:  Starting value for reference.
        scenario_label: Chart title suffix.

    Returns:
        Plotly Figure.
    """
    years = pct_df.index.values
    fig = go.Figure()

    # 5–95 outer band
    fig.add_trace(go.Scatter(
        x=np.concatenate([years, years[::-1]]),
        y=np.concatenate([pct_df["p95"].values, pct_df["p5"].values[::-1]]),
        fill="toself",
        fillcolor="rgba(88, 166, 255, 0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="5–95th pct",
        showlegend=True,
    ))

    # 25–75 inner band
    fig.add_trace(go.Scatter(
        x=np.concatenate([years, years[::-1]]),
        y=np.concatenate([pct_df["p75"].values, pct_df["p25"].values[::-1]]),
        fill="toself",
        fillcolor="rgba(88, 166, 255, 0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="25–75th pct",
        showlegend=True,
    ))

    # Median line
    fig.add_trace(go.Scatter(
        x=years,
        y=pct_df["p50"].values,
        mode="lines",
        name="Median (50th)",
        line=dict(color="#f0883e", width=2.5),
    ))

    # Initial value
    fig.add_hline(
        y=initial_value,
        line=dict(color="#8b949e", dash="dash", width=1.2),
        annotation_text="Initial",
        annotation_position="top left",
        annotation_font=dict(color="#8b949e", size=10),
    )

    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(text=f"Confidence Bands — {scenario_label}", font=dict(size=16)),
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig

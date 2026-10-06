"""
scenario.py
-----------
Forward-looking scenario and Monte Carlo simulation engine.

Provides:
  - simulate_portfolio_paths    : Geometric Brownian Motion simulation
  - run_predefined_scenario     : Apply stress scenario shocks
  - calculate_outcome_distribution : Summary stats from simulation paths
  - summarise_future_metrics    : Percentile time-series DataFrame
  - explain_scenario_results    : Natural language explainability
  - plot_simulation_paths       : Plotly chart of sample paths
  - plot_final_distribution     : Plotly histogram of final values
  - plot_confidence_bands       : Plotly confidence interval bands
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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

# ---------------------------------------------------------------------------
# Core simulation
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

    Args:
        weights:         Portfolio weight vector (must sum to 1).
        mu_annual:       Annualised mean returns per asset.
        cov_annual:      Annualised covariance matrix.
        initial_value:   Starting portfolio value in USD.
        years:           Simulation horizon in years.
        n_sims:          Number of Monte Carlo paths.
        monthly_contrib: Monthly cash contribution in USD (added at start of each month).
        trading_days:    Trading days per year (default 252).
        seed:            Random seed for reproducibility.

    Returns:
        ndarray of shape (n_sims, steps + 1) with portfolio values at each step.
        Step 0 is always initial_value.
    """
    steps = int(round(years * trading_days))
    rng = np.random.default_rng(seed)

    # Daily parameters
    mu_daily = mu_annual / trading_days          # (n_assets,)
    cov_daily = cov_annual / trading_days        # (n_assets, n_assets)

    # Simulate all daily asset returns at once: (n_sims, steps, n_assets)
    daily_asset_returns = rng.multivariate_normal(
        mean=mu_daily, cov=cov_daily, size=(n_sims, steps)
    )

    # Portfolio daily returns: (n_sims, steps)
    daily_port_returns = daily_asset_returns @ weights

    # Initialise paths array
    paths = np.empty((n_sims, steps + 1))
    paths[:, 0] = initial_value

    # Determine which step corresponds to each month boundary (~21 trading days)
    days_per_month = trading_days / 12

    for t in range(steps):
        # Geometric return: V_{t+1} = V_t * exp(r_t)
        paths[:, t + 1] = paths[:, t] * np.exp(daily_port_returns[:, t])

        # Monthly contribution at the start of each approximate month
        if monthly_contrib > 0 and t > 0 and (t % int(round(days_per_month)) == 0):
            paths[:, t + 1] += monthly_contrib

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
        mu_annual:      Base annualised mean returns.
        cov_annual:     Base annualised covariance matrix.
        scenario:       One of 'Normal', 'Market Crash', 'Bull Market',
                        'High Volatility', or 'Custom'.
        return_shock:   Return shock to apply (used only for 'Custom').
        vol_multiplier: Volatility multiplier (used only for 'Custom').

    Returns:
        Tuple of (mu_shocked, cov_shocked, scenario_info_dict).
    """
    if scenario == "Custom":
        info = {
            "label": "Custom Scenario",
            "return_shock": return_shock,
            "vol_multiplier": vol_multiplier,
            "description": (
                f"User-defined scenario: return shock {return_shock:+.1%}, "
                f"vol multiplier {vol_multiplier:.2f}x."
            ),
        }
    else:
        info = SCENARIO_DEFINITIONS.get(scenario, SCENARIO_DEFINITIONS["Normal"])
        return_shock = info["return_shock"]
        vol_multiplier = info["vol_multiplier"]

    mu_shocked = mu_annual + return_shock
    cov_shocked = cov_annual * (vol_multiplier ** 2)

    return mu_shocked, cov_shocked, info


# ---------------------------------------------------------------------------
# Outcome statistics
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
    drawdowns = (running_max - paths) / running_max
    return drawdowns.max(axis=1)


def calculate_outcome_distribution(
    paths: np.ndarray,
    initial_value: float,
) -> dict:
    """
    Compute summary statistics from simulation paths.

    Args:
        paths:         ndarray of shape (n_sims, steps + 1).
        initial_value: Starting portfolio value.

    Returns:
        Dictionary with keys:
          - expected_value: mean final portfolio value
          - median_value:   median final portfolio value
          - worst_5pct:     5th percentile final value
          - best_5pct:      95th percentile final value
          - prob_loss:      probability of ending below initial value
          - mean_max_drawdown: average max drawdown across paths
          - final_values:   array of all final values
    """
    final_values = paths[:, -1]

    return {
        "expected_value": float(np.mean(final_values)),
        "median_value": float(np.median(final_values)),
        "worst_5pct": float(np.percentile(final_values, 5)),
        "best_5pct": float(np.percentile(final_values, 95)),
        "prob_loss": float(np.mean(final_values < initial_value)),
        "mean_max_drawdown": float(_max_drawdown_vectorised(paths).mean()),
        "final_values": final_values,
    }


def summarise_future_metrics(
    paths: np.ndarray,
    initial_value: float,
    years: float,
    trading_days: int = 252,
) -> tuple[dict, pd.DataFrame]:
    """
    Summarise simulation results and build a percentile time-series DataFrame.

    Args:
        paths:         ndarray of shape (n_sims, steps + 1).
        initial_value: Starting portfolio value.
        years:         Simulation horizon in years.
        trading_days:  Trading days per year.

    Returns:
        Tuple of (metrics_dict, pct_df) where:
          - metrics_dict: output of calculate_outcome_distribution
          - pct_df: DataFrame indexed by year fraction with columns
                    [p5, p25, p50, p75, p95]
    """
    metrics = calculate_outcome_distribution(paths, initial_value)

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
# Natural language explainability
# ---------------------------------------------------------------------------

def explain_scenario_results(
    metrics: dict,
    scenario_info: dict,
    initial_value: float,
) -> str:
    """
    Generate a dynamic natural language summary of scenario simulation results.

    Args:
        metrics:       Output from calculate_outcome_distribution / summarise_future_metrics.
        scenario_info: Scenario metadata dict (label, return_shock, description, etc.).
        initial_value: Starting portfolio value.

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

    expected_gain_pct = (expected - initial_value) / initial_value * 100
    worst_pct = (worst - initial_value) / initial_value * 100
    best_pct = (best - initial_value) / initial_value * 100

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

    # Expected outcome
    direction = "grow" if expected_gain_pct >= 0 else "decline"
    lines.append(
        f"Your portfolio is expected to {direction} by **{abs(expected_gain_pct):.1f}%** on average, "
        f"reaching **${expected:,.0f}** from an initial **${initial_value:,.0f}**."
    )

    # Worst / best case
    lines.append(
        f"In the worst 5% of outcomes, the portfolio would reach **${worst:,.0f}** "
        f"({worst_pct:+.1f}%). In the best 5%, it would reach **${best:,.0f}** ({best_pct:+.1f}%)."
    )

    # Probability of loss
    if prob_loss > 0.50:
        lines.append(
            f"There is a **{prob_loss:.0%} probability** of finishing below the initial investment — "
            f"this scenario carries significant downside risk."
        )
    elif prob_loss > 0.20:
        lines.append(
            f"There is a **{prob_loss:.0%} probability** of not recovering the initial capital — "
            f"moderate downside risk is present."
        )
    elif prob_loss > 0:
        lines.append(
            f"The probability of a net loss is low at **{prob_loss:.0%}**."
        )
    else:
        lines.append("In all simulated paths, the portfolio ends above the initial investment.")

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

    # Vertical markers
    vlines = [
        (metrics["expected_value"], "#3fb950", "Mean"),
        (metrics["median_value"], "#f0883e", "Median"),
        (metrics["worst_5pct"], "#f85149", "Worst 5%"),
        (initial_value, "#8b949e", "Initial"),
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

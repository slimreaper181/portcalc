"""
app.py
------
Portfolio Analytics & Risk Dashboard — Streamlit entry point.

Tabs:
  1. 📈 Overview      — positions, returns, correlation/covariance
  2. ⚠️  Risk          — VaR (parametric, historical, Monte Carlo) + CVaR
  3. 🎯 Optimise      — efficient frontier, max Sharpe, min variance, target constraints
  4. 📊 Scenario      — Monte Carlo forward simulation with stress scenarios
  5. 📉 Performance   — benchmark comparison, drawdowns, downside risk, rolling stats
  6. 🔍 Security Detail — single-holding analytics + TradingView chart

Run locally:
    streamlit run app.py

The app is designed to be modular so the analytics modules can be imported
and used independently of Streamlit (e.g. in a desktop GUI or CLI).

Numerical conventions:
  * Market-data returns are daily LOG returns (see analytics.returns).
  * Weight vectors, mu vectors and covariance matrices always share the
    canonical (sorted) ticker order defined in analytics.validation.
"""

import os
import json

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from data.market_data import (
    fetch_price_history,
    fetch_benchmark_history,
    fetch_current_prices,
    fetch_exchange_code,
    fetch_risk_free_rate,
)
from analytics.returns import (
    daily_returns,
    annualised_mean_returns,
    annualised_cov_matrix,
    correlation_matrix,
    cumulative_growth_from_log_returns,
)
from analytics.portfolio import (
    portfolio_expected_return,
    portfolio_std,
    sharpe_ratio,
    risk_contributions,
    weights_from_shares,
)
from analytics.var import (
    parametric_var,
    historical_var,
    monte_carlo_var,
    to_daily,
    mean_to_daily,
)
from analytics.optimisation import (
    min_variance,
    max_sharpe,
    target_return,
    target_volatility,
    efficient_frontier,
    feasible_return_range,
    feasible_volatility_range,
    rebalance_trades,
)
from analytics.scenario import (
    simulate_portfolio_paths,
    run_predefined_scenario,
    summarise_future_metrics,
    explain_scenario_results,
    plot_simulation_paths,
    plot_final_distribution,
    plot_confidence_bands,
    SCENARIO_DEFINITIONS,
)
from analytics.performance import (
    align_return_series,
    best_worst_periods,
    drawdown_series,
    growth_of_capital,
    historical_cvar,
    log_returns_from_prices,
    monthly_returns,
    plot_drawdown,
    plot_growth_comparison,
    rolling_annualised_return,
    rolling_sharpe,
    rolling_volatility,
    sharpe_from_log,
    summarise_performance,
    underwater_episodes,
)
from analytics.security import (
    period_returns,
    security_summary,
)
from components.tradingview import (
    render_tradingview_chart,
    resolve_tradingview_symbol,
)
from analytics.validation import (
    align_market_data,
    aligned_portfolio_returns,
    canonical_tickers,
    sanitize_prices,
    validate_horizon_years,
    validate_monthly_contrib,
    validate_risk_free_rate,
    validate_shares,
    validate_simulation_count,
    validate_ticker_symbol,
    validate_weight_bounds,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "portfolio.json")

CHART_THEME = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Mono, monospace", color="#c9d1d9", size=12),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    margin=dict(l=60, r=30, t=60, b=60),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30363d"),
)

DEFAULT_PORTFOLIO = {
    "AAPL":  {"shares": 10, "avg_cost": 150.0},
    "MSFT":  {"shares": 5,  "avg_cost": 300.0},
    "GOOGL": {"shares": 3,  "avg_cost": 130.0},
    "AMZN":  {"shares": 4,  "avg_cost": 180.0},
    "NVDA":  {"shares": 8,  "avg_cost": 500.0},
}

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Portfolio Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
  }
  h1, h2, h3, h4 { font-family: 'IBM Plex Mono', monospace; color: #e6edf3; }
  .stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 14px; }
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
  }
  .metric-card .value { font-size: 26px; font-weight: 600; color: #58a6ff; }
  .metric-card .label { font-size: 12px; color: #8b949e; margin-top: 4px; }
  div[data-testid="stSidebarContent"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
  }
  .stButton>button {
    background-color: #238636;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
  }
  .stButton>button:hover { background-color: #2ea043; }
  div[data-testid="stMetricValue"] { color: #58a6ff; font-family: 'IBM Plex Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Portfolio persistence helpers
# ---------------------------------------------------------------------------

def load_portfolio() -> dict:
    """Load portfolio from JSON file, falling back to defaults."""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                data = json.load(f)
            # Normalise keys on load (upper-case, stripped).
            cleaned = {}
            for k, v in data.items():
                try:
                    sym = validate_ticker_symbol(k)
                except ValueError:
                    continue
                cleaned[sym] = v
            if cleaned:
                return cleaned
        except (json.JSONDecodeError, IOError):
            pass
    return DEFAULT_PORTFOLIO.copy()


def save_portfolio(portfolio: dict) -> None:
    """Persist portfolio to JSON file."""
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(portfolio, f, indent=2)
    except IOError as e:
        st.warning(f"Could not save portfolio: {e}")


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_portfolio()

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def style_matrix(df: pd.DataFrame, fmt: str = ".4f") -> Styler:
    """Apply background gradient styling to a numeric DataFrame."""
    return (
        df.style
        .format("{:" + fmt + "}")
        .background_gradient(cmap="Blues", axis=None)
    )


def fmt_usd(value: float) -> str:
    return f"${value:,.2f}"


def fmt_pct(value: float) -> str:
    return f"{value:.2%}"


def build_core_analytics(
    tickers: list[str],
    prices: pd.DataFrame,
    current_prices: dict,
    rf_rate: float,
) -> dict:
    """Sanitise market data, align to canonical ticker order, compute core stats.

    Raises:
        ValueError: with a user-facing message if data is unusable.
    """
    tickers = canonical_tickers(tickers)
    clean_prices = sanitize_prices(prices, tickers)
    if len(clean_prices) < 30:
        st.warning(
            f"Only {len(clean_prices)} trading days of overlapping history — "
            "statistics and VaR estimates will be noisy. "
            "Consider a longer history period or fewer tickers."
        )

    # Current prices must be present and positive for every ticker.
    bad = [t for t in tickers
           if not np.isfinite(float(current_prices.get(t, np.nan)))
           or float(current_prices.get(t, 0.0)) <= 0]
    if bad:
        raise ValueError(
            f"No usable current price for: {', '.join(bad)}. "
            "Check the ticker symbols."
        )

    shares_dict = {t: st.session_state.portfolio[t]["shares"] for t in tickers}
    for t, sh in shares_dict.items():
        if not np.isfinite(sh) or sh <= 0:
            raise ValueError(
                f"Position {t} has invalid shares ({sh!r}); "
                "shares must be positive."
            )
    weights, w_tickers = weights_from_shares(shares_dict, current_prices)
    if list(w_tickers) != list(tickers) or len(weights) != len(tickers):
        raise ValueError("Internal error: weight/ticker ordering mismatch.")

    portfolio_value = float(sum(
        st.session_state.portfolio[t]["shares"] * float(current_prices[t])
        for t in tickers
    ))
    if not np.isfinite(portfolio_value) or portfolio_value <= 0:
        raise ValueError(
            "Portfolio value is zero — check share counts and prices."
        )

    try:
        rf_rate = validate_risk_free_rate(rf_rate)
    except ValueError:
        rf_rate = 0.05

    rets = daily_returns(clean_prices)
    rets = rets.dropna(how="any")
    if rets.empty or len(rets) < 2:
        raise ValueError("Not enough return observations to compute statistics.")
    aligned = align_market_data(tickers, returns=rets)
    rets = aligned["returns"]

    mu = annualised_mean_returns(rets)
    cov = annualised_cov_matrix(rets)
    corr = correlation_matrix(rets)
    # Defensive reindex: mu/cov/corr rows/cols follow the canonical order.
    mu = align_market_data(tickers, mu=mu)["mu"]
    cov = align_market_data(tickers, cov=cov)["cov"]
    corr = corr.reindex(index=tickers, columns=tickers)

    if not np.all(np.isfinite(mu.values)):
        raise ValueError("Expected returns contain non-finite values.")
    if not np.all(np.isfinite(cov.values)):
        raise ValueError("Covariance matrix contains non-finite values.")

    port_return = portfolio_expected_return(weights, mu.values)
    port_vol = portfolio_std(weights, cov.values)
    port_sharpe = sharpe_ratio(port_return, port_vol, rf_rate)
    port_daily_returns = aligned_portfolio_returns(rets, weights, tickers)
    rc = risk_contributions(weights, cov.values)

    for name, val in (("expected return", port_return), ("volatility", port_vol)):
        if not np.isfinite(val):
            raise ValueError(f"Portfolio {name} is not finite — check market data.")

    return {
        "tickers": tickers,
        "prices": clean_prices,
        "current_prices": {t: float(current_prices[t]) for t in tickers},
        "rf_rate": rf_rate,
        "weights": weights,
        "portfolio_value": portfolio_value,
        "rets": rets,
        "mu": mu,
        "cov": cov,
        "corr": corr,
        "port_return": port_return,
        "port_vol": port_vol,
        "port_sharpe": port_sharpe,
        "port_daily_returns": port_daily_returns,
        "rc": rc,
    }


# ---------------------------------------------------------------------------
# Sidebar — Portfolio Editor
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📈 Portfolio Editor")
    st.markdown("---")

    # --- Add position ---
    with st.expander("➕ Add / Update Position", expanded=False):
        new_ticker = st.text_input("Ticker", placeholder="e.g. TSLA").upper().strip()
        new_shares = st.number_input("Shares", min_value=0.0, step=0.5, value=1.0)
        new_cost   = st.number_input("Avg Cost ($)", min_value=0.0, step=1.0, value=100.0)

        if st.button("Add Position"):
            try:
                sym = validate_ticker_symbol(new_ticker)
                sh = validate_shares(float(new_shares))
                if not np.isfinite(new_cost) or new_cost < 0:
                    raise ValueError(f"Avg cost must be >= 0, got {new_cost!r}.")
            except ValueError as e:
                st.error(str(e))
            else:
                st.session_state.portfolio[sym] = {
                    "shares": sh,
                    "avg_cost": float(new_cost),
                }
                save_portfolio(st.session_state.portfolio)
                st.session_state.data_loaded = False
                st.success(f"Added {sym}")
                st.rerun()

    # --- Current positions ---
    st.markdown("### Current Positions")
    to_remove = []
    for ticker, pos in list(st.session_state.portfolio.items()):
        col1, col2 = st.columns([3, 1])
        col1.markdown(f"**{ticker}** — {pos['shares']} sh @ ${pos['avg_cost']:.2f}")
        if col2.button("✕", key=f"rm_{ticker}"):
            to_remove.append(ticker)

    for t in to_remove:
        del st.session_state.portfolio[t]
        save_portfolio(st.session_state.portfolio)
        st.session_state.data_loaded = False
        st.rerun()

    st.markdown("---")

    if st.button("🔄 Refresh Market Data"):
        st.session_state.data_loaded = False
        st.rerun()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def load_market_data(tickers: list[str], period: str):
    prices  = fetch_price_history(tickers, period=period)
    current = fetch_current_prices(tickers)
    rf      = fetch_risk_free_rate()
    return prices, current, rf


@st.cache_data(ttl=300, show_spinner=False)
def load_benchmark_data(benchmark_ticker: str, period: str):
    """Cached benchmark history for the same period as the portfolio."""
    return fetch_benchmark_history(benchmark_ticker, period=period)


@st.cache_data(ttl=3600, show_spinner=False)
def load_exchange_hint(ticker: str):
    """Cached yfinance exchange code (TradingView resolution hint only)."""
    try:
        return fetch_exchange_code(ticker)
    except Exception:
        return None


try:
    tickers = canonical_tickers(list(st.session_state.portfolio.keys()))
except ValueError as e:
    st.warning(f"Portfolio problem: {e} Add at least one valid position in the sidebar.")
    st.stop()

if not tickers:
    st.warning("Add at least one position in the sidebar to get started.")
    st.stop()

# De-duplicated tickers are canonicalised; warn if the raw keys differed.
if len(tickers) != len(st.session_state.portfolio):
    st.info("Duplicate tickers were merged (symbols are case-insensitive).")

with st.spinner("Loading market data…"):
    # Default history period — can be overridden per-tab
    try:
        prices, current_prices, rf_rate = load_market_data(tickers, "2y")
    except ValueError as e:
        st.error(f"Market data error: {e}")
        st.stop()
    except Exception:
        st.error(
            "Could not load market data (network or API error). "
            "Check your connection and click Refresh Market Data to retry."
        )
        st.stop()

try:
    core = build_core_analytics(tickers, prices, current_prices, rf_rate)
except ValueError as e:
    st.error(f"Data error: {e}")
    st.stop()

prices = core["prices"]
current_prices = core["current_prices"]
rf_rate = core["rf_rate"]
weights = core["weights"]
portfolio_value = core["portfolio_value"]
rets = core["rets"]
mu = core["mu"]
cov = core["cov"]
corr = core["corr"]
port_return = core["port_return"]
port_vol = core["port_vol"]
port_sharpe = core["port_sharpe"]
port_daily_returns = core["port_daily_returns"]
rc = core["rc"]

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_overview, tab_risk, tab_optimise, tab_scenario, tab_perf, tab_security = st.tabs([
    "📈 Overview",
    "⚠️  Risk",
    "🎯 Optimise",
    "📊 Scenario Analysis",
    "📉 Performance",
    "🔍 Security Detail",
])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tab_overview:
    st.markdown("## Portfolio Overview")

    # History period selector lives here
    hist_col, _ = st.columns([2, 8])
    with hist_col:
        hist_period = st.selectbox(
            "History period",
            ["6mo", "1y", "2y", "5y"],
            index=2,
            key="hist_period_tab1",
        )

    if hist_period != "2y":
        with st.spinner("Fetching data…"):
            try:
                prices, current_prices, rf_rate = load_market_data(tickers, hist_period)
                core = build_core_analytics(tickers, prices, current_prices, rf_rate)
            except ValueError as e:
                st.error(f"Data error: {e}")
                st.stop()
            except Exception:
                st.error("Could not load market data for the selected period.")
                st.stop()
            prices = core["prices"]
            current_prices = core["current_prices"]
            rf_rate = core["rf_rate"]
            weights = core["weights"]
            portfolio_value = core["portfolio_value"]
            rets = core["rets"]
            mu = core["mu"]
            cov = core["cov"]
            corr = core["corr"]
            port_return = core["port_return"]
            port_vol = core["port_vol"]
            port_sharpe = core["port_sharpe"]
            rc = core["rc"]
            port_daily_returns = core["port_daily_returns"]

    # Top metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Portfolio Value", fmt_usd(portfolio_value))
    c2.metric("Expected Return", fmt_pct(port_return))
    c3.metric("Volatility (σ)", fmt_pct(port_vol))
    c4.metric("Sharpe Ratio", f"{port_sharpe:.2f}")
    c5.metric("Risk-Free Rate", fmt_pct(rf_rate))

    st.markdown("---")

    # Positions table
    st.markdown("### Positions")
    rows = []
    for t in tickers:
        price = current_prices.get(t, 0.0)
        sh    = st.session_state.portfolio[t]["shares"]
        cost  = st.session_state.portfolio[t]["avg_cost"]
        val   = sh * price
        pnl   = val - sh * cost
        rows.append({
            "Ticker": t,
            "Shares": sh,
            "Avg Cost": cost,
            "Current Price": price,
            "Market Value ($)": val,
            "P&L ($)": pnl,
            "P&L (%)": pnl / (sh * cost) * 100 if cost > 0 else 0.0,
            "Weight": weights[tickers.index(t)],
        })
    pos_df = pd.DataFrame(rows)
    st.dataframe(
        pos_df.style
        .format({
            "Avg Cost": "${:.2f}", "Current Price": "${:.2f}",
            "Market Value ($)": "${:,.2f}", "P&L ($)": "${:+,.2f}",
            "P&L (%)": "{:+.2f}%", "Weight": "{:.1%}",
        })
        .map(lambda v: "color: #3fb950" if isinstance(v, (int, float)) and v > 0
                  else ("color: #f85149" if isinstance(v, (int, float)) and v < 0 else ""),
                  subset=["P&L ($)", "P&L (%)"]),
        use_container_width=True,
    )

    st.markdown("---")

    col_weights, col_rc = st.columns(2)

    with col_weights:
        st.markdown("### Weight Allocation")
        fig_pie = go.Figure(go.Pie(
            labels=tickers,
            values=weights,
            hole=0.45,
            marker=dict(colors=px.colors.qualitative.Plotly),
            textinfo="label+percent",
            textfont=dict(family="IBM Plex Mono", size=13),
        ))
        fig_pie.update_layout(**CHART_THEME, title="Portfolio Weights")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_rc:
        st.markdown("### Risk Contributions")
        fig_rc = go.Figure(go.Bar(
            x=tickers, y=rc,
            marker_color="#58a6ff",
            text=[f"{v:.1%}" for v in rc],
            textposition="outside",
        ))
        fig_rc.update_layout(
            **CHART_THEME,
            title="Risk Contribution per Asset",
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig_rc, use_container_width=True)

    st.markdown("---")

    col_cov, col_corr = st.columns(2)

    with col_cov:
        st.markdown("### Covariance Matrix")
        st.dataframe(style_matrix(cov, ".5f"), use_container_width=True)

    with col_corr:
        st.markdown("### Correlation Matrix")
        st.dataframe(style_matrix(corr, ".4f"), use_container_width=True)

    # Cumulative returns chart (log returns -> exp(cumsum), NOT (1+r).cumprod())
    st.markdown("---")
    st.markdown("### Cumulative Returns")

    cum_ret = cumulative_growth_from_log_returns(rets)
    fig_cum = go.Figure()
    for col in tickers:
        fig_cum.add_trace(go.Scatter(
            x=cum_ret.index, y=cum_ret[col],
            mode="lines", name=col,
        ))
    fig_cum.update_layout(
        **CHART_THEME,
        title="Cumulative Return (base = 1, from log returns)",
        xaxis_title="Date", yaxis_title="Growth of $1",
    )
    st.plotly_chart(fig_cum, use_container_width=True)


# ============================================================
# TAB 2 — RISK
# ============================================================
with tab_risk:
    st.markdown("## Risk Analysis")

    r_col1, r_col2 = st.columns([1, 3])

    with r_col1:
        var_horizon = st.number_input(
            "VaR Horizon (days)", min_value=1, max_value=30, value=1, step=1
        )
        st.caption("VaR confidence fixed at 95% (industry standard)")

    st.markdown("---")

    # Daily parameters (log-return space)
    port_std_daily  = to_daily(port_vol)
    mu_daily        = mean_to_daily(mu.values)
    cov_daily       = cov.values / 252

    try:
        var_param = parametric_var(port_std_daily, portfolio_value, 0.95, int(var_horizon))
    except ValueError as e:
        var_param = None
        st.error(f"Parametric VaR error: {e}")
    try:
        var_hist = historical_var(
            port_daily_returns, portfolio_value, 0.95, int(var_horizon)
        )
    except ValueError as e:
        var_hist = None
        hist_err = str(e)
    try:
        var_mc, mc_sim_rets = monte_carlo_var(
            weights, mu_daily, cov_daily, portfolio_value, 0.95, int(var_horizon), 10_000
        )
    except ValueError as e:
        var_mc, mc_sim_rets = None, None
        st.error(f"Monte Carlo VaR error: {e}")

    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown("#### Parametric VaR")
        st.metric("95% VaR", fmt_usd(var_param) if var_param is not None else "n/a")
        st.caption(
            "Assumes normally distributed returns. "
            f"At 95% confidence over {int(var_horizon)}d, potential loss ≤ "
            f"{fmt_usd(var_param) if var_param is not None else 'n/a'}."
        )
    with v2:
        st.markdown("#### Historical VaR")
        st.metric("95% VaR", fmt_usd(var_hist) if var_hist is not None else "n/a")
        if var_hist is None:
            st.caption(f"Historical VaR unavailable: {hist_err}")
        else:
            n_win = len(port_daily_returns) - int(var_horizon) + 1
            st.caption(
                "Uses genuine rolling "
                f"{int(var_horizon)}-day historical returns ({n_win:,} overlapping "
                "windows). No normality or √T-scaling assumption."
            )
    with v3:
        st.markdown("#### Monte Carlo VaR")
        st.metric("95% VaR", fmt_usd(var_mc) if var_mc is not None else "n/a")
        st.caption(
            "10,000 simulated return paths using multivariate normal. "
            "Log returns are converted to simple P&L via exp(r)−1. "
            "Captures correlation structure."
        )

    # Historical CVaR (Expected Shortfall) at the same horizon
    st.markdown("---")
    st.markdown("### Historical CVaR (Expected Shortfall)")
    try:
        cvar_out = historical_cvar(
            port_daily_returns, portfolio_value, 0.95, int(var_horizon)
        )
    except ValueError as e:
        cvar_out = None
        st.warning(f"Historical CVaR unavailable: {e}")

    if cvar_out is not None:
        cc1, cc2 = st.columns(2)
        with cc1:
            st.metric(
                f"95% Historical VaR ({int(var_horizon)}d)",
                fmt_usd(cvar_out["var_currency"]),
                help="Loss threshold exceeded in the worst 5% of historical periods.",
            )
            st.caption(f"Cutoff return: {cvar_out['var_pct']:.2%}")
        with cc2:
            st.metric(
                f"95% Historical CVaR ({int(var_horizon)}d)",
                fmt_usd(cvar_out["cvar_currency"]),
                help="Average loss once the VaR threshold has been exceeded.",
            )
            st.caption(
                f"Average tail loss: {cvar_out['cvar_pct']:.2%} "
                f"over {cvar_out['n_tail']:,} of {cvar_out['n_windows']:,} "
                "genuine rolling windows."
            )
        st.caption(
            "VaR estimates the loss threshold exceeded in the worst 5% of periods. "
            "CVaR estimates the average loss once that threshold has been exceeded."
        )

    st.markdown("---")

    # Return distribution
    st.markdown("### Daily Return Distribution")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=port_daily_returns,
        nbinsx=60,
        name="Daily Returns",
        marker_color="rgba(88, 166, 255, 0.6)",
        marker_line=dict(color="rgba(88,166,255,0.9)", width=0.5),
    ))
    var_pct_5 = np.percentile(port_daily_returns, 5)
    fig_dist.add_vline(x=var_pct_5, line=dict(color="#f85149", dash="dash", width=2))
    fig_dist.add_annotation(
        x=var_pct_5, text="5th pct",
        showarrow=True, arrowcolor="#f85149",
        font=dict(color="#f85149"), yref="paper", y=0.9,
    )
    fig_dist.update_layout(
        **CHART_THEME,
        title="Portfolio Daily Return Distribution",
        xaxis_title="Daily Return", yaxis_title="Count",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # MC simulation return histogram
    if mc_sim_rets is not None:
        st.markdown("### Monte Carlo Simulated Returns")
        fig_mc = go.Figure(go.Histogram(
            x=mc_sim_rets,
            nbinsx=60,
            marker_color="rgba(240, 136, 62, 0.6)",
            marker_line=dict(color="rgba(240,136,62,0.9)", width=0.5),
        ))
        var_mc_pct = np.percentile(mc_sim_rets, 5)
        fig_mc.add_vline(x=var_mc_pct, line=dict(color="#f85149", dash="dash", width=2))
        fig_mc.add_annotation(
            x=var_mc_pct, text="VaR 5th pct",
            showarrow=True, arrowcolor="#f85149",
            font=dict(color="#f85149"), yref="paper", y=0.9,
        )
        fig_mc.update_layout(
            **CHART_THEME,
            title=f"MC Simulated {int(var_horizon)}-Day Portfolio Returns (simple returns)",
            xaxis_title="Portfolio Return", yaxis_title="Count",
        )
        st.plotly_chart(fig_mc, use_container_width=True)


# ============================================================
# TAB 3 — OPTIMISE
# ============================================================
with tab_optimise:
    st.markdown("## Portfolio Optimisation")

    mu_arr  = mu.values
    cov_arr = cov.values

    # Constraints
    st.markdown("### Constraints")
    oc1, oc2 = st.columns(2)
    with oc1:
        min_w = st.slider("Min weight per asset", 0.0, 0.20, 0.0, 0.01, format="%.2f")
    with oc2:
        max_w = st.slider("Max weight per asset", 0.20, 1.0, 1.0, 0.01, format="%.2f")

    try:
        validate_weight_bounds(len(tickers), float(min_w), float(max_w))
        bounds_ok = True
    except ValueError as e:
        bounds_ok = False
        st.error(f"Invalid weight bounds: {e}")

    st.markdown("---")

    if not bounds_ok:
        st.warning("Fix the weight bounds above to run optimisations.")
    else:
        # Run optimisations (failures render as errors in their result cards;
        # failed results carry no weights and are never shown as solutions).
        res_mv = min_variance(mu_arr, cov_arr, rf_rate, min_w, max_w)
        res_ms = max_sharpe(mu_arr, cov_arr, rf_rate, min_w, max_w)

        # Target return slider — bounded by the feasible return range.
        try:
            r_lo, r_hi = feasible_return_range(mu_arr, min_w, max_w)
        except ValueError as e:
            st.error(f"Cannot determine feasible returns: {e}")
            r_lo, r_hi = None, None

        if r_lo is None:
            res_tr = None
            target_r_val = None
        else:
            r_min_bound = float(np.clip(r_lo, -0.30, 0.50))
            r_max_bound = float(np.clip(r_hi, r_lo + 0.01, 1.0))
            if "target_r_val" not in st.session_state:
                st.session_state.target_r_val = float(
                    np.clip(port_return, r_min_bound, r_max_bound))
            # Clamp a stale session value into the current bounds.
            st.session_state.target_r_val = float(
                np.clip(st.session_state.target_r_val, r_min_bound, r_max_bound))
            target_r_val = st.slider(
                "Target annual return",
                r_min_bound, r_max_bound,
                st.session_state.target_r_val,
                0.005,
                format="%.1f%%",
                key="target_r_val",
            )
            res_tr = target_return(mu_arr, cov_arr, target_r_val, rf_rate, min_w, max_w)

        # Target volatility slider — bounded by feasible portfolio volatility
        # (from the covariance structure), NOT by return dispersion.
        try:
            v_lo, v_hi = feasible_volatility_range(mu_arr, cov_arr, min_w, max_w)
        except ValueError as e:
            st.error(f"Cannot determine feasible volatility range: {e}")
            v_lo, v_hi = None, None

        if v_lo is None:
            res_tv = None
            target_v_val = None
        else:
            v_min_bound = float(max(v_lo, 0.005))
            v_max_bound = float(max(v_hi, v_min_bound + 0.005))
            if "target_v_val" not in st.session_state:
                st.session_state.target_v_val = float(
                    np.clip(port_vol, v_min_bound, v_max_bound))
            st.session_state.target_v_val = float(
                np.clip(st.session_state.target_v_val, v_min_bound, v_max_bound))
            st.caption(
                f"Feasible volatility under current bounds: "
                f"{v_lo:.1%} – {v_hi:.1%}."
            )
            target_v_val = st.slider(
                "Target annual volatility",
                v_min_bound, v_max_bound,
                st.session_state.target_v_val,
                0.005,
                format="%.1f%%",
                key="target_v_val",
            )
            res_tv = target_volatility(mu_arr, cov_arr, target_v_val, rf_rate, min_w, max_w)

        st.markdown("---")
        st.markdown("### Optimisation Results")

        def _result_card(label: str, res, tickers: list[str]) -> None:
            if res is None or not res.success or res.weights is None:
                reason = ""
                if res is not None and res.message:
                    reason = f": {res.message}"
                st.error(f"❌ {label} — optimisation failed{reason}")
                st.caption(
                    "Your current portfolio is unchanged; no fallback weights "
                    "are shown as a solution."
                )
                return
            cols = st.columns(4)
            cols[0].metric(f"✅ {label} — Return", fmt_pct(res.expected_return))
            cols[1].metric("Volatility", fmt_pct(res.volatility))
            cols[2].metric("Sharpe", f"{res.sharpe:.2f}")
            w_df = pd.DataFrame({"Ticker": tickers, "Weight": res.weights}).set_index("Ticker")
            cols[3].dataframe(w_df.style.format({"Weight": "{:.1%}"}), height=160)

        _result_card("Min Variance", res_mv, tickers)
        st.markdown("---")
        _result_card("Max Sharpe", res_ms, tickers)
        st.markdown("---")
        _result_card(
            f"Target Return {target_r_val:.1%}" if target_r_val is not None else "Target Return",
            res_tr, tickers)
        st.markdown("---")
        _result_card(
            f"Target Vol {target_v_val:.1%}" if target_v_val is not None else "Target Vol",
            res_tv, tickers)

        st.markdown("---")

        # Efficient frontier
        st.markdown("### Efficient Frontier")
        with st.spinner("Tracing efficient frontier…"):
            ef_df = efficient_frontier(mu_arr, cov_arr, rf_rate, 60, min_w, max_w)

        if ef_df.empty:
            st.warning(
                "Efficient frontier could not be traced under the current "
                "constraints (no target-return point converged)."
            )
        else:
            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(
                x=ef_df["volatility"], y=ef_df["return"],
                mode="lines",
                name="Efficient Frontier",
                line=dict(color="#58a6ff", width=2),
            ))
            # Mark current portfolio
            fig_ef.add_trace(go.Scatter(
                x=[port_vol], y=[port_return],
                mode="markers", name="Current",
                marker=dict(color="#f85149", size=12, symbol="star"),
            ))
            # Mark max Sharpe
            if (res_ms.success and res_ms.weights is not None
                    and np.isfinite(res_ms.volatility) and res_ms.volatility > 0):
                fig_ef.add_trace(go.Scatter(
                    x=[res_ms.volatility], y=[res_ms.expected_return],
                    mode="markers", name="Max Sharpe",
                    marker=dict(color="#3fb950", size=12, symbol="diamond"),
                ))
                # CML
                cml_vol = np.linspace(0, ef_df["volatility"].max() * 1.2, 50)
                cml_ret = rf_rate + (res_ms.expected_return - rf_rate) / res_ms.volatility * cml_vol
                fig_ef.add_trace(go.Scatter(
                    x=cml_vol, y=cml_ret,
                    mode="lines", name="Capital Market Line",
                    line=dict(color="#f0883e", width=1.5, dash="dash"),
                ))
            fig_ef.update_layout(
                **CHART_THEME,
                title="Efficient Frontier with CML",
                xaxis_title="Volatility (σ)", yaxis_title="Expected Return",
                xaxis_tickformat=".0%", yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig_ef, use_container_width=True)

        # Rebalancing trades
        st.markdown("---")
        st.markdown("### Rebalancing Trades")

        target_choice = st.selectbox(
            "Rebalance to",
            ["Max Sharpe", "Min Variance", "Target Return", "Target Volatility"],
            key="rebalance_target",
        )
        target_map = {
            "Max Sharpe": res_ms,
            "Min Variance": res_mv,
            "Target Return": res_tr,
            "Target Volatility": res_tv,
        }
        chosen = target_map[target_choice]

        if (chosen is not None and chosen.success
                and chosen.weights is not None
                and np.all(np.isfinite(chosen.weights))):
            try:
                trades_df = rebalance_trades(weights, chosen.weights, tickers, portfolio_value)
            except ValueError as e:
                st.warning(f"Could not compute rebalancing trades: {e}")
            else:
                def _colour_action(val):
                    if val == "BUY":
                        return "color: #3fb950; font-weight: bold"
                    if val == "SELL":
                        return "color: #f85149; font-weight: bold"
                    return "color: #8b949e"

                st.dataframe(
                    trades_df.style
                    .format({
                        "Current Weight": "{:.1%}", "Target Weight": "{:.1%}",
                        "Δ Weight": "{:+.1%}", "Trade ($)": "${:+,.2f}",
                    })
                    .map(_colour_action, subset=["Action"]),
                    use_container_width=True,
                )
        else:
            detail = ""
            if chosen is not None and not chosen.success:
                detail = f" ({chosen.message})"
            st.warning(f"Optimisation did not converge for selected target.{detail}")


# ============================================================
# TAB 4 — SCENARIO ANALYSIS
# ============================================================
with tab_scenario:
    st.markdown("## Scenario & Monte Carlo Simulation")

    sc1, sc2 = st.columns([1, 2])

    with sc1:
        st.markdown("### Simulation Settings")

        horizon_years = st.slider(
            "Time horizon (years)", 1, 10, 5, 1, key="sc_horizon"
        )
        initial_investment = st.number_input(
            "Initial investment ($)",
            min_value=1_000.0,
            value=float(max(portfolio_value, 1_000.0)),
            step=1_000.0,
            key="sc_initial",
        )
        monthly_contrib = st.number_input(
            "Monthly contribution ($)",
            min_value=0.0, value=0.0, step=100.0,
            key="sc_monthly",
        )
        n_simulations = st.slider(
            "Number of simulations", 1_000, 20_000, 5_000, 1_000, key="sc_nsims"
        )

        scenario_choice = st.selectbox(
            "Scenario",
            ["Normal", "Market Crash", "Bull Market", "High Volatility", "Custom"],
            key="sc_scenario",
        )

        if scenario_choice == "Custom":
            custom_shock = st.slider(
                "Return shock", -0.50, 0.50, 0.0, 0.01,
                format="%+.0f%%", key="sc_custom_shock"
            )
            custom_vol = st.slider(
                "Volatility multiplier", 0.5, 3.0, 1.0, 0.1,
                format="%.1fx", key="sc_custom_vol"
            )
        else:
            custom_shock = 0.0
            custom_vol   = 1.0

        run_btn = st.button("▶  Run Simulation", use_container_width=True)

    with sc2:
        if run_btn or st.session_state.get("sc_ran"):
            st.session_state.sc_ran = True

            # Validate simulation inputs before running.
            try:
                _years = validate_horizon_years(float(horizon_years))
                _n_sims = validate_simulation_count(int(n_simulations))
                _monthly = validate_monthly_contrib(float(monthly_contrib))
                if not np.isfinite(initial_investment) or initial_investment <= 0:
                    raise ValueError("Initial investment must be positive.")
                if scenario_choice == "Custom" and custom_vol <= 0:
                    raise ValueError("Volatility multiplier must be positive.")
            except ValueError as e:
                st.error(f"Invalid simulation settings: {e}")
                st.stop()

            with st.spinner(f"Running {_n_sims:,} simulations…"):
                try:
                    mu_s, cov_s, sc_info = run_predefined_scenario(
                        mu.values, cov.values,
                        scenario=scenario_choice,
                        return_shock=custom_shock,
                        vol_multiplier=custom_vol,
                    )

                    paths = simulate_portfolio_paths(
                        weights=weights,
                        mu_annual=mu_s,
                        cov_annual=cov_s,
                        initial_value=float(initial_investment),
                        years=_years,
                        n_sims=_n_sims,
                        monthly_contrib=_monthly,
                    )

                    metrics, pct_df = summarise_future_metrics(
                        paths, float(initial_investment), _years,
                        monthly_contrib=_monthly,
                    )
                except ValueError as e:
                    st.error(f"Simulation error: {e}")
                    st.stop()
                except Exception:
                    st.error(
                        "The simulation failed unexpectedly. Try fewer "
                        "simulations or a shorter horizon."
                    )
                    st.stop()

            # Key metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Expected Final Value",     fmt_usd(metrics["expected_value"]))
            m2.metric("Median Outcome",           fmt_usd(metrics["median_value"]))
            m3.metric("Worst 5% Outcome",         fmt_usd(metrics["worst_5pct"]))
            m4.metric("Probability of Loss",      fmt_pct(metrics["prob_loss"]))
            if metrics.get("contributions", 0) > 0:
                st.caption(
                    f"Total invested: {fmt_usd(metrics['total_contributed'])} "
                    f"(${metrics['initial_capital']:,.0f} initial + "
                    f"${metrics['contributions']:,.0f} contributions). "
                    f"Expected P&L vs invested: "
                    f"${metrics['expected_profit_loss']:+,.0f} "
                    f"({metrics['return_on_invested']:+.1%})."
                )

            st.markdown("---")

            # Explainability
            explanation = explain_scenario_results(
                metrics, sc_info, float(initial_investment),
                monthly_contrib=_monthly, years=_years,
            )
            st.info(explanation)

            # Charts
            st.plotly_chart(
                plot_simulation_paths(
                    paths, pct_df, float(initial_investment), sc_info["label"],
                    total_contributed=metrics["total_contributed"],
                ),
                use_container_width=True,
            )
            st.plotly_chart(
                plot_confidence_bands(pct_df, float(initial_investment), sc_info["label"]),
                use_container_width=True,
            )
            st.plotly_chart(
                plot_final_distribution(paths, metrics, float(initial_investment), sc_info["label"]),
                use_container_width=True,
            )

        else:
            st.markdown(
                """
                ### How to use this tab
                1. Set your **time horizon** and **initial investment** on the left.
                2. Optionally add a **monthly contribution**.
                3. Choose a **scenario** (Normal, Market Crash, Bull Market, High Volatility, or Custom).
                4. Click **▶ Run Simulation**.

                The engine simulates thousands of forward paths using Geometric Brownian Motion
                calibrated to your portfolio's historical returns and covariance.
                """
            )

    # Scenario comparison (always shown)
    st.markdown("---")
    st.markdown("### Scenario Comparison")

    if st.button("Compare All Scenarios", key="sc_compare"):
        try:
            _years_c = validate_horizon_years(float(horizon_years))
            _monthly_c = validate_monthly_contrib(float(monthly_contrib))
            if not np.isfinite(initial_investment) or initial_investment <= 0:
                raise ValueError("Initial investment must be positive.")
        except ValueError as e:
            st.error(f"Invalid comparison settings: {e}")
            st.stop()

        comp_rows = []
        compare_sims = 2_000  # lighter weight for comparison

        try:
            for sc_name in ["Normal", "Market Crash", "Bull Market", "High Volatility"]:
                mu_c, cov_c, info_c = run_predefined_scenario(mu.values, cov.values, sc_name)
                p = simulate_portfolio_paths(
                    weights, mu_c, cov_c, float(initial_investment),
                    years=_years_c, n_sims=compare_sims,
                    monthly_contrib=_monthly_c,
                )
                m, _ = summarise_future_metrics(
                    p, float(initial_investment), _years_c,
                    monthly_contrib=_monthly_c,
                )
                comp_rows.append({
                    "Scenario": info_c["label"],
                    "Expected Value": m["expected_value"],
                    "Median": m["median_value"],
                    "Worst 5%": m["worst_5pct"],
                    "Best 5%": m["best_5pct"],
                    "P(Loss)": m["prob_loss"],
                    "Avg Max Drawdown": m["mean_max_drawdown"],
                })
        except ValueError as e:
            st.error(f"Comparison failed: {e}")
            st.stop()
        except Exception:
            st.error("Scenario comparison failed unexpectedly.")
            st.stop()

        comp_df = pd.DataFrame(comp_rows).set_index("Scenario")
        if _monthly_c > 0:
            st.caption(
                f"Comparison includes monthly contributions of ${float(_monthly_c):,.0f} "
                f"(loss probabilities vs total invested capital)."
            )
        st.dataframe(
            comp_df.style.format({
                "Expected Value": "${:,.0f}",
                "Median": "${:,.0f}",
                "Worst 5%": "${:,.0f}",
                "Best 5%": "${:,.0f}",
                "P(Loss)": "{:.1%}",
                "Avg Max Drawdown": "{:.1%}",
            }),
            use_container_width=True,
        )


# ============================================================
# TAB 5 — PERFORMANCE vs BENCHMARK
# ============================================================
with tab_perf:
    st.markdown("## Performance vs Benchmark")

    if "bench_ticker" not in st.session_state:
        st.session_state.bench_ticker = "SPY"

    set1, set2, set3 = st.columns([2, 2, 3])
    with set1:
        st.text_input("Benchmark ticker", key="bench_ticker",
                      help="Any Yahoo Finance ticker, e.g. SPY, QQQ, ^GSPC, ^FTSE.")
    with set2:
        roll_win = st.selectbox(
            "Rolling window (trading days)",
            [21, 63, 126, 252], index=1, key="perf_rollwin",
            help="21d ≈ 1 month, 63d ≈ 1 quarter, 126d ≈ half year, 252d ≈ 1 year.",
        )
    with set3:
        bench_period = st.session_state.get("hist_period_tab1", "2y")
        st.caption(
            f"Benchmark is loaded over the same **{bench_period}** history period "
            "as the portfolio and all comparisons use only common trading dates."
        )

    try:
        bench_sym = validate_ticker_symbol(st.session_state.bench_ticker)
    except ValueError as e:
        st.error(f"Invalid benchmark ticker: {e}")
        bench_sym = None

    if bench_sym is not None:
        with st.spinner(f"Loading benchmark {bench_sym}…"):
            try:
                bench_prices = load_benchmark_data(bench_sym, bench_period)
            except ValueError as e:
                st.error(f"Benchmark error: {e}")
                bench_prices = None
            except Exception:
                st.error(
                    "Could not load benchmark data (network or API error). "
                    "Check the ticker and your connection."
                )
                bench_prices = None

        if bench_prices is not None:
            try:
                bench_rets_full = log_returns_from_prices(bench_prices)
                p_al, b_al = align_return_series(port_daily_returns, bench_rets_full)
            except ValueError as e:
                st.error(f"Benchmark comparison unavailable: {e}")
                p_al = None
            else:
                try:
                    perf = summarise_performance(
                        port_daily_returns, bench_rets_full, rf_rate)
                except ValueError as e:
                    st.warning(f"Relative metrics unavailable: {e}")
                    perf = summarise_performance(
                        port_daily_returns, None, rf_rate)
                perf_p = perf["portfolio"]
                perf_b = perf["benchmark"]
                perf_r = perf["relative"]

                # ---- Growth of $10,000 ----
                st.markdown("---")
                st.markdown("### Growth of $10,000")
                g_p = growth_of_capital(p_al, 10_000.0)
                g_b = growth_of_capital(b_al, 10_000.0)
                gh1, gh2 = st.columns(2)
                gh1.metric("Portfolio", fmt_usd(float(g_p.iloc[-1])),
                           help="Final value of $10,000 in the portfolio.")
                gh2.metric(bench_sym, fmt_usd(float(g_b.iloc[-1])),
                           help=f"Final value of $10,000 in {bench_sym}.")
                st.plotly_chart(
                    plot_growth_comparison(g_p, g_b, 10_000.0, bench_sym),
                    use_container_width=True,
                )
                st.caption(
                    f"Both series start at $10,000 on {g_p.index[0].date()} "
                    f"({len(p_al):,} common trading dates)."
                )

                # ---- Performance summary panel ----
                st.markdown("---")
                st.markdown("### Performance Summary")
                s1, s2, s3 = st.columns(3)
                with s1:
                    st.markdown("#### PORTFOLIO PERFORMANCE")
                    st.metric("Annualised Return", fmt_pct(perf_p["ann_return"]),
                              help="Mean daily log return × 252.")
                    st.metric("Annualised Volatility", fmt_pct(perf_p["ann_vol"]),
                              help="Daily std × √252.")
                    st.metric("Sharpe", f"{perf_p['sharpe']:.2f}",
                              help="Excess return over cash per unit of total volatility.")
                    st.metric("Sortino", f"{perf_p['sortino']:.2f}",
                              help="Excess return per unit of downside volatility (losses only).")
                    st.metric("Calmar", f"{perf_p['calmar']:.2f}",
                              help="Annualised return per unit of maximum drawdown.")
                    st.metric("Maximum Drawdown", fmt_pct(perf_p["max_drawdown"]),
                              help="Worst peak-to-trough fall on the aligned history.")
                with s2:
                    st.markdown("#### BENCHMARK")
                    if perf_b is not None:
                        st.metric("Annualised Return", fmt_pct(perf_b["ann_return"]))
                        st.metric("Annualised Volatility", fmt_pct(perf_b["ann_vol"]))
                        st.metric("Sharpe", f"{perf_b['sharpe']:.2f}")
                        st.metric("Maximum Drawdown", fmt_pct(perf_b["max_drawdown"]))
                    else:
                        st.caption("Benchmark metrics unavailable.")
                with s3:
                    st.markdown("#### RELATIVE PERFORMANCE")
                    if perf_r["beta"] is not None:
                        beta = perf_r["beta"]
                        st.metric("Beta", f"{beta:.2f}",
                                  help="Portfolio sensitivity to benchmark moves.")
                        gap = (beta - 1.0) * 100
                        direction = "more" if gap >= 0 else "less"
                        st.caption(
                            f"Beta {beta:.2f}: the portfolio has historically moved "
                            f"≈{abs(gap):.0f}% {direction} than {bench_sym}. "
                            "Association, not causality."
                        )
                        st.metric(
                            "Historical Alpha", f"{perf_r['historical_alpha']:+.2%}",
                            help="Past CAPM-style excess return. Historical and "
                                 "model-dependent — not expected future alpha.",
                        )
                        st.metric("Tracking Error", fmt_pct(perf_r["tracking_error"]),
                                  help="Annualised std of portfolio-minus-benchmark returns.")
                        st.metric("Information Ratio", f"{perf_r['information_ratio']:.2f}",
                                  help="Annualised active return per unit of tracking error.")
                    else:
                        st.caption("Relative metrics unavailable for this benchmark.")

                # ---- Portfolio vs benchmark table ----
                st.markdown("---")
                st.markdown("### Portfolio vs Benchmark")
                p_simple = np.exp(p_al) - 1
                b_simple = np.exp(b_al) - 1

                def _fmt_cell(metric: str, v: float) -> str:
                    if v is None or not np.isfinite(v):
                        return "n/a"
                    if metric == "Sharpe":
                        return f"{v:.2f}"
                    if "Day" in metric:
                        return f"{v:+.2%}"
                    return f"{v:.2%}"

                _metrics = ["Annualised Return", "Volatility", "Sharpe",
                            "Max Drawdown", "Best Day", "Worst Day"]
                _p_vals = [perf_p["ann_return"], perf_p["ann_vol"],
                           perf_p["sharpe"], perf_p["max_drawdown"],
                           p_simple.max(), p_simple.min()]
                _b_vals = ([perf_b["ann_return"], perf_b["ann_vol"],
                            perf_b["sharpe"], perf_b["max_drawdown"],
                            b_simple.max(), b_simple.min()]
                           if perf_b is not None else [np.nan] * 6)
                perf_table = pd.DataFrame(
                    {
                        "Portfolio": [_fmt_cell(m, v) for m, v in zip(_metrics, _p_vals)],
                        bench_sym: [_fmt_cell(m, v) for m, v in zip(_metrics, _b_vals)],
                    },
                    index=_metrics,
                )
                st.dataframe(perf_table, use_container_width=True)
                st.caption(
                    f"Best day: portfolio {p_simple.idxmax().date()} "
                    f"({p_simple.max():.2%}) vs {bench_sym} {b_simple.idxmax().date()} "
                    f"({b_simple.max():.2%}). Worst day: portfolio "
                    f"{p_simple.idxmin().date()} ({p_simple.min():.2%}) vs "
                    f"{bench_sym} {b_simple.idxmin().date()} ({b_simple.min():.2%})."
                )

                # ---- Drawdown chart + underwater episodes ----
                st.markdown("---")
                st.markdown("### Drawdowns")
                dd_p = drawdown_series(g_p)
                dd_b = drawdown_series(g_b)
                st.plotly_chart(
                    plot_drawdown(dd_p, dd_b, bench_sym, perf_p["max_drawdown"]),
                    use_container_width=True,
                )
                st.markdown("#### Five Largest Drawdowns (Portfolio, Full History)")
                wealth_full = growth_of_capital(port_daily_returns, 1.0)
                try:
                    uw = underwater_episodes(wealth_full, top_n=5)
                except ValueError as e:
                    st.warning(f"Underwater analysis unavailable: {e}")
                    uw = None
                if uw is not None and not uw.empty:
                    disp = uw.copy()
                    for col in ("Peak", "Trough"):
                        disp[col] = pd.to_datetime(disp[col]).dt.strftime("%Y-%m-%d")
                    disp["Recovery"] = disp["Recovery"].apply(
                        lambda d: "Not yet recovered"
                        if pd.isna(d) else pd.Timestamp(d).strftime("%Y-%m-%d")
                    )
                    st.dataframe(
                        disp.style.format({
                            "Drawdown %": "{:.2%}",
                            "Duration (days)": "{:.0f}",
                        }),
                        use_container_width=True,
                    )
                else:
                    st.caption("No drawdown episodes found.")

                # ---- Rolling analytics ----
                st.markdown("---")
                st.markdown(f"### Rolling Analytics ({int(roll_win)}-Day Window)")
                if len(p_al) < int(roll_win):
                    st.warning(
                        f"Only {len(p_al)} common observations — need at least "
                        f"{int(roll_win)} for the selected rolling window. "
                        "Choose a shorter window or load a longer history."
                    )
                else:
                    try:
                        rr_p = rolling_annualised_return(p_al, int(roll_win))
                        rr_b = rolling_annualised_return(b_al, int(roll_win))
                        rv_p = rolling_volatility(p_al, int(roll_win))
                        rv_b = rolling_volatility(b_al, int(roll_win))
                        rs_p = rolling_sharpe(p_al, rf_rate, int(roll_win))
                        rs_b = rolling_sharpe(b_al, rf_rate, int(roll_win))
                    except ValueError as e:
                        st.warning(f"Rolling analytics unavailable: {e}")
                    else:
                        def _roll_fig(p_s, b_s, title, ytitle, fmt):
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=p_s.index, y=p_s.values, mode="lines",
                                name="Portfolio", line=dict(color="#58a6ff", width=2)))
                            fig.add_trace(go.Scatter(
                                x=b_s.index, y=b_s.values, mode="lines",
                                name=bench_sym, line=dict(color="#f0883e", width=1.5)))
                            fig.update_layout(**CHART_THEME, title=title,
                                              xaxis_title="Date", yaxis_title=ytitle,
                                              yaxis_tickformat=fmt)
                            return fig

                        st.plotly_chart(_roll_fig(
                            rr_p, rr_b, "Rolling Annualised Return",
                            "Return", ".0%"), use_container_width=True)
                        st.plotly_chart(_roll_fig(
                            rv_p, rv_b, "Rolling Annualised Volatility",
                            "Volatility (σ)", ".0%"), use_container_width=True)
                        st.plotly_chart(_roll_fig(
                            rs_p, rs_b, "Rolling Sharpe Ratio",
                            "Sharpe", ".1f"), use_container_width=True)

                # ---- Best / worst periods ----
                st.markdown("---")
                st.markdown("### Best / Worst Periods (Portfolio)")
                try:
                    bw = best_worst_periods(port_daily_returns)
                except ValueError as e:
                    st.warning(f"Best/worst periods unavailable: {e}")
                else:
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Best Day",
                              f"{bw['best_day'][1]:+.2%} — {bw['best_day'][0].date()}")
                    b2.metric("Worst Day",
                              f"{bw['worst_day'][1]:+.2%} — {bw['worst_day'][0].date()}")
                    b3.metric("Best Month", f"{bw['best_month'][1]:+.2%} — {bw['best_month'][0]}")
                    b4.metric("Worst Month",
                              f"{bw['worst_month'][1]:+.2%} — {bw['worst_month'][0]}")
                    st.caption(
                        "Monthly returns compound daily log returns "
                        "(exp of summed logs − 1), never simple averages."
                    )

    # ---- Metric glossary ----
    with st.expander("Metric glossary"):
        st.markdown(
            """
            **Sharpe** — excess return over cash per unit of total volatility. Higher is better.
            **Sortino** — like Sharpe but only downside volatility counts; upside swings are not penalised.
            **Calmar** — annualised return divided by the worst peak-to-trough drawdown.
            **Beta** — how strongly the portfolio has historically moved with the benchmark (1.0 = in step).
            **Alpha (historical)** — past return left over after accounting for beta; model-dependent, not a forecast.
            **Tracking Error** — annualised volatility of portfolio-minus-benchmark returns.
            **Information Ratio** — active return per unit of tracking error.
            **VaR** — loss threshold crossed only in the worst 5% of periods.
            **CVaR / Expected Shortfall** — average loss once the VaR threshold is crossed.
            **Maximum Drawdown** — largest observed peak-to-trough fall; 0 means a prior peak.
            """
        )


# ============================================================
# TAB 6 — SECURITY DETAIL
# ============================================================
with tab_security:
    st.markdown("## Security Detail")

    if st.session_state.get("security_ticker") not in tickers:
        st.session_state.security_ticker = tickers[0]
    sel = st.selectbox(
        "Security", tickers, key="security_ticker",
        help="Holdings come from your current portfolio. Analytics below "
             "follow this selection.",
    )
    sel_idx = tickers.index(sel)

    # ---- Top security summary (position facts, not performance) ----
    sel_price = float(current_prices[sel])
    sel_shares = float(st.session_state.portfolio[sel]["shares"])
    sel_value = sel_price * sel_shares
    sel_weight = float(weights[sel_idx])
    sel_cost = st.session_state.portfolio[sel].get("avg_cost")

    st.markdown(f"### {sel}")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Current Price", fmt_usd(sel_price))
    h2.metric("Position Value", fmt_usd(sel_value))
    h3.metric("Portfolio Weight", fmt_pct(sel_weight),
              help="Share of total portfolio market value.")
    h4.metric("Shares", f"{sel_shares:g}")
    # Unrealised P&L only — cost basis is captured per position. Omit it
    # entirely when no valid cost basis exists rather than inventing one.
    if sel_cost is not None and np.isfinite(sel_cost) and sel_cost > 0:
        upl = sel_value - sel_shares * float(sel_cost)
        upl_pct = upl / (sel_shares * float(sel_cost))
        st.caption(f"Unrealised P&L (vs avg cost ${float(sel_cost):,.2f}): "
                   f"${upl:+,.2f} ({upl_pct:+.2%}).")

    st.markdown("---")

    # ---- Holding analytics (project's own data pipeline only) ----
    st.markdown("### Holding Analytics")
    sec_rets = rets[sel].dropna()

    # Benchmark leg for beta: reuse the Performance tab benchmark (SPY
    # default) through the same cached loader — no extra request.
    bench_period = st.session_state.get("hist_period_tab1", "2y")
    bench_rets_sec = None
    sec_bench_sym = None
    try:
        sec_bench_sym = validate_ticker_symbol(
            st.session_state.get("bench_ticker", "SPY"))
    except ValueError:
        sec_bench_sym = None
    if sec_bench_sym is not None and sec_bench_sym != sel:
        try:
            _bp = load_benchmark_data(sec_bench_sym, bench_period)
            bench_rets_sec = log_returns_from_prices(_bp)
        except Exception:
            bench_rets_sec = None  # beta shows as unavailable, tab continues

    try:
        sec = security_summary(
            sec_rets, weights, cov.values, sel_idx, rf_rate,
            portfolio_log_returns=port_daily_returns,
            benchmark_log_returns=bench_rets_sec,
        )
    except ValueError as e:
        st.error(f"Holding analytics unavailable: {e}")
        sec = None

    if sec is not None:
        a1, a2, a3 = st.columns(3)
        a1.metric("Annualised Return", fmt_pct(sec["annualised_return"]),
                  help="Mean daily log return × 252, this holding only.")
        a2.metric("Annualised Volatility", fmt_pct(sec["annualised_volatility"]),
                  help="Daily std × √252, this holding only.")
        a3.metric("Sharpe Ratio", f"{sec['sharpe']:.2f}",
                  help="Holding excess return per unit of its own volatility.")
        b1, b2, b3 = st.columns(3)
        with b1:
            if sec["beta_vs_benchmark"] is not None:
                st.metric(
                    f"Beta vs {sec_bench_sym}", f"{sec['beta_vs_benchmark']:.2f}",
                    help="Historical sensitivity to the benchmark. "
                         "Association, not causality.")
            else:
                st.metric("Beta", "n/a")
                st.caption(sec.get("beta_error")
                           or "Benchmark unavailable for beta.")
        with b2:
            st.metric("Contribution to Portfolio Risk",
                      fmt_pct(sec["risk_contribution"]),
                      help="This holding's share of portfolio volatility "
                           "(same model as the Overview risk breakdown).")
        with b3:
            if sec["correlation_to_portfolio"] is not None:
                st.metric("Correlation to Portfolio",
                          f"{sec['correlation_to_portfolio']:.2f}",
                          help="Co-movement with the portfolio return series. "
                               "Association, not causality.")
                st.caption(sec["correlation_note"])
            else:
                st.metric("Correlation to Portfolio", "n/a")
                st.caption(sec.get("correlation_error")
                           or "Correlation unavailable.")

    st.markdown("---")

    # ---- TradingView advanced chart (visual analysis only) ----
    st.markdown("### Interactive Chart")
    hints = {}
    for t in tickers:
        try:
            hints[t] = load_exchange_hint(t)
        except Exception:
            hints[t] = None
    watchlist = []
    for t in tickers:
        try:
            watchlist.append(resolve_tradingview_symbol(t, hints.get(t)).symbol)
        except ValueError:
            continue

    override_key = f"tv_override_{sel}"
    with st.expander("TradingView symbol override (advanced)"):
        st.text_input(
            "TradingView symbol", key=override_key, placeholder="e.g. NASDAQ:AAPL",
            help="Chart display only. Never affects portfolio data or analytics.",
        )
        st.caption(f"Yahoo ticker: {sel}")
    override_val = (st.session_state.get(override_key) or "").strip() or None

    tv_failed = False
    try:
        resolved = resolve_tradingview_symbol(sel, hints.get(sel), override_val)
    except ValueError as e:
        tv_failed = True
        st.warning(f"TradingView chart unavailable for {sel}: {e} "
                   "Portfolio Calc analytics above are unaffected; you can "
                   "change the symbol inside the widget once it loads.")
        resolved = None

    if not tv_failed and resolved is not None:
        if not resolved.resolved:
            st.caption(f"ℹ️ {resolved.note}")
        try:
            render_tradingview_chart(resolved.symbol, watchlist, height=680)
        except Exception:
            st.warning(
                "The TradingView widget could not display this symbol. "
                "Portfolio Calc analytics above are unaffected — try the "
                "widget's symbol search or the override above.")
        st.caption(
            "Streamlit selector drives Portfolio Calc analytics; the widget "
            "watchlist switches only the embedded chart. Chart data is "
            "TradingView's own and is never used in calculations.")

    # ---- Reproducible internal price chart (project data) ----
    with st.expander("Portfolio data chart (reproducible)"):
        st.caption("Plotted from Portfolio Calc's own yfinance history — "
                   "the reproducible application data, unlike the widget above.")
        fig_sec = go.Figure()
        fig_sec.add_trace(go.Scatter(
            x=prices.index, y=prices[sel],
            mode="lines", name=sel, line=dict(color="#58a6ff", width=2),
        ))
        fig_sec.update_layout(**CHART_THEME, title=f"{sel} Price History",
                              xaxis_title="Date", yaxis_title="Price ($)")
        st.plotly_chart(fig_sec, use_container_width=True)

    # ---- Performance snapshot (project's own prices) ----
    st.markdown("---")
    st.markdown("### Performance Snapshot")
    try:
        snap = period_returns(prices[sel])
    except ValueError as e:
        st.warning(f"Performance snapshot unavailable: {e}")
        snap = None
    if snap is not None:
        cols = st.columns(5)
        for col, label in zip(cols, ["1M", "3M", "6M", "YTD", "1Y"]):
            entry = snap[label]
            if entry["return"] is None:
                col.metric(f"{label} Return", "n/a",
                           help="Not enough history for this window.")
            else:
                col.metric(
                    f"{label} Return", f"{entry['return']:+.2%}",
                    help=f"{entry['start']} → {entry['end']}, from actual prices.")
        st.caption("Window returns use date-based ranges on Portfolio Calc "
                   "price history — never scraped from TradingView.")

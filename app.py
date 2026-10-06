"""
app.py
------
Portfolio Analytics & Risk Dashboard — Streamlit entry point.

Tabs:
  1. 📈 Overview      — positions, returns, correlation/covariance
  2. ⚠️  Risk          — VaR (parametric, historical, Monte Carlo)
  3. 🎯 Optimise      — efficient frontier, max Sharpe, min variance, target constraints
  4. 📊 Scenario      — Monte Carlo forward simulation with stress scenarios

Run locally:
    streamlit run app.py

The app is designed to be modular so the analytics modules can be imported
and used independently of Streamlit (e.g. in a desktop GUI or CLI).
"""

import os
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from data.market_data import (
    fetch_price_history,
    fetch_current_prices,
    fetch_risk_free_rate,
)
from analytics.returns import (
    daily_returns,
    annualised_mean_returns,
    annualised_cov_matrix,
    correlation_matrix,
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
                return json.load(f)
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

def style_matrix(df: pd.DataFrame, fmt: str = ".4f") -> pd.io.formats.style.Styler:
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
            if new_ticker:
                st.session_state.portfolio[new_ticker] = {
                    "shares": new_shares,
                    "avg_cost": new_cost,
                }
                save_portfolio(st.session_state.portfolio)
                st.session_state.data_loaded = False
                st.success(f"Added {new_ticker}")
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


tickers = sorted(st.session_state.portfolio.keys())

if not tickers:
    st.warning("Add at least one position in the sidebar to get started.")
    st.stop()

with st.spinner("Loading market data…"):
    # Default history period — can be overridden per-tab
    prices, current_prices, rf_rate = load_market_data(tickers, "2y")

# Compute core analytics
shares_dict = {t: st.session_state.portfolio[t]["shares"] for t in tickers}
weights, _ = weights_from_shares(shares_dict, current_prices)

portfolio_value = sum(
    st.session_state.portfolio[t]["shares"] * current_prices.get(t, 0.0)
    for t in tickers
)

rets   = daily_returns(prices)
mu     = annualised_mean_returns(rets)
cov    = annualised_cov_matrix(rets)
corr   = correlation_matrix(rets)

port_return  = portfolio_expected_return(weights, mu.values)
port_vol     = portfolio_std(weights, cov.values)
port_sharpe  = sharpe_ratio(port_return, port_vol, rf_rate)
port_daily_returns = (rets * weights).sum(axis=1)

rc = risk_contributions(weights, cov.values)

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_overview, tab_risk, tab_optimise, tab_scenario = st.tabs([
    "📈 Overview",
    "⚠️  Risk",
    "🎯 Optimise",
    "📊 Scenario Analysis",
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
            prices, current_prices, rf_rate = load_market_data(tickers, hist_period)
            rets = daily_returns(prices)
            mu   = annualised_mean_returns(rets)
            cov  = annualised_cov_matrix(rets)
            corr = correlation_matrix(rets)
            port_return = portfolio_expected_return(weights, mu.values)
            port_vol    = portfolio_std(weights, cov.values)
            port_sharpe = sharpe_ratio(port_return, port_vol, rf_rate)
            rc = risk_contributions(weights, cov.values)
            port_daily_returns = (rets * weights).sum(axis=1)

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
        .applymap(lambda v: "color: #3fb950" if isinstance(v, (int, float)) and v > 0
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

    # Cumulative returns chart
    st.markdown("---")
    st.markdown("### Cumulative Returns")

    cum_ret = (1 + rets).cumprod()
    fig_cum = go.Figure()
    for col in tickers:
        fig_cum.add_trace(go.Scatter(
            x=cum_ret.index, y=cum_ret[col],
            mode="lines", name=col,
        ))
    fig_cum.update_layout(
        **CHART_THEME,
        title="Cumulative Return (base = 1)",
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

    # Daily parameters
    port_std_daily  = to_daily(port_vol)
    mu_daily        = mean_to_daily(mu.values)
    cov_daily       = cov.values / 252

    var_param = parametric_var(port_std_daily, portfolio_value, 0.95, var_horizon)
    var_hist  = historical_var(port_daily_returns, portfolio_value, 0.95, var_horizon)
    var_mc, mc_sim_rets = monte_carlo_var(
        weights, mu_daily, cov_daily, portfolio_value, 0.95, var_horizon, 10_000
    )

    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown("#### Parametric VaR")
        st.metric("95% VaR", fmt_usd(var_param))
        st.caption(
            "Assumes normally distributed returns. "
            f"At 95% confidence over {var_horizon}d, potential loss ≤ {fmt_usd(var_param)}."
        )
    with v2:
        st.markdown("#### Historical VaR")
        st.metric("95% VaR", fmt_usd(var_hist))
        st.caption(
            "Uses the actual empirical return distribution from historical data. "
            "No normality assumption."
        )
    with v3:
        st.markdown("#### Monte Carlo VaR")
        st.metric("95% VaR", fmt_usd(var_mc))
        st.caption(
            "10,000 simulated return paths using multivariate normal. "
            "Captures correlation structure."
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
        x=var_pct_5, y=0, text="5th pct",
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
        x=var_mc_pct, y=0, text="VaR 5th pct",
        showarrow=True, arrowcolor="#f85149",
        font=dict(color="#f85149"), yref="paper", y=0.9,
    )
    fig_mc.update_layout(
        **CHART_THEME,
        title=f"MC Simulated {var_horizon}-Day Portfolio Returns",
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

    st.markdown("---")

    # Run optimisations
    res_mv = min_variance(mu_arr, cov_arr, rf_rate, min_w, max_w)
    res_ms = max_sharpe(mu_arr, cov_arr, rf_rate, min_w, max_w)

    # Target return slider
    display_return = port_return
    r_min_bound = float(np.clip(mu_arr.min(), -0.30, 0.50))
    r_max_bound = float(np.clip(mu_arr.max(), r_min_bound + 0.01, 1.0))

    if "target_r_val" not in st.session_state:
        st.session_state.target_r_val = float(np.clip(display_return, r_min_bound, r_max_bound))
    target_r_val = st.slider(
        "Target annual return",
        r_min_bound, r_max_bound,
        st.session_state.target_r_val,
        0.005,
        format="%.1f%%",
        key="target_r_val",
    )
    res_tr = target_return(mu_arr, cov_arr, target_r_val, rf_rate, min_w, max_w)

    # Target volatility slider
    v_min_bound = float(np.clip(res_mv.volatility, 0.01, 0.99))
    v_max_bound = float(np.clip(mu_arr.std() * 5, v_min_bound + 0.01, 1.0))
    if "target_v_val" not in st.session_state:
        st.session_state.target_v_val = float(np.clip(port_vol, v_min_bound, v_max_bound))
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
        status = "✅" if res.success else "⚠️"
        cols = st.columns(4)
        cols[0].metric(f"{status} {label} — Return", fmt_pct(res.expected_return))
        cols[1].metric("Volatility", fmt_pct(res.volatility))
        cols[2].metric("Sharpe", f"{res.sharpe:.2f}")
        w_df = pd.DataFrame({"Ticker": tickers, "Weight": res.weights}).set_index("Ticker")
        cols[3].dataframe(w_df.style.format({"Weight": "{:.1%}"}), height=160)

    _result_card("Min Variance", res_mv, tickers)
    st.markdown("---")
    _result_card("Max Sharpe", res_ms, tickers)
    st.markdown("---")
    _result_card(f"Target Return {target_r_val:.1%}", res_tr, tickers)
    st.markdown("---")
    _result_card(f"Target Vol {target_v_val:.1%}", res_tv, tickers)

    st.markdown("---")

    # Efficient frontier
    st.markdown("### Efficient Frontier")
    with st.spinner("Tracing efficient frontier…"):
        ef_df = efficient_frontier(mu_arr, cov_arr, rf_rate, 60, min_w, max_w)

    if not ef_df.empty:
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

    if chosen.success:
        trades_df = rebalance_trades(weights, chosen.weights, tickers, portfolio_value)

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
            .applymap(_colour_action, subset=["Action"]),
            use_container_width=True,
        )
    else:
        st.warning("Optimisation did not converge for selected target.")


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

            with st.spinner(f"Running {n_simulations:,} simulations…"):
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
                    initial_value=initial_investment,
                    years=horizon_years,
                    n_sims=n_simulations,
                    monthly_contrib=monthly_contrib,
                )

                metrics, pct_df = summarise_future_metrics(
                    paths, initial_investment, horizon_years
                )

            # Key metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Expected Final Value",     fmt_usd(metrics["expected_value"]))
            m2.metric("Median Outcome",           fmt_usd(metrics["median_value"]))
            m3.metric("Worst 5% Outcome",         fmt_usd(metrics["worst_5pct"]))
            m4.metric("Probability of Loss",      fmt_pct(metrics["prob_loss"]))

            st.markdown("---")

            # Explainability
            explanation = explain_scenario_results(metrics, sc_info, initial_investment)
            st.info(explanation)

            # Charts
            st.plotly_chart(
                plot_simulation_paths(paths, pct_df, initial_investment, sc_info["label"]),
                use_container_width=True,
            )
            st.plotly_chart(
                plot_confidence_bands(pct_df, initial_investment, sc_info["label"]),
                use_container_width=True,
            )
            st.plotly_chart(
                plot_final_distribution(paths, metrics, initial_investment, sc_info["label"]),
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
        comp_rows = []
        compare_sims = 2_000  # lighter weight for comparison

        for sc_name in ["Normal", "Market Crash", "Bull Market", "High Volatility"]:
            mu_c, cov_c, info_c = run_predefined_scenario(mu.values, cov.values, sc_name)
            p = simulate_portfolio_paths(
                weights, mu_c, cov_c, initial_investment,
                years=horizon_years, n_sims=compare_sims,
            )
            m, _ = summarise_future_metrics(p, initial_investment, horizon_years)
            comp_rows.append({
                "Scenario": info_c["label"],
                "Expected Value": m["expected_value"],
                "Median": m["median_value"],
                "Worst 5%": m["worst_5pct"],
                "Best 5%": m["best_5pct"],
                "P(Loss)": m["prob_loss"],
                "Avg Max Drawdown": m["mean_max_drawdown"],
            })

        comp_df = pd.DataFrame(comp_rows).set_index("Scenario")
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

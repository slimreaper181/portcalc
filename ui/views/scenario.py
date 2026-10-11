"""views.scenario — Scenario Analysis tab (rendered by the app shell).

Extracted from the application shell; analytics live in
``analytics/``, fetching in ``data/``. Receives the shared
context dict and renders into the active tab.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from analytics.currency import currency_symbol
from analytics.positions import format_money
from analytics.scenario import (
    explain_scenario_results,
    plot_confidence_bands,
    plot_final_distribution,
    plot_simulation_paths,
    run_predefined_scenario,
    simulate_portfolio_paths,
    summarise_future_metrics,
)
from analytics.validation import (
    validate_horizon_years,
    validate_monthly_contrib,
    validate_simulation_count,
)
from ui.formatting import fmt_pct, fmt_usd


def render(ctx):
    """Render the Scenario Analysis tab from the shared context."""
    base_ccy = ctx["base_ccy"]
    cov = ctx["cov"]
    mu = ctx["mu"]
    portfolio_value = ctx["portfolio_value"]
    weights = ctx["weights"]

    st.markdown("## Scenario & Monte Carlo Simulation")

    sc1, sc2 = st.columns([1, 2])

    with sc1:
        st.markdown("### Simulation Settings")

        horizon_years = st.slider(
            "Time horizon (years)", 1, 10, 5, 1, key="sc_horizon"
        )
        initial_investment = st.number_input(
            f"Initial investment ({currency_symbol(base_ccy).strip() or base_ccy})",
            min_value=1_000.0,
            value=float(max(portfolio_value, 1_000.0)),
            step=1_000.0,
            key="sc_initial",
        )
        monthly_contrib = st.number_input(
            f"Monthly contribution ({currency_symbol(base_ccy).strip() or base_ccy})",
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

        run_btn = st.button("▶  Run Simulation", width="stretch")

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
                return  # section-local return (was st.stop()): sibling tabs keep rendering

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
                    return  # section-local return (was st.stop()): sibling tabs keep rendering
                except Exception:
                    st.error(
                        "The simulation failed unexpectedly. Try fewer "
                        "simulations or a shorter horizon."
                    )
                    return  # section-local return (was st.stop()): sibling tabs keep rendering

            # Key metrics: large currency values get 2 columns x 2 rows
            # so full "$6,589.31"-style values always fit.
            r1a, r1b = st.columns(2)
            r1a.metric("Expected Final Value", fmt_usd(metrics["expected_value"]))
            r1b.metric("Median Outcome",        fmt_usd(metrics["median_value"]))
            r2a, r2b = st.columns(2)
            r2a.metric("Worst 5% Outcome",     fmt_usd(metrics["worst_5pct"]))
            r2b.metric("Probability of Loss",  fmt_pct(metrics["prob_loss"]))
            if metrics.get("contributions", 0) > 0:
                _csym = currency_symbol(base_ccy)
                st.caption(
                    f"Total invested: {fmt_usd(metrics['total_contributed'])} "
                    f"({_csym}{metrics['initial_capital']:,.0f} initial + "
                    f"{_csym}{metrics['contributions']:,.0f} contributions). "
                    f"Expected P&L vs invested: "
                    f"{_csym}{metrics['expected_profit_loss']:+,.0f} "
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
                width="stretch",
            )
            st.plotly_chart(
                plot_confidence_bands(pct_df, float(initial_investment), sc_info["label"]),
                width="stretch",
            )
            st.plotly_chart(
                plot_final_distribution(paths, metrics, float(initial_investment), sc_info["label"]),
                width="stretch",
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
            return  # section-local return (was st.stop()): sibling tabs keep rendering

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
            return  # section-local return (was st.stop()): sibling tabs keep rendering
        except Exception:
            st.error("Scenario comparison failed unexpectedly.")
            return  # section-local return (was st.stop()): sibling tabs keep rendering

        comp_df = pd.DataFrame(comp_rows).set_index("Scenario")
        if _monthly_c > 0:
            st.caption(
                f"Comparison includes monthly contributions of "
                f"{format_money(float(_monthly_c), base_ccy)} "
                f"(loss probabilities vs total invested capital)."
            )
        _csym = currency_symbol(base_ccy)
        st.dataframe(
            comp_df.style.format({
                "Expected Value": f"{_csym}{{:,.0f}}",
                "Median": f"{_csym}{{:,.0f}}",
                "Worst 5%": f"{_csym}{{:,.0f}}",
                "Best 5%": f"{_csym}{{:,.0f}}",
                "P(Loss)": "{:.1%}",
                "Avg Max Drawdown": "{:.1%}",
            }),
            width="stretch",
        )


# ============================================================

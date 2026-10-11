"""views.performance — Performance tab (rendered by the app shell).

Extracted from the application shell; analytics live in
``analytics/``, fetching in ``data/``. Receives the shared
context dict and renders into the active tab.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.currency import currency_symbol
from analytics.performance import (
    align_return_series,
    best_worst_periods,
    drawdown_series,
    growth_of_capital,
    log_returns_from_prices,
    plot_drawdown,
    plot_growth_comparison,
    rolling_annualised_return,
    rolling_sharpe,
    rolling_volatility,
    summarise_performance,
    underwater_episodes,
)
from analytics.validation import validate_ticker_symbol
from data.pipeline import base_benchmark_series, load_benchmark_data
from ui.formatting import fmt_pct, fmt_usd
from ui.theme import CHART_THEME


def render(ctx):
    """Render the Performance tab from the shared context."""
    base_ccy = ctx["base_ccy"]
    ccy = ctx["ccy"]
    port_daily_returns = ctx["port_daily_returns"]
    rf_rate = ctx["rf_rate"]

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
                # Translate the benchmark into base currency first, so all
                # relative metrics live in the investor's return space.
                bench_prices = base_benchmark_series(
                    bench_prices, bench_sym, base_ccy, ccy.get("fx_hist"))
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
                _g_sym = currency_symbol(base_ccy)
                st.markdown(f"### Growth of {_g_sym}10,000")
                g_p = growth_of_capital(p_al, 10_000.0)
                g_b = growth_of_capital(b_al, 10_000.0)
                gh1, gh2 = st.columns(2)
                gh1.metric("Portfolio", fmt_usd(float(g_p.iloc[-1])),
                           help=f"Final value of {_g_sym}10,000 in the portfolio.")
                gh2.metric(bench_sym, fmt_usd(float(g_b.iloc[-1])),
                           help=f"Final value of {_g_sym}10,000 in {bench_sym}.")
                st.plotly_chart(
                    plot_growth_comparison(g_p, g_b, 10_000.0, bench_sym,
                                           currency=base_ccy),
                    width="stretch",
                )
                st.caption(
                    f"Both series start at {_g_sym}10,000 on {g_p.index[0].date()} "
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
                st.dataframe(perf_table, width="stretch")
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
                    width="stretch",
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
                        width="stretch",
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
                            "Return", ".0%"), width="stretch")
                        st.plotly_chart(_roll_fig(
                            rv_p, rv_b, "Rolling Annualised Volatility",
                            "Volatility (σ)", ".0%"), width="stretch")
                        st.plotly_chart(_roll_fig(
                            rs_p, rs_b, "Rolling Sharpe Ratio",
                            "Sharpe", ".1f"), width="stretch")

                # ---- Best / worst periods ----
                st.markdown("---")
                st.markdown("### Best / Worst Periods (Portfolio)")
                try:
                    bw = best_worst_periods(port_daily_returns)
                except ValueError as e:
                    st.warning(f"Best/worst periods unavailable: {e}")
                else:
                    # Day/month values carry a date suffix ("% — date"), so
                    # use 2 columns x 2 rows instead of one tight row of 4.
                    bw1, bw2 = st.columns(2)
                    bw1.metric("Best Day",
                               f"{bw['best_day'][1]:+.2%} — {bw['best_day'][0].date()}")
                    bw2.metric("Worst Day",
                               f"{bw['worst_day'][1]:+.2%} — {bw['worst_day'][0].date()}")
                    bw3, bw4 = st.columns(2)
                    bw3.metric("Best Month", f"{bw['best_month'][1]:+.2%} — {bw['best_month'][0]}")
                    bw4.metric("Worst Month",
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

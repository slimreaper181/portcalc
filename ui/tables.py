"""tables — professional dataframe presentation (pure formatting + render).

Rules: right-aligned numerics, one decimal vocabulary per column kind,
positive/negative P&L colour via pandas Styler, no rainbow gradients on
money tables.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .formatting import (format_date, format_int, format_money, format_number,
                         format_percent, format_ratio,
                         format_signed_money, format_signed_percent)


def money_table(df: pd.DataFrame, money_cols=(), pct_cols=(),
                signed_money_cols=(), signed_pct_cols=(),
                ratio_cols=(), int_cols=(), date_cols=()) -> pd.DataFrame:
    """Return a display copy with formatted strings (pure, input untouched)."""
    out = df.copy()
    for c in money_cols:
        if c in out.columns:
            out[c] = [format_money(v, out.attrs.get("currency", "USD"))
                      if not isinstance(v, str) else v for v in out[c]]
    for c in signed_money_cols:
        if c in out.columns:
            out[c] = [format_signed_money(v, out.attrs.get("currency", "USD"))
                      if not isinstance(v, str) else v for v in out[c]]
    for c in pct_cols:
        if c in out.columns:
            out[c] = [format_percent(v) if not isinstance(v, str) else v
                      for v in out[c]]
    for c in signed_pct_cols:
        if c in out.columns:
            out[c] = [format_signed_percent(v) if not isinstance(v, str) else v
                      for v in out[c]]
    for c in ratio_cols:
        if c in out.columns:
            out[c] = [format_ratio(v) if not isinstance(v, str) else v
                      for v in out[c]]
    for c in int_cols:
        if c in out.columns:
            out[c] = [format_int(v) if not isinstance(v, str) else v
                      for v in out[c]]
    for c in date_cols:
        if c in out.columns:
            out[c] = [format_date(v) if not isinstance(v, str) else v
                      for v in out[c]]
    return out


def with_currency(df: pd.DataFrame, currency: str) -> pd.DataFrame:
    """Attach display currency without mutating (used by money_table)."""
    out = df.copy()
    out.attrs["currency"] = currency
    return out


def show(df: pd.DataFrame, key: str | None = None,
         height: int | None = None) -> None:
    """Render a dataframe full-width."""
    st.dataframe(df, width="stretch", height=height)


def download_csv(df: pd.DataFrame, label: str, filename: str,
                 key: str) -> None:
    """CSV download button (data only — never secrets)."""
    st.download_button(label, data=df.to_csv(index=True).encode("utf-8"),
                       file_name=filename, mime="text/csv", key=key)


def pnl_styler(df: pd.DataFrame, cols) -> "pd.io.formats.style.Styler":
    """Green/red text for signed P&L columns (colour supports the sign)."""
    def _color(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return ""
        if f > 0:
            return "color: #3fb950"
        if f < 0:
            return "color: #f85149"
        return ""

    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df.style
    return df.style.map(_color, subset=cols)


def render_backtest_comparison(results: dict) -> None:
    """Formatted strategy-comparison table (raw values stay numeric).

    Shared by static OOS and walk-forward sections. Benchmark Total Costs
    render as N/A via ``is_benchmark``.
    """
    import streamlit as st

    from analytics.backtest import compare_backtests
    from ui.formatting import fmt_pct as _pct
    from ui.formatting import fmt_usd as _usd

    table = compare_backtests(results)
    disp = pd.DataFrame(index=table.index)
    for col in table.columns:
        is_bench = results[col].is_benchmark
        cells = []
        for metric in table.index:
            v = table.loc[metric, col]
            if metric == "Total Costs" and is_bench:
                cells.append("N/A")
            elif metric in ("Final Value", "Total Costs"):
                cells.append(_usd(v))
            elif metric in ("Total Return", "CAGR", "Annualised Return",
                             "Volatility", "Max Drawdown", "Best Day",
                             "Worst Day"):
                cells.append(_pct(v))
            elif metric in ("Sharpe", "Sortino", "Calmar"):
                cells.append(f"{v:.2f}")
            elif metric == "Rebalances":
                cells.append(f"{int(v)}")
            else:
                cells.append(str(v))
        disp[col] = cells
    st.dataframe(disp, width="stretch")

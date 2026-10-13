"""
report.py
---------
Institutional HTML report generation (Task 10, Milestone 9) — pure logic.

Builds a self-contained, printable HTML research report from a JSON-safe
payload assembled by the UI layer (Overview / Research). All
user-controlled strings (portfolio names, tickers, notes, tags, analysis
names) are escaped with :mod:`html` — never interpolated raw. The builder
takes an explicit payload only and never reads environment variables,
credentials, or secrets, so reports cannot leak API keys by construction.

Sections (each optional except metadata):
  Portfolio Summary / Holdings / P&L / Performance / Risk / Attribution /
  Stress / Saved Analyses. Metadata always includes portfolio, generated
  time (UTC), base currency, benchmark, data-through dates and sources,
  plus an explicit freshness disclaimer (never claims real-time).
"""

from __future__ import annotations

import html
from datetime import datetime, timezone


def _esc(value) -> str:
    """HTML-escape anything user-controlled (None → em dash)."""
    if value is None:
        return "—"
    try:
        import math
        if isinstance(value, float) and (
                value != value or value in (float("inf"), float("-inf"))):
            return "—"
    except Exception:
        pass
    return html.escape(str(value), quote=True)


def _money(value, ccy: str) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    import math
    if not math.isfinite(f):
        return "—"
    return f"{_esc(ccy)} {f:+,.2f}" if f < 0 else f"{_esc(ccy)} {f:,.2f}"


def _pct(value) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    import math
    if not math.isfinite(f):
        return "—"
    return f"{f:+.2%}" if f < 0 else f"{f:.2%}"


def _table(headers: list[str], rows: list[list[str]],
           caption: str = "") -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cap = f"<caption>{_esc(caption)}</caption>" if caption else ""
    return (f"<table>{cap}<thead><tr>{head}</tr></thead>"
            f"<tbody>{body or '<tr><td>—</td></tr>'}</tbody></table>")


def build_report_html(payload: dict) -> str:
    """Render a complete printable HTML report from a JSON-safe payload.

    Expected keys (all optional; missing sections render as '—'):
      portfolio_name, base_ccy, benchmark, generated_utc, data_through,
      fx_through, sources (list[str]), freshness_note,
      nav, cash_base, day_pnl, n_positions,
      holdings (list[dict]: ticker/shares/mark/value/weight/pnl/currency),
      pnl (dict: unrealised/realised/income/fees/total/net_flows),
      performance (dict: twr/xirr/window),
      risk (dict: volatility/sharpe/var_95_1d/max_drawdown/note),
      attribution (dict: rows list[{ticker, contribution}] + note),
      stress (dict: rows list[{window, return, drawdown}] + note),
      analyses (list[dict]: name/type/created/last_run/tags),
      notes (str, plain text).
    """
    p = dict(payload or {})
    base = str(p.get("base_ccy") or "USD")
    bench = str(p.get("benchmark") or "SPY")
    pname = str(p.get("portfolio_name") or "Portfolio")
    gen = str(p.get("generated_utc") or
              datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    data_through = p.get("data_through")
    fx_through = p.get("fx_through")
    sources = p.get("sources") or [
        "Historical prices: Yahoo Finance (adjusted closes)",
        "Current US prices: Alpaca IEX (Yahoo fallback)",
        "FX: Yahoo Finance", "Factors: Kenneth French library",
        "Charts: TradingView (display only)"]

    holdings = p.get("holdings") or []
    h_rows = []
    for h in holdings:
        if not isinstance(h, dict):
            continue
        h_rows.append([
            _esc(h.get("ticker")), _esc(h.get("shares")),
            _esc(h.get("mark")), _esc(h.get("value")),
            _esc(h.get("weight")), _esc(h.get("pnl")),
            _esc(h.get("currency"))])

    pnl = p.get("pnl") or {}
    perf = p.get("performance") or {}
    risk = p.get("risk") or {}
    attr = p.get("attribution") or {}
    stress = p.get("stress") or {}
    analyses = p.get("analyses") or []
    a_rows = []
    for a in analyses:
        if not isinstance(a, dict):
            continue
        a_rows.append([
            _esc(a.get("name")), _esc(a.get("type")),
            _esc(a.get("created")), _esc(a.get("last_run")),
            _esc(a.get("tags"))])

    attr_rows = []
    for r in (attr.get("rows") or []):
        if isinstance(r, dict):
            attr_rows.append([_esc(r.get("ticker")),
                              _esc(r.get("contribution"))])
    stress_rows = []
    for r in (stress.get("rows") or []):
        if isinstance(r, dict):
            stress_rows.append([_esc(r.get("window")), _esc(r.get("return")),
                                _esc(r.get("drawdown"))])

    src_list = "".join(f"<li>{_esc(s)}</li>" for s in sources)
    now = _esc(gen)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_esc(pname)} — Portfolio Report</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', Arial, sans-serif;
         max-width: 900px; margin: 24px auto; padding: 0 16px;
         color: #111; font-size: 14px; line-height: 1.45; }}
  h1 {{ font-size: 22px; margin-bottom: 2px; }}
  h2 {{ font-size: 16px; margin-top: 26px; border-bottom: 1px solid #ccc;
        padding-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 5px 8px; text-align: left;
            font-size: 13px; }}
  th {{ background: #f2f2f2; }}
  caption {{ caption-side: top; text-align: left; font-weight: bold;
             padding-bottom: 4px; }}
  .meta {{ color: #444; font-size: 13px; }}
  .grid {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .card {{ border: 1px solid #ccc; padding: 8px 12px; min-width: 150px; }}
  .fresh {{ background: #fffbe6; border: 1px solid #e0c341; padding: 8px 12px; }}
  @media print {{ body {{ max-width: none; margin: 0; }}
    h2 {{ break-after: avoid; }} table {{ break-inside: auto; }} }}
</style>
</head>
<body>
<h1>{_esc(pname)} — Portfolio Report</h1>
<p class="meta">Generated {now} · Base {_esc(base)} · Benchmark {_esc(bench)} ·
Data through {_esc(data_through)} · FX through {_esc(fx_through)}</p>
<div class="fresh"><strong>Data freshness:</strong>
{_esc(p.get("freshness_note") or "Current marks are point-in-time quotes with stated sources; nothing here is claimed real-time.")}</div>

<h2>Portfolio Summary</h2>
<div class="grid">
<div class="card"><strong>NAV</strong><br>{_esc(p.get("nav"))}</div>
<div class="card"><strong>Cash</strong><br>{_esc(p.get("cash_base"))}</div>
<div class="card"><strong>Day P&amp;L</strong><br>{_esc(p.get("day_pnl"))}</div>
<div class="card"><strong>Positions</strong><br>{_esc(p.get("n_positions"))}</div>
</div>

<h2>Holdings</h2>
{_table(["Ticker", "Shares", "Mark", "Value", "Weight", "P&L", "Ccy"], h_rows)}

<h2>Accounting (P&amp;L)</h2>
<div class="grid">
<div class="card"><strong>Unrealised</strong><br>{_esc((pnl or {}).get("unrealised"))}</div>
<div class="card"><strong>Realised</strong><br>{_esc((pnl or {}).get("realised"))}</div>
<div class="card"><strong>Income</strong><br>{_esc((pnl or {}).get("income"))}</div>
<div class="card"><strong>Fees</strong><br>{_esc((pnl or {}).get("fees"))}</div>
<div class="card"><strong>Total</strong><br>{_esc((pnl or {}).get("total"))}</div>
<div class="card"><strong>Net flows</strong><br>{_esc((pnl or {}).get("net_flows"))}</div>
</div>
<p class="meta">Deposits/withdrawals are capital, never investment return.</p>

<h2>Performance</h2>
<p>TWR: {_esc(perf.get("twr"))} · XIRR: {_esc(perf.get("xirr"))} ·
Window: {_esc(perf.get("window"))}</p>

<h2>Risk</h2>
<p>Volatility: {_esc(risk.get("volatility"))} · Sharpe: {_esc(risk.get("sharpe"))} ·
VaR (95%, 1d): {_esc(risk.get("var_95_1d"))} ·
Max drawdown: {_esc(risk.get("max_drawdown"))}</p>
<p class="meta">{_esc(risk.get("note") or "Historical estimates; live exposure combined without refitting on ticks.")}</p>

<h2>Attribution</h2>
{_table(["Ticker", "Contribution"], attr_rows,
        caption=str(attr.get("note") or "")) if attr_rows else "<p>—</p>"}

<h2>Stress</h2>
{_table(["Window", "Return", "Drawdown"], stress_rows,
        caption=str(stress.get("note") or "")) if stress_rows else "<p>—</p>"}

<h2>Saved Analyses</h2>
{_table(["Name", "Type", "Created", "Last Run", "Tags"], a_rows)
  if a_rows else "<p>No saved analyses referenced.</p>"}

<h2>Notes</h2>
<p>{_esc(p.get("notes"))}</p>

<h2>Sources &amp; Methodology</h2>
<ul>{src_list}</ul>
<p class="meta">Historical analytics use date-matched historical FX in base
currency; live quotes never enter historical estimation. TradingView is
display only. No broker orders are sent by this report. No investment
recommendation is made.</p>
</body>
</html>"""

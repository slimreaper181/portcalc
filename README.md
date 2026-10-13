# portcalc — Institutional Portfolio Workstation

A modular Python portfolio research workstation with a Streamlit web
interface: a persistent multi-portfolio SQLite workspace, streaming market
data with polling fallback, a corrected transaction ledger with audit
trail, institutional risk estimation, constrained optimisation, a
walk-forward robustness lab, a research-only execution simulator, a
persistent research library with re-runs and comparison, and printable
HTML institutional reports — plus live marks and NAV, FIFO lots and cash
accounting, TWR/XIRR performance, return attribution, and USD/GBP/EUR
multi-currency support.

## Features

- **Persistent workspace** — local SQLite store (`portcalc.db`,
  git-ignored) with versioned migrations; multiple portfolios (create,
  rename, switch, duplicate with explicit research opt-in, archive,
  guarded delete); session working copy with hash-guarded atomic saves;
  portable JSON import/export round-trips independently of the DB
- **Transaction accounting 2.0** — edit/delete with confirmation,
  deterministic rebuild from the ledger (lots/cash/realised/holdings never
  mutated directly), immutable revision audit trail (CREATE/UPDATE/DELETE
  with before/after payloads), running cash/position display, FIFO lot
  drilldown, notes, ticker/type/date/text filters, atomic CSV import with
  preview and duplicate warnings
- **Streaming market data** — provider abstraction, Alpaca WebSocket
  (trades + quotes) with SIP→IEX→REST→Yahoo fallback and cached
  capability, single connection manager, incremental subscription
  (holdings + watchlist + selected ticker), thread-safe quote store
  (newest timestamp wins), backoff reconnect/resubscribe, heartbeat
  (STREAMING/POLLING/FALLBACK/STALE/CLOSED), 30s UI fragment refresh that
  never recomputes historical analytics
- **Live market data** — quote model (bid/ask/last/previous close), explicit
  mark policy (mid → last → fallback, never invented), freshness
  (LIVE/DELAYED/STALE/CLOSED/FALLBACK), US session detection, feed fallback
  (SIP→IEX), Yahoo fallback; no background threads, no auto-rerun loops
- **Live portfolio** — NAV, Day P&L vs previous close, live weights, live
  risk snapshot (historical estimates labelled as such)
- **Transaction ledger** — BUY/SELL/DIVIDEND/FEE/DEPOSIT/WITHDRAWAL/INTEREST
  with validation, deterministic entry ordering and stable IDs
- **Multi-lot FIFO accounting** — open lots, realised P&L (native + base +
  FX effect), weighted-average cost display, cash-exempt classic entries
- **Cash accounts** — currency-keyed balances, negative-cash rejection,
  full movement audit
- **Performance from cash flows** — TWR (flows never count as return),
  XIRR (structured failure when unsolvable), flow-matched benchmark legs
- **Attribution** — beginning-weight contributions, winners/losers, exact
  local×FX identity, separate recorded income
- **Rebalancing planner** — optimiser/custom targets, cash-aware trades
  with buffer/target/thresholds, cost estimates, before/after comparison,
  OrderIntent objects; research-only, never executes
- **Watchlist** — marks without holdings, opens in Security Detail
- **Historical stress lab** — COVID/2022 presets + custom windows on real
  data (current-composition hypothetical, labelled)
- **Advanced risk** — component/marginal VaR (reconciled), diversification
  benefit, Herfindahl, skew/excess kurtosis, Omega, capture ratios,
  trailing rolling beta/correlation/tracking error
- **Portfolio overview** — positions, cost basis, P&L, weights, insights,
  quote-currency exposure, dividends, data quality
- **Risk analysis** — Parametric, Historical, and Monte Carlo VaR, CVaR,
  risk contributions, correlation
- **Portfolio optimisation** — Min variance, Max Sharpe, target return/volatility,
  efficient frontier, Equal Risk Contribution, Maximum Diversification,
  Black-Litterman (absolute/relative views), side-by-side comparison
- **Scenario & Monte Carlo simulation** — forward simulation with stress
  scenarios (crash, bull, high-vol, custom), contributions, comparison
- **Performance analytics** — benchmark comparison, drawdowns, rolling
  metrics, beta/alpha/tracking error, currency-aware benchmark
- **Security detail** — live quote header, position card, lots, performance,
  risk relationship, TradingView chart (display only), intraday display,
  Alpaca live US prices with fallback
- **Historical backtesting** — buy-and-hold, periodic rebalancing, exact
  transaction-cost accounting, three gated workflows (Historical, Static
  OOS, Walk-Forward)
- **Walk-forward optimisation** — rolling/expanding re-estimation with
  strict no-look-ahead (training dates < execution date), turnover and
  cost-drag diagnostics, static-vs-dynamic comparison
- **Factor analysis** — CAPM, FF3, FF5, FF5+Momentum on French data with
  attribution and rolling exposures
- **Purchase dates & cost basis** — split-adjusted estimates, GBp
  normalisation, manual execution-price override
- **Multi-currency** — USD/GBP/EUR base with date-matched historical FX
- **Risk estimation lab** — Sample (default, preserves history), EWMA
  (configurable decay, correct normalisation), Ledoit-Wolf shrinkage;
  estimation windows (1y/2y/3y/5y/expanding) on Base Currency returns;
  diagnostics (condition number, eigenvalues, effective rank, PSD safety,
  correlation stability, estimator comparison); training slices only,
  never any look-ahead
- **Constrained optimisation** — per-asset min/max table with feasibility
  explainer, turnover limit (one-way Σ|Δ|/2 vs current), cost-averse
  planning penalty (not backtest accounting), quote-currency caps, cash
  target, custom ERC risk budgets; before/after + stability diagnostics
- **Walk-forward robustness lab** — grid over lookbacks × frequencies ×
  risk models × costs reusing the unchanged no-look-ahead engine; result
  matrix, Sharpe heatmap, robustness scorecard (median/worst/best/
  dispersion, turnover, drawdown), cost sensitivity + break-even, result
  cache, progress + size guard, CSV export
- **Execution simulator** — MARKET/LIMIT research fills (BUY near ask,
  SELL near bid + slippage bps + flat/bps commission, conservative LIMIT
  rule, deterministic partial fills), blotter (NEW/PARTIAL/FILLED/
  CANCELLED/REJECTED), sandbox-only application, rebalance-plan
  integration, cost comparison; SIMULATION ONLY banner, no broker orders
  ever sent
- **Research workspace** — saved analyses (Scenario/Optimisation/Static
  Backtest/OOS/Walk-Forward/WF Robustness/Stress) with full config +
  result summary (never giant arrays), per-view “Save to Research”,
  library table (Name/Type/Portfolio/Created/Last Run/Tags), plain-text
  notes, re-run on current data (Saved vs Recomputed, explicit
  overwrite), compatible-type comparison with exact config differences,
  rename/delete, per-portfolio isolation
- **Institutional reporting** — self-contained printable HTML report
  (Portfolio Summary/Holdings/P&L/Performance/Risk/Attribution/Stress/
  Saved Analyses + metadata: portfolio, UTC time, base, benchmark,
  data-through, sources, freshness disclaimer); all user content escaped,
  no secrets embedded, print CSS for PDF-via-browser
- **Portfolio persistence** — schema v2 JSON (positions, ledger,
  watchlist, settings) with v1 import compatibility and deterministic
  aggregate→lot migration (`portfolio.json`), analysis snapshot export

## Live Data & Accounting Methodology

- **Mark policy:** fresh bid/ask midpoint → fresh last trade → aged legs
  (labelled) → previous close → unavailable. Nothing invented.
- **Freshness:** LIVE (≤120s in session), DELAYED, STALE (>30min),
  CLOSED (session shut, last-known shown), FALLBACK (Yahoo).
- **Cash:** currency-keyed; buys spend, sells/dividends add, fees/deposits/
  withdrawals move the matching balance; negative cash rejected by default.
- **FIFO:** sells consume oldest lots first; realised P&L = proceeds −
  allocated cost − fees (native and base, plus FX effect identity).
- **TWR:** sub-period linking around external flows (idle cash dilutes
  honestly); **XIRR:** guarded solver with structured failure.
- **Attribution:** contribution = beginning weight × simple return;
  foreign legs split exactly into local/FX via
  (1+R_local)(1+R_fx)−1 = R_base; recorded income stays separate from
  adjusted-price returns.
- **Planner:** first-order cost estimates on displayed notionals (the
  historical simulator uses the exact fixed-point solver); net buys plus
  costs draw deployable cash; shortfalls reported, never assumed.
- **Ledger vs analytics:** the ledger is canonical for accounting (NAV,
  lots, realised, cash, TWR/XIRR); aggregate positions remain the universe
  definition for historical analytics (backtest/OOS/WF/factors on Yahoo
  history). Live quotes never enter historical estimation.

## Project Structure

```
portcalc/
├── app.py                  # Streamlit shell (session, sidebar, navigation)
├── ui/                     # Presentation layer (no numerical logic)
│   ├── theme.py            # Design tokens + global stylesheet
│   ├── formatting.py       # Money/percent/date vocabulary
│   ├── charts.py           # Plotly design system
│   ├── tables.py           # Table presentation + comparison renderer
│   ├── feedback.py         # Rerun-surviving flash messages
│   ├── errors.py           # Section error boundaries
│   ├── state.py            # Session defaults + result cache
│   ├── insights.py         # Deterministic portfolio observations
│   ├── bl.py               # Black-Litterman UI inputs
│   ├── ledger.py           # Ledger session-state helpers
│   ├── workspace.py        # DB workspace ↔ session mapping
│   └── views/              # One render module per research tab
│       └── overview.py risk.py optimise.py scenario.py
│           performance.py security.py backtest.py factors.py
│           research.py accounting.py
├── data/
│   ├── __init__.py
│   ├── market_data.py      # yfinance wrappers (+intraday, prev closes)
│   ├── live_market.py      # Quotes, marks, store, ingestor, snapshot
│   ├── ws_stream.py        # Streaming provider + connection manager
│   ├── pipeline.py         # Cached loaders + composed data plumbing
│   ├── fx.py               # FX + instrument metadata
│   ├── alpaca.py           # Live US prices (valuation only)
│   └── factors.py          # French data library access
├── storage/                # SQLite workspace (no ORM, no secrets)
│   ├── database.py         # Connection factory + atomic() helper
│   ├── migrations.py       # Deterministic idempotent schema versions
│   ├── repositories.py     # Typed access (portfolios/ledger/research)
│   └── models.py           # Plain dataclasses
└── analytics/
    ├── __init__.py
    ├── returns.py          # Log returns, mean, covariance, correlation
    ├── portfolio.py        # Portfolio metrics, Sharpe, risk contributions, CML
    ├── var.py              # Parametric, historical, Monte Carlo VaR + CVaR
    ├── risk_extra.py       # Component/marginal VaR, skew, Omega, capture
    ├── estimation.py       # Sample / EWMA / Ledoit-Wolf + diagnostics
    ├── constraints.py      # Real-world optimisation constraint model
    ├── optimisation.py     # SLSQP optimisation, efficient frontier, rebalancing
    ├── scenario.py         # Monte Carlo forward simulation, stress scenarios
    ├── allocation.py       # ERC, MaxDiv, Black-Litterman, OOS windows
    ├── backtest.py         # Static backtesting engine (exact costs)
    ├── walkforward.py      # Walk-forward optimisation engine
    ├── robustness.py       # WF robustness grid + scorecard
    ├── execution.py        # Research execution simulator (no brokerage)
    ├── research.py         # Saved-analysis re-runners + comparison
    ├── report.py           # Printable HTML institutional report
    ├── currency.py         # FX maths (no look-ahead)
    ├── positions.py        # Cost basis, valuation, splits, dividends
    ├── performance.py      # Drawdowns, rolling metrics, benchmark stats
    ├── performance_cashflows.py  # TWR/XIRR, flow-matched benchmark
    ├── ledger.py           # Transactions, FIFO lots, cash, realised P&L
    ├── ledger_csv.py       # Atomic CSV import with preview
    ├── persistence.py      # Schema v2, migration, import validation
    ├── history.py          # Transaction-aware NAV reconstruction
    ├── attribution.py      # Contributions, winners/losers, FX split
    ├── rebalance_plan.py   # Research-only planner (no execution)
    ├── stress.py           # Historical stress windows
    └── factors.py          # Factor regressions + attribution
```

UI modules must not contain numerical logic; analytics modules stay
importable without Streamlit session state (the cached pipeline in
`data/pipeline.py` is the only Streamlit-aware data layer).

## Installation

```bash
# Clone the repo
git clone https://github.com/slimreaper181/portcalc.git
cd portcalc

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running

### Web (Streamlit)
```bash
streamlit run app.py
```

### Programmatic / Desktop use

All analytics modules are importable independently of Streamlit:

```python
from data.market_data import fetch_price_history, fetch_risk_free_rate
from analytics.returns import daily_returns, annualised_mean_returns, annualised_cov_matrix
from analytics.portfolio import portfolio_expected_return, portfolio_std, sharpe_ratio
from analytics.var import parametric_var, monte_carlo_var
from analytics.optimisation import max_sharpe, efficient_frontier
from analytics.scenario import simulate_portfolio_paths, run_predefined_scenario
```

## Scenarios

| Scenario | Return Shock | Vol Multiplier |
|---|---|---|
| Normal | 0% | 1.0x |
| Market Crash | -20% | 1.8x |
| Bull Market | +40% | 0.85x |
| High Volatility | 0% | 2.0x |
| Custom | User-defined | User-defined |

## Alpaca live prices (optional)

Current US equity prices can come from the Alpaca Market Data API
(latest trade, IEX feed). Historical analytics always stay on Yahoo
Finance; TradingView remains charting-only.

PowerShell (applies to that session only, unless persisted):

```powershell
$env:APCA_API_KEY_ID="..."
$env:APCA_API_SECRET_KEY="..."

python -m streamlit run app.py
```

To persist for future sessions instead, set them as user environment
variables (e.g. via System Properties → Environment Variables).

macOS/Linux:

```bash
export APCA_API_KEY_ID="..."
export APCA_API_SECRET_KEY="..."
```

Alternatively, Streamlit secrets are supported as a fallback
(`.streamlit/secrets.toml`, never committed):

```toml
APCA_API_KEY_ID = "..."
APCA_API_SECRET_KEY = "..."
```

Without credentials the app runs normally on Yahoo Finance prices and
labels them as such. Non-US symbols (e.g. `BARC.L`, `^FTSE`) always use
Yahoo Finance.

## Purchase dates & cost basis

Each position records **Ticker / Shares / Purchase Date**. The app looks
up the raw market close on that date (split-adjusted to current shares)
as the **Estimated Purchase Price** — your actual broker fill may have
differed. Non-trading dates offer the previous/next close explicitly
(never silently substituted). An **actual execution price override** is
available per position; old `avg_cost` entries are preserved as
legacy/manual cost basis. One aggregate purchase per position is
supported (no tax lots yet).

## Base currency & multi-currency

Pick a portfolio **Base Currency** (USD/GBP/EUR) in the sidebar. Pipeline:

```text
Security native price
  ↓ × quote scale (÷100 for GBp pence)
Current/historical FX translation (date-matched, Yahoo Finance)
  ↓
Portfolio base-currency value
```

```text
Purchase-date native price
  ↓ × purchase-date FX
Historical base-currency cost basis
```

Current values use current FX; historical analytics/backtests use
date-matched historical FX, so currency movement is part of
base-currency returns. LSE pence quotes (e.g. `BARC.L` 320 GBp = £3.20)
are normalised — no 100× errors. Unrealised P&L is price/FX P&L;
dividends are reported separately as estimated income, never merged
into backtest wealth paths (which use adjusted prices).

## Methodology & limitations

- Market-data delays: Yahoo daily closes; Alpaca IEX latest trade
  (10s cache) where configured.
- Estimated purchase close: raw close on the selected trading date,
  split-adjusted to current shares; execution-time/fills excluded.
- Single acquisition date per position; FIFO/LIFO not modelled.
- Dividend estimates: cash amounts × shares × payment-date FX, only for
  payments after purchase; broker-exact entitlement not claimed.
- Currency conversion via Yahoo FX (`GBPUSD=X`, `EURUSD=X`), USD-routed;
  FX gaps beyond 5 trading days raise instead of inventing data.
- Factor data lags the portfolio (French library updates periodically);
  US factors stay in USD — non-USD bases carry an explicit warning.
- Backtests use currently-held securities only (survivorship/selection
  bias); rebalances execute at the next eligible close (approximation);
  static out-of-sample weights are never re-estimated in-test.
- TradingView charts show native market quotation, display only.
- Data model notes for future work: positions are single-lot aggregates
  (`purchase_date` may be null, prices are per current split-adjusted
  share); cash balances, multi-lot tax lots and realised-P&L ledgers are
  not modelled yet, so those extensions need no redesign of the analytics
  layer, only new position kinds and ledger views.

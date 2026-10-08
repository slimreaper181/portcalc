# portcalc — Portfolio Analytics & Risk Dashboard

A modular Python portfolio analytics system with a Streamlit web interface.

## Features

- **Live market data** via yfinance (prices, risk-free rate)
- **Portfolio overview** — positions, P&L, weights, risk contributions
- **Risk analysis** — Parametric, Historical, and Monte Carlo VaR
- **Portfolio optimisation** — Min variance, Max Sharpe, target return/volatility, efficient frontier
- **Scenario & Monte Carlo simulation** — forward simulation with stress scenarios (crash, bull, high-vol, custom)
- **Portfolio persistence** — positions saved to `portfolio.json` between sessions

## Project Structure

```
portcalc/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── portfolio.json          # Saved positions (auto-created)
├── data/
│   ├── __init__.py
│   └── market_data.py      # yfinance wrappers
└── analytics/
    ├── __init__.py
    ├── returns.py           # Log returns, mean, covariance, correlation
    ├── portfolio.py         # Portfolio metrics, Sharpe, risk contributions, CML
    ├── var.py               # Parametric, historical, Monte Carlo VaR
    ├── optimisation.py      # SLSQP optimisation, efficient frontier, rebalancing
    └── scenario.py          # Monte Carlo forward simulation, stress scenarios
```

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

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

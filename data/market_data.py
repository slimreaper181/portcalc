"""
market_data.py
--------------
Fetches live and historical stock price data via yfinance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def fetch_price_history(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.

    Args:
        tickers: List of stock ticker symbols (e.g. ['AAPL', 'MSFT'])
        period:  yfinance period string ('1y', '2y', '6mo', etc.)

    Returns:
        DataFrame of daily adjusted close prices, columns = tickers.
    """
    if not tickers:
        return pd.DataFrame()

    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    prices.dropna(how="all", inplace=True)
    return prices


def fetch_current_prices(tickers: list[str]) -> dict[str, float]:
    """
    Fetch the most recent closing price for each ticker.

    Returns:
        Dict mapping ticker -> latest price (USD).
    """
    prices = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            prices[t] = float(info.last_price)
        except Exception:
            # Fall back to last close from recent history
            hist = yf.Ticker(t).history(period="5d")
            prices[t] = float(hist["Close"].iloc[-1]) if not hist.empty else np.nan
    return prices


def fetch_risk_free_rate() -> float:
    """
    Approximate the annualised risk-free rate using the 13-week US T-Bill (^IRX).
    Returns a decimal (e.g. 0.053 for 5.3%).
    """
    try:
        tbill = yf.Ticker("^IRX").history(period="5d")
        rate = float(tbill["Close"].iloc[-1]) / 100
    except Exception:
        rate = 0.05  # sensible default
    return rate

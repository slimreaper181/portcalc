"""Tests for data.fx (instruments, FX pairs, raw history). Network mocked."""

import numpy as np
import pandas as pd
import pytest

import yfinance as yf

from data import fx as fx_mod
from data.fx import (
    EXCHANGE_TIMEZONES,
    FX_PAIRS,
    SUPPORTED_BASE_CURRENCIES,
    Instrument,
    fetch_current_fx,
    fetch_dividends,
    fetch_fx_history,
    fetch_raw_closes,
    fetch_splits,
    get_instrument,
)


class _FakeFastInfo:
    def __init__(self, currency="USD", exchange="NMS"):
        self.currency = currency
        self.exchange = exchange


class _FakeTicker:
    """Stand-in for yfinance.Ticker with scripted responses."""

    def __init__(self, symbol, currency="USD", exchange="NMS",
                 closes=None, splits=None, dividends=None, fail=False):
        self._symbol = symbol
        self._currency = currency
        self._exchange = exchange
        self._closes = closes
        self._splits = splits
        self._dividends = dividends
        self._fail = fail

    @property
    def fast_info(self):
        if self._fail:
            raise RuntimeError("boom")
        info = _FakeFastInfo(self._currency, self._exchange)
        return info

    def history(self, period="max", auto_adjust=True):
        if self._fail:
            raise RuntimeError("boom")
        return self._closes

    @property
    def splits(self):
        if self._fail:
            raise RuntimeError("boom")
        return self._splits if self._splits is not None else pd.Series(
            dtype=float)

    @property
    def dividends(self):
        if self._fail:
            raise RuntimeError("boom")
        return self._dividends if self._dividends is not None else pd.Series(
            dtype=float)


def _patch_ticker(monkeypatch, mapping):
    """Map symbol -> _FakeTicker kwargs; anything else raises."""
    def factory(symbol):
        if symbol not in mapping:
            raise RuntimeError(f"unexpected symbol {symbol}")
        return _FakeTicker(symbol, **mapping[symbol])
    monkeypatch.setattr(yf, "Ticker", factory)


def _bday_closes(values, start="2024-01-02"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.DataFrame(
        {"Close": values},
        index=idx.tz_localize("America/New_York"))


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------

def test_instrument_usd(monkeypatch):
    _patch_ticker(monkeypatch, {"AAPL": {}})
    inst = get_instrument("aapl")
    assert isinstance(inst, Instrument)
    assert inst.ticker == "AAPL"
    assert inst.native_currency == "USD"
    assert inst.quote_unit == "USD" and inst.quote_scale == 1.0
    assert inst.timezone == "America/New_York"
    assert inst.yahoo_symbol == "AAPL"


def test_instrument_gbp_pence():
    # BARC.L quotes in pence: native GBP, 0.01 scale. No network needed
    # beyond the unit mapping — exercise via _normalise path with a stub.
    from data.fx import _normalise_currency
    assert _normalise_currency("GBp") == ("GBP", "GBp", 0.01)
    assert _normalise_currency("GBX") == ("GBP", "GBp", 0.01)
    assert _normalise_currency("USD") == ("USD", "USD", 1.0)
    assert _normalise_currency("EUR") == ("EUR", "EUR", 1.0)
    with pytest.raises(ValueError):
        _normalise_currency(None)
    with pytest.raises(ValueError):
        _normalise_currency("???")
    assert "LSE" in EXCHANGE_TIMEZONES


def test_instrument_pence_end_to_end(monkeypatch):
    _patch_ticker(monkeypatch, {"BARC.L": {"currency": "GBp",
                                           "exchange": "LSE"}})
    inst = get_instrument("BARC.L")
    assert inst.native_currency == "GBP"
    assert inst.quote_unit == "GBp"
    assert inst.quote_scale == pytest.approx(0.01)
    assert inst.timezone == "Europe/London"
    assert inst.alpaca_supported is False


def test_instrument_index_forces_scale_one(monkeypatch):
    _patch_ticker(monkeypatch, {"^FTSE": {"currency": "GBp",
                                          "exchange": "FTSE"}})
    inst = get_instrument("^FTSE")
    assert inst.quote_scale == 1.0
    assert inst.native_currency == "GBP"


def test_instrument_missing_currency_refuses_usd_assumption(monkeypatch):
    _patch_ticker(monkeypatch, {"XYZ": {"currency": None}})
    with pytest.raises(ValueError, match="refusing"):
        get_instrument("XYZ")
    _patch_ticker(monkeypatch, {"XYZ": {}})
    # monkeypatch a failing Ticker via fail flag
    def boom(symbol):
        return _FakeTicker(symbol, fail=True)
    monkeypatch.setattr(yf, "Ticker", boom)
    with pytest.raises(ValueError, match="[Nn]etwork|metadata"):
        get_instrument("XYZ")


# ---------------------------------------------------------------------------
# FX history
# ---------------------------------------------------------------------------

def _fx_frame():
    idx = pd.bdate_range("2024-01-02", periods=6)
    cols = pd.MultiIndex.from_product([["Close"], ["GBPUSD=X", "EURUSD=X"]])
    vals = np.array([[1.27, 1.08], [1.28, 1.09], [1.26, 1.07],
                     [1.29, 1.10], [1.30, 1.09], [1.31, 1.11]])
    return pd.DataFrame(vals, index=idx, columns=cols)


def test_fetch_fx_history(monkeypatch):
    monkeypatch.setattr(yf, "download", lambda *a, **k: _fx_frame())
    out = fetch_fx_history()
    assert list(out.columns) == ["GBP", "EUR"]
    assert out["GBP"].iloc[0] == pytest.approx(1.27)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is None
    # Unknown pair rejected, not guessed.
    with pytest.raises(ValueError):
        fetch_fx_history(["JPYUSD=X"])


def test_fetch_fx_history_single_pair_and_period(monkeypatch):
    seen = {}

    def fake_dl(tickers, period="max", auto_adjust=True, progress=False):
        seen["tickers"] = tickers
        seen["period"] = period
        assert tickers == ["GBPUSD=X"]
        idx = pd.bdate_range("2024-01-02", periods=3)
        return pd.DataFrame({"Close": [1.27, 1.28, 1.29]}, index=idx)

    monkeypatch.setattr(yf, "download", fake_dl)
    out = fetch_fx_history(["GBPUSD=X"], period="5d")
    assert list(out.columns) == ["GBP"]
    assert seen["period"] == "5d"
    assert list(out["GBP"]) == pytest.approx([1.27, 1.28, 1.29])


def test_fetch_fx_history_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr(yf, "download", boom)
    with pytest.raises(ValueError, match="[Nn]etwork"):
        fetch_fx_history()


def test_fetch_current_fx(monkeypatch):
    monkeypatch.setattr(yf, "download", lambda *a, **k: _fx_frame())
    rate, ts = fetch_current_fx("GBPUSD=X")
    assert rate == pytest.approx(1.31)
    assert ts is not None and ts.tzinfo is not None
    with pytest.raises(ValueError):
        fetch_current_fx("JPYUSD=X")


# ---------------------------------------------------------------------------
# Raw closes / splits / dividends
# ---------------------------------------------------------------------------

def test_fetch_raw_closes_and_splits(monkeypatch):
    closes = _bday_closes([100.0, 101.0, 25.5, 26.0])
    splits = pd.Series(
        [4.0], index=pd.DatetimeIndex(["2024-01-04"]).tz_localize(
            "America/New_York"))
    divs = pd.Series(
        [0.5], index=pd.DatetimeIndex(["2024-01-03"]).tz_localize(
            "America/New_York"))

    class T(_FakeTicker):
        @property
        def splits(self):
            return splits

        @property
        def dividends(self):
            return divs

        def history(self, period="max", auto_adjust=True):
            return closes

    monkeypatch.setattr(yf, "Ticker", lambda s: T(s))
    px = fetch_raw_closes("AAPL")
    assert list(px) == [100.0, 101.0, 25.5, 26.0]
    assert px.index.tz is None
    sp = fetch_splits("AAPL")
    assert list(sp) == [4.0]
    dv = fetch_dividends("AAPL")
    assert list(dv) == [0.5]


def test_fetch_empty_history(monkeypatch):
    class T(_FakeTicker):
        def history(self, period="max", auto_adjust=True):
            return pd.DataFrame()
    monkeypatch.setattr(yf, "Ticker", lambda s: T(s))
    with pytest.raises(ValueError):
        fetch_raw_closes("AAPL")


def test_fetch_splits_dividends_empty(monkeypatch):
    _patch_ticker(monkeypatch, {"AAPL": {}})
    # _FakeTicker defaults splits/dividends to empty series
    assert len(fetch_splits("AAPL")) == 0
    assert len(fetch_dividends("AAPL")) == 0


def test_supported_base_currencies():
    assert set(SUPPORTED_BASE_CURRENCIES) == {"USD", "GBP", "EUR"}
    assert set(FX_PAIRS) == {"GBPUSD=X", "EURUSD=X"}

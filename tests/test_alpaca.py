"""Tests for data.alpaca (live prices). All HTTP is mocked — no real API calls."""

from datetime import datetime, timezone

import pytest
import requests

from data.alpaca import (
    AlpacaConfigError,
    AlpacaError,
    Credentials,
    LatestTrade,
    build_price_view,
    fetch_latest_trade,
    fetch_latest_trades,
    get_alpaca_credentials,
    is_alpaca_supported_symbol,
    is_fresh,
)

CREDS = Credentials(key_id="KEY", secret="SECRET")
TRADE_TS = "2026-10-06T14:30:00.123456789Z"


class FakeResponse:
    def __init__(self, payload=None, status=200, bad_json=False):
        self._payload = payload
        self.status_code = status
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("No JSON")
        return self._payload


def _trades_payload():
    return {
        "trades": {
            "AAPL": {"p": 250.10, "t": TRADE_TS, "x": "V"},
            "MSFT": {"p": 529.87, "t": TRADE_TS, "x": "Q"},
        }
    }


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def test_credentials_env_priority(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "ENVKEY")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "ENVSECRET")
    creds = get_alpaca_credentials()
    assert (creds.key_id, creds.secret) == ("ENVKEY", "ENVSECRET")
    # Secrets never appear in repr/logs.
    assert "ENVKEY" not in repr(creds) and "ENVSECRET" not in repr(creds)


def test_credentials_missing(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(AlpacaConfigError):
        get_alpaca_credentials()


# ---------------------------------------------------------------------------
# Symbol support
# ---------------------------------------------------------------------------

def test_supported_symbols():
    for sym in ("AAPL", "MSFT", "NVDA", "JPM", "SPY"):
        assert is_alpaca_supported_symbol(sym), sym
    for sym in ("BARC.L", "BRK.B", "^FTSE", "^GSPC", "BTC-USD", "",
                "AAPL;X", "A B"):
        assert not is_alpaca_supported_symbol(sym), sym


# ---------------------------------------------------------------------------
# Batch fetch (mocked)
# ---------------------------------------------------------------------------

def test_batch_parsing(monkeypatch):
    seen = {}

    def fake_get(url, headers=None, params=None, timeout=None):
        seen.update(params or {})
        assert url.endswith("/v2/stocks/trades/latest")
        assert headers["APCA-API-KEY-ID"] == "KEY"
        assert headers["APCA-API-SECRET-KEY"] == "SECRET"  # transmitted…
        return FakeResponse(_trades_payload())
        # …but never logged: covered by the redacted Credentials repr test.

    monkeypatch.setattr(requests, "get", fake_get)
    out = fetch_latest_trades(["AAPL", "MSFT"], credentials=CREDS)
    assert set(out) == {"AAPL", "MSFT"}
    assert out["MSFT"].price == pytest.approx(529.87)
    assert out["AAPL"].exchange == "V"
    assert out["AAPL"].feed == "iex"
    assert isinstance(out["AAPL"].timestamp, datetime)
    assert seen["feed"] == "iex"


def test_batch_skips_unsupported_without_http(monkeypatch):
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(params["symbols"])
        return FakeResponse(_trades_payload())

    monkeypatch.setattr(requests, "get", fake_get)
    out = fetch_latest_trades(["AAPL", "BARC.L", "^FTSE"], credentials=CREDS)
    assert calls == ["AAPL"]  # non-US never requested
    assert set(out) == {"AAPL"}
    # Nothing supportable → no HTTP at all.
    assert fetch_latest_trades(["BARC.L"], credentials=CREDS) == {}
    assert len(calls) == 1


def test_batch_missing_and_malformed_trades(monkeypatch):
    payload = {"trades": {"AAPL": {"p": 250.0, "t": TRADE_TS, "x": "V"},
                          "MSFT": {"x": "Q"},          # no price
                          "NVDA": {"p": -5.0, "t": TRADE_TS}}}  # invalid
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: FakeResponse(payload))
    out = fetch_latest_trades(["AAPL", "MSFT", "NVDA"], credentials=CREDS)
    assert set(out) == {"AAPL"}  # bad entries fall back per symbol

    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: FakeResponse({"oops": 1}))
    with pytest.raises(AlpacaError):
        fetch_latest_trades(["AAPL"], credentials=CREDS)

    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: FakeResponse(bad_json=True))
    with pytest.raises(AlpacaError):
        fetch_latest_trades(["AAPL"], credentials=CREDS)


def test_single_trade_shapes(monkeypatch):
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **k: FakeResponse(
            {"trade": {"p": 100.5, "t": TRADE_TS, "x": "N"}}))
    t = fetch_latest_trade("JPM", credentials=CREDS)
    assert isinstance(t, LatestTrade) and t.price == pytest.approx(100.5)
    assert fetch_latest_trade("BARC.L", credentials=CREDS) is None


def test_auth_rate_limit_timeout_failures(monkeypatch):
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: FakeResponse({}, status=401))
    with pytest.raises(AlpacaError) as e:
        fetch_latest_trades(["AAPL"], credentials=CREDS)
    assert e.value.status == 401

    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: FakeResponse({}, status=403))
    with pytest.raises(AlpacaError) as e:
        fetch_latest_trades(["AAPL"], credentials=CREDS)
    assert e.value.status == 403

    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: FakeResponse({}, status=429))
    with pytest.raises(AlpacaError) as e:
        fetch_latest_trades(["AAPL"], credentials=CREDS)
    assert e.value.status == 429

    def _timeout(*a, **k):
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "get", _timeout)
    with pytest.raises(AlpacaError) as e:
        fetch_latest_trades(["AAPL"], credentials=CREDS)
    assert e.value.status is None


# ---------------------------------------------------------------------------
# Injection safety
# ---------------------------------------------------------------------------

def test_malicious_tickers_never_reach_http(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, "get", lambda *a, **k: calls.append(a) or FakeResponse({}))
    for evil in ('AAPL"; DROP TABLE', "AAPL&feed=sip", "../stocks",
                 "AAPL\nX-APCA: 1", "MSFT<script>"):
        with pytest.raises(ValueError):
            fetch_latest_trades([evil], credentials=CREDS)
        with pytest.raises(ValueError):
            fetch_latest_trade(evil, credentials=CREDS)
    assert calls == []


# ---------------------------------------------------------------------------
# Price view / valuation
# ---------------------------------------------------------------------------

def test_price_view_sources_and_fallback():
    now = datetime.now(timezone.utc)
    trades = {"MSFT": LatestTrade("MSFT", 529.87, now, "Q", "iex")}
    view = build_price_view(["MSFT", "BARC.L"],
                            {"MSFT": 500.0, "BARC.L": 300.0}, trades)
    assert view["MSFT"].source == "Alpaca IEX" and view["MSFT"].live is True
    assert view["BARC.L"].source == "Yahoo" and view["BARC.L"].live is False
    assert view["BARC.L"].price == pytest.approx(300.0)
    # No Alpaca at all → everything Yahoo, nothing claims to be live.
    view2 = build_price_view(["MSFT"], {"MSFT": 500.0}, {})
    assert view2["MSFT"].source == "Yahoo" and view2["MSFT"].live is False


def test_freshness_labelling():
    now = datetime.now(timezone.utc)
    assert is_fresh(now) is True
    assert is_fresh(None) is False
    old = datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert is_fresh(old) is False


def test_market_value_and_pnl_with_live_price():
    view = build_price_view(
        ["MSFT"],
        {"MSFT": 500.0},
        {"MSFT": LatestTrade("MSFT", 529.87,
                             datetime.now(timezone.utc), "Q", "iex")},
    )
    shares, cost = 5.0, 300.0
    mv = shares * view["MSFT"].price
    pnl = (view["MSFT"].price - cost) * shares
    assert mv == pytest.approx(5 * 529.87)
    assert pnl == pytest.approx((529.87 - 300.0) * 5)


def test_inputs_not_mutated(monkeypatch):
    symbols = ["MSFT", "AAPL"]
    before = list(symbols)
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: FakeResponse(_trades_payload()))
    fetch_latest_trades(symbols, credentials=CREDS)
    assert symbols == before

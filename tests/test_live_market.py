"""Tests for data.live_market (quotes, marks, store, ingestor — all mocked).

No test touches the network or requires markets to be open: providers are
injected fakes and timestamps are explicit.
"""

from datetime import datetime, timedelta, timezone

import pytest

from data import live_market as lm
from data.live_market import (
    LiveStore,
    Quote,
    StreamIngestor,
    build_mark_snapshot,
    market_session_us,
    quote_status,
    select_mark,
    yahoo_fallback_quote,
)

NOW = datetime(2026, 10, 6, 15, 30, tzinfo=timezone.utc)  # Tue 11:30 ET


def _q(**kw):
    base = dict(symbol="AAPL", source="Alpaca", feed="iex",
                received_at=NOW)
    base.update(kw)
    return Quote(**base)


# ---------------------------------------------------------------------------
# Quote model
# ---------------------------------------------------------------------------

def test_mid_spread_and_day_change():
    q = _q(bid=100.0, ask=101.0, previous_close=99.0,
           last_trade=100.6, last_trade_timestamp=NOW)
    assert q.mid == pytest.approx(100.5)
    assert q.spread == pytest.approx(1.0)
    assert q.spread_bps == pytest.approx(1.0 / 100.5 * 10000.0)
    assert q.day_change == pytest.approx(1.6)
    assert q.day_change_pct == pytest.approx(1.6 / 99.0)
    assert q.age_seconds(NOW + timedelta(seconds=30)) == pytest.approx(30.0)


def test_partial_fields_never_fabricated():
    q = _q()
    assert q.mid is None and q.spread is None and q.day_change is None
    assert q.day_change_pct is None
    q2 = _q(bid=100.0)  # one-sided book: no mid
    assert q2.mid is None


def test_market_session_us_weekday_and_weekend():
    assert market_session_us(NOW) == "regular"
    sat = datetime(2026, 10, 10, 15, 30, tzinfo=timezone.utc)
    assert market_session_us(sat) == "closed"
    pre = datetime(2026, 10, 6, 11, 30, tzinfo=timezone.utc)  # 07:30 ET
    assert market_session_us(pre) == "pre"
    after = datetime(2026, 10, 6, 22, 30, tzinfo=timezone.utc)  # 18:30 ET
    assert market_session_us(after) == "after"


# ---------------------------------------------------------------------------
# Freshness + mark policy
# ---------------------------------------------------------------------------

def test_status_ladder():
    live = _q(bid=100.0, ask=101.0, quote_timestamp=NOW)
    assert quote_status(live, NOW) == "LIVE"
    aged = _q(bid=100.0, ask=101.0,
              quote_timestamp=NOW - timedelta(minutes=10))
    assert quote_status(aged, NOW) == "DELAYED"
    stale = _q(bid=100.0, ask=101.0,
               quote_timestamp=NOW - timedelta(hours=2))
    assert quote_status(stale, NOW) == "STALE"
    sat = datetime(2026, 10, 10, 15, 30, tzinfo=timezone.utc)
    assert quote_status(live, sat) == "CLOSED"
    assert quote_status(yahoo_fallback_quote("AAPL", 100.0), NOW) == "FALLBACK"
    assert quote_status(None, NOW) == "STALE"


def test_mark_policy_priority():
    both = _q(bid=100.0, ask=102.0, last_trade=100.0,
              quote_timestamp=NOW, last_trade_timestamp=NOW)
    mark, method, _ = select_mark(both, NOW)
    assert mark == pytest.approx(101.0) and "Mid" in method
    last_only = _q(last_trade=99.5, last_trade_timestamp=NOW)
    mark, method, _ = select_mark(last_only, NOW)
    assert mark == pytest.approx(99.5) and "Last" in method
    aged = _q(bid=100.0, ask=102.0,
              quote_timestamp=NOW - timedelta(hours=3))
    mark, method, _ = select_mark(aged, NOW)
    assert mark == pytest.approx(101.0) and "aged" in method
    prev = _q(previous_close=98.0)
    mark, method, _ = select_mark(prev, NOW)
    assert mark == pytest.approx(98.0) and "Previous" in method
    mark, method, _ = select_mark(None, NOW)
    assert mark is None and method == "Unavailable"


def test_yahoo_fallback_never_live():
    q = yahoo_fallback_quote("AAPL", 250.0, previous_close=248.0)
    assert q is not None and q.bid is None and q.ask is None
    assert quote_status(q, NOW) == "FALLBACK"
    mark, method, _ = select_mark(q, NOW)
    assert mark == pytest.approx(250.0)
    assert yahoo_fallback_quote("!!!bad!!!", 1.0) is None
    assert yahoo_fallback_quote("AAPL", float("nan")) is None


# ---------------------------------------------------------------------------
# LiveStore sequencing + subscriptions
# ---------------------------------------------------------------------------

def test_store_newest_wins_out_of_order_dropped():
    store = LiveStore()
    old = _q(bid=100.0, ask=101.0, quote_timestamp=NOW - timedelta(minutes=5))
    new = _q(bid=100.5, ask=101.5, quote_timestamp=NOW)
    assert store.update(old) is True
    assert store.update(new) is True
    assert store.update(old) is False  # older must not regress state
    assert store.get("AAPL").bid == pytest.approx(100.5)
    assert store.get("NOPEXYZ") is None
    assert store.get("!!!") is None
    assert store.symbols() == ["AAPL"]
    store.remove("AAPL")
    assert store.get("AAPL") is None


def test_store_snapshot_consistent():
    store = LiveStore()
    store.update(_q(symbol="AAPL", bid=1.0, ask=2.0, quote_timestamp=NOW))
    store.update(_q(symbol="MSFT", bid=3.0, ask=4.0, quote_timestamp=NOW))
    snap = store.snapshot(["MSFT", "AAPL", "!!!"])
    assert set(snap) == {"MSFT", "AAPL", "!!!"}
    assert snap["MSFT"].bid == pytest.approx(3.0)
    assert snap["!!!"] is None


# ---------------------------------------------------------------------------
# StreamIngestor (deterministic message sequences)
# ---------------------------------------------------------------------------

def test_ingest_sequence_and_stats():
    store = LiveStore()
    ing = StreamIngestor(store)
    assert ing.subscribe(["AAPL", "MSFT"]) == ["AAPL", "MSFT"]
    assert ing.unsubscribe(["MSFT"]) == ["AAPL"]
    assert ing.connect() is False  # no transport configured
    assert ing.stats.last_error is not None
    assert ing.ingest({"symbol": "AAPL", "bid": 100.0, "ask": 101.0,
                       "quote_ts": NOW.isoformat(), "source": "stream",
                       "feed": "iex"}, received_at=NOW) is True
    # out-of-order (older quote ts) must not regress
    assert ing.ingest({"symbol": "AAPL", "bid": 1.0, "ask": 2.0,
                       "quote_ts": (NOW - timedelta(minutes=10)).isoformat(),
                       "source": "stream"}, received_at=NOW) is False
    assert store.get("AAPL").bid == pytest.approx(100.0)
    assert ing.stats.messages == 2 and ing.stats.applied == 1
    assert ing.stats.dropped_stale == 1
    assert ing.ingest({"symbol": "!!!bad!!!"}) is False
    assert ing.ingest("not-a-dict") is False


def test_ingestor_backoff_and_transport():
    ing = StreamIngestor()
    d1 = ing.note_failure("drop")
    d2 = ing.note_failure("drop again")
    assert d2 > d1 and d2 <= 60.0
    assert ing.stats.reconnects == 2
    seen = {}

    def _factory(subs):
        seen["subs"] = subs
        return object()

    ing2 = StreamIngestor(transport_factory=_factory)
    ing2.subscribe(["AAPL"])
    assert ing2.connect() is True
    assert seen["subs"] == ["AAPL"]
    ing2.disconnect()
    assert ing2.connected is False


def test_feed_fallback_on_403_then_success(monkeypatch):
    from data import alpaca as _alp

    calls = []

    def _fake_fetch(symbols, credentials=None, feed="iex", timeout=10):
        calls.append(feed)
        if feed == "sip":
            raise _alp.AlpacaError("entitlement", status=403)
        return {"AAPL": Quote(symbol="AAPL", bid=1.0, ask=2.0,
                              source="Alpaca", feed=feed)}

    monkeypatch.setattr(lm, "fetch_alpaca_quotes", _fake_fetch)
    quotes, used = lm.fetch_with_feed_fallback(["AAPL"], feeds=("sip", "iex"))
    assert used == "iex" and calls == ["sip", "iex"]
    assert quotes["AAPL"].bid == pytest.approx(1.0)


def test_feed_fallback_propagates_non_403(monkeypatch):
    from data import alpaca as _alp

    def _boom(symbols, credentials=None, feed="iex", timeout=10):
        raise _alp.AlpacaError("auth", status=401)

    monkeypatch.setattr(lm, "fetch_alpaca_quotes", _boom)
    with pytest.raises(_alp.AlpacaError):
        lm.fetch_with_feed_fallback(["AAPL"], feeds=("iex",))


def test_build_mark_snapshot_mixed_sources():
    quotes = {
        "AAPL": _q(bid=100.0, ask=102.0, quote_timestamp=NOW),
        "MSFT": yahoo_fallback_quote("MSFT", 50.0, previous_close=49.0),
        "BARC.L": None,
    }
    snap = build_mark_snapshot(quotes, NOW)
    assert snap["AAPL"].mark == pytest.approx(101.0)
    assert snap["AAPL"].status == "LIVE"
    assert snap["MSFT"].status == "FALLBACK"
    assert snap["BARC.L"].mark is None and snap["BARC.L"].status == "STALE"
    assert snap["MSFT"].day_change_pct == pytest.approx(1.0 / 49.0)

"""
live_market.py
--------------
Live/current market-data subsystem (quotes, marks, freshness, sessions).

Architecture (deliberately poll-based, no background threads by default):

* Providers implement small fetch functions (Alpaca REST trades/quotes,
  Yahoo fallback). A ``StreamIngestor`` consumes quote *messages* from an
  injected transport, so streaming logic (sequencing, subscriptions,
  reconnect/backoff state) is fully unit-testable without markets open.
  No transport is started automatically: with no credentials or no
  transport, the subsystem degrades to controlled REST/fallback polling.
  Exactly one ``LiveStore`` per process holds current quotes; updates keep
  only the newest observation per symbol (out-of-order messages cannot
  corrupt state). Locks make it thread-/session-safe.

Isolation contract (frozen):
* Live quotes feed CURRENT valuation/display only (marks, NAV, Day P&L,
  live risk snapshot). They NEVER enter historical returns, covariance,
  optimiser/OOS/WF estimation, or factor regressions.

All timestamps are tz-aware UTC internally; ages are computed against
explicit ``now`` parameters (injectable in tests — never hidden clocks).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from analytics.validation import validate_ticker_symbol

# Freshness thresholds (documented policy, not guesses).
FRESH_TRADE_SECONDS = 120      # last trade this fresh in-session → LIVE
STALE_AFTER_SECONDS = 30 * 60  # older than this → STALE (intraday context)
FEED_PRIORITY = ("sip", "iex")  # tried in order; 403 → next (no SIP assumption)


# ---------------------------------------------------------------------------
# Quote model
# ---------------------------------------------------------------------------

@dataclass
class Quote:
    """One current quote snapshot for a symbol (partial fields allowed)."""

    symbol: str
    last_trade: float | None = None
    last_trade_size: int | None = None
    last_trade_timestamp: datetime | None = None
    bid: float | None = None
    bid_size: int | None = None
    ask: float | None = None
    ask_size: int | None = None
    quote_timestamp: datetime | None = None
    previous_close: float | None = None
    source: str = "unknown"   # e.g. "Alpaca", "Yahoo"
    feed: str | None = None   # e.g. "iex", "sip", None for fallback
    received_at: datetime | None = None

    # -- derived ---------------------------------------------------------
    @property
    def mid(self) -> float | None:
        """Midpoint when both sides are positive and finite."""
        if (self.bid is not None and self.ask is not None
                and math.isfinite(float(self.bid))
                and math.isfinite(float(self.ask))
                and float(self.bid) > 0 and float(self.ask) > 0):
            return (float(self.bid) + float(self.ask)) / 2.0
        return None

    @property
    def spread(self) -> float | None:
        """Absolute spread when both sides are usable."""
        if self.mid is None:
            return None
        return float(self.ask) - float(self.bid)

    @property
    def spread_bps(self) -> float | None:
        """Spread in basis points of mid."""
        mid = self.mid
        if mid is None or mid <= 0:
            return None
        return float(self.ask - self.bid) / mid * 10_000.0

    @property
    def day_change(self) -> float | None:
        """Last (or mid fallback) minus previous close."""
        ref = self.last_trade if self.last_trade else self.mid
        if ref is None or self.previous_close in (None, 0):
            return None
        try:
            if not math.isfinite(float(ref)) \
                    or not math.isfinite(float(self.previous_close)):
                return None
        except (TypeError, ValueError):
            return None
        return float(ref) - float(self.previous_close)

    @property
    def day_change_pct(self) -> float | None:
        """Day change as a fraction of previous close."""
        if self.day_change is None or not self.previous_close:
            return None
        return self.day_change / float(self.previous_close)

    def age_seconds(self, now: datetime | None = None) -> float | None:
        """Age of the newest observation (quote ts preferred, else trade)."""
        ts = self.quote_timestamp or self.last_trade_timestamp
        if ts is None:
            return None
        now = now or datetime.now(timezone.utc)
        base = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        ref = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        return max(0.0, (ref - base).total_seconds())


# ---------------------------------------------------------------------------
# Market session (US equities; holidays ignored and documented)
# ---------------------------------------------------------------------------

def market_session_us(ts: datetime | None = None) -> str:
    """Classify a US-equity session instant.

    Returns one of ``pre`` / ``regular`` / ``after`` / ``closed`` using
    America/New_York wall time (weekends always closed; exchange holidays
    are NOT modelled — a holiday renders as ``regular`` with stale data,
    which the freshness layer still labels STALE, never LIVE).
    """
    from zoneinfo import ZoneInfo

    ts = ts or datetime.now(timezone.utc)
    try:
        et = (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
              ).astimezone(ZoneInfo("America/New_York"))
    except Exception:
        return "closed"
    if et.weekday() >= 5:
        return "closed"
    mins = et.hour * 60 + et.minute
    if mins < 4 * 60:
        return "closed"
    if mins < 9 * 60 + 30:
        return "pre"
    if mins < 16 * 60:
        return "regular"
    if mins < 20 * 60:
        return "after"
    return "closed"


# ---------------------------------------------------------------------------
# Freshness + mark policy
# ---------------------------------------------------------------------------

def quote_status(quote: Quote | None, now: datetime | None = None,
                 stale_after: int = STALE_AFTER_SECONDS) -> str:
    """Freshness category: LIVE / DELAYED / STALE / CLOSED / FALLBACK.

    * CLOSED — US session not open (any quote shown is last-known, never live).
    * LIVE — fresh provider observation inside an open session.
    * DELAYED — open session but observation aged past the LIVE window.
    * STALE — older than ``stale_after`` (or no timestamp at all).
    * FALLBACK — Yahoo/snapshot source (never called live by construction).
    """
    if quote is None:
        return "STALE"
    now = now or datetime.now(timezone.utc)
    if (quote.source or "").strip().lower() in ("yahoo", "snapshot", "fallback"):
        return "FALLBACK"
    age = quote.age_seconds(now)
    if age is None:
        return "STALE"
    if market_session_us(now) == "closed":
        return "CLOSED"
    if age <= FRESH_TRADE_SECONDS:
        return "LIVE"
    if age <= stale_after:
        return "DELAYED"
    return "STALE"


def select_mark(quote: Quote | None, now: datetime | None = None,
                stale_after: int = STALE_AFTER_SECONDS
                ) -> tuple[float | None, str, str]:
    """Explicit mark-price policy → ``(mark, method, source_label)``.

    Priority (first usable wins):
      1. valid fresh bid+ask → ``mid`` ("Mid (bid/ask)")
      2. fresh latest trade → ``last`` ("Last trade")
      3. any usable last/mid regardless of age → same labels + " (aged)"
      4. previous close → ("Previous close")
      5. otherwise → (None, "Unavailable", source)

    Freshness here means "not STALE" (DELAYED still counts — labelled).
    Nothing is ever invented: missing legs fall through, never assumed.
    """
    label_src = (quote.source if quote else None) or "unknown"
    if quote is None:
        return None, "Unavailable", label_src
    now = now or datetime.now(timezone.utc)
    fresh = quote_status(quote, now, stale_after) in ("LIVE", "DELAYED")
    if quote.mid is not None and fresh:
        return quote.mid, "Mid (bid/ask)", label_src
    trade_ok = (quote.last_trade is not None
                and math.isfinite(float(quote.last_trade))
                and float(quote.last_trade) > 0)
    if trade_ok and fresh:
        return float(quote.last_trade), "Last trade", label_src
    if quote.mid is not None:
        return quote.mid, "Mid (bid/ask, aged)", label_src
    if trade_ok:
        return float(quote.last_trade), "Last trade (aged)", label_src
    if quote.previous_close is not None \
            and math.isfinite(float(quote.previous_close)) \
            and float(quote.previous_close) > 0:
        return float(quote.previous_close), "Previous close", label_src
    return None, "Unavailable", label_src


# ---------------------------------------------------------------------------
# Thread-safe quote store (one per process; newest-wins sequencing)
# ---------------------------------------------------------------------------

class LiveStore:
    """Registry of current quotes with newest-wins update semantics."""

    def __init__(self):
        self._lock = threading.Lock()
        self._quotes: dict[str, Quote] = {}

    def _newer(self, old: Quote | None, new: Quote) -> bool:
        if old is None:
            return True

        def _ts(q: Quote | None):
            for attr in ("quote_timestamp", "last_trade_timestamp",
                         "received_at"):
                ts = getattr(q, attr, None)
                if ts is not None:
                    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            return None

        t_old, t_new = _ts(old), _ts(new)
        if t_old is None:
            return True
        if t_new is None:
            return False
        return t_new >= t_old  # equal timestamps: latest message wins

    def update(self, quote: Quote) -> bool:
        """Store ``quote`` iff newer than the held one. Returns stored flag."""
        sym = validate_ticker_symbol(quote.symbol)
        with self._lock:
            if self._newer(self._quotes.get(sym), quote):
                self._quotes[sym] = quote
                return True
            return False

    def get(self, symbol: str) -> Quote | None:
        """Current quote or None (never raises on unknown symbols)."""
        try:
            sym = validate_ticker_symbol(symbol)
        except ValueError:
            return None
        with self._lock:
            return self._quotes.get(sym)

    def snapshot(self, symbols: list[str]) -> dict[str, Quote | None]:
        """One consistent read of several symbols (single lock hold)."""
        out: dict[str, Quote | None] = {}
        with self._lock:
            for s in symbols or []:
                try:
                    out[s] = self._quotes.get(validate_ticker_symbol(s))
                except ValueError:
                    out[s] = None
        return out

    def symbols(self) -> list[str]:
        """Currently tracked symbols (sorted)."""
        with self._lock:
            return sorted(self._quotes)

    def remove(self, symbol: str) -> None:
        """Unsubscribe one symbol (no error when absent)."""
        try:
            sym = validate_ticker_symbol(symbol)
        except ValueError:
            return
        with self._lock:
            self._quotes.pop(sym, None)

    def clear(self) -> None:
        """Drop all quotes (tests / session reset)."""
        with self._lock:
            self._quotes.clear()


# ---------------------------------------------------------------------------
# Streaming ingestor (transport-injected; disabled by default)
# ---------------------------------------------------------------------------

@dataclass
class IngestorStats:
    """Counters describing ingestor lifecycle (no secrets)."""
    messages: int = 0
    applied: int = 0
    dropped_stale: int = 0
    reconnects: int = 0
    last_error: str | None = None


class StreamIngestor:
    """Quote-message consumer with sequencing + subscription registry.

    Messages are plain dicts (symbol, bid/ask/last(+sizes), timestamps,
    feed). No socket is opened here: the caller feeds ``ingest()`` from
    whatever transport exists (or tests). Reconnect/backoff counters model
    lifecycle policy deterministically; ``transport_factory`` is None by
    default (streaming disabled — REST/fallback polling applies).
    """

    def __init__(self, store: LiveStore | None = None,
                 transport_factory=None):
        self.store = store or LiveStore()
        self.transport_factory = transport_factory
        self.subscriptions: set[str] = set()
        self.stats = IngestorStats()
        self.connected = False
        self._backoff_s = 1.0

    # -- subscriptions ---------------------------------------------------
    def subscribe(self, symbols) -> list[str]:
        """Add symbols (validated); returns the sorted subscription set."""
        for s in symbols or []:
            self.subscriptions.add(validate_ticker_symbol(s))
        return sorted(self.subscriptions)

    def unsubscribe(self, symbols) -> list[str]:
        """Remove symbols (and drop their quotes); no reconnect forced."""
        for s in symbols or []:
            try:
                sym = validate_ticker_symbol(s)
            except ValueError:
                continue
            self.subscriptions.discard(sym)
            self.store.remove(sym)
        return sorted(self.subscriptions)

    # -- lifecycle --------------------------------------------------------
    def connect(self) -> bool:
        """Attach transport if configured; False when streaming disabled."""
        if self.transport_factory is None:
            self.connected = False
            self.stats.last_error = "streaming transport not configured"
            return False
        try:
            self.transport_factory(sorted(self.subscriptions))
        except Exception as e:
            self.note_failure(str(e))
            return False
        self.connected = True
        self._backoff_s = 1.0
        return True

    def disconnect(self) -> None:
        """Detach (idempotent); store keeps last-known quotes."""
        self.connected = False

    def note_failure(self, message: str) -> float:
        """Record a drop/failure; returns the next backoff delay (capped)."""
        self.stats.reconnects += 1
        self.stats.last_error = str(message)[:200]
        self.connected = False
        delay = self._backoff_s
        self._backoff_s = min(self._backoff_s * 2.0, 60.0)
        return delay

    # -- ingest -----------------------------------------------------------
    @staticmethod
    def _f(value) -> float | None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) and v > 0 else None

    @staticmethod
    def _ts(value) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        try:
            from data.alpaca import _parse_timestamp
            return _parse_timestamp(value)
        except Exception:
            return None

    def ingest(self, message: dict, received_at: datetime | None = None
               ) -> bool:
        """Apply one quote message; out-of-order messages never regress state.

        Returns True when the store now holds this message's observation.
        Unknown/malformed symbols are ignored (False), never raised.
        """
        if not isinstance(message, dict):
            return False
        try:
            sym = validate_ticker_symbol(message.get("symbol", ""))
        except ValueError:
            return False
        now = received_at or datetime.now(timezone.utc)
        quote = Quote(
            symbol=sym,
            last_trade=self._f(message.get("last")),
            last_trade_size=message.get("last_size"),
            last_trade_timestamp=self._ts(message.get("last_ts")),
            bid=self._f(message.get("bid")),
            bid_size=message.get("bid_size"),
            ask=self._f(message.get("ask")),
            ask_size=message.get("ask_size"),
            quote_timestamp=self._ts(message.get("quote_ts")),
            previous_close=self._f(message.get("previous_close")),
            source=str(message.get("source") or "stream"),
            feed=message.get("feed"),
            received_at=now if now.tzinfo else now.replace(tzinfo=timezone.utc),
        )
        self.stats.messages += 1
        stored = self.store.update(quote)
        if stored:
            self.stats.applied += 1
        else:
            self.stats.dropped_stale += 1
        return stored


# ---------------------------------------------------------------------------
# REST providers (Alpaca quotes + Yahoo fallback quote objects)
# ---------------------------------------------------------------------------

def fetch_alpaca_quotes(symbols, credentials=None, feed: str = "iex",
                        timeout: int = 10) -> dict[str, Quote]:
    """Latest bid/ask quotes via ``GET /v2/stocks/quotes/latest``.

    Returns symbol → :class:`Quote` (bid/ask legs only; last-trade merge is
    the caller's job). Missing/malformed symbols are absent, never fabricated.
    Raises :class:`AlpacaError` on transport/auth/entitlement failures.
    """
    from data import alpaca as _alp

    creds = credentials or _alp.get_alpaca_credentials()
    wanted = []
    for s in symbols or []:
        sym = validate_ticker_symbol(s)
        if _alp.is_alpaca_supported_symbol(sym) and sym not in wanted:
            wanted.append(sym)
    if not wanted:
        return {}
    data = _alp._request_json(
        "/v2/stocks/quotes/latest", creds,
        {"symbols": ",".join(wanted), "feed": feed}, timeout)
    quotes = data.get("quotes")
    if not isinstance(quotes, dict):
        raise _alp.AlpacaError("Alpaca response missing 'quotes'.")
    out: dict[str, Quote] = {}
    for sym in wanted:
        entry = quotes.get(sym)
        if not isinstance(entry, dict):
            continue
        bid = StreamIngestor._f(entry.get("bp"))
        ask = StreamIngestor._f(entry.get("ap"))
        if bid is None and ask is None:
            continue
        out[sym] = Quote(
            symbol=sym, bid=bid,
            bid_size=entry.get("bs"), ask=ask, ask_size=entry.get("as"),
            quote_timestamp=_alp._parse_timestamp(entry.get("t")),
            source="Alpaca", feed=feed,
            received_at=datetime.now(timezone.utc))
    return out


def fetch_with_feed_fallback(symbols, credentials=None,
                             feeds: tuple = FEED_PRIORITY,
                             timeout: int = 10) -> tuple[dict, str | None]:
    """Try feeds in order; 403 entitlement failure falls through to the next.

    Returns ``(quotes, feed_used)``; ``feed_used`` is None when every feed
    failed (caller falls back to Yahoo). Other :class:`AlpacaError` kinds
    propagate (network/auth/rate-limit are not entitlement guesses).
    """
    from data import alpaca as _alp

    last = None
    for feed in feeds or FEED_PRIORITY:
        try:
            return fetch_alpaca_quotes(symbols, credentials, feed, timeout), feed
        except _alp.AlpacaError as e:
            if getattr(e, "status", None) == 403:
                last = e
                continue
            raise
    raise last if last is not None else _alp.AlpacaError(
        "No feed available and no entitlement error recorded.")


def yahoo_fallback_quote(symbol: str, last: float | None,
                         previous_close: float | None = None,
                         received_at: datetime | None = None) -> Quote | None:
    """Yahoo current price as an explicit FALLBACK quote (never 'live').

    Bid/ask legs are unknown (None) by construction; the mark policy falls
    through to last/previous-close with the fallback label.
    """
    try:
        sym = validate_ticker_symbol(symbol)
    except ValueError:
        return None
    px = StreamIngestor._f(last)
    if px is None:
        return None
    now = received_at or datetime.now(timezone.utc)
    return Quote(symbol=sym, last_trade=px, source="Yahoo", feed=None,
                 previous_close=StreamIngestor._f(previous_close),
                 received_at=now if now.tzinfo else now.replace(
                     tzinfo=timezone.utc))


# ---------------------------------------------------------------------------
# Snapshot builder (one consistent read → marks for every holding)
# ---------------------------------------------------------------------------

@dataclass
class MarkResult:
    """Valuation mark for one symbol from the snapshot quote."""

    symbol: str
    mark: float | None
    method: str
    source: str
    feed: str | None
    status: str
    age_seconds: float | None
    day_change_pct: float | None


def build_mark_snapshot(quotes: dict[str, Quote | None],
                        now: datetime | None = None,
                        stale_after: int = STALE_AFTER_SECONDS
                        ) -> dict[str, MarkResult]:
    """Apply the mark policy to a consistent quote mapping (pure)."""
    now = now or datetime.now(timezone.utc)
    out: dict[str, MarkResult] = {}
    for sym, quote in (quotes or {}).items():
        try:
            key = validate_ticker_symbol(sym)
        except ValueError:
            continue
        mark, method, source = select_mark(quote, now, stale_after)
        status = quote_status(quote, now, stale_after)
        out[key] = MarkResult(
            symbol=key, mark=mark, method=method, source=source,
            feed=(quote.feed if quote else None),
            status=status,
            age_seconds=(quote.age_seconds(now) if quote else None),
            day_change_pct=(quote.day_change_pct if quote else None))
    return out


def build_snapshot_marks(symbols: list[str], yahoo_current: dict,
                         yahoo_prev_close: dict | None = None,
                         credentials=None,
                         now: datetime | None = None) -> dict:
    """One consistent mark snapshot for a symbol set (never raises).

    Tries Alpaca trades+quotes (feed fallback SIP→IEX) when credentials are
    available, merges last-trade legs, and falls back per symbol to Yahoo
    (explicit FALLBACK status). Returns symbol → plain mark dicts
    (JSON-safe for session state) plus a ``_status`` summary entry.
    """
    from data import alpaca as _alp

    now = now or datetime.now(timezone.utc)
    symbols = [validate_ticker_symbol(s) for s in (symbols or [])]
    yahoo_current = dict(yahoo_current or {})
    yahoo_prev_close = dict(yahoo_prev_close or {})
    quotes: dict[str, Quote] = {}
    feed_used, feed_note = None, "Yahoo fallback"
    trades: dict = {}
    if credentials is not None:
        try:
            trades = _alp.fetch_latest_trades(symbols, credentials)
        except (_alp.AlpacaError, ValueError):
            trades = {}
        try:
            qmap, feed_used = fetch_with_feed_fallback(symbols, credentials)
            feed_note = f"Alpaca {feed_used.upper()}"
        except (_alp.AlpacaError, ValueError):
            qmap = {}
        for sym, q in qmap.items():
            trade = trades.get(sym) if isinstance(trades, dict) else None
            quotes[sym] = Quote(
                symbol=sym, bid=q.bid, bid_size=q.bid_size, ask=q.ask,
                ask_size=q.ask_size, quote_timestamp=q.quote_timestamp,
                last_trade=(trade.price if trade else None),
                last_trade_timestamp=(trade.timestamp if trade else None),
                last_trade_size=None,
                previous_close=(float(yahoo_prev_close[sym])
                                if yahoo_prev_close.get(sym) not in (None,)
                                and math.isfinite(float(
                                    yahoo_prev_close.get(sym, float("nan"))))
                                else None),
                source="Alpaca", feed=q.feed or feed_used,
                received_at=now)
    for sym in symbols:
        if sym in quotes:
            continue
        prev = yahoo_prev_close.get(sym)
        quotes[sym] = yahoo_fallback_quote(
            sym, yahoo_current.get(sym),
            previous_close=prev if prev is not None and math.isfinite(
                float(prev)) else None,
            received_at=now) or Quote(symbol=sym, source="unknown",
                                      received_at=now)
    marks = build_mark_snapshot(quotes, now)
    out = {}
    for sym, m in marks.items():
        out[sym] = {"mark": m.mark, "method": m.method, "source": m.source,
                    "feed": m.feed, "status": m.status,
                    "age_seconds": m.age_seconds,
                    "day_change_pct": m.day_change_pct,
                    "prev_close": (quotes[sym].previous_close
                                   if quotes[sym] else None)}
    out["_status"] = {"feed": feed_used, "feed_note": feed_note,
                      "n_symbols": len(symbols),
                      "at": now.isoformat()}
    return out

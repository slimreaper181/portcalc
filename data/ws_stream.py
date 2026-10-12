"""ws_stream — true streaming market data (websockets transport + manager).

Task 9 built the poll path plus a transport-injected ``StreamIngestor`` /
``LiveStore`` pair. This module supplies the missing real transport and
the single process-wide connection manager:

* Transport — :mod:`websockets` sync client (mature RFC 6455 framing,
  buffering, ping→pong, reassembly). A previous hand-rolled stdlib
  codec dropped pipelined handshake bytes under TCP coalescing and
  stalled reconnects; the library transport replaces it.
* :class:`AlpacaStreamClient` — Alpaca stock-data v2 protocol (auth,
  trades/quotes subscribe, error frames) with SIP→IEX entitlement
  fallback. The working feed is cached process-wide so a rejected SIP
  feed is never re-probed every rerun.
* :class:`StreamManager` — exactly one logical live connection per
  process: owns the store + ingestor, runs one daemon worker thread
  (which never touches Streamlit), applies incremental subscription
  diffs, tracks heartbeat, and reports STREAMING / POLLING / FALLBACK /
  STALE / CLOSED status. Stream failure degrades to REST/fallback —
  the portfolio keeps working.
* :func:`overlay_stream_marks` — merge stream-fresher marks into a poll
  snapshot using the SAME mark policy (timestamp precedence only; live
  data never touches historical estimation).

Secrets: credentials travel only inside the auth frame. Errors, logs,
status dicts and ``repr`` never contain key material (tested).
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

try:
    from websockets.sync.client import connect as _ws_lib_connect
    from websockets.exceptions import ConnectionClosed as _WsClosed
    _WS_LIB_AVAILABLE = True
except ImportError:  # pragma: no cover - websockets is a hard dependency
    _ws_lib_connect = None  # type: ignore
    _WsClosed = Exception  # type: ignore
    _WS_LIB_AVAILABLE = False

from analytics.validation import validate_ticker_symbol
from data.live_market import (
    LiveStore,
    StreamIngestor,
    market_session_us,
    quote_status,
    select_mark,
)

logger = logging.getLogger("portcalc.stream")

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
ALPACA_WS_HOST = "stream.data.alpaca.markets"
CONNECT_TIMEOUT = 10
RECV_TIMEOUT = 1.0


class StreamError(Exception):
    """Streaming failure (transport, auth, entitlement — never secrets)."""


class StreamAuthError(StreamError):
    """Authentication rejected (check keys; nothing secret is exposed)."""


class StreamEntitlementError(StreamError):
    """Feed not entitled (SIP→IEX fallback applies)."""


# ---------------------------------------------------------------------------
# WebSocket transport (mature `websockets` library — framing, buffering,
# ping→pong, fragmented reassembly handled by the library).
# ---------------------------------------------------------------------------

def _require_ws_lib():
    if not _WS_LIB_AVAILABLE or _ws_lib_connect is None:
        raise StreamError("Stream transport unavailable "
                          "(websockets package missing).")


def ws_connect(url: str, extra_headers: dict | None = None,
               timeout: int = CONNECT_TIMEOUT):
    """Open a WebSocket and return the library connection handle.

    Kept under the historic ``ws_*`` names so callers/tests keep working;
    framing is delegated to :mod:`websockets` (no hand-rolled RFC 6455).
    """
    _require_ws_lib()
    try:
        conn = _ws_lib_connect(
            url,
            open_timeout=timeout,
            additional_headers=extra_headers or None,
            ping_interval=None,  # mock + Alpaca: no client pings needed
            max_size=2 ** 20,
        )
        return conn
    except StreamError:
        raise
    except Exception as e:
        raise StreamError("Stream connection failed.") from e


def ws_send_text(conn, text: str) -> None:
    """Send one text message (masked client frame via the library)."""
    try:
        conn.send(text)
    except _WsClosed as e:
        raise ConnectionError("stream closed by peer") from e
    except StreamError:
        raise
    except Exception as e:
        raise StreamError("Stream send failed.") from e


def ws_recv_message(conn, timeout: float | None = None):
    """Next application message → ``(kind, payload)``.

    ``kind`` is ``"text"`` (str payload), ``"close"`` (peer is going
    away), or ``"none"`` (timeout — caller keeps polling its stop flag).
    Ping/pong, fragmentation and buffering are handled by the library;
    binary frames are ignored (``"none"`` — caller polls again).
    """
    wait = RECV_TIMEOUT if timeout is None else timeout
    while True:
        try:
            message = conn.recv(timeout=wait)
        except TimeoutError:
            return "none", None
        except _WsClosed as e:
            # Abnormal drops (TCP reset / no close frame, e.g. code 1006)
            # must surface as failures so the run loop records a
            # "disconnected" heartbeat and reconnects with backoff; only a
            # clean close frame maps to ("close", None).
            try:
                from websockets.exceptions import ConnectionClosedOK
                if isinstance(e, ConnectionClosedOK):
                    return "close", None
            except ImportError:
                pass
            raise ConnectionError("stream closed by peer") from e
        except StreamError:
            raise
        except Exception as e:
            raise StreamError("Stream receive failed.") from e
        if isinstance(message, str):
            return "text", message
        # binary (unexpected — ignore and poll the next message)
        continue


def ws_close(conn) -> None:
    """Best-effort orderly close."""
    try:
        conn.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Alpaca streaming protocol
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """One parsed Alpaca message batch outcome (no secrets)."""
    kind: str  # trade | quote | info | error
    payload: dict | None = None
    code: int | None = None
    message: str = ""


class AlpacaStreamClient:
    """Alpaca stock-data v2 streaming client (trades + quotes).

    Message callbacks receive ingestor-shaped dicts
    (``{"symbol","last","last_ts","bid","ask","quote_ts","feed"}``).
    ``on_state`` receives ``"connected"`` / ``"disconnected"`` plus a
    short human note (never credentials).
    """

    def __init__(self, key_id: str, secret: str, feed: str = "iex",
                 on_message=None, on_state=None,
                 backoff_base: float = 1.0, backoff_cap: float = 60.0,
                 url_override: str | None = None):
        self._key_id = key_id
        self._secret = secret
        self.feed = str(feed or "iex").lower()
        self.on_message = on_message
        self.on_state = on_state
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.url_override = url_override
        self.last_message_at: datetime | None = None
        self.connected = False

    def __repr__(self):  # secrets never leak through repr/logs
        return (f"AlpacaStreamClient(feed={self.feed!r}, "
                f"connected={self.connected})")

    # -- connection ------------------------------------------------------
    def _url(self, feed: str) -> str:
        if self.url_override:
            if "{feed}" in self.url_override:
                return self.url_override.format(feed=feed)
            return self.url_override
        return f"wss://{ALPACA_WS_HOST}/v2/{feed}"

    def _note(self, text: str):
        if self.on_state:
            try:
                self.on_state("note", text)
            except Exception:
                pass

    def _handshake_session(self, conn, symbols: set[str], feed: str):
        """Auth + subscribe on an open connection (raises StreamError)."""
        first = self._recv_batch(conn, timeout=15)
        if not any(m.get("T") == "success" for m in first):
            raise StreamError("Stream greeting unexpected.")
        ws_send_text(conn, json.dumps(
            {"action": "auth", "key": self._key_id,
             "secret": self._secret}))
        auth = self._recv_batch(conn, timeout=15)
        for m in auth:
            if m.get("T") == "error":
                code = m.get("code")
                if code in (401, 403):
                    raise StreamAuthError(
                        "Stream authentication rejected (check API keys).")
                raise StreamError(f"Stream auth failed ({code or '?'}).")
        if not any(m.get("T") == "success" for m in auth):
            raise StreamError("Stream auth unexpected reply.")
        self._send_subscribe(conn, symbols)
        sub = self._recv_batch(conn, timeout=15)
        for m in sub:
            if m.get("T") == "error":
                code = m.get("code")
                if code == 403:
                    raise StreamEntitlementError(
                        f"Feed '{feed}' not entitled.")
                raise StreamError(
                    f"Stream subscribe rejected ({code or '?'}).")

    def _recv_batch(self, conn, timeout: int) -> list:
        kind, payload = ws_recv_message(conn, timeout=timeout)
        if kind != "text":
            raise StreamError("Stream ended during handshake.")
        try:
            data = json.loads(payload)
        except ValueError as e:
            raise StreamError("Stream sent malformed JSON.") from e
        if not isinstance(data, list):
            raise StreamError("Stream sent a malformed batch.")
        return data

    def _send_subscribe(self, sock, symbols: set[str]):
        wanted = sorted(symbols or [])
        ws_send_text(sock, json.dumps(
            {"action": "subscribe", "trades": wanted,
             "quotes": wanted}))

    # -- message dispatch --------------------------------------------------
    def _dispatch(self, batch: list, feed: str):
        for m in batch:
            if not isinstance(m, dict):
                continue
            tag = m.get("T")
            if tag == "t":
                self._emit({"symbol": m.get("S"), "last": m.get("p"),
                            "last_size": m.get("s"), "last_ts": m.get("t"),
                            "source": "stream", "feed": feed})
            elif tag == "q":
                self._emit({"symbol": m.get("S"), "bid": m.get("bp"),
                            "ask": m.get("ap"), "bid_size": m.get("bs"),
                            "ask_size": m.get("as"), "quote_ts": m.get("t"),
                            "source": "stream", "feed": feed})
            elif tag == "error":
                self._note(f"stream notice: {m.get('msg') or '?'}")
            # 'success' / 'subscription' / bars: nothing to ingest.

    def _emit(self, message: dict):
        self.last_message_at = datetime.now(timezone.utc)
        if self.on_message:
            try:
                self.on_message(message)
            except Exception as e:
                logger.warning("stream message handler failed: %s", e)

    # -- run loop ------------------------------------------------------------
    def run(self, stop_event: threading.Event,
            symbols_fn, feed: str | None = None):
        """Connect → stream → reconnect with backoff until stopped.

        ``symbols_fn`` returns the current subscription set (called
        under no locks — pass a snapshot callable). Re-subscribes the
        full set after every reconnect and on set changes.
        """
        _require_ws_lib()
        delay = self.backoff_base
        feed = (feed or self.feed or "iex").lower()
        while not stop_event.is_set():
            symbols = set(symbols_fn() or [])
            if not symbols:
                if stop_event.wait(1.0):
                    break
                continue
            try:
                try:
                    lib_conn = _ws_lib_connect(
                        self._url(feed),
                        open_timeout=CONNECT_TIMEOUT,
                        ping_interval=None,
                        max_size=2 ** 20,
                    )
                except _WsClosed as e:
                    raise StreamError("Stream ended during handshake.") from e
                except StreamError:
                    raise
                except Exception as e:
                    raise StreamError("Stream connection failed.") from e
                with lib_conn as conn:
                    try:
                        self._handshake_session(conn, symbols, feed)
                    except StreamEntitlementError:
                        # SIP→IEX fallback inside the loop: remember the
                        # refusal (never re-probed this process) and retry
                        # immediately on the fallback feed.
                        note_feed_result(feed, False)
                        if feed == "sip":
                            feed = "iex"
                            self.feed = "iex"
                            if self.on_state:
                                self.on_state(
                                    "disconnected",
                                    f"feed 'sip' not entitled "
                                    f"(fell back to iex)")
                            continue
                        raise
                    note_feed_result(feed, True)
                    self.connected = True
                    if self.on_state:
                        self.on_state("connected", f"feed {feed}")
                    delay = self.backoff_base
                    active = set(symbols)
                    while not stop_event.is_set():
                        current = set(symbols_fn() or [])
                        if current != active and current:
                            self._send_subscribe(conn, current)
                            active = set(current)
                        kind, payload = ws_recv_message(conn)
                        if kind == "text":
                            try:
                                batch = json.loads(payload)
                            except ValueError:
                                continue
                            if isinstance(batch, list):
                                self._dispatch(batch, feed)
                        elif kind == "close":
                            break
                        # "none": poll again (stop flag / subscription tick)
            except (StreamError, OSError, ConnectionError, _WsClosed) as e:
                if self.on_state:
                    self.on_state("disconnected", str(e)[:120])
            except Exception as e:  # never let the thread die silently
                logger.warning("stream loop error: %s", e)
                if self.on_state:
                    self.on_state("disconnected", "internal stream error")
            finally:
                self.connected = False
            if stop_event.is_set():
                break
            if stop_event.wait(delay):
                break
            delay = min(delay * 2.0 + random.uniform(0, 0.5),
                        self.backoff_cap)


# ---------------------------------------------------------------------------
# Feed resolution (SIP→IEX, cached — never re-probed every rerun)
# ---------------------------------------------------------------------------

_FEED_LOCK = threading.Lock()
_FEED_OK: str | None = None       # working feed once proven
_SIP_BLOCKED = False              # hard entitlement refusal remembered


def resolve_stream_feed() -> str:
    """Preferred stream feed ('sip' unless blocked → 'iex')."""
    with _FEED_LOCK:
        if _FEED_OK is not None:
            return _FEED_OK
        return "iex" if _SIP_BLOCKED else "sip"


def note_feed_result(feed: str, ok: bool):
    """Remember a proven/blocked feed for the rest of the process."""
    global _FEED_OK, _SIP_BLOCKED
    with _FEED_LOCK:
        if ok:
            _FEED_OK = str(feed or "iex").lower()
        elif str(feed or "").lower() == "sip":
            _SIP_BLOCKED = True


def reset_feed_cache_for_tests():
    """Clear feed memory (tests only)."""
    global _FEED_OK, _SIP_BLOCKED
    with _FEED_LOCK:
        _FEED_OK, _SIP_BLOCKED = None, False


# ---------------------------------------------------------------------------
# Connection manager (one logical connection per process)
# ---------------------------------------------------------------------------

_MANAGER_LOCK = threading.Lock()
_MANAGER = None


def _creds_available() -> bool:
    try:
        from data.alpaca import get_alpaca_credentials
        get_alpaca_credentials()
        return True
    except Exception:
        return False


class StreamManager:
    """Owns the store, ingestor and the single worker thread."""

    def __init__(self):
        self.store = LiveStore()
        self.ingestor = StreamIngestor(self.store)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._client: AlpacaStreamClient | None = None
        self._failures = 0
        self._last_error: str | None = None
        self._feed: str | None = None
        self._disabled = os.environ.get("PORTCALC_NO_STREAM", "").strip() \
            .lower() in ("1", "true", "yes")
        self._started_with_creds = False

    # -- lifecycle ---------------------------------------------------------
    def ensure(self, symbols, client_factory=None) -> dict:
        """Reconcile subscriptions; start the worker if streamable.

        Returns the current status dict (see :meth:`status`). Never
        raises, never touches Streamlit. ``client_factory`` is a test
        seam (default builds the real Alpaca client).
        """
        wanted: set[str] = set()
        for s in symbols or []:
            try:
                sym = validate_ticker_symbol(s)
            except ValueError:
                continue
            from data.alpaca import is_alpaca_supported_symbol
            if is_alpaca_supported_symbol(sym):
                wanted.add(sym)
        with self._lock:
            current = set(self.ingestor.subscriptions)
            for sym in sorted(wanted - current):
                self.ingestor.subscribe([sym])
            for sym in sorted(current - wanted):
                self.ingestor.unsubscribe([sym])
            if (not self._disabled and wanted and
                    (self._thread is None or not self._thread.is_alive())):
                if _creds_available():
                    self._start_worker(client_factory)
        return self.status()

    def _start_worker(self, client_factory=None):
        from data.alpaca import get_alpaca_credentials
        try:
            creds = get_alpaca_credentials()
        except Exception as e:
            self._last_error = str(e)[:160]
            return
        feed = resolve_stream_feed()
        if client_factory is not None:
            client = client_factory(feed)
        else:
            client = AlpacaStreamClient(
                creds.key_id, creds.secret, feed,
                on_message=self._on_message, on_state=self._on_state)
        self._client = client
        self._feed = feed
        self._stop.clear()
        self._started_with_creds = True
        thread = threading.Thread(
            target=self._run_client, args=(client,), daemon=True,
            name="portcalc-stream")
        self._thread = thread
        thread.start()

    def _run_client(self, client: AlpacaStreamClient):
        try:
            client.run(self._stop, self._symbols_snapshot)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("stream worker failed: %s", e)
        finally:
            with self._lock:
                if self._thread is not None and \
                        not self._thread.is_alive():
                    pass

    def _symbols_snapshot(self) -> set[str]:
        with self._lock:
            return set(self.ingestor.subscriptions)

    def _on_message(self, message: dict):
        self.ingestor.ingest(message)

    def _on_state(self, kind: str, note: str):
        with self._lock:
            if kind == "connected":
                self._failures = 0
                self._last_error = None
                text = str(note or "")
                if text.startswith("feed "):
                    self._feed = text[5:].strip() or self._feed
                    note_feed_result(self._feed, True)
                elif self._feed:
                    note_feed_result(self._feed, True)
            else:
                self._failures += 1
                if "not entitled" in (note or "").lower() and self._feed:
                    note_feed_result(self._feed, False)
                    self._feed = "iex" if self._feed == "sip" else self._feed
                self._last_error = str(note or "")[:160]

    def stop(self):
        """Stop the worker (idempotent; tests + cleanup)."""
        self._stop.set()
        thread = None
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)
        with self._lock:
            self._thread = None

    def reset_for_tests(self):
        """Stop + clear store/subscriptions/feed memory (tests only)."""
        self.stop()
        with self._lock:
            self.store.clear()
            self.ingestor.subscriptions.clear()
            self.ingestor.stats.messages = 0
            self.ingestor.stats.applied = 0
            self.ingestor.stats.dropped_stale = 0
            self.ingestor.stats.reconnects = 0
            self.ingestor.stats.last_error = None
            self._failures = 0
            self._last_error = None
            self._feed = None
            self._started_with_creds = False
        reset_feed_cache_for_tests()

    # -- status --------------------------------------------------------------
    def status(self, now: datetime | None = None) -> dict:
        """Connection status (JSON-safe; no secrets).

        Modes: STREAMING (live ticks flowing) · POLLING (no stream
        credentials/transport — REST path) · FALLBACK (stream failed,
        REST covers) · STALE (connected but quiet in an open session) ·
        CLOSED (market closed, nothing expected).
        """
        now = now or datetime.now(timezone.utc)
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            connected = bool(self._client is not None and
                             self._client.connected)
            last_msg = (self._client.last_message_at
                        if self._client else None)
            failures = self._failures
            last_error = self._last_error
            feed = self._feed
            n_symbols = len(self.ingestor.subscriptions)
            started = self._started_with_creds
        if last_msg is not None:
            base = last_msg if last_msg.tzinfo else last_msg.replace(
                tzinfo=timezone.utc)
            msg_age = max(0.0, (now - base).total_seconds())
        else:
            msg_age = None
        session = market_session_us(now)
        newest_age = None
        for sym in self.store.symbols():
            q = self.store.get(sym)
            if q is None:
                continue
            age = q.age_seconds(now)
            if age is not None and (newest_age is None or age < newest_age):
                newest_age = age
        if self._disabled or (not started and not _creds_available()):
            mode, note = "POLLING", "stream credentials absent (REST path)"
        elif session == "closed" and (msg_age is None or msg_age > 300):
            mode, note = "CLOSED", "market closed (no ticks expected)"
        elif connected and msg_age is not None and msg_age <= 120:
            mode, note = "STREAMING", f"live ticks ({feed or '?'} feed)"
        elif connected:
            mode, note = ("STALE", "connected but quiet "
                           "(no ticks in an open session)")
        elif failures > 0 or (started and not alive):
            mode, note = "FALLBACK", (last_error or "stream unavailable "
                                      "(REST covers)")
        elif alive:
            mode, note = "POLLING", "connecting… (REST covers meanwhile)"
        else:
            mode, note = "POLLING", "REST path"
        return {"mode": mode, "note": note, "feed": feed,
                "connected": connected,
                "last_message_age_s": (round(msg_age, 1)
                                       if msg_age is not None else None),
                "newest_quote_age_s": (round(newest_age, 1)
                                       if newest_age is not None else None),
                "n_symbols": n_symbols, "failures": failures,
                "last_error": last_error,
                "at": now.isoformat()}


def get_stream_manager() -> StreamManager:
    """Process-wide singleton (one logical connection per app scope)."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = StreamManager()
            try:
                import atexit
                atexit.register(_MANAGER.stop)
            except Exception:
                pass
        return _MANAGER


def reset_stream_manager_for_tests():
    """Drop the singleton (tests only — stops the worker first)."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is not None:
            try:
                _MANAGER.reset_for_tests()
            except Exception:
                pass
            _MANAGER = None


# ---------------------------------------------------------------------------
# Snapshot overlay (stream-fresher marks win; same mark policy)
# ---------------------------------------------------------------------------

def _parse_snapshot_time(raw) -> datetime | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def overlay_stream_marks(poll_snapshot: dict, store: LiveStore,
                         now: datetime | None = None) -> dict:
    """Overlay stream-fresher marks onto a poll snapshot (pure).

    A stream quote replaces a poll entry only when its observation is
    strictly newer than the poll snapshot time; the replacement mark
    comes from the SAME :func:`select_mark` policy. Entries gain a
    ``streamed: True`` flag. Poll entries (and their freshness labels)
    are otherwise untouched — and when the stream is quiet, the poll
    snapshot passes through unchanged.
    """
    now = now or datetime.now(timezone.utc)
    out = dict(poll_snapshot or {})
    status = dict(out.get("_status", {}) or {})
    snap_at = _parse_snapshot_time(status.get("at"))
    overlaid = 0
    for sym, entry in list(out.items()):
        if sym == "_status" or not isinstance(entry, dict):
            continue
        try:
            quote = store.get(sym)
        except Exception:
            continue
        if quote is None or quote.received_at is None:
            continue
        if (quote.source or "") != "stream":
            continue
        recv = quote.received_at if quote.received_at.tzinfo \
            else quote.received_at.replace(tzinfo=timezone.utc)
        if snap_at is not None and recv <= snap_at:
            continue  # poll is fresher — keep it
        work = quote
        if quote.previous_close is None and entry.get("prev_close"):
            try:
                from dataclasses import replace as _replace
                work = _replace(quote,
                                previous_close=float(entry["prev_close"]))
            except (TypeError, ValueError):
                pass
        mark, method, source = select_mark(work, now)
        if mark is None:
            continue
        merged = dict(entry)
        merged.update({
            "mark": mark, "method": method, "source": source,
            "feed": quote.feed or merged.get("feed"),
            "status": quote_status(work, now),
            "age_seconds": work.age_seconds(now),
            "day_change_pct": (work.day_change_pct
                               if work.day_change_pct is not None
                               else merged.get("day_change_pct")),
            "last": quote.last_trade if quote.last_trade is not None
            else merged.get("last"),
            "bid": quote.bid if quote.bid is not None
            else merged.get("bid"),
            "ask": quote.ask if quote.ask is not None
            else merged.get("ask"),
            "streamed": True,
        })
        out[sym] = merged
        overlaid += 1
    status["stream_overlaid"] = overlaid
    out["_status"] = status
    return out

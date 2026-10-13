"""M3 — true streaming: real stdlib WS transport against a mock server.

No internet is used: the mock Alpaca server speaks real RFC 6455 over
localhost sockets (handshake, masked client frames, server text frames,
close). Sequences: quote1 → quote2 → stale quote → disconnect →
reconnect → quote3, plus SIP→IEX fallback, auth rejection, secret
safety, store precedence and status mapping.
"""

import base64
import hashlib
import json
import socket
import struct
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from data import ws_stream as ws
from data.live_market import LiveStore, StreamIngestor, select_mark
from data.ws_stream import (
    AlpacaStreamClient,
    StreamManager,
    get_stream_manager,
    overlay_stream_marks,
    reset_stream_manager_for_tests,
)

SECRET = "sekrit-007-do-not-log"
KEY = "key-123-canary"


# ---------------------------------------------------------------------------
# mock Alpaca WS server (real sockets, scripted per connection)
# ---------------------------------------------------------------------------

def _server_send_text(conn: socket.socket, text: str):
    payload = text.encode("utf-8")
    head = bytearray([0x81])
    if len(payload) < 126:
        head.append(len(payload))
    elif len(payload) < 65536:
        head.append(126)
        head += struct.pack(">H", len(payload))
    else:
        head.append(127)
        head += struct.pack(">Q", len(payload))
    conn.sendall(bytes(head) + payload)


def _server_read_frame(conn: socket.socket) -> tuple[str, str]:
    conn.settimeout(10)
    head = b""
    while len(head) < 2:
        chunk = conn.recv(2 - len(head))
        if not chunk:
            raise ConnectionError("client closed")
        head += chunk
    b1, b2 = head
    opcode = b1 & 0x0F
    length = b2 & 0x7F
    if length == 126:
        buf = b""
        while len(buf) < 2:
            buf += conn.recv(2 - len(buf))
        length = struct.unpack(">H", buf)[0]
    elif length == 127:
        buf = b""
        while len(buf) < 8:
            buf += conn.recv(8 - len(buf))
        length = struct.unpack(">Q", buf)[0]
    mask = b""
    while len(mask) < 4:
        mask += conn.recv(4 - len(mask))
    payload = b""
    while len(payload) < length:
        chunk = conn.recv(length - len(payload))
        if not chunk:
            raise ConnectionError("client closed mid-frame")
        payload += chunk
    raw = bytes(c ^ mask[i % 4] for i, c in enumerate(payload))
    if opcode == 0x8:
        return "close", ""
    return "text", raw.decode("utf-8")


class MockAlpacaServer:
    """Scripted multi-connection WS server on localhost."""

    def __init__(self, handler):
        self.handler = handler  # fn(conn_no, path, inbound, send) -> None
        self.inbound: list[tuple[int, str, dict]] = []
        self.paths: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self._n = 0
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def url(self, feed="{feed}"):
        return f"ws://127.0.0.1:{self.port}/v2/{feed}"

    def stop(self):
        self._stop.set()
        try:
            socket.create_connection(("127.0.0.1", self.port),
                                     timeout=2).close()
        except OSError:
            pass
        self.thread.join(timeout=5)
        self.sock.close()

    # -- internals ---------------------------------------------------------
    def _serve(self):
        self.sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with self._lock:
                self._n += 1
                n = self._n
            threading.Thread(target=self._handle, args=(conn, n),
                             daemon=True).start()

    def _handle(self, conn: socket.socket, n: int):
        try:
            conn.settimeout(10)
            request = b""
            while b"\r\n\r\n" not in request:
                chunk = conn.recv(4096)
                if not chunk:
                    conn.close()
                    return
                request += chunk
            head = request.split(b"\r\n\r\n", 1)[0].decode("latin-1")
            lines = head.split("\r\n")
            path = lines[0].split(" ")[1] if lines else "/"
            key = ""
            for line in lines[1:]:
                if line.lower().startswith("sec-websocket-key:"):
                    key = line.split(":", 1)[1].strip()
            accept = base64.b64encode(hashlib.sha1(
                (key + ws.WS_GUID).encode("ascii")).digest()).decode()
            conn.sendall((
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())
            with self._lock:
                self.paths.append(path)

            def send(obj):
                _server_send_text(conn, json.dumps(obj))

            def recv():
                kind, text = _server_read_frame(conn)
                data = json.loads(text) if kind == "text" and text else {}
                with self._lock:
                    self.inbound.append((n, path, data))
                return data

            self.handler(n, path, recv, send, conn)
        except (ConnectionError, OSError, ValueError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass


def _auth_ok(n, path, recv, send, conn):
    send([{"T": "success", "msg": "connected"}])
    auth = recv()
    assert auth.get("action") == "auth"
    send([{"T": "success", "msg": "authenticated"}])


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "000Z"


# ---------------------------------------------------------------------------
# 3.16 deterministic harness: q1, q2, stale, drop, reconnect, q3
# ---------------------------------------------------------------------------

def test_stream_sequence_reconnect_and_stale_drop():
    now = datetime.now(timezone.utc)
    events: dict = {"subs": []}

    def handler(n, path, recv, send, conn):
        _auth_ok(n, path, recv, send, conn)
        sub = recv()
        assert sub.get("action") == "subscribe"
        events["subs"].append((n, sorted(sub.get("quotes", []))))
        send([{"T": "subscription", "trades": ["AAPL"],
               "quotes": ["AAPL"]}])
        if n == 1:
            send([{"T": "t", "S": "AAPL", "p": 100.0, "s": 10,
                   "t": _ts(now - timedelta(seconds=60)), "x": "V"}])
            send([{"T": "q", "S": "AAPL", "bp": 101.0, "ap": 101.5,
                   "bs": 5, "as": 5,
                   "t": _ts(now - timedelta(seconds=50))}])
            send([{"T": "q", "S": "AAPL", "bp": 1.0, "ap": 1.5,
                   "bs": 5, "as": 5,
                   "t": _ts(now - timedelta(hours=2))}])  # stale
            conn.close()  # hard drop → client must reconnect
        else:
            send([{"T": "t", "S": "AAPL", "p": 103.0, "s": 7,
                   "t": _ts(now - timedelta(seconds=5)), "x": "V"}])
            # hold the connection until the test stops the client
            conn.settimeout(0.5)
            deadline = time.time() + 20
            while time.time() < deadline:
                try:
                    kind, _text = _server_read_frame(conn)
                    if kind == "close":
                        return
                except (ConnectionError, OSError):
                    return
                except socket.timeout:
                    continue

    server = MockAlpacaServer(handler).start()
    store = LiveStore()
    ing = StreamIngestor(store)
    notes: list = []
    client = AlpacaStreamClient(
        KEY, SECRET, "iex", on_message=ing.ingest,
        on_state=lambda k, m: notes.append((k, m)),
        backoff_base=0.05, backoff_cap=0.2,
        url_override=server.url())
    stop = threading.Event()
    thread = threading.Thread(target=client.run,
                              args=(stop, lambda: {"AAPL"}),
                              daemon=True)
    thread.start()
    try:
        deadline = time.time() + 15
        while time.time() < deadline:
            q = store.get("AAPL")
            if q is not None and q.last_trade == 103.0:
                break
            time.sleep(0.05)
        q = store.get("AAPL")
        assert q is not None and q.last_trade == 103.0
        # stale quote never regressed the book: bid/ask from quote2
        assert q.bid == 101.0 and q.ask == 101.5
        assert ing.stats.applied == 3
        assert ing.stats.dropped_stale == 1
        assert len(events["subs"]) >= 2, "must resubscribe after reconnect"
        assert any(k == "disconnected" for k, _m in notes)
        assert any(k == "connected" for k, _m in notes)
    finally:
        stop.set()
        thread.join(timeout=10)
        server.stop()
    assert not thread.is_alive(), "worker thread must stop (no orphans)"


def test_sip_entitlement_falls_back_to_iex_and_caches():
    ws.reset_feed_cache_for_tests()
    now = datetime.now(timezone.utc)

    def handler(n, path, recv, send, conn):
        _auth_ok(n, path, recv, send, conn)
        sub = recv()
        if path.endswith("/sip"):
            send([{"T": "error", "code": 403,
                   "msg": "subscription/downgrade to IEX"}])
            return
        events["iex_subs"].append(sorted(sub.get("quotes", [])))
        send([{"T": "subscription", "trades": [], "quotes": []}])
        send([{"T": "t", "S": "MSFT", "p": 50.0, "s": 3,
               "t": _ts(now), "x": "V"}])
        conn.settimeout(0.5)
        try:
            while True:
                _server_read_frame(conn)
        except (ConnectionError, OSError, socket.timeout):
            return

    events: dict = {"iex_subs": []}
    server = MockAlpacaServer(handler).start()
    store = LiveStore()
    ing = StreamIngestor(store)
    client = AlpacaStreamClient(
        KEY, SECRET, "sip", on_message=ing.ingest,
        backoff_base=0.05, backoff_cap=0.2,
        url_override=server.url())
    stop = threading.Event()
    thread = threading.Thread(target=client.run,
                              args=(stop, lambda: {"MSFT"}),
                              daemon=True)
    thread.start()
    try:
        deadline = time.time() + 15
        while time.time() < deadline:
            if store.get("MSFT") is not None:
                break
            time.sleep(0.05)
        assert store.get("MSFT") is not None
        assert store.get("MSFT").feed == "iex"
        assert events["iex_subs"] == [["MSFT"]]
        assert ws.resolve_stream_feed() == "iex", "refusal cached"
        assert server.paths[0].endswith("/sip")
    finally:
        stop.set()
        thread.join(timeout=10)
        server.stop()
    ws.reset_feed_cache_for_tests()


def test_auth_rejection_exposes_no_secrets():
    def handler(n, path, recv, send, conn):
        send([{"T": "success", "msg": "connected"}])
        recv()
        send([{"T": "error", "code": 401, "msg": "invalid key"}])
        conn.close()

    server = MockAlpacaServer(handler).start()
    client = AlpacaStreamClient(
        KEY, SECRET, "iex", backoff_base=0.05, backoff_cap=0.1,
        url_override=server.url())
    stop = threading.Event()
    notes: list = []
    client.on_state = lambda k, m: notes.append((k, m))
    thread = threading.Thread(target=client.run,
                              args=(stop, lambda: {"AAPL"}),
                              daemon=True)
    thread.start()
    time.sleep(1.5)
    stop.set()
    thread.join(timeout=10)
    server.stop()
    assert repr(client) and KEY not in repr(client)
    assert SECRET not in repr(client)
    for _k, msg in notes:
        assert KEY not in msg and SECRET not in msg
    assert any("authentication rejected" in m for _k, m in notes)


# ---------------------------------------------------------------------------
# store precedence, overlay, status mapping (no sockets)
# ---------------------------------------------------------------------------

def test_overlay_newer_stream_wins_older_poll_kept():
    # Fixed Monday 10:00 ET (regular session) so freshness labels are live.
    now = datetime(2026, 10, 12, 14, 0, tzinfo=timezone.utc)
    store = LiveStore()
    ing = StreamIngestor(store)
    poll_at = now - timedelta(seconds=30)
    poll = {
        "AAPL": {"mark": 100.0, "method": "Last trade", "source": "Alpaca",
                 "feed": "iex", "status": "DELAYED", "age_seconds": 25.0,
                 "day_change_pct": 0.01, "prev_close": 99.0,
                 "last": 100.0, "bid": None, "ask": None},
        "MSFT": {"mark": 50.0, "method": "Last trade", "source": "Alpaca",
                 "feed": "iex", "status": "DELAYED", "age_seconds": 25.0,
                 "day_change_pct": 0.0, "prev_close": 50.0,
                 "last": 50.0, "bid": None, "ask": None},
        "_status": {"feed": "iex", "at": poll_at.isoformat()},
    }
    # fresher stream quote for AAPL (bid/ask → mid policy, not last print)
    assert ing.ingest({"symbol": "AAPL", "bid": 104.0, "ask": 104.4,
                       "quote_ts": _ts(now - timedelta(seconds=5)),
                       "source": "stream", "feed": "iex"},
                      received_at=now) is True
    # older stream quote for MSFT (predates the poll → ignored)
    assert ing.ingest({"symbol": "MSFT", "last": 70.0,
                       "last_ts": _ts(now - timedelta(minutes=5)),
                       "source": "stream", "feed": "iex"},
                      received_at=now - timedelta(minutes=5)) is True
    out = overlay_stream_marks(poll, store, now=now)
    assert out["AAPL"]["mark"] == pytest.approx(104.2)  # mid, not last
    assert out["AAPL"]["method"] == "Mid (bid/ask)"
    assert out["AAPL"]["streamed"] is True
    assert out["MSFT"]["mark"] == pytest.approx(50.0)  # poll kept
    assert "streamed" not in out["MSFT"]
    assert out["_status"]["stream_overlaid"] == 1


def test_status_modes_without_network(monkeypatch):
    monkeypatch.setenv("PORTCALC_NO_STREAM", "1")
    mgr = StreamManager()
    st = mgr.status()
    assert st["mode"] == "POLLING"
    assert "REST" in st["note"]
    assert mgr.ensure(["AAPL"])["mode"] == "POLLING"
    assert mgr._thread is None, "disabled manager starts no threads"
    monkeypatch.delenv("PORTCALC_NO_STREAM")


def test_manager_subscription_diff_is_incremental():
    import os
    os.environ["PORTCALC_NO_STREAM"] = "1"
    try:
        mgr = StreamManager()
        mgr.ensure(["AAPL", "MSFT"])
        assert mgr.ingestor.subscriptions == {"AAPL", "MSFT"}
        mgr.ensure(["AAPL", "TSLA"])
        assert mgr.ingestor.subscriptions == {"AAPL", "TSLA"}
        assert mgr._thread is None
    finally:
        del os.environ["PORTCALC_NO_STREAM"]


def test_mark_policy_prefers_bid_ask_over_stale_last():
    # A random trade print must not override fresher bid/ask logic:
    # mid wins whenever both sides are valid and fresh.
    now = datetime(2026, 10, 12, 14, 0, tzinfo=timezone.utc)
    store = LiveStore()
    ing = StreamIngestor(store)
    ing.ingest({"symbol": "NVDA", "last": 500.0,
                "last_ts": _ts(now - timedelta(seconds=10)),
                "bid": 100.0, "ask": 100.2,
                "quote_ts": _ts(now - timedelta(seconds=10)),
                "source": "stream", "feed": "iex"}, received_at=now)
    mark, method, _src = select_mark(store.get("NVDA"), now)
    assert mark == pytest.approx(100.1)
    assert method == "Mid (bid/ask)"

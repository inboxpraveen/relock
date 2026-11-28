"""HTTP/JSON API: plug relock into a harness written in any language.

    RELOCK_TOKEN=change-me relock serve --port 8765

Rust, TypeScript, Go, Java, C# or a shell script all use relock the same way. Open an episode,
recall into the prompt, report each tool call's real outcome, then close the episode. The OpenAPI
3.1 description is served at ``GET /v1/openapi.json``.

This API is for your *harness*: the code that runs tools and checks results. Each request's
signal is trusted according to the ``source`` it declares. Never expose it to the model. Give
models the MCP server, where nothing the agent writes is trusted.

The server has no dependencies (``http.server``). It handles requests on threads and closes
episodes left idle longer than ``idle_timeout``. It refuses to listen on a non-loopback address
without a token.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import re
import signal as _signal
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .. import __version__
from ..episode import Episode
from ..memory import Memory
from ..signals import _PROGRAMMATIC_STATUS, _TRANSIENT_STATUS, ErrorClass, Signal
from ..types import (
    ConflictError,
    EpisodeClosedError,
    Item,
    NotFoundError,
    NotLabileError,
    Observation,
    RelockError,
    Verdict,
)

log = logging.getLogger("relock.http")

MAX_BODY = 1 << 20  # 1 MiB
SOCKET_TIMEOUT = 30.0  # seconds a connection may sit idle or stall mid-request
MAX_CONNECTIONS = 256  # connections served at once; more wait in the listen backlog


class ApiError(Exception):
    def __init__(self, status: int, message: str, code: str = "bad_request") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


# --------------------------------------------------------------------------- service
class Service:
    """The API's logic, independent of HTTP (easy to test and to embed)."""

    def __init__(self, memory: Memory, *, idle_timeout: float | None = 1800.0, max_episodes: int = 10_000,
                 k: int = 3) -> None:
        self.memory = memory
        self.idle_timeout = idle_timeout
        self.max_episodes = max_episodes
        self.k = k
        self._eps: dict[str, list[Any]] = {}  # id -> [episode, last_used, busy]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sweeper: threading.Thread | None = None

    # ---- episode registry
    def _expired(self, last: float, busy: int) -> bool:
        return bool(self.idle_timeout) and busy == 0 and last < time.monotonic() - (self.idle_timeout or 0)

    def _sweep(self) -> None:
        if not self.idle_timeout:
            return
        with self._lock:
            stale = [eid for eid, (_, last, busy) in self._eps.items() if self._expired(last, busy)]
            eps = [self._eps.pop(eid)[0] for eid in stale]
        for ep in eps:
            self._relock(ep)

    @staticmethod
    def _relock(ep: Episode) -> None:
        try:
            ep.close()
            log.info("re-locked idle episode %s", ep.id)
        except Exception:  # pragma: no cover - closing must never break a request
            log.exception("failed to close idle episode %s", ep.id)

    def start_sweeper(self) -> None:
        """Re-lock idle episodes in the background, even ones no request touches again."""
        if not self.idle_timeout or self._sweeper is not None:
            return
        interval = max(0.05, min(60.0, self.idle_timeout / 4))

        def loop() -> None:
            while not self._stop.wait(interval):
                self._sweep()
        self._sweeper = threading.Thread(target=loop, name="relock-sweeper", daemon=True)
        self._sweeper.start()

    def _use(self, eid: str) -> Episode:
        expired = None
        with self._lock:
            entry = self._eps.get(eid)
            if entry is not None and self._expired(entry[1], entry[2]):
                expired, entry = self._eps.pop(eid)[0], None
            if entry is not None:
                entry[1] = time.monotonic()
                entry[2] += 1
                return entry[0]
        if expired is not None:
            self._relock(expired)
        raise ApiError(404, f"no open episode {eid!r} (closed, idle-expired or unknown)", "episode_not_found")

    def _release(self, eid: str) -> None:
        with self._lock:
            entry = self._eps.get(eid)
            if entry is not None:
                entry[1] = time.monotonic()
                entry[2] = max(0, entry[2] - 1)

    def with_episode(self, eid: str, fn: Callable[[Episode], Any]) -> Any:
        ep = self._use(eid)
        try:
            return fn(ep)
        finally:
            self._release(eid)

    # ---- operations
    def health(self) -> dict[str, Any]:
        with self._lock:
            open_eps = len(self._eps)
        return {"ok": True, "version": __version__, "open_episodes": open_eps}

    def add_item(self, body: dict[str, Any]) -> dict[str, Any]:
        item = self.memory.add(
            _str(body, "title"), _str_list(body, "clauses"), key=_opt_str(body, "key"),
            kind=_opt_str(body, "kind") or "procedure", tags=_str_list(body, "tags", required=False),
            tools=_str_list(body, "tools", required=False), meta=_opt_dict(body, "meta"),
            exist_ok=_opt_bool(body, "exist_ok", False), author=_opt_str(body, "author") or "api")
        return item_json(item)

    def list_items(self, limit: int = 500, offset: int = 0) -> dict[str, Any]:
        items = self.memory._active_items()  # cached, ordered by id
        page = items[offset:offset + limit]
        nxt = offset + limit if offset + limit < len(items) else None
        return {"items": [item_json(i) for i in page], "total": len(items), "next_offset": nxt}

    def get_item(self, ref: str, version: int | None) -> dict[str, Any]:
        return item_json(self.memory.get(ref, version))

    def history(self, ref: str) -> dict[str, Any]:
        return {"versions": [item_json(i) for i in self.memory.history(ref)]}

    def begin(self, body: dict[str, Any]) -> dict[str, Any]:
        self._sweep()
        with self._lock:
            if len(self._eps) >= self.max_episodes:
                raise ApiError(429, f"too many open episodes ({self.max_episodes}); close some first",
                               "too_many_episodes")
        ep = self.memory.episode(_opt_str(body, "task") or "", **(_opt_dict(body, "meta") or {}))
        with self._lock:
            self._eps[ep.id] = [ep, time.monotonic(), 0]
        return {"episode_id": ep.id}

    def recall(self, eid: str, body: dict[str, Any]) -> dict[str, Any]:
        query = _str(body, "query")
        k = _opt_int(body, "k") or self.k
        budget = _opt_int(body, "budget_tokens")

        def run(ep: Episode) -> dict[str, Any]:
            rec = ep.recall(query, k=k)
            return {"context": rec.render(budget_tokens=budget),
                    "items": [{**item_json(h.item), "score": round(h.score, 6), "rank": h.rank} for h in rec]}
        return self.with_episode(eid, run)

    def use(self, eid: str, body: dict[str, Any]) -> dict[str, Any]:
        ref = _str(body, "item")
        return self.with_episode(eid, lambda ep: item_json(ep.use(ref)))

    def expect(self, eid: str, body: dict[str, Any]) -> dict[str, Any]:
        item, statement = _str(body, "item"), _opt_str(body, "prediction") or ""
        clauses, tool = _str_list(body, "clauses", required=False) or None, _opt_str(body, "tool")
        exp = self.with_episode(eid, lambda ep: ep.expect(item, statement, clauses=clauses, tool=tool))
        return {"expectation_id": exp.id}

    def observe(self, eid: str, body: dict[str, Any]) -> dict[str, Any]:
        exp_id, signal = _str(body, "expectation_id"), signal_from(body)
        return obs_json(self.with_episode(eid, lambda ep: ep.observe(exp_id, signal=signal)))

    def tool_result(self, eid: str, body: dict[str, Any]) -> dict[str, Any]:
        tool, signal = _str(body, "tool"), signal_from(body)
        items = _str_list(body, "items", required=False) or None
        result = body.get("result")
        obs = self.with_episode(eid, lambda ep: ep.tool_result(tool, result, signal=signal, items=items))
        return {"observations": [obs_json(o) for o in obs]}

    def correct(self, eid: str, body: dict[str, Any]) -> dict[str, Any]:
        item, correction = _str(body, "item"), _str(body, "correction")
        clauses = _str_list(body, "clauses", required=False) or None
        authenticated = _opt_bool(body, "authenticated", False)
        principal = _opt_str(body, "principal")
        return obs_json(self.with_episode(eid, lambda ep: ep.correct(
            item, correction, principal=principal, authenticated=authenticated, clauses=clauses)))

    def close(self, eid: str) -> dict[str, Any]:
        with self._lock:
            entry = self._eps.pop(eid, None)
        if entry is None:
            raise ApiError(404, f"no open episode {eid!r}", "episode_not_found")
        return entry[0].close()

    def close_all(self) -> None:
        self._stop.set()
        with self._lock:
            eps = [e[0] for e in self._eps.values()]
            self._eps.clear()
        for ep in eps:
            ep.close()


# --------------------------------------------------------------------------- JSON helpers
def item_json(item: Item) -> dict[str, Any]:
    keep = ("id", "key", "version", "status", "title", "kind", "clauses", "tags", "tools", "meta", "confidence",
            "confirmations", "failures", "change_kind", "reason", "author", "created_at", "archived")
    d = item.to_dict()
    return {"ref": item.ref, **{k: d[k] for k in keep}}


def obs_json(obs: Observation) -> dict[str, Any]:
    return {"expectation_id": obs.expectation_id, "matched": obs.matched, **obs.decision.to_dict()}


def signal_from(body: dict[str, Any]) -> Signal:
    """Build a trusted-harness signal from ``{"ok": bool, "error": {...}, "source": ...}``."""
    ok = body.get("ok")
    if not isinstance(ok, bool):
        raise ApiError(400, "'ok' must be true or false")
    source = body.get("source", "tool")
    if not isinstance(source, str):
        raise ApiError(400, "'source' must be a string")
    source = source.lower()
    err = body.get("error") or {}
    if isinstance(err, str):
        err = {"message": err}
    if not isinstance(err, dict):
        raise ApiError(400, "'error' must be an object or a string")
    message = _opt_str(err, "message", "error.message") or ""
    correction = _opt_str(err, "correction", "error.correction")
    status = err.get("status")
    if status is not None and (not isinstance(status, int) or isinstance(status, bool)
                               or not 100 <= status <= 599):
        raise ApiError(400, "'error.status' must be an integer HTTP status (100-599)")
    transient = _opt_bool(err, "transient", False, "error.transient")
    detail_trusted = _opt_bool(err, "detail_trusted", True, "error.detail_trusted")
    if source == "tool":
        if ok:
            return Signal.ok()
        if transient or status in _TRANSIENT_STATUS:
            return Signal.transient(message or "transient failure", status=status)
        cls = _opt_str(err, "class", "error.class")
        if cls is None and status in _PROGRAMMATIC_STATUS:
            cls = ErrorClass.PROGRAMMATIC
        try:
            error_class = ErrorClass(cls) if cls else None
        except ValueError:
            raise ApiError(400, f"unknown error class {cls!r}; use one of "
                                f"{[c.value for c in ErrorClass]}") from None
        return Signal.tool_error(message or "tool failed", error_class=error_class, correction=correction,
                                 detail_trusted=detail_trusted, status=status)
    if source == "verifier":
        return Signal.verifier(ok, message, correction=correction, name=_opt_str(body, "verifier"))
    if source == "agent":
        return Signal.agent_report(ok, message, correction=correction)
    if source == "environment":
        return Signal.observation(message, passed=ok, correction=correction)
    raise ApiError(400, f"unknown source {source!r}; use tool, verifier, agent or environment "
                        "(user corrections go to /correct)")


def _str(body: dict[str, Any], name: str) -> str:
    v = body.get(name)
    if not isinstance(v, str) or not v.strip():
        raise ApiError(400, f"'{name}' is required and must be a non-empty string")
    return v


def _opt_str(body: dict[str, Any], name: str, label: str | None = None) -> str | None:
    v = body.get(name)
    if v is None:
        return None
    if not isinstance(v, str):
        raise ApiError(400, f"'{label or name}' must be a string")
    return v


def _opt_bool(body: dict[str, Any], name: str, default: bool, label: str | None = None) -> bool:
    """A JSON boolean, strictly: "false" or 0 is rejected, never read as truthy."""
    v = body.get(name)
    if v is None:
        return default
    if not isinstance(v, bool):
        raise ApiError(400, f"'{label or name}' must be true or false")
    return v


def _opt_int(body: dict[str, Any], name: str) -> int | None:
    v = body.get(name)
    if v is None:
        return None
    if not isinstance(v, int) or isinstance(v, bool) or v < 1:
        raise ApiError(400, f"'{name}' must be a positive integer")
    return v


def _opt_dict(body: dict[str, Any], name: str) -> dict[str, Any] | None:
    v = body.get(name)
    if v is None:
        return None
    if not isinstance(v, dict):
        raise ApiError(400, f"'{name}' must be an object")
    return v


def _str_list(body: dict[str, Any], name: str, *, required: bool = True) -> list[str]:
    v = body.get(name)
    if v is None and not required:
        return []
    if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v) or (required and not v):
        raise ApiError(400, f"'{name}' must be a{' non-empty' if required else ''} list of non-empty strings")
    return v


# --------------------------------------------------------------------------- routing
_ID = r"(?P<eid>[A-Za-z0-9_\-]+)"
_REF = r"(?P<ref>[^/]+)"


def _routes(svc: Service) -> list[tuple[str, re.Pattern[str], Callable[..., Any]]]:
    r = [
        ("GET", r"/v1/health", lambda m, b, q: svc.health()),
        ("GET", r"/v1/openapi\.json", lambda m, b, q: openapi()),
        ("POST", r"/v1/items", lambda m, b, q: (201, svc.add_item(b))),
        ("GET", r"/v1/items", lambda m, b, q: svc.list_items(_int_q(q, "limit", 500, 1, 5000),
                                                               _int_q(q, "offset", 0, 0, None))),
        ("GET", rf"/v1/items/{_REF}/history", lambda m, b, q: svc.history(unquote(m["ref"]))),
        ("GET", rf"/v1/items/{_REF}", lambda m, b, q: svc.get_item(unquote(m["ref"]), _version(q))),
        ("POST", r"/v1/episodes", lambda m, b, q: (201, svc.begin(b))),
        ("POST", rf"/v1/episodes/{_ID}/recall", lambda m, b, q: svc.recall(m["eid"], b)),
        ("POST", rf"/v1/episodes/{_ID}/use", lambda m, b, q: svc.use(m["eid"], b)),
        ("POST", rf"/v1/episodes/{_ID}/expect", lambda m, b, q: (201, svc.expect(m["eid"], b))),
        ("POST", rf"/v1/episodes/{_ID}/observe", lambda m, b, q: svc.observe(m["eid"], b)),
        ("POST", rf"/v1/episodes/{_ID}/tool-result", lambda m, b, q: svc.tool_result(m["eid"], b)),
        ("POST", rf"/v1/episodes/{_ID}/correct", lambda m, b, q: svc.correct(m["eid"], b)),
        ("POST", rf"/v1/episodes/{_ID}/close", lambda m, b, q: svc.close(m["eid"])),
    ]
    return [(meth, re.compile(f"^{pat}$"), fn) for meth, pat, fn in r]


def _int_q(q: dict[str, list[str]], name: str, default: int, lo: int, hi: int | None) -> int:
    if name not in q:
        return default
    try:
        value = int(q[name][0])
    except ValueError:
        raise ApiError(400, f"'{name}' must be an integer") from None
    if value < lo or (hi is not None and value > hi):
        raise ApiError(400, f"'{name}' must be between {lo} and {hi}" if hi is not None else f"'{name}' must be >= {lo}")
    return value


def _version(q: dict[str, list[str]]) -> int | None:
    if "version" not in q:
        return None
    try:
        return int(q["version"][0])
    except ValueError:
        raise ApiError(400, "'version' must be an integer") from None


PUBLIC = {"/v1/health", "/v1/openapi.json"}


def make_handler(svc: Service, token: str | None) -> type[BaseHTTPRequestHandler]:
    routes = _routes(svc)

    class Handler(BaseHTTPRequestHandler):
        server_version = f"relock/{__version__}"
        protocol_version = "HTTP/1.1"
        timeout = SOCKET_TIMEOUT  # idle keep-alive and stalled clients are disconnected

        def log_message(self, fmt: str, *args: Any) -> None:  # route access logs to logging
            log.debug("%s - %s", self.address_string(), fmt % args)

        def _send(self, status: int, payload: Any) -> None:
            data = json.dumps(payload, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, code: str, message: str) -> None:
            self._send(status, {"error": {"code": code, "message": message}})

        def _fail(self, status: int, code: str, message: str) -> None:
            """Answer and drop the connection: the request's framing can't be trusted."""
            self.close_connection = True
            self._error(status, code, message)

        def _read_body(self) -> bytes | None:
            """The request body, or None after answering an error. Never reads past MAX_BODY."""
            te = self.headers.get("Transfer-Encoding")
            cl = self.headers.get("Content-Length")
            if te is not None:
                if cl is not None:
                    self._fail(400, "bad_request", "both Transfer-Encoding and Content-Length")
                    return None
                if te.strip().lower() != "chunked":
                    self._fail(501, "not_implemented", f"unsupported Transfer-Encoding {te!r}")
                    return None
                return self._read_chunked()
            if cl is None or not cl.strip():
                return b""
            if not cl.strip().isdigit():
                self._fail(400, "bad_request", "invalid Content-Length")
                return None
            length = int(cl)
            if length > MAX_BODY:
                # Drain a moderately oversized body so the client reads the 413, not a reset.
                if length <= 8 * MAX_BODY:
                    while length > 0:
                        chunk = self.rfile.read(min(length, 1 << 16))
                        if not chunk:
                            break
                        length -= len(chunk)
                self._fail(413, "too_large", f"request body over {MAX_BODY} bytes")
                return None
            return self.rfile.read(length) if length else b""

        def _read_chunked(self) -> bytes | None:
            out = bytearray()
            while True:
                line = self.rfile.readline(1024)
                size_s = line.split(b";", 1)[0].strip()
                if not line.endswith(b"\n") or not size_s or any(c not in b"0123456789abcdefABCDEF" for c in size_s):
                    self._fail(400, "bad_request", "malformed chunked body")
                    return None
                size = int(size_s, 16)
                if size == 0:
                    for _ in range(64):  # trailer fields, then the blank line
                        if self.rfile.readline(1024) in (b"\r\n", b"\n", b""):
                            return bytes(out)
                    self._fail(400, "bad_request", "too many trailer fields")
                    return None
                if len(out) + size > MAX_BODY:
                    self._fail(413, "too_large", f"request body over {MAX_BODY} bytes")
                    return None
                data = self.rfile.read(size)
                if len(data) < size or self.rfile.readline(4) not in (b"\r\n", b"\n"):
                    self._fail(400, "bad_request", "malformed chunked body")
                    return None
                out += data

        def _dispatch(self, method: str) -> None:
            path = urlsplit(self.path)
            raw = self._read_body()
            if raw is None:
                return None
            if token and path.path not in PUBLIC:
                given = self.headers.get("Authorization", "")
                if not (given.startswith("Bearer ") and hmac.compare_digest(given[7:].encode(), token.encode())):
                    return self._error(401, "unauthorized", "missing or wrong bearer token")
            allowed = []
            for meth, rx, fn in routes:
                m = rx.match(path.path)
                if not m:
                    continue
                if meth != method:
                    allowed.append(meth)
                    continue
                try:
                    body = json.loads(raw) if raw.strip() else {}
                    if not isinstance(body, dict):
                        raise ApiError(400, "request body must be a JSON object")
                    out = fn(m.groupdict(), body, parse_qs(path.query))
                except json.JSONDecodeError as e:
                    return self._error(400, "bad_json", f"invalid JSON: {e}")
                except RecursionError:
                    return self._error(400, "bad_json", "JSON nested too deeply")
                except ApiError as e:
                    return self._error(e.status, e.code, str(e))
                except NotFoundError as e:
                    return self._error(404, "not_found", str(e).strip("'\""))
                except (NotLabileError, EpisodeClosedError, ConflictError) as e:
                    return self._error(409, type(e).__name__, str(e))
                except (RelockError, ValueError, TypeError) as e:
                    return self._error(400, "bad_request", str(e))
                except Exception:
                    log.exception("unhandled error on %s %s", method, path.path)
                    return self._error(500, "internal", "internal server error")
                status, payload = out if isinstance(out, tuple) else (200, out)
                return self._send(status, payload)
            if allowed:
                return self._error(405, "method_not_allowed", f"use {' or '.join(sorted(set(allowed)))}")
            return self._error(404, "no_route", f"no route for {method} {path.path}")

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

    return Handler


class _Server(ThreadingHTTPServer):
    """A thread per connection, at most ``max_connections`` at once (the rest wait to be accepted)."""

    daemon_threads = True

    def __init__(self, addr: tuple[str, int], handler: type[BaseHTTPRequestHandler], *,
                 max_connections: int = MAX_CONNECTIONS) -> None:
        self._slots = threading.BoundedSemaphore(max_connections)
        super().__init__(addr, handler)

    def process_request(self, request: Any, client_address: Any) -> None:
        self._slots.acquire()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def make_server(memory: Memory, host: str = "127.0.0.1", port: int = 8765, *, token: str | None = None,
                idle_timeout: float | None = 1800.0, k: int = 3, max_connections: int = MAX_CONNECTIONS,
                ) -> tuple[ThreadingHTTPServer, Service]:
    """Build (but don't start) the server; idle episodes are swept in the background.

    Refuses a non-loopback bind without a token.
    """
    if not token and not _is_loopback(host):
        raise ValueError(f"refusing to listen on {host!r} without a token: set RELOCK_TOKEN "
                         "(anyone who can reach this port could rewrite your agent's memory)")
    svc = Service(memory, idle_timeout=idle_timeout, k=k)
    server = _Server((host, port), make_handler(svc, token), max_connections=max_connections)
    svc.start_sweeper()
    return server, svc


def serve(db: str | None = None, *, host: str = "127.0.0.1", port: int = 8765, token: str | None = None,
          rewriter: Any = None, idle_timeout: float | None = 1800.0, k: int = 3) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    memory = Memory(db, rewriter=rewriter)
    server, svc = make_server(memory, host, port, token=token, idle_timeout=idle_timeout, k=k)
    where = f"http://{host}:{server.server_address[1]}"
    print(f"relock {__version__} serving {where}/v1 (OpenAPI: {where}/v1/openapi.json)"
          f"{'' if token else ' - no token: loopback only'}", flush=True)

    def _terminate(*_: Any) -> None:  # docker stop / systemd: re-lock and close cleanly
        raise KeyboardInterrupt
    try:
        _signal.signal(_signal.SIGTERM, _terminate)
    except ValueError:  # pragma: no cover - not the main thread
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        svc.close_all()
        memory.close()


# --------------------------------------------------------------------------- OpenAPI
def openapi() -> dict[str, Any]:
    """OpenAPI 3.1 description of the API (also committed as ``docs/openapi.json``)."""
    S = {"type": "string"}
    strs = {"type": "array", "items": S}

    def ref(name: str) -> dict[str, str]:
        return {"$ref": f"#/components/schemas/{name}"}

    def body(schema: dict[str, Any]) -> dict[str, Any]:
        return {"required": True, "content": {"application/json": {"schema": schema}}}

    def ok(schema: dict[str, Any], code: str = "200", desc: str = "OK") -> dict[str, Any]:
        return {code: {"description": desc, "content": {"application/json": {"schema": schema}}},
                "default": {"description": "Error", "content": {"application/json": {"schema": ref("Error")}}}}

    eid = [{"name": "episode_id", "in": "path", "required": True, "schema": S}]
    iref = [{"name": "ref", "in": "path", "required": True, "schema": S, "description": "Item key or id."}]
    outcome = {
        "type": "object", "required": ["ok"],
        "properties": {
            "ok": {"type": "boolean", "description": "Did the call succeed / the prediction hold?"},
            "source": {"type": "string", "enum": ["tool", "verifier", "agent", "environment"], "default": "tool",
                       "description": "Who observed the outcome. tool and verifier are trusted; agent and "
                                      "environment are logged but never rewrite memory (by default)."},
            "error": {"oneOf": [S, ref("ErrorDetail")]},
            "verifier": {**S, "description": "Verifier name, for source=verifier."},
        },
    }
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "relock",
            "version": __version__,
            "summary": "Agent memory that fixes the rule that fired, in place and only on trusted "
                       "evidence, so corrected mistakes don't come back.",
            "description": (
                "Language-neutral API for relock. Call it from your harness, the code that runs tools "
                "and checks results, never from the model. Loop: POST /episodes, /recall (put "
                "`context` in the prompt), report each tool call with /tool-result, then /close. "
                "Authenticate with `Authorization: Bearer $RELOCK_TOKEN` when the server has a token."),
            "license": {"name": "Apache-2.0", "identifier": "Apache-2.0"},
        },
        "servers": [{"url": "http://127.0.0.1:8765"}],
        "components": {
            "securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}},
            "schemas": {
                "Error": {"type": "object", "properties": {"error": {"type": "object", "properties": {
                    "code": S, "message": S}}}},
                "Clause": {"type": "object", "properties": {"id": S, "text": S}},
                "Item": {"type": "object", "properties": {
                    "ref": S, "id": S, "key": {"type": ["string", "null"]}, "version": {"type": "integer"},
                    "status": S, "title": S, "kind": S, "clauses": {"type": "array", "items": ref("Clause")},
                    "tags": strs, "tools": strs, "meta": {"type": "object"}, "confidence": {"type": "number"},
                    "confirmations": {"type": "integer"}, "failures": {"type": "integer"},
                    "change_kind": S, "reason": {"type": ["string", "null"]}, "author": {"type": ["string", "null"]},
                    "created_at": S, "archived": {"type": "boolean"}}},
                "ErrorDetail": {"type": "object", "properties": {
                    "message": S,
                    "correction": {**S, "description": "The rule that would have been right, e.g. "
                                                        "'Pass dates as YYYY-MM-DD.' Drives the rewrite."},
                    "status": {"type": "integer", "minimum": 100, "maximum": 599,
                               "description": "HTTP status of the failed call, if any. "
                                              "408/425/429/5xx count as transient."},
                    "transient": {"type": "boolean", "description": "Timeouts, rate limits: never rewrite."},
                    "class": {"type": "string", "enum": [c.value for c in ErrorClass]},
                    "detail_trusted": {"type": "boolean", "default": True,
                                       "description": "false if message/correction echo remote content; "
                                                      "the failure still counts but its text never reaches "
                                                      "the rewriter."}}},
                "Outcome": outcome,
                "Decision": {"type": "object", "properties": {
                    "expectation_id": S, "matched": {"type": "boolean"}, "reason": S,
                    "verdict": {"type": "string", "enum": [v.value for v in Verdict]},
                    "item_id": S, "version": {"type": "integer"}, "new_version": {"type": ["integer", "null"]},
                    "evidence": {"type": "number"}, "required": {"type": "number"}, "delta": {}}},
            },
        },
        "security": [{"bearer": []}],
        "paths": {
            "/v1/health": {"get": {"summary": "Liveness and version", "security": [], "responses": ok(
                {"type": "object", "properties": {"ok": {"type": "boolean"}, "version": S,
                                                  "open_episodes": {"type": "integer"}}})}},
            "/v1/openapi.json": {"get": {"summary": "This document", "security": [],
                                         "responses": ok({"type": "object"})}},
            "/v1/items": {
                "get": {"summary": "List active items, ordered by id, a page at a time", "parameters": [
                    {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 500, "minimum": 1,
                                                                "maximum": 5000}},
                    {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0, "minimum": 0}}],
                    "responses": ok({"type": "object", "properties": {
                        "items": {"type": "array", "items": ref("Item")}, "total": {"type": "integer"},
                        "next_offset": {"type": ["integer", "null"], "description": "null on the last page"}}})},
                "post": {"summary": "Add a procedure", "requestBody": body({
                    "type": "object", "required": ["title", "clauses"], "properties": {
                        "title": S, "clauses": strs, "key": S, "kind": S, "tags": strs,
                        "tools": {**strs, "description": "Tools this item governs; their outcomes resolve it."},
                        "meta": {"type": "object"}, "exist_ok": {"type": "boolean", "default": False}}}),
                    "responses": ok(ref("Item"), "201", "Created")},
            },
            "/v1/items/{ref}": {"get": {"summary": "Get the active (or a given) version", "parameters": [
                *iref, {"name": "version", "in": "query", "schema": {"type": "integer"}}], "responses": ok(ref("Item"))}},
            "/v1/items/{ref}/history": {"get": {"summary": "All versions, oldest first", "parameters": iref,
                                                "responses": ok({"type": "object", "properties": {
                                                    "versions": {"type": "array", "items": ref("Item")}}})}},
            "/v1/episodes": {"post": {"summary": "Open an episode (a reconsolidation window)", "requestBody": body(
                {"type": "object", "properties": {"task": S, "meta": {"type": "object"}}}),
                "responses": ok({"type": "object", "properties": {"episode_id": S}}, "201", "Created")}},
            "/v1/episodes/{episode_id}/recall": {"post": {
                "summary": "Retrieve procedures; recalled items become editable in this episode",
                "parameters": eid, "requestBody": body({"type": "object", "required": ["query"], "properties": {
                    "query": S, "k": {"type": "integer", "minimum": 1},
                    "budget_tokens": {"type": "integer", "minimum": 1}}}),
                "responses": ok({"type": "object", "properties": {
                    "context": {**S, "description": "Prompt block to give the model."},
                    "items": {"type": "array", "items": ref("Item")}}})}},
            "/v1/episodes/{episode_id}/use": {"post": {
                "summary": "Load an item by key (also makes it editable)", "parameters": eid,
                "requestBody": body({"type": "object", "required": ["item"], "properties": {"item": S}}),
                "responses": ok(ref("Item"))}},
            "/v1/episodes/{episode_id}/expect": {"post": {
                "summary": "Record a prediction before acting (optional; tool-result creates one if absent)",
                "parameters": eid, "requestBody": body({"type": "object", "required": ["item"], "properties": {
                    "item": S, "prediction": S, "clauses": {**strs, "description": "Clause ids relied on; "
                                                                               "limits what can be rewritten."},
                    "tool": {**S, "description": "Bind to a tool so /tool-result resolves it."}}}),
                "responses": ok({"type": "object", "properties": {"expectation_id": S}}, "201", "Created")}},
            "/v1/episodes/{episode_id}/observe": {"post": {
                "summary": "Resolve one expectation with an outcome", "parameters": eid,
                "requestBody": body({"allOf": [outcome, {"type": "object", "required": ["expectation_id"],
                                                         "properties": {"expectation_id": S}}]}),
                "responses": ok(ref("Decision"))}},
            "/v1/episodes/{episode_id}/tool-result": {"post": {
                "summary": "Report a tool call's real outcome (what @mem.wrap does in Python)",
                "parameters": eid,
                "requestBody": body({"allOf": [outcome, {"type": "object", "required": ["tool"], "properties": {
                    "tool": S, "result": {}, "items": {**strs, "description": "Attribute to these items "
                                                                         "instead of by tool name."}}}]}),
                "responses": ok({"type": "object", "properties": {
                    "observations": {"type": "array", "items": ref("Decision")}}})}},
            "/v1/episodes/{episode_id}/correct": {"post": {
                "summary": "A human correction (rewrites only with authenticated=true)", "parameters": eid,
                "requestBody": body({"type": "object", "required": ["item", "correction"], "properties": {
                    "item": S, "correction": S, "clauses": strs, "principal": S,
                    "authenticated": {"type": "boolean", "default": False,
                                      "description": "Set only after your app verified the user."}}}),
                "responses": ok(ref("Decision"))}},
            "/v1/episodes/{episode_id}/close": {"post": {
                "summary": "Re-lock: the episode's items are no longer editable", "parameters": eid,
                "responses": ok({"type": "object"})}},
        },
    }


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(openapi(), indent=2))

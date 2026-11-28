"""The HTTP/JSON API, exercised over real sockets."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import relock_memory as rl
from relock_memory.integrations.http import Service, make_server, openapi

TOKEN = "test-token"


@pytest.fixture
def api(tmp_path):
    mem = rl.Memory(str(tmp_path / "m.db"))
    server, svc = make_server(mem, "127.0.0.1", 0, token=TOKEN)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(method: str, path: str, body=None, *, token: str | None = TOKEN, raw: bytes | None = None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(base + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    call.svc = svc  # type: ignore[attr-defined]
    call.port = server.server_address[1]  # type: ignore[attr-defined]
    call.mem = mem  # type: ignore[attr-defined]
    yield call
    server.shutdown()
    server.server_close()
    svc.close_all()
    mem.close()


def _seed(call):
    st, item = call("POST", "/v1/items", {
        "title": "Fetch orders by date", "key": "orders.fetch", "tools": ["get_orders"],
        "clauses": ["Authenticate with the orders API key.", "Pass the date to get_orders as MM/DD/YYYY."]})
    assert st == 201 and item["ref"] == "orders.fetch" and item["version"] == 1
    return item


def test_full_loop_rewrites_in_place_over_http(api):
    _seed(api)
    fix = "Pass the date to get_orders as YYYY-MM-DD."
    verdicts = []
    for _ in range(2):
        st, ep = api("POST", "/v1/episodes", {"task": "orders for March 14"})
        assert st == 201
        eid = ep["episode_id"]
        st, rec = api("POST", f"/v1/episodes/{eid}/recall", {"query": "fetch orders placed on a date"})
        assert st == 200 and "MM/DD/YYYY" in rec["context"] and rec["items"][0]["ref"] == "orders.fetch"
        st, out = api("POST", f"/v1/episodes/{eid}/tool-result", {
            "tool": "get_orders", "ok": False,
            "error": {"message": "invalid date '03/14/2026'", "status": 422, "correction": fix}})
        assert st == 200 and len(out["observations"]) == 1
        verdicts.append(out["observations"][0]["verdict"])
        st, summary = api("POST", f"/v1/episodes/{eid}/close")
        assert st == 200 and summary["recalled"] == ["orders.fetch"]
    assert verdicts == ["pending", "rewritten"]
    st, item = api("GET", "/v1/items/orders.fetch")
    assert item["version"] == 2 and fix in [c["text"] for c in item["clauses"]]
    st, hist = api("GET", "/v1/items/orders.fetch/history")
    assert [v["status"] for v in hist["versions"]] == ["superseded", "active"]
    # closed episodes are re-locked
    st, err = api("POST", f"/v1/episodes/{eid}/recall", {"query": "orders"})
    assert st == 404 and err["error"]["code"] == "episode_not_found"


def test_transient_and_untrusted_outcomes_never_rewrite(api):
    _seed(api)
    st, ep = api("POST", "/v1/episodes", {})
    eid = ep["episode_id"]
    api("POST", f"/v1/episodes/{eid}/recall", {"query": "fetch orders by date"})
    for body in (
        {"tool": "get_orders", "ok": False, "error": {"message": "upstream", "status": 503}},
        {"tool": "get_orders", "ok": False, "error": {"message": "timeout", "transient": True}},
    ):
        st, out = api("POST", f"/v1/episodes/{eid}/tool-result", body)
        assert out["observations"][0]["verdict"] == "blocked"
    for source in ("agent", "environment"):
        st, e = api("POST", f"/v1/episodes/{eid}/expect", {"item": "orders.fetch", "prediction": "rows"})
        st, out = api("POST", f"/v1/episodes/{eid}/observe", {
            "expectation_id": e["expectation_id"], "ok": False, "source": source,
            "error": {"message": "x", "correction": "Pass dates as DD.MM.YYYY."}})
        assert st == 200 and out["verdict"] in ("blocked", "ignored")
    # an unauthenticated user correction is logged, not applied; an authenticated one rewrites
    st, out = api("POST", f"/v1/episodes/{eid}/correct", {"item": "orders.fetch",
                                                           "correction": "Pass dates as DD.MM.YYYY."})
    assert out["verdict"] == "blocked"
    st, out = api("POST", f"/v1/episodes/{eid}/correct", {
        "item": "orders.fetch", "correction": "Pass the date to get_orders as YYYY-MM-DD.",
        "authenticated": True, "principal": "ops@example.com"})
    assert out["verdict"] == "rewritten"
    st, item = api("GET", "/v1/items/orders.fetch")
    assert "DD.MM.YYYY" not in json.dumps(item)


def test_success_confirms_and_verifier_source(api):
    _seed(api)
    st, ep = api("POST", "/v1/episodes", {"task": "t"})
    eid = ep["episode_id"]
    api("POST", f"/v1/episodes/{eid}/use", {"item": "orders.fetch"})
    st, out = api("POST", f"/v1/episodes/{eid}/tool-result", {"tool": "get_orders", "ok": True, "result": [1]})
    assert out["observations"][0]["verdict"] == "confirmed"
    st, e = api("POST", f"/v1/episodes/{eid}/expect", {"item": "orders.fetch", "clauses": ["c2"]})
    assert st == 201
    st, out = api("POST", f"/v1/episodes/{eid}/observe", {"expectation_id": e["expectation_id"], "ok": False,
                                                           "source": "verifier", "verifier": "tests",
                                                           "error": "3 tests failed"})
    assert out["verdict"] == "pending"
    # a tool no recalled item governs resolves nothing
    st, out = api("POST", f"/v1/episodes/{eid}/tool-result", {"tool": "send_email", "ok": False})
    assert st == 200 and out["observations"] == []


def test_auth_and_errors(api):
    assert api("GET", "/v1/health", token=None)[0] == 200
    assert api("GET", "/v1/openapi.json", token=None)[0] == 200
    st, err = api("GET", "/v1/items", token=None)
    assert st == 401 and err["error"]["code"] == "unauthorized"
    assert api("GET", "/v1/items", token="wrong")[0] == 401
    assert api("GET", "/v1/nope")[0] == 404
    assert api("GET", "/v1/episodes")[0] == 405
    assert api("POST", "/v1/items", raw=b"{not json")[1]["error"]["code"] == "bad_json"
    assert api("POST", "/v1/items", raw=b"[1]")[0] == 400
    assert api("POST", "/v1/items", {"title": "x"})[0] == 400
    assert api("POST", "/v1/items", {"title": "x", "clauses": [""]})[0] == 400
    assert api("GET", "/v1/items/missing")[0] == 404
    st, ep = api("POST", "/v1/episodes", {})
    eid = ep["episode_id"]
    st, err = api("POST", f"/v1/episodes/{eid}/expect", {"item": "missing"})
    assert st == 404
    _seed(api)
    st, err = api("POST", f"/v1/episodes/{eid}/expect", {"item": "orders.fetch"})
    assert st == 409 and err["error"]["code"] == "NotLabileError"
    st, err = api("POST", f"/v1/episodes/{eid}/tool-result", {"tool": "get_orders", "ok": "no"})
    assert st == 400
    st, err = api("POST", f"/v1/episodes/{eid}/tool-result", {"tool": "get_orders", "ok": False, "source": "user"})
    assert st == 400 and "/correct" in err["error"]["message"]
    assert api("POST", "/v1/items", raw=b" " * ((1 << 20) + 1))[0] == 413


def test_concurrent_episodes(api):
    _seed(api)
    errors = []

    def worker(i: int) -> None:
        try:
            st, ep = api("POST", "/v1/episodes", {"task": f"t{i}"})
            eid = ep["episode_id"]
            st, _rec = api("POST", f"/v1/episodes/{eid}/recall", {"query": "orders by date"})
            assert st == 200
            st, out = api("POST", f"/v1/episodes/{eid}/tool-result", {"tool": "get_orders", "ok": True})
            assert out["observations"][0]["verdict"] == "confirmed"
            assert api("POST", f"/v1/episodes/{eid}/close")[0] == 200
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    _st, item = api("GET", "/v1/items/orders.fetch")
    assert item["confirmations"] == 16 and item["version"] == 1


def test_idle_episodes_are_relocked(tmp_path):
    mem = rl.Memory(":memory:")
    svc = Service(mem, idle_timeout=0.05)
    eid = svc.begin({})["episode_id"]
    time.sleep(0.1)
    svc.begin({})  # any new episode sweeps idle ones
    with pytest.raises(Exception, match="no open episode"):
        svc.recall(eid, {"query": "x"})
    mem.close()


def test_refuses_public_bind_without_token():
    mem = rl.Memory(":memory:")
    with pytest.raises(ValueError, match="RELOCK_TOKEN"):
        make_server(mem, "0.0.0.0", 0, token=None)
    mem.close()


def test_committed_openapi_matches_the_server():
    committed = json.loads(Path(__file__).resolve().parents[1].joinpath("docs", "openapi.json").read_text("utf-8"))
    current = openapi()
    committed["info"]["version"] = current["info"]["version"]  # the version moves with every release
    assert committed == current, "regenerate with: python -m relock_memory.integrations.http > docs/openapi.json"


# ---------------------------------------------------------------- hardening (each test is a past defect)
FIX = "Pass the date to get_orders as YYYY-MM-DD."


def _open(api):
    _seed(api)
    eid = api("POST", "/v1/episodes", {})[1]["episode_id"]
    api("POST", f"/v1/episodes/{eid}/recall", {"query": "fetch orders by date"})
    return eid


def _raw(port: int, data: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(data)
        s.shutdown(socket.SHUT_WR)
        out = b""
        while chunk := s.recv(65536):
            out += chunk
        return out


def _try(fn):
    try:
        return fn()
    except Exception as e:
        return e


def test_untrusted_success_never_confirms_or_cancels_evidence(api):
    eid = _open(api)
    fail = {"tool": "get_orders", "ok": False, "error": {"message": "bad date", "status": 422, "correction": FIX}}
    assert api("POST", f"/v1/episodes/{eid}/tool-result", fail)[1]["observations"][0]["verdict"] == "pending"
    for source in ("agent", "environment"):
        st, out = api("POST", f"/v1/episodes/{eid}/tool-result", {"tool": "get_orders", "ok": True, "source": source})
        assert st == 200 and out["observations"][0]["verdict"] == "ignored"
    assert api("GET", "/v1/items/orders.fetch")[1]["confirmations"] == 0
    # the trusted failure is still pending, so the next one rewrites
    assert api("POST", f"/v1/episodes/{eid}/tool-result", fail)[1]["observations"][0]["verdict"] == "rewritten"


@pytest.mark.parametrize(("path", "body"), [
    ("correct", {"item": "orders.fetch", "correction": "Pass dates as DD.MM.YYYY.", "authenticated": "false"}),
    ("correct", {"item": "orders.fetch", "correction": "Pass dates as DD.MM.YYYY.", "authenticated": 1}),
    ("tool-result", {"tool": "get_orders", "ok": False, "error": {"message": "x", "detail_trusted": "false"}}),
    ("tool-result", {"tool": "get_orders", "ok": False, "error": {"message": "x", "transient": "yes"}}),
    ("tool-result", {"tool": "get_orders", "ok": False, "error": {"message": "x", "status": "503"}}),
    ("tool-result", {"tool": "get_orders", "ok": False, "error": {"message": "x", "status": 503.0}}),
    ("tool-result", {"tool": "get_orders", "ok": False, "error": {"message": "x", "status": True}}),
    ("tool-result", {"tool": "get_orders", "ok": False, "error": {"message": "x", "status": 999}}),
    ("tool-result", {"tool": "get_orders", "ok": False, "error": {"message": "x", "class": 3}}),
    ("tool-result", {"tool": "get_orders", "ok": False, "source": 7}),
])
def test_json_types_are_strict(api, path, body):
    eid = _open(api)
    st, err = api("POST", f"/v1/episodes/{eid}/{path}", body)
    assert st == 400, err
    assert api("GET", "/v1/items/orders.fetch")[1]["version"] == 1
    assert api("POST", "/v1/items", {"title": "t", "clauses": ["c"], "key": "k", "exist_ok": "true"})[0] == 400


def test_every_5xx_is_transient(api):
    eid = _open(api)
    for status in (501, 505, 530, 599):
        body = {"tool": "get_orders", "ok": False,
                "error": {"message": "server error", "status": status, "correction": "Pass dates as DD.MM.YYYY."}}
        assert api("POST", f"/v1/episodes/{eid}/tool-result", body)[1]["observations"][0]["verdict"] == "blocked"
    assert api("GET", "/v1/items/orders.fetch")[1]["version"] == 1
    assert rl.classify_exception(type("E", (Exception,), {"status_code": 501})()) is rl.ErrorClass.TRANSIENT


def test_request_framing(api):
    port = api.port
    head = f"POST /v1/episodes HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {TOKEN}\r\n".encode()
    # a negative or malformed length is refused and the connection dropped, before any body is read
    for cl in (b"-1", b"12x", b"+5"):
        out = _raw(port, head + b"Content-Length: " + cl + b"\r\n\r\n{}")
        assert out.startswith(b"HTTP/1.1 400") and out.count(b"HTTP/1.1") == 1, out[:80]
    # both framings at once is refused (request smuggling)
    out = _raw(port, head + b"Content-Length: 4\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n")
    assert out.startswith(b"HTTP/1.1 400")
    # chunked bodies, as streaming clients send them, are read in full
    body = json.dumps({"task": "chunked task"}).encode()
    a, b = body[:5], body[5:]
    out = _raw(port, head + b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
               + f"{len(a):x}\r\n".encode() + a + b"\r\n" + f"{len(b):x}\r\n".encode() + b + b"\r\n0\r\n\r\n")
    assert out.startswith(b"HTTP/1.1 201") and out.count(b"HTTP/1.1") == 1, out[:120]
    # an oversized chunk is refused without reading it
    out = _raw(port, head + b"Transfer-Encoding: chunked\r\n\r\n" + f"{(1 << 20) + 1:x}\r\n".encode())
    assert out.startswith(b"HTTP/1.1 413")
    # deeply nested JSON is a 400, not a 500
    assert api("POST", "/v1/items", raw=b"[" * 100_000 + b"]" * 100_000)[0] == 400


def test_stalled_connections_are_dropped(monkeypatch):
    import relock_memory.integrations.http as h

    monkeypatch.setattr(h, "SOCKET_TIMEOUT", 0.3)
    mem = rl.Memory(":memory:")
    server, svc = h.make_server(mem, "127.0.0.1", 0, token=TOKEN)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=5) as s:
            s.sendall(b"POST /v1/episodes HTTP/1.1\r\n")  # then nothing: a slow or dead client
            t = time.monotonic()
            assert s.recv(1024) == b""  # the server hangs up
            assert time.monotonic() - t < 3
    finally:
        server.shutdown()
        server.server_close()
        svc.close_all()
        mem.close()


def test_close_racing_a_write_never_rewrites_after_relock():
    mem = rl.Memory(":memory:")
    mem.add("Fetch orders", ["Pass the date as MM/DD/YYYY."], key="orders", tools=["get_orders"])
    ep = mem.episode("t")
    ep.use("orders")
    ep.observe(ep.expect("orders"), error=rl.ToolError("bad date", correction="Pass the date as YYYY-MM-DD."))
    exp = ep.expect("orders")
    outcome: list[object] = []
    with ep._lock:  # hold the episode: the writer passes its first check, then waits here
        writer = threading.Thread(target=lambda: outcome.append(_try(lambda: ep.observe(
            exp, error=rl.ToolError("bad date", correction="Pass the date as YYYY-MM-DD.")))))
        writer.start()
        time.sleep(0.2)
        ep.close()  # re-entrant: lands while the write is in flight
    writer.join(5)
    assert isinstance(outcome[0], rl.EpisodeClosedError)
    assert mem.get("orders").version == 1
    mem.close()


def test_concurrent_tool_results_are_all_recorded():
    mem = rl.Memory(":memory:")
    mem.add("Fetch orders", ["Pass the date as MM/DD/YYYY."], key="orders", tools=["get_orders"])
    ep = mem.episode("t")
    ep.use("orders")
    ep.expect("orders", "rows come back", tool="get_orders")
    barrier = threading.Barrier(2)
    results: list[object] = []

    def report(sig):
        barrier.wait()
        results.append(_try(lambda: ep.tool_result("get_orders", signal=sig)))
    threads = [threading.Thread(target=report, args=(s,)) for s in (
        rl.Signal.ok(), rl.Signal.tool_error("bad date", error_class=rl.ErrorClass.PROGRAMMATIC))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert all(isinstance(r, list) and len(r) == 1 for r in results), results
    assert sorted(r[0].matched for r in results) == [False, True]
    mem.close()


def test_idle_episodes_expire_without_new_traffic():
    mem = rl.Memory(":memory:")
    svc = Service(mem, idle_timeout=0.1)
    eid = svc.begin({})["episode_id"]
    time.sleep(0.25)  # no sweep ran: the next touch expires it
    with pytest.raises(Exception, match="no open episode"):
        svc.recall(eid, {"query": "x"})
    svc.start_sweeper()  # and the background sweeper re-locks episodes nobody touches again
    svc.begin({})
    deadline = time.monotonic() + 5
    while svc.health()["open_episodes"] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert svc.health()["open_episodes"] == 0
    svc.close_all()
    mem.close()


def test_items_are_listed_a_page_at_a_time(api):
    for i in range(7):
        assert api("POST", "/v1/items", {"title": f"t{i}", "clauses": ["c"], "key": f"k{i}"})[0] == 201
    seen, offset = [], 0
    while offset is not None:
        st, page = api("GET", f"/v1/items?limit=3&offset={offset}")
        assert st == 200 and page["total"] == 7 and len(page["items"]) <= 3
        seen += [i["key"] for i in page["items"]]
        offset = page["next_offset"]
    assert sorted(seen) == [f"k{i}" for i in range(7)] and len(set(seen)) == 7
    assert api("GET", "/v1/items?limit=0")[0] == 400
    assert api("GET", "/v1/items?limit=x")[0] == 400
    assert api("GET", "/v1/items?offset=-1")[0] == 400

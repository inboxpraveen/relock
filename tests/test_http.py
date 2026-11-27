"""The HTTP/JSON API, exercised over real sockets."""

from __future__ import annotations

import json
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

"""OpenAICompatible against a local stub server (no network, no model)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import relock_memory as rl
from relock_memory.llm import OpenAICompatible


class _Stub:
    def __init__(self, script):
        self.script = list(script)  # list of (status, body) served in order; last one repeats
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                stub.requests.append({"path": self.path, "headers": dict(self.headers),
                                      "body": json.loads(self.rfile.read(length))})
                status, body = stub.script.pop(0) if len(stub.script) > 1 else stub.script[0]
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def _reply(text):
    return (200, {"choices": [{"message": {"role": "assistant", "content": text}}]})


def test_request_shape_and_reply():
    with _Stub([_reply("hello")]) as stub:
        llm = OpenAICompatible(stub.url, model="qwen3-30b", api_key="sk-test", system="be brief",
                               max_tokens=77, json_mode=True, extra_body={"top_p": 0.9})
        assert llm("hi there") == "hello"
    req = stub.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer sk-test"
    body = req["body"]
    assert body["model"] == "qwen3-30b" and body["max_tokens"] == 77 and body["temperature"] == 0.0
    assert body["messages"] == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi there"}]
    assert body["response_format"] == {"type": "json_object"} and body["top_p"] == 0.9


def test_no_auth_header_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RELOCK_LLM_API_KEY", raising=False)
    with _Stub([_reply("ok")]) as stub:
        OpenAICompatible(stub.url, model="m")("x")
    assert "Authorization" not in stub.requests[0]["headers"]


def test_transient_status_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr("relock_memory.llm.time.sleep", lambda s: None)
    with _Stub([(503, {"error": "overloaded"}), _reply("recovered")]) as stub:
        assert OpenAICompatible(stub.url, model="m", retries=2)("x") == "recovered"
    assert len(stub.requests) == 2


def test_persistent_503_raises_transient(monkeypatch):
    monkeypatch.setattr("relock_memory.llm.time.sleep", lambda s: None)
    with _Stub([(503, {"error": "overloaded"})]) as stub, pytest.raises(rl.TransientToolError):
        OpenAICompatible(stub.url, model="m", retries=1)("x")
    assert len(stub.requests) == 2
    assert rl.classify_exception(rl.TransientToolError("x")) is rl.ErrorClass.TRANSIENT


def test_client_error_is_programmatic_and_not_retried():
    with _Stub([(400, {"error": "bad model"})]) as stub, pytest.raises(rl.ToolError) as info:
        OpenAICompatible(stub.url, model="m", retries=3)("x")
    assert not isinstance(info.value, rl.TransientToolError)
    assert len(stub.requests) == 1


def test_unreachable_endpoint_is_transient(monkeypatch):
    monkeypatch.setattr("relock_memory.llm.time.sleep", lambda s: None)
    with pytest.raises(rl.TransientToolError):
        OpenAICompatible("http://127.0.0.1:9/v1", model="m", retries=0, timeout=2)("x")


def test_end_to_end_llm_rewrite_through_http(tmp_path):
    reply = json.dumps({"ops": [{"op": "replace", "id": "c1", "text": "Call get with YYYY-MM-DD."}],
                        "rationale": "iso"})
    with _Stub([_reply(f"<think>the date format is wrong</think>{reply}")]) as stub:
        llm = OpenAICompatible(stub.url, model="m")
        mem = rl.Memory(str(tmp_path / "m.db"), rewriter=rl.LLMRewriter(llm), policy=rl.WritePolicy.eager())
        item = mem.add("Fetch", ["Call get with MM/DD/YYYY."], key="f")
        with mem.episode("t") as ep:
            ep.recall("fetch get")
            obs = ep.observe(ep.expect(item, "get returns rows", clauses=["c1"]),
                             error=rl.ToolError("ValidationError: date must be ISO-8601"))
        assert obs.verdict is rl.Verdict.REWRITTEN
        assert mem.get("f").clause("c1").text == "Call get with YYYY-MM-DD."
        assert mem.get("f").author == "llm:m"
        assert "[c1] Call get with MM/DD/YYYY." in stub.requests[0]["body"]["messages"][-1]["content"]
        mem.close()

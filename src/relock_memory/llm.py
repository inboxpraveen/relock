"""A tiny, dependency-free client for OpenAI-compatible chat endpoints.

Works with vLLM, SGLang, Ollama (``http://localhost:11434/v1``), llama.cpp
server, LM Studio, TGI, OpenRouter and OpenAI itself. Anything else can be
plugged in as a plain ``Callable[[str], str]``.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from .signals import ToolError, TransientToolError

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by reasoning models (Qwen3, DeepSeek-R1...)."""
    text = _THINK.sub("", text)
    if "</think>" in text.lower():  # unterminated opening tag cut off by the server
        text = text[text.lower().rindex("</think>") + len("</think>"):]
    return text.strip()


def extract_json(text: str) -> Any:
    """Return the first JSON object or array found in ``text`` (tolerates fences and chatter)."""
    text = strip_reasoning(text)
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start = cand.find(opener)
            while start != -1:
                depth, in_str, esc = 0, False, False
                for i in range(start, len(cand)):
                    ch = cand[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                        continue
                    if ch == '"':
                        in_str = True
                    elif ch == opener:
                        depth += 1
                    elif ch == closer:
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(cand[start : i + 1])
                            except json.JSONDecodeError:
                                break
                start = cand.find(opener, start + 1)
    raise ValueError("no JSON object found in model output")


class OpenAICompatible:
    """Callable chat client: ``llm("prompt") -> "reply"``.

    >>> llm = OpenAICompatible("http://localhost:8000/v1", model="Qwen/Qwen3-30B-A3B-Instruct-2507")
    >>> llm("Say hi")  # doctest: +SKIP
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        system: str | None = None,
        json_mode: bool = False,
        retries: int = 2,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("RELOCK_LLM_BASE_URL") or "http://localhost:8000/v1").rstrip("/")
        self.model = model or os.environ.get("RELOCK_LLM_MODEL") or "default"
        self.api_key = api_key or os.environ.get("RELOCK_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.system = system
        self.json_mode = json_mode
        self.retries = retries
        self.extra_body = extra_body or {}

    def __repr__(self) -> str:
        return f"OpenAICompatible({self.base_url!r}, model={self.model!r})"

    def chat(self, messages: list[dict[str, str]], **overrides: Any) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": overrides.pop("temperature", self.temperature),
            "max_tokens": overrides.pop("max_tokens", self.max_tokens),
        }
        if overrides.pop("json_mode", self.json_mode):
            body["response_format"] = {"type": "json_object"}
        body.update(self.extra_body)
        body.update(overrides)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body).encode()
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(f"{self.base_url}/chat/completions", data=data, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode())
                return payload["choices"][0]["message"].get("content") or ""
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if e.code in (408, 429, 500, 502, 503, 504) and attempt < self.retries:
                    last = e
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if e.code in (408, 429, 500, 502, 503, 504):
                    raise TransientToolError(f"LLM endpoint HTTP {e.code}: {detail}") from e
                raise ToolError(f"LLM endpoint HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = e
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise TransientToolError(f"LLM endpoint unreachable: {e}") from e
        raise TransientToolError(f"LLM endpoint failed: {last}")

    def __call__(self, prompt: str, **overrides: Any) -> str:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **overrides)

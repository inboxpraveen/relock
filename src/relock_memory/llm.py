"""A tiny, dependency-free client for OpenAI-compatible chat endpoints.

Works with vLLM, SGLang, Ollama (``http://localhost:11434/v1``), llama.cpp
server, LM Studio, TGI, OpenRouter and OpenAI itself. Anything else can be
plugged in as a plain ``Callable[[str], str]``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit

from .signals import ToolError, TransientToolError

log = logging.getLogger("relock_memory")

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by reasoning models (Qwen3, DeepSeek-R1...)."""
    text = _THINK.sub("", text)
    lower = text.lower()
    if "</think>" in lower:  # the server cut off the opening tag: drop everything before the close
        text = text[lower.rindex("</think>") + len("</think>"):]
        lower = text.lower()
    if "<think>" in lower:  # reasoning cut off before it closed (max_tokens): it holds no answer
        text = text[: lower.index("<think>")]
    return text.strip()


def extract_json(text: str, *, prefer_key: str | None = None) -> Any:
    """Return the first JSON object or array found in ``text`` (tolerates fences and chatter).

    With ``prefer_key``, the first object that has that key wins over earlier JSON values (a
    model may quote an example before its answer); without one, the first value is returned.
    """
    first: list[Any] = []
    for value in _json_values(strip_reasoning(text)):
        if prefer_key is None or (isinstance(value, dict) and prefer_key in value):
            return value
        if not first:
            first.append(value)
    if first:
        return first[0]
    raise ValueError("no JSON object found in model output")


def _json_values(text: str) -> Iterator[Any]:
    """Every JSON object or array in ``text``, fenced blocks first, in order of appearance."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            yield json.loads(cand)
            continue
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
                                yield json.loads(cand[start : i + 1])
                            except json.JSONDecodeError:
                                pass
                            break
                start = cand.find(opener, start + 1)


_OPENAI_HOSTS = frozenset({"api.openai.com"})
_warned_openai_key = False


def _openai_key_for(base_url: str) -> str | None:
    """``$OPENAI_API_KEY``, but only for OpenAI's own API: it is never sent to another host."""
    global _warned_openai_key
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    host = (urlsplit(base_url).hostname or "").lower()
    if host in _OPENAI_HOSTS:
        return key
    if not _warned_openai_key:
        _warned_openai_key = True
        log.warning("relock: OPENAI_API_KEY is set but is not sent to %s (it is used only for api.openai.com). "
                    "Set RELOCK_LLM_API_KEY or pass api_key= if that endpoint needs a key.", host or base_url)
    return None


def _reply_text(body: str) -> str:
    """The assistant text of a chat-completions response, or a clear :class:`ToolError`."""
    try:
        payload = json.loads(body)
        content = payload["choices"][0]["message"].get("content")
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        raise ToolError(f"LLM endpoint returned an unexpected response: {body[:300]!r}") from None
    if isinstance(content, list):  # content parts: [{"type": "text", "text": "..."}]
        content = "".join(p.get("text") or "" for p in content if isinstance(p, dict) and p.get("type") == "text")
    return content if isinstance(content, str) else ""


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
        self.api_key = api_key or os.environ.get("RELOCK_LLM_API_KEY") or _openai_key_for(self.base_url)
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
                    body_text = resp.read().decode("utf-8", errors="replace")
                return _reply_text(body_text)
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

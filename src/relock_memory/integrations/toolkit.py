"""Framework-agnostic agent tools.

The split that keeps memory safe is: **the model predicts, the harness judges.**

* ``recall_memory`` and ``state_expectation`` are given to the model.
* Real outcomes come from your own tools wrapped with :meth:`Memory.wrap`
  (trusted), from verifiers, or from authenticated users.
* ``report_outcome`` lets the model describe what happened, but its reports
  are ``Source.AGENT`` and do not open writes unless the policy trusts them.

The toolkit works with anything that accepts Python callables or JSON tool
schemas: OpenAI Agents SDK, PydanticAI, smolagents, CrewAI, LangChain/LangGraph,
LlamaIndex, AutoGen, raw OpenAI / Anthropic / vLLM tool calling.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from ..episode import Episode
from ..memory import Memory
from ..policy import WritePolicy
from ..signals import ErrorClass, Signal
from ..types import RelockError

# The words report_outcome accepts. Others are refused rather than guessed: reading "held" or
# "yes" as a failure (as relock < 0.3.1 did) records a false report.
_OUTCOMES = {**dict.fromkeys(("confirmed", "success", "succeeded", "passed", "ok", "true"), True),
             **dict.fromkeys(("failed", "fail", "failure", "error", "false"), False)}


class _BadArgument(ValueError):
    """A model-supplied argument of the wrong type: answered with an ``error:`` string."""


def _message(e: BaseException) -> str:
    """An exception's message. ``str()`` of a KeyError (so of NotFoundError) adds repr quotes."""
    return e.args[0] if e.args and isinstance(e.args[0], str) else str(e)


def _text(name: str, value: Any, *, required: bool = False, default: str | None = None) -> str | None:
    if value is None:
        if required:
            raise _BadArgument(f"'{name}' is required and must be a string")
        return default
    if not isinstance(value, str):
        raise _BadArgument(f"'{name}' must be a string, not {type(value).__name__}")
    return value


def _texts(name: str, value: Any, *, required: bool = False) -> list[str] | None:
    """A list of strings. A bare string is one element, never split into characters."""
    if value is None:
        if required:
            raise _BadArgument(f"'{name}' is required and must be a list of strings")
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return list(value)
    raise _BadArgument(f"'{name}' must be a list of strings")


class Toolkit:
    def __init__(
        self,
        memory: Memory,
        *,
        k: int = 3,
        budget_tokens: int | None = None,
        allow_reports: bool = True,
        allow_teach: bool = False,
        task: str = "",
        idle_timeout: float | None = 1800.0,
        policy: WritePolicy | None = None,
    ) -> None:
        """``idle_timeout``: seconds without a toolkit call after which an episode the toolkit
        opened itself is closed (re-locked) and the next call starts a fresh one. Clients that
        never call ``end`` would otherwise keep their recalled items editable indefinitely.
        ``None`` disables it. Episodes you open yourself are never closed by the toolkit.
        ``policy`` applies to episodes the toolkit opens itself (default: the memory's policy)."""
        self.memory = memory
        self.policy = policy
        self.k = k
        self.budget_tokens = budget_tokens
        self.allow_reports = allow_reports
        self.allow_teach = allow_teach
        self.task = task
        self.idle_timeout = idle_timeout
        self._own: Episode | None = None
        self._last_used = time.monotonic()
        # Guards check-and-open of the toolkit's own episode: concurrent first calls (an MCP
        # server's worker threads, parallel tool calls) must share one episode, not leak several.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ episode plumbing
    def episode(self) -> Episode:
        """The active episode if you opened one, else one this toolkit opens and owns.

        Each toolkit owns its own episode, so two agents with their own toolkits never share
        one. The toolkit's episode is made active in the calling context, so your wrapped tools
        called next in the same thread or task are attributed to it.
        """
        with self._lock:
            now = time.monotonic()
            idle = now - self._last_used
            self._last_used = now
            if (self._own is not None and not self._own.closed and self.idle_timeout is not None
                    and idle > self.idle_timeout):
                self._own.close()  # re-lock: whatever the client recalled before going quiet
            ep = self.memory.current_episode()
            if ep is not None and ep._owner in (None, self):  # yours, or this toolkit's
                return ep
            own = self._own if self._own is not None and not self._own.closed else self._open_own(
                self.task or "toolkit session")
            self.memory._ctx.set(own)
            return own

    def _open_own(self, task: str) -> Episode:
        own = self.memory.episode(task, policy=self.policy)
        own._owner = self
        self._own = own
        return own

    def begin_episode(self, task: str = "") -> str:
        """Start a fresh memory episode for a new task. Returns the episode id."""
        with self._lock:
            self.end_episode()
            own = self._open_own(task or self.task or "toolkit session")
            self.memory._ctx.set(own)
            self._last_used = time.monotonic()
            return own.id

    def end_episode(self) -> str:
        """Close the memory episode for the current task (re-lock). Returns a summary."""
        with self._lock:
            if self._own is None or self._own.closed:
                return "{}"
            own, self._own = self._own, None
            summary = own.close()
            if self.memory._ctx.get() is own:
                self.memory._ctx.set(None)
            return json.dumps(summary)

    # ------------------------------------------------------------------ model-facing tools
    def recall_memory(self, query: str) -> str:
        """Look up procedures learned from earlier tasks that are relevant to what you are about to do.

        Args:
            query: what you are trying to do, e.g. "fetch orders for a date range".

        Returns the matching procedures with their ids and clause ids. Follow them, and cite the
        item and clause ids when you call state_expectation.
        """
        try:
            query = _text("query", query, required=True)
        except _BadArgument as e:
            return f"error: {e}"
        recall = self.episode().recall(query, k=self.k)
        if not recall:
            return "No relevant procedures in memory."
        return recall.render(budget_tokens=self.budget_tokens)

    def state_expectation(self, item: str, prediction: str, clause_ids: list[str] | None = None,
                          tool: str | None = None) -> str:
        """Before acting on a recalled procedure, state what you expect to happen.

        Args:
            item: the procedure's key or id, as shown by recall_memory.
            prediction: the concrete outcome you expect, e.g. "get_orders returns at least one order".
            clause_ids: the clause ids (like "c2") your next action relies on.
            tool: the name of the tool you are about to call, if any.

        Returns an expectation id. The tool's real result is checked against it automatically.
        """
        try:
            item = _text("item", item, required=True)
            prediction = _text("prediction", prediction, required=True)
            clause_ids = _texts("clause_ids", clause_ids)
            tool = _text("tool", tool)
            exp = self.episode().expect(item, prediction, clauses=clause_ids or None, tool=tool or None,
                                        author="agent")
        except (RelockError, ValueError, KeyError) as e:
            return f"error: {_message(e)}"
        return exp.id

    def report_outcome(self, expectation_id: str, outcome: str, details: str = "", correction: str = "",
                       error_type: str = "unknown") -> str:
        """Report whether a stated expectation held, when no tool result could check it.

        Args:
            expectation_id: the id returned by state_expectation.
            outcome: "confirmed" or "failed".
            details: what actually happened.
            correction: if it failed, the instruction that would have been right.
            error_type: if it failed: "transient" (timeout, rate limit), "programmatic"
                (validation error, bad request) or "semantic" (ran fine but the result was wrong).
        """
        if not self.allow_reports:
            return "error: outcome reports are disabled; outcomes are checked by the harness"
        try:
            expectation_id = _text("expectation_id", expectation_id, required=True)
            outcome = _text("outcome", outcome, required=True)
            details = _text("details", details, default="")
            correction = _text("correction", correction, default="")
            error_type = _text("error_type", error_type, default="unknown")
        except _BadArgument as e:
            return f"error: {e}"
        word = outcome.strip().removesuffix(".").strip().lower()
        if word not in _OUTCOMES:
            return f"error: outcome must be 'confirmed' or 'failed', not {outcome!r}"
        try:
            cls = ErrorClass(error_type.strip().lower()) if error_type.strip() else ErrorClass.UNKNOWN
        except ValueError:
            cls = ErrorClass.UNKNOWN
        signal = Signal.agent_report(_OUTCOMES[word], details, correction=correction or None, error_class=cls)
        try:
            obs = self.episode().observe(expectation_id, signal=signal)
        except (RelockError, ValueError, KeyError) as e:
            return f"error: {_message(e)}"
        return json.dumps({"matched": obs.matched, **obs.decision.to_dict()})

    def teach_memory(self, title: str, clauses: list[str], key: str | None = None,
                     tools: list[str] | None = None) -> str:
        """Save a new procedure you discovered so future tasks can reuse it.

        Args:
            title: short name of the procedure.
            clauses: the steps, one short imperative sentence each.
            key: optional stable name such as "orders.fetch".
            tools: names of the tools this procedure is about.
        """
        if not self.allow_teach:
            return "error: teaching is disabled"
        try:
            title = _text("title", title, required=True)
            # A bare string is handed to Memory.add whole, which reads one clause per line.
            steps = clauses if isinstance(clauses, str) else _texts("clauses", clauses, required=True)
            key = _text("key", key)
            tools = _texts("tools", tools)
            item = self.memory.add(title, steps, key=key or None, tools=tools or (),
                                   meta={"origin": "agent"}, author="agent")
        except (RelockError, ValueError) as e:
            return f"error: {_message(e)}"
        return f"saved {item.ref} v{item.version}"

    # ------------------------------------------------------------------ exports
    def functions(self) -> list[Callable[..., str]]:
        """Plain callables (with docstrings and type hints) to hand to an agent framework."""
        fns: list[Callable[..., str]] = [self.recall_memory, self.state_expectation]
        if self.allow_reports:
            fns.append(self.report_outcome)
        if self.allow_teach:
            fns.append(self.teach_memory)
        return fns

    def schemas(self) -> list[dict[str, Any]]:
        """Neutral ``{name, description, parameters}`` JSON-Schema tool definitions."""
        out = [
            {"name": "recall_memory",
             "description": "Look up procedures learned from earlier tasks that are relevant to what you are "
                            "about to do. Follow them and cite their ids in state_expectation.",
             "parameters": {"type": "object", "properties": {
                 "query": {"type": "string", "description": "What you are trying to do."}},
                 "required": ["query"]}},
            {"name": "state_expectation",
             "description": "Before acting on a recalled procedure, state the concrete outcome you expect. "
                            "Returns an expectation id; the real tool result is checked against it.",
             "parameters": {"type": "object", "properties": {
                 "item": {"type": "string", "description": "Procedure key or id from recall_memory."},
                 "prediction": {"type": "string", "description": "The outcome you expect."},
                 "clause_ids": {"type": "array", "items": {"type": "string"},
                                "description": "Clause ids (like c2) your action relies on."},
                 "tool": {"type": "string", "description": "Tool you are about to call, if any."}},
                 "required": ["item", "prediction"]}},
        ]
        if self.allow_reports:
            out.append({"name": "report_outcome",
                        "description": "Report whether a stated expectation held when no tool result could check it.",
                        "parameters": {"type": "object", "properties": {
                            "expectation_id": {"type": "string"},
                            "outcome": {"type": "string", "enum": ["confirmed", "failed"]},
                            "details": {"type": "string", "description": "What actually happened."},
                            "correction": {"type": "string",
                                           "description": "If it failed, the instruction that would have been right."},
                            "error_type": {"type": "string",
                                           "enum": ["transient", "programmatic", "semantic", "unknown"],
                                           "description": "If it failed: transient (timeout, rate limit), "
                                                          "programmatic (validation error), semantic (wrong result)."}},
                            "required": ["expectation_id", "outcome"]}})
        if self.allow_teach:
            out.append({"name": "teach_memory",
                        "description": "Save a new procedure so future tasks can reuse it.",
                        "parameters": {"type": "object", "properties": {
                            "title": {"type": "string"},
                            "clauses": {"type": "array", "items": {"type": "string"}},
                            "key": {"type": "string"},
                            "tools": {"type": "array", "items": {"type": "string"}}},
                            "required": ["title", "clauses"]}})
        return out

    def openai_tools(self) -> list[dict[str, Any]]:
        """``tools=`` for OpenAI-compatible chat completions (OpenAI, vLLM, Ollama, llama.cpp...)."""
        return [{"type": "function", "function": s} for s in self.schemas()]

    def anthropic_tools(self) -> list[dict[str, Any]]:
        """``tools=`` for the Anthropic Messages API."""
        return [{"name": s["name"], "description": s["description"], "input_schema": s["parameters"]}
                for s in self.schemas()]

    def handles(self, name: str) -> bool:
        return name in {s["name"] for s in self.schemas()}

    def call(self, name: str, arguments: dict[str, Any] | str | None = None) -> str:
        """Dispatch a tool call from any provider. ``arguments`` may be a dict or a JSON string.

        A model's mistakes (bad JSON, missing or mistyped arguments) come back as an
        ``error: ...`` string the model can act on; they are never raised.
        """
        fns = {f.__name__: f for f in self.functions()}
        if not isinstance(name, str) or name not in fns:
            return f"error: unknown relock tool {name!r}"
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments) if arguments.strip() else {}
            except (ValueError, RecursionError) as e:
                return f"error: arguments for {name} are not valid JSON ({e}); send a JSON object"
        else:
            args = {} if arguments is None else arguments
        if not isinstance(args, Mapping):
            return f"error: arguments for {name} must be a JSON object, not {type(args).__name__}"
        try:
            return fns[name](**args)
        except TypeError as e:
            return f"error: bad arguments for {name}: {e}"

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
from collections.abc import Callable
from typing import Any

from ..episode import Episode
from ..memory import Memory
from ..signals import ErrorClass, Signal
from ..types import RelockError


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
    ) -> None:
        self.memory = memory
        self.k = k
        self.budget_tokens = budget_tokens
        self.allow_reports = allow_reports
        self.allow_teach = allow_teach
        self.task = task
        self._own: Episode | None = None

    # ------------------------------------------------------------------ episode plumbing
    def episode(self) -> Episode:
        """The active episode (yours, if you opened one), else one the toolkit opens and owns."""
        ep = self.memory.current_episode()
        if ep is not None:
            return ep
        if self._own is None or self._own.closed:
            self._own = self.memory.episode(self.task or "toolkit session")
        return self._own

    def begin_episode(self, task: str = "") -> str:
        """Start a fresh memory episode for a new task. Returns the episode id."""
        self.end_episode()
        self._own = self.memory.episode(task or self.task or "toolkit session")
        return self._own.id

    def end_episode(self) -> str:
        """Close the memory episode for the current task (re-lock). Returns a summary."""
        if self._own is not None and not self._own.closed:
            summary = self._own.close()
            self._own = None
            return json.dumps(summary)
        return "{}"

    # ------------------------------------------------------------------ model-facing tools
    def recall_memory(self, query: str) -> str:
        """Look up procedures learned from earlier tasks that are relevant to what you are about to do.

        Args:
            query: what you are trying to do, e.g. "fetch orders for a date range".

        Returns the matching procedures with their ids and clause ids. Follow them, and cite the
        item and clause ids when you call state_expectation.
        """
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
            exp = self.episode().expect(item, prediction, clauses=clause_ids or None, tool=tool or None,
                                        author="agent")
        except (RelockError, ValueError, KeyError) as e:
            return f"error: {e}"
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
        passed = outcome.strip().lower() in ("confirmed", "success", "succeeded", "passed", "ok", "true")
        try:
            cls = ErrorClass(error_type.strip().lower()) if error_type else ErrorClass.UNKNOWN
        except ValueError:
            cls = ErrorClass.UNKNOWN
        signal = Signal.agent_report(passed, details, correction=correction or None, error_class=cls)
        try:
            obs = self.episode().observe(expectation_id, signal=signal)
        except (RelockError, ValueError, KeyError) as e:
            return f"error: {e}"
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
            item = self.memory.add(title, clauses, key=key or None, tools=tools or (),
                                   meta={"origin": "agent"}, author="agent")
        except (RelockError, ValueError) as e:
            return f"error: {e}"
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
        """Dispatch a tool call from any provider. ``arguments`` may be a dict or a JSON string."""
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        fns = {f.__name__: f for f in self.functions()}
        if name not in fns:
            return f"error: unknown relock tool {name!r}"
        try:
            return fns[name](**args)
        except TypeError as e:
            return f"error: bad arguments for {name}: {e}"

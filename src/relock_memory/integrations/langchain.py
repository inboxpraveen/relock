"""LangChain / LangGraph adapter (``pip install relock-memory[langchain]``).

Two pieces:

* :class:`RelockCallbackHandler` - zero-touch: add it to ``callbacks`` and every
  tool run inside an open episode resolves the expectations of the recalled
  items that govern that tool (trusted tool outcomes, no tool changes needed).
* :func:`langchain_tools` - ``recall_memory`` / ``state_expectation`` as
  ``StructuredTool`` objects for a LangGraph ``create_react_agent`` or ``ToolNode``.
"""

from __future__ import annotations

import threading
from typing import Any
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.tools import StructuredTool
except ImportError as e:  # pragma: no cover - depends on environment
    raise ImportError("the LangChain adapter needs langchain-core: pip install 'relock-memory[langchain]'") from e

from ..memory import Memory
from .toolkit import Toolkit


class RelockCallbackHandler(BaseCallbackHandler):
    """Feeds real tool outcomes into the active relock episode.

    LangChain calls ``on_tool_error`` when a tool raises (including inside
    LangGraph's ``ToolNode``, which catches the error afterwards). Tools that
    *return* error strings instead are seen as successes unless their output is
    a ``ToolMessage`` with ``status="error"``.
    """

    raise_error = False
    run_inline = True

    def __init__(self, memory: Memory, *, ignore: set[str] | None = None) -> None:
        super().__init__()
        self.memory = memory
        self.ignore = set(ignore or ()) | {"recall_memory", "state_expectation", "report_outcome", "teach_memory"}
        self._runs: dict[UUID, tuple[Any, list[Any]]] = {}
        self._lock = threading.Lock()

    def on_tool_start(self, serialized: dict[str, Any] | None, input_str: str, *, run_id: UUID,
                      **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or ""
        if not name or name in self.ignore:
            return
        ep = self.memory.current_episode()
        if ep is None:
            return
        exps = ep._bind_tool(name, None)
        if exps:
            with self._lock:
                self._runs[run_id] = (ep, exps)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            entry = self._runs.pop(run_id, None)
        if entry is None:
            return
        ep, exps = entry
        if getattr(output, "status", None) == "error":
            ep._observe_quietly(exps, error=f"tool returned an error: {getattr(output, 'content', output)}")
        else:
            ep._observe_quietly(exps, result=getattr(output, "content", output))

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            entry = self._runs.pop(run_id, None)
        if entry is None:
            return
        ep, exps = entry
        ep._observe_quietly(exps, error=error)


def langchain_tools(toolkit: Toolkit) -> list[StructuredTool]:
    """The toolkit's model-facing functions as LangChain ``StructuredTool`` objects."""
    return [StructuredTool.from_function(fn, name=fn.__name__, description=(fn.__doc__ or "").strip())
            for fn in toolkit.functions()]

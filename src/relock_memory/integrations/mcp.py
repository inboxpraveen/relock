"""MCP server: plug relock into any MCP-capable agent runtime.

Claude Code, Cursor, the OpenAI Agents SDK, LangGraph (via langchain-mcp-adapters),
PydanticAI, CrewAI, VS Code and others all speak MCP, so this one adapter
covers most current agent stacks without code changes::

    relock mcp --db ./relock.db --verifier tests="pytest -q" --verifier lint="ruff check ."

Trust model: the agent calls these tools, so anything it *reports* is
``Source.AGENT`` and cannot rewrite memory (unless ``--trust-agent-reports``).
Trusted outcomes come from **verifiers**: shell commands fixed by whoever
starts the server. The agent can only choose a verifier by name; the server
runs it and judges the exit code itself.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from ..memory import Memory
from ..policy import WritePolicy
from ..signals import Signal, Source
from ..types import RelockError
from .toolkit import Toolkit

INSTRUCTIONS = """relock is a procedural memory that corrects itself.
1. At the start of a task call relock_begin, then relock_recall with what you are about to do.
2. Follow the recalled procedures. Before relying on one, call relock_expect with the item id,
   the clause ids you rely on and the concrete outcome you predict.
3. Resolve the expectation: prefer relock_verify (the server runs a trusted check); otherwise
   relock_report what happened. Wrong procedures are rewritten in place once the evidence is trusted.
4. Call relock_end when the task is finished."""


def _load_server_class() -> Any:
    try:
        from mcp.server import MCPServer  # mcp >= 2

        return MCPServer
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP  # mcp 1.x

        return FastMCP
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError("the MCP server needs the 'mcp' package: pip install 'relock-memory[mcp]'") from e


def run_verifier(command: str, *, timeout: float = 600.0, cwd: str | None = None) -> Signal:
    """Run a configured verifier command and turn its exit status into a trusted signal."""
    try:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:
        return Signal.transient(f"verifier timed out after {timeout:.0f}s")
    except OSError as e:
        return Signal.transient(f"verifier could not start: {e}")
    output = (proc.stdout or "") + (proc.stderr or "")
    tail = output[-1500:].strip()
    if proc.returncode == 0:
        return Signal.verifier(True, f"exit 0\n{tail}"[:1600])
    return Signal.verifier(False, f"exit {proc.returncode}\n{tail}"[:1600])


def build_server(
    memory: Memory,
    *,
    verifiers: dict[str, str] | None = None,
    verifier_timeout: float = 600.0,
    verifier_cwd: str | None = None,
    trust_agent_reports: bool = False,
    allow_teach: bool = False,
    idle_timeout: float | None = 1800.0,
    k: int = 3,
    name: str = "relock",
) -> Any:
    """Create an MCP server (``mcp`` 1.x ``FastMCP`` or 2.x ``MCPServer``) exposing relock tools."""
    if trust_agent_reports and Source.AGENT not in memory.policy.trusted_sources:
        memory.policy = memory.policy.with_(trusted_sources=memory.policy.trusted_sources | {Source.AGENT})
    verifiers = dict(verifiers or {})
    kit = Toolkit(memory, k=k, allow_reports=True, allow_teach=allow_teach, idle_timeout=idle_timeout)
    server_cls = _load_server_class()
    server = server_cls(name, instructions=INSTRUCTIONS)

    @server.tool(name="relock_begin", description="Start a memory episode for a new task.")
    def relock_begin(task: str) -> str:
        return kit.begin_episode(task)

    @server.tool(name="relock_recall",
                 description="Recall learned procedures relevant to what you are about to do. "
                             "Returns items with ids and clause ids.")
    def relock_recall(query: str) -> str:
        return kit.recall_memory(query)

    @server.tool(name="relock_expect",
                 description="Before acting on a recalled procedure, state the outcome you predict. "
                             "Returns an expectation id.")
    def relock_expect(item: str, prediction: str, clause_ids: list[str] | None = None,
                      tool: str | None = None) -> str:
        return kit.state_expectation(item, prediction, clause_ids, tool)

    @server.tool(name="relock_report",
                 description="Report whether an expectation held: outcome is 'confirmed' or 'failed'; "
                             "error_type is transient, programmatic, semantic or unknown. "
                             "Agent reports are logged; they rewrite memory only if the server trusts them.")
    def relock_report(expectation_id: str, outcome: str, details: str = "", correction: str = "",
                      error_type: str = "unknown") -> str:
        return kit.report_outcome(expectation_id, outcome, details, correction, error_type)

    @server.tool(name="relock_verify",
                 description="Resolve an expectation with a trusted verifier run by the server. "
                             f"Available verifiers: {', '.join(sorted(verifiers)) or 'none configured'}. "
                             "A suggested correction is only used if the server trusts agent reports.")
    def relock_verify(expectation_id: str, verifier: str, correction: str = "") -> str:
        if verifier not in verifiers:
            return f"error: unknown verifier {verifier!r}; available: {sorted(verifiers)}"
        signal = run_verifier(verifiers[verifier], timeout=verifier_timeout, cwd=verifier_cwd)
        signal = signal.with_(data={**signal.data, "verifier": verifier})
        # The verifier's verdict is trusted; the agent's suggested fix is not. Attaching it to a
        # VERIFIER signal would let the agent write arbitrary text into memory whenever the check
        # is red for any reason, so it is used only when the operator trusts agent reports.
        correction_used = bool(correction) and not signal.passed and trust_agent_reports
        if correction_used:
            signal = signal.with_(correction=correction)
        try:
            obs = kit.episode().observe(expectation_id, signal=signal)
        except (RelockError, ValueError, KeyError) as e:
            return f"error: {e}"
        return json.dumps({"verifier": verifier, "passed": signal.passed, "matched": obs.matched,
                           **obs.decision.to_dict(), "output": signal.message[-600:],
                           "correction_used": correction_used})

    if allow_teach:
        @server.tool(name="relock_teach", description="Save a new procedure (title plus one clause per step).")
        def relock_teach(title: str, clauses: list[str], key: str | None = None,
                         tools: list[str] | None = None) -> str:
            return kit.teach_memory(title, clauses, key, tools)

    @server.tool(name="relock_history", description="Show every version of a memory item and what changed.")
    def relock_history(item: str) -> str:
        try:
            versions = memory.history(item)
        except (RelockError, KeyError) as e:
            return f"error: {e}"
        lines = []
        for v in versions:
            lines.append(f"v{v.version} [{v.status}] {v.change_kind} by {v.author or '?'}"
                         + (f" - {v.reason}" if v.reason else ""))
            lines.extend(f"  ({c.id}) {c.text}" for c in v.clauses)
        return "\n".join(lines)

    @server.tool(name="relock_end", description="Finish the task: closes (re-locks) the memory episode.")
    def relock_end() -> str:
        return kit.end_episode()

    server.relock_toolkit = kit  # type: ignore[attr-defined]
    return server


def serve(
    db: str | None = None,
    *,
    transport: str = "stdio",
    rewriter: Any = None,
    policy: WritePolicy | None = None,
    **kwargs: Any,
) -> None:
    memory = Memory(db, rewriter=rewriter, policy=policy)
    server = build_server(memory, **kwargs)
    try:
        server.run(transport=transport)
    finally:
        memory.close()

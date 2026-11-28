"""MCP server: plug relock into any MCP-capable agent runtime.

Claude Code, Cursor, the OpenAI Agents SDK, LangGraph (via langchain-mcp-adapters),
PydanticAI, CrewAI, VS Code and others all speak MCP, so this one adapter
covers most current agent stacks without code changes::

    relock mcp --db ./relock.db --verifier tests="pytest -q" --verifier lint="ruff check ."

Trust model: the agent calls these tools, so anything it *reports* is
``Source.AGENT`` and cannot rewrite memory (unless ``--trust-agent-reports``).
Trusted outcomes come from **verifiers**: shell commands fixed by whoever
starts the server. The agent can only choose a verifier by name; the server
runs it and judges the exit code itself. When the agent can also edit what a
verifier prints (its tests, say), pass ``trust_verifier_output=False``: the
pass/fail still counts, but the output text never reaches the rewriter.

The server holds one episode at a time. Serve one client per server: over a
network transport, every connected client shares that episode.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import signal as _signal
import subprocess
from typing import Any, Literal

from ..memory import Memory
from ..policy import WritePolicy
from ..signals import ErrorClass, Signal, Source
from ..types import RelockError
from .toolkit import Toolkit, _message

log = logging.getLogger("relock.mcp")

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


_WINDOWS = os.name == "nt"
_KILL_GRACE = 5.0  # seconds to collect output once a timed-out verifier is killed; never longer
# cmd.exe's own messages when the program can't be started. Matched only when they are the
# whole output: verifier output is partly agent-controlled, so text alone must not turn a red
# run into non-evidence.
_CMD_FAILED = re.compile(r"'.*' is not recognized as an internal or external command,"
                         r"|operable program or batch file\.|The system cannot find the path specified\.")
_SHELL_WORDS = frozenset({"cd", "echo", "exit", "set", "export", "call", "if", "for", "type", "start",
                          "pushd", "popd", "exec", "source", ".", ":", "[", "test", "true", "false"})


def _not_evidence(message: str) -> Signal:
    """A verifier run that says nothing about the procedure: logged, never evidence."""
    return Signal.verifier(False, message).with_(error_class=ErrorClass.TRANSIENT)


def _could_not_run(returncode: int, output: str) -> bool:
    if not _WINDOWS:
        return returncode in (126, 127)  # the shell's "not executable" / "not found"
    if returncode == 9009:
        return True
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    return returncode == 1 and bool(lines) and all(_CMD_FAILED.fullmatch(ln) for ln in lines)


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    """Kill the verifier and everything it started. ``Popen.kill`` alone stops only the shell,
    and its children keep the output pipes open."""
    try:
        if _WINDOWS:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=_KILL_GRACE, check=False)
        else:
            os.killpg(proc.pid, _signal.SIGKILL)  # its own session: the group is everything it started
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def run_verifier(command: str, *, timeout: float | None = 600.0, cwd: str | None = None) -> Signal:
    """Run a configured verifier command and turn its exit status into a trusted signal.

    The command runs in a shell with no stdin (under the stdio transport the server's stdin
    is the JSON-RPC stream), in its own process group, so a timeout kills everything it
    started. A run that times out or could not start is transient: it is never evidence.
    """
    group: dict[str, Any] = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if _WINDOWS
                             else {"start_new_session": True})
    try:
        proc = subprocess.Popen(command, shell=True, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", **group)
    except OSError as e:
        return _not_evidence(f"verifier could not run: {e}")
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            proc.communicate(timeout=_KILL_GRACE)
        except subprocess.TimeoutExpired:  # something escaped the kill and still holds the pipes
            log.warning("verifier %r: a process it started survived the timeout kill; abandoning its output",
                        command)
        return _not_evidence(f"verifier timed out after {timeout:g}s")
    output = (stdout or "") + (stderr or "")
    tail = output[-1500:].strip()
    if proc.returncode == 0:
        return Signal.verifier(True, f"exit 0\n{tail}"[:1600])
    if _could_not_run(proc.returncode, output):
        return _not_evidence(f"verifier could not run (exit {proc.returncode}); check its command\n{tail}"[:1600])
    return Signal.verifier(False, f"exit {proc.returncode}\n{tail}"[:1600])


def _missing_program(command: str, cwd: str | None) -> str | None:
    """The program a verifier command starts, if it can't be found (best effort: shells vary)."""
    try:
        words = shlex.split(command, posix=not _WINDOWS)
    except ValueError:
        return None
    words = [w for w in words if not re.match(r"[A-Za-z_][A-Za-z0-9_]*=", w)]  # VAR=value prefixes
    if not words:
        return None
    program = words[0].strip('"') if _WINDOWS else words[0]
    if program.lower() in _SHELL_WORDS or any(c in program for c in "&|;<>()$`%"):
        return None
    path = program
    if cwd and not os.path.isabs(program) and any(sep and sep in program for sep in (os.sep, os.altsep)):
        path = os.path.join(cwd, program)
    return None if shutil.which(path) else program


def build_server(
    memory: Memory,
    *,
    verifiers: dict[str, str] | None = None,
    verifier_timeout: float = 600.0,
    verifier_cwd: str | None = None,
    trust_agent_reports: bool = False,
    trust_verifier_output: bool = True,
    allow_teach: bool = False,
    idle_timeout: float | None = 1800.0,
    k: int = 3,
    name: str = "relock",
) -> Any:
    """Create an MCP server (``mcp`` 1.x ``FastMCP`` or 2.x ``MCPServer``) exposing relock tools.

    ``trust_verifier_output=False`` marks verifier signals ``detail_trusted=False``: a red run
    still counts as evidence, but its output (and any correction the agent suggests with it)
    never reaches the rewriter. Use it when the agent can edit what the verifier prints.
    """
    # Widen trust for this server's episodes only: the Memory may be shared with other code.
    policy = memory.policy
    if trust_agent_reports and Source.AGENT not in policy.trusted_sources:
        policy = policy.with_(trusted_sources=policy.trusted_sources | {Source.AGENT})
    verifiers = dict(verifiers or {})
    for vname, command in verifiers.items():
        missing = _missing_program(command, verifier_cwd)
        if missing:
            log.warning("verifier %r: program %r was not found; relock_verify will report that the "
                        "verifier could not run", vname, missing)
    kit = Toolkit(memory, k=k, allow_reports=True, allow_teach=allow_teach, idle_timeout=idle_timeout,
                  policy=policy)
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
    def relock_report(expectation_id: str, outcome: Literal["confirmed", "failed"], details: str = "",
                      correction: str = "", error_type: str = "unknown") -> str:
        return kit.report_outcome(expectation_id, outcome, details, correction, error_type)

    @server.tool(name="relock_verify",
                 description="Resolve an expectation with a trusted verifier run by the server. "
                             f"Available verifiers: {', '.join(sorted(verifiers)) or 'none configured'}. "
                             "A suggested correction is only used if the server trusts agent reports.")
    def relock_verify(expectation_id: str, verifier: str, correction: str = "") -> str:
        if verifier not in verifiers:
            return f"error: unknown verifier {verifier!r}; available: {sorted(verifiers)}"
        ep = kit.episode()  # before the run: a long test suite must not count as idle time
        try:
            ep._expectation(expectation_id)  # open in this episode? (a lookup: observes nothing)
        except (RelockError, KeyError) as e:
            return f"error: {_message(e)}"  # don't run a whole suite for an id it can't resolve
        signal = run_verifier(verifiers[verifier], timeout=verifier_timeout, cwd=verifier_cwd)
        signal = signal.with_(data={**signal.data, "verifier": verifier})
        if not trust_verifier_output:
            signal = signal.with_(detail_trusted=False)
        # The verifier's verdict is trusted; the agent's suggested fix is not. Attaching it to a
        # VERIFIER signal would let the agent write arbitrary text into memory whenever the check
        # is red for any reason, so it is used only when the operator trusts agent reports (and
        # the run's text may reach the rewriter at all).
        correction_used = (bool(correction) and not signal.passed and trust_agent_reports and trust_verifier_output
                           and signal.error_class is not ErrorClass.TRANSIENT)
        if correction_used:
            signal = signal.with_(correction=correction)
        try:
            obs = ep.observe(expectation_id, signal=signal)
        except (RelockError, ValueError, KeyError) as e:
            return f"error: {_message(e)}"
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
            return f"error: {_message(e)}"
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
    trust_verifier_output: bool = True,
    **kwargs: Any,
) -> None:
    if transport != "stdio":
        log.warning("relock MCP over %s: this server holds one episode at a time, shared by every client "
                    "that connects, so one client's relock_begin takes over another's episode. Serve a "
                    "single client (run one server per agent).", transport)
    memory = Memory(db, rewriter=rewriter, policy=policy)
    server = build_server(memory, trust_verifier_output=trust_verifier_output, **kwargs)
    try:
        server.run(transport=transport)
    finally:
        memory.close()

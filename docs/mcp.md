# The MCP server

`relock mcp` serves relock over the Model Context Protocol, so Claude Code, Cursor, VS Code, the
OpenAI Agents SDK, LangGraph, PydanticAI and any other MCP client can use it with configuration
only.

Here the agent itself calls the memory tools. That changes the trust picture: **nothing the agent
writes is trusted by default**. Trusted outcomes come from **verifiers**, commands that whoever
starts the server fixes in advance.

## Set it up

Most clients take a JSON server entry:

```json
{
  "mcpServers": {
    "relock": {
      "command": "uvx",
      "args": ["--from", "relock-memory[mcp]", "relock", "--db", "/path/to/relock.db", "mcp",
               "--verifier", "tests=pytest -q", "--verifier", "lint=ruff check ."]
    }
  }
}
```

[`uvx`](https://docs.astral.sh/uv/) fetches the package the first time the client starts the
server. If you installed `relock-memory[mcp]` with pip instead, use `"command": "relock"` and drop
the first two arguments. In Claude Code the same server is one command:

```
claude mcp add relock -- uvx --from "relock-memory[mcp]" relock --db relock.db mcp --verifier tests="pytest -q"
```

Verifier commands run in the server's working directory, which depends on the client. Pass
`--cwd /path/to/project` to pin it.

Seed the procedures the agent should follow with the [CLI](cli.md), against the same database:

```
relock --db relock.db add "Run the tests" "Run the tests with pytest -q." --key tests.run --tool pytest
```

## Tools the agent gets

| tool | what it does |
|---|---|
| `relock_begin(task)` | Starts an episode for a new task. |
| `relock_recall(query)` | Returns matching procedures with item and clause ids. Recalled items become editable. |
| `relock_expect(item, prediction, clause_ids?, tool?)` | States what the agent expects before it relies on a procedure. Returns an expectation id. |
| `relock_verify(expectation_id, verifier)` | The server runs a configured verifier and resolves the expectation from its exit code. This is the trusted path. |
| `relock_report(expectation_id, outcome, ...)` | The agent's own account of what happened, with `outcome` `confirmed` or `failed`. Logged; never rewrites memory by default. |
| `relock_history(item)` | Every version of an item and what changed. |
| `relock_end()` | Finishes the task and re-locks. |
| `relock_teach(title, clauses, ...)` | Saves a new procedure. Only with `--allow-teach`. |

The server also sends instructions that tell the agent to call these in order. If a client never
calls `relock_end`, its episode is re-locked once it has been idle for 30 minutes
(`--idle-timeout`): the next call closes it before doing anything else.

## What the agent can and can't change

Because the agent calls these tools, anything it writes could come from a prompt injection. By
default:

- its `relock_report`s are logged but never rewrite memory;
- the prediction text it gives `relock_expect`, and the clause ids it cites, are withheld from the
  rewriter;
- a `correction` it suggests to `relock_verify` is ignored.

The agent can only pick a verifier by name. The server runs the command and reads the exit code
itself, so two red verifier runs are enough evidence for a rewrite. The agent's own words never
reach the rewriter. The verifier's output does, so that an [LLM rewriter](configuration.md#rewriters)
can work out the fix from a failing check. That output can include text the agent influenced,
such as messages in tests it wrote, so review rewrites made this way (`relock history`). Where the
agent can edit what a verifier prints, start the server with `--untrusted-verifier-output`: a red
run still counts as evidence, but its output never reaches the rewriter, which then knows only
that the check failed.

A verifier that times out, or can't start at all (a missing program), is never evidence. The
server warns at startup when a verifier's program isn't found, and a timeout stops everything the
verifier started.

Items the agent teaches with `--allow-teach` are marked `meta.origin = "agent"` and author `agent`,
so you can review or archive them.

For a single-user local sandbox, `--trust-agent-reports` opts in to all three kinds of agent
text. Don't use it where the agent reads untrusted content.

## Options

| option | default | meaning |
|---|---|---|
| `--verifier NAME=COMMAND` | none | A trusted check the agent may run by name. Repeatable. |
| `--verifier-timeout S` | 600 | A verifier that runs longer counts as a transient failure, which is never evidence. |
| `--cwd DIR` | server's working directory | Where verifier commands run. |
| `--allow-teach` | off | Expose `relock_teach`. |
| `--trust-agent-reports` | off | Let the agent's reports, predictions and corrections count. |
| `--untrusted-verifier-output` | off | Keep verifier output from the rewriter; pass/fail still counts. |
| `--idle-timeout S` | 1800 | Re-lock an episode after this many idle seconds; 0 disables. |
| `-k N` | 3 | Items per recall. |
| `--llm-base-url`, `--llm-model`, `--llm-api-key` | hint rewriter | An OpenAI-compatible model to write rewrites. Also read from `RELOCK_LLM_*`. |
| `--transport` | `stdio` | `stdio`, `sse` or `streamable-http`. The server holds one episode at a time, so serve a single client from each server; it warns on the network transports. |

The database is `--db` (before or after `mcp`), or `$RELOCK_DB`, or `./relock.db`. CI tests the
server on both the `mcp` 1.x and 2.x lines.

## From Python

`build_server(memory, verifiers={...})` in `relock_memory.integrations.mcp` returns the server
object, if you want to embed it or share a `Memory` with other code. `trust_agent_reports=True`
there widens trust for the server's own episodes only, and `trust_verifier_output=False` is the
`--untrusted-verifier-output` switch.

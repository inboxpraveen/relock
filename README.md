<p align="center">
  <img src="docs/assets/logo.svg" alt="relock" width="252" height="64">
</p>

<p align="center">
  <strong>Agent memory that fixes the rule that fired, in place and only on trusted evidence,<br>
  so corrected mistakes don't come back.</strong>
</p>

<p align="center">
  <a href="https://github.com/inboxpraveen/relock/actions/workflows/ci.yml"><img alt="ci" src="https://github.com/inboxpraveen/relock/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/relock-memory/"><img alt="pypi" src="https://img.shields.io/pypi/v/relock-memory"></a>
  <a href="https://pypi.org/project/relock-memory/"><img alt="python" src="https://img.shields.io/pypi/pyversions/relock-memory"></a>
  <a href="https://github.com/inboxpraveen/relock/pkgs/container/relock"><img alt="docker" src="https://img.shields.io/badge/docker-ghcr.io%2Finboxpraveen%2Frelock-blue"></a>
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
</p>

<p align="center">
  <a href="docs/getting-started.md">Getting started</a> ·
  <a href="docs/concepts.md">How it works</a> ·
  <a href="docs/README.md">Documentation</a> ·
  <a href="docs/benchmarks.md">Benchmarks</a> ·
  <a href="CHANGELOG.md">Changelog</a>
</p>

You corrected your agent: the orders API takes ISO dates now. Its memory saved the lesson as a
note. Days later, deep into a long session, it sends `03/14/2026` again. The old rule was still in
memory, next to the note, and the model followed it.

![Same correction, two memories: a typical memory appends a note and the agent repeats the mistake; relock rewrites the rule and the next call succeeds](docs/assets/hero.svg)

Lesson notes, rules files and playbooks all keep both rules and hope the model picks the right
one. relock edits the rule that fired, in place, under three conditions:

- **Only on trusted evidence.** A typed tool error, a verifier or an authenticated user can
  trigger a rewrite. The model's own claims, a web page, a timeout or a one-off flaky check can't.
- **Only while that memory is recalled** in the current task. The retrieval trace records which
  memory was in play, so the model never has to guess.
- **The old version is kept, but never retrieved again.** Every version and every decision is
  stored for audit and rollback.

It is built for agents that call tools or APIs. It is not for chat history or document RAG.

## Results

![Correct tool calls per model in a long session: append-notes, append-gated and relock](docs/results/models/by-model.svg)

8 hosted models and 10 changed tool contracts, in long sessions with stale docs and a prompt
injection in context. With relock, agents made the corrected call **98%** of the time.

- A memory that appends each lesson as a note managed **60%**.
- Appending the correction to the procedure, like a rules file, managed **52%**.
- Keeping relock's trust gate but still appending managed **74%**: the gate keeps the poisoned
  "fix" out, and replacing the rule is what stops the relapse.
- relock was ahead on all 8 models, with the 95% intervals apart for all 8.

The design was committed before the run, and every call is published: 11,520 probe calls, $7.33
in API spend. See [Benchmarks](docs/benchmarks.md) and the [full results](docs/results/models/).

## Install

```
pip install relock-memory            # core: zero dependencies, Python 3.10+
pip install "relock-memory[all]"     # + Postgres, MCP, LangChain/LangGraph
docker run -p 8765:8765 -e RELOCK_TOKEN=change-me -v relock-data:/data ghcr.io/inboxpraveen/relock   # HTTP API, any language
```

The import is `relock_memory`. The PyPI name `relock` belongs to an unrelated SDK.

## Quickstart

This runs as is. A stand-in for your model follows whatever date format memory gives it:

```python
import re
import relock_memory as relock

mem = relock.Memory("agent.db")                    # or ":memory:", or "postgresql://..."
mem.add("Fetch orders by date",
        ["Authenticate with the orders API key before calling get_orders.",
         "Pass the date to get_orders as MM/DD/YYYY."],   # stale: the API now wants ISO dates
        key="orders.fetch", tools=["get_orders"], exist_ok=True)

@mem.wrap                                          # your real tool; its outcome is the trusted signal
def get_orders(date: str) -> list[dict]:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise relock.ToolError(f"invalid date {date!r}",
                               correction="Pass the date to get_orders as YYYY-MM-DD.")
    return [{"id": 1, "date": date}]

for attempt in range(1, 4):
    with mem.episode("orders for March 14") as ep:  # one task; recalled items are editable in it
        context = ep.recall("fetch the orders placed on a date").render()   # put this in the prompt
        date = "2026-03-14" if "YYYY-MM-DD" in context else "03/14/2026"     # stand-in for your model
        try:
            print(attempt, get_orders(date))
        except relock.ToolError as e:
            print(attempt, e)
```

```
1 invalid date '03/14/2026'
2 invalid date '03/14/2026'
3 [{'id': 1, 'date': '2026-03-14'}]
```

After two trusted failures (configurable), clause `c2` was rewritten in place. The old version is
kept for audit, and never retrieved again:

```
$ relock --db agent.db history orders.fetch
v1 [superseded] create by author at 2026-11-28T05:41:21.511337+00:00
    (c1) Authenticate with the orders API key before calling get_orders.
    (c2) Pass the date to get_orders as MM/DD/YYYY.
v2 [active] reconsolidate by hint at 2026-11-28T05:41:21.514296+00:00 (episode ep_01m8gar8f97ntcc6rqcc)
    reason: replace c2 with trusted correction
  - (c2) Pass the date to get_orders as MM/DD/YYYY.
  + (c2) Pass the date to get_orders as YYYY-MM-DD.
```

[Getting started](docs/getting-started.md) walks through each step, and shows how to inspect,
roll back, and let a model write the fixes.

## How it works

Unlock on recall, rewrite on surprise, re-lock.

![The lifecycle of a memory: locked, recalled and editable, through a trust gate, evidence, rewritten in place, re-locked](docs/assets/lifecycle.svg)

1. **Recall unlocks.** Only items recalled in the current task can change.
2. **Real outcomes are the evidence.** Tool errors, verifiers and authenticated users count.
   Timeouts, the model's own reports and anything it read never do.
3. **Enough evidence rewrites the clause that fired**, as a small edit. The old version is
   superseded, never deleted.
4. **The task ends, memory re-locks.**

The idea comes from memory reconsolidation in neuroscience. [How relock works](docs/concepts.md)
covers the trust gate, the evidence threshold and how relock compares with other agent memories.

## Plug it into your stack

relock sits beside your agent framework; it doesn't replace it. The model reads memory, and only
real outcomes can change it.

| You use | Integrate with | Guide |
|---|---|---|
| Any Python agent: OpenAI Agents SDK, PydanticAI, smolagents, CrewAI, LlamaIndex, AutoGen, your own loop | `@mem.wrap` on your tools, `with mem.episode()` around each task | [Python](docs/python.md) |
| Raw tool calling: OpenAI-compatible, Anthropic, vLLM, Ollama | `relock.Toolkit(mem).openai_tools()` and `toolkit.call(name, args)` | [Toolkit](docs/python.md#raw-tool-calling-toolkit) |
| LangChain / LangGraph | `RelockCallbackHandler(mem)` in `config={"callbacks": [...]}` | [LangChain](docs/python.md#langchain-and-langgraph) |
| Any MCP client: Claude Code, Cursor, VS Code... | `relock mcp --verifier tests="pytest -q"` | [MCP](docs/mcp.md) |
| Any language: Rust, TypeScript, Go, Java, C#... | `relock serve` or the Docker image, then four HTTP calls per task | [HTTP](docs/http.md) |

Clients for [Rust](examples/http/rust/src/main.rs), [TypeScript](examples/http/typescript/relock.ts)
and [curl](examples/http/curl.sh) run the HTTP loop against a live server in CI.

## Documentation

- [Getting started](docs/getting-started.md): install, first rewrite, inspect and undo.
- [How relock works](docs/concepts.md): the lifecycle, the trust gate, and comparison with
  other memories.
- Integrations: [Python](docs/python.md), [MCP](docs/mcp.md), [HTTP API](docs/http.md).
- [Configuration](docs/configuration.md): write policy, rewriters and small open models,
  retrieval, storage.
- [CLI](docs/cli.md) and [Production](docs/production.md): logs, retention, Postgres, scale.
- [Design](docs/design.md): the specification and threat model.
- [Benchmarks](docs/benchmarks.md): RelapseBench, and how to run it on your own model.

## Status

v0.3, beta. CI tests every change on Python 3.10–3.14, on SQLite and Postgres, with `mcp` 1.x and
2.x, and from Rust, TypeScript and curl over HTTP. Breaking changes are listed in the
[changelog](CHANGELOG.md). The [roadmap](docs/README.md#roadmap) covers what's next.

## Contributing

Bug reports, integrations, benchmark runs on other models and docs fixes are all welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for setup and the invariants a change must keep. Please report
vulnerabilities privately, as described in [SECURITY.md](SECURITY.md).

## License

Apache-2.0. See [LICENSE](LICENSE).

<p align="center">
  <a href="../README.md"><img src="assets/logo.svg" alt="relock" width="189" height="48"></a>
</p>

# relock documentation

relock is agent memory that fixes the rule that fired, in place and only on trusted evidence, so
corrected mistakes don't come back. These pages go from a first run to production. The
[project README](../README.md) is the one-page overview.

## Start here

| page | what it covers |
|---|---|
| [Getting started](getting-started.md) | Install, store a procedure, wrap a tool, watch the first rewrite, inspect and undo it. About ten minutes. |
| [How relock works](concepts.md) | The lifecycle of a memory, the trust gate, what counts as evidence, and how this differs from other memories. |

## Connect it to your agent

| page | for |
|---|---|
| [Python](python.md) | Any Python agent: `@mem.wrap`, episodes, explicit predictions, verifiers, human corrections, threads and asyncio, raw tool calling with `Toolkit`, LangChain and LangGraph. |
| [MCP server](mcp.md) | Claude Code, Cursor, VS Code and any other MCP client. Configuration only, no code. |
| [HTTP API](http.md) | Any language: Rust, TypeScript, Go, Java, C#, shell. One Docker image, four calls per task. |

## Run it

| page | what it covers |
|---|---|
| [Configuration](configuration.md) | Write policy and its presets, retrieval, storage, rewriters (including small open models), event hooks, training-data export. |
| [CLI](cli.md) | Every `relock` command: inspect items, explain pending evidence, read the audit log, edit, roll back, prune. |
| [Production](production.md) | Logs, errors, several agents in one process, retention, Postgres, upgrades and measured scale. |

## Reference

| page | what it covers |
|---|---|
| [Design](design.md) | The precise specification: data model, lifecycle, gates, error taxonomy, storage schema, [threat model](design.md#threat-model). |
| [OpenAPI](openapi.json) | The HTTP contract (OpenAPI 3.1), for generating typed clients. |
| [Benchmarks](benchmarks.md) | RelapseBench: what it measures, the structural and behavioural results, and how to run it on your own model. |
| [Behavioural results](results/models/) | The 8-model study: every table, interval, cost and caveat, plus every probe call. |

## Project

- [Changelog](../CHANGELOG.md)
- [Contributing](../CONTRIBUTING.md), including the invariants a change must keep
- [Security policy](../SECURITY.md)
- [Examples](../examples/): a 60-line quickstart, an LLM rewriter, LangGraph callbacks, and HTTP
  clients in Rust, TypeScript and curl

## Roadmap

- Behavioural RelapseBench on self-hosted open models, plus a long-horizon variant with real
  context compaction.
- A weight tier: periodically distil the accumulated rewrite traces for a skill into adapter
  weights. Fast edits stay in the harness, durable ones move to the weights.
- pgvector retrieval inside Postgres, and published client packages for the HTTP API (npm,
  crates.io).
- Per-tenant ledgers and signed user corrections.

Ideas and results on other models are welcome in
[issues](https://github.com/inboxpraveen/relock/issues).

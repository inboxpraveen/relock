# CLI

`pip install relock-memory` installs the `relock` command. It works on the same database as your
agent, so you can see what memory holds, why it changed, and undo a change.

```
relock [--db DB] [--log-level LEVEL] <command> ...
```

- **`--db`** is a SQLite path or a `postgresql://` URL. Without it, `$RELOCK_DB`, then
  `./relock.db`. Put it before the command: `relock --db agent.db ls`.
- **`--log-level`** (`DEBUG`, `INFO`, `WARNING`, `ERROR`, or `$RELOCK_LOG_LEVEL`) logs relock
  events to stderr. `relock serve` logs at INFO when it is unset.
- `relock --version` prints the version, and `relock <command> --help` shows every option.

## Read memory

```
relock ls [--all] [--json]                   # active items, with confidence and counters
relock show orders.fetch [--version 1] [--json]
relock history orders.fetch [--json]         # every version, as clause diffs
relock pending orders.fetch                  # pending evidence vs. the rewrite threshold, and what was blocked
relock events [--item REF] [--episode ID] [--type TYPE] [--limit 50] [--json]
relock stats                                 # row counts, to watch growth
```

An item is referred to by its key (`orders.fetch`) or its id (`itm_...`). `relock pending`
answers the most common question, "why hasn't it rewritten yet?":

```
$ relock pending orders.fetch
{
  "item": "orders.fetch",
  "version": 1,
  "confidence": 0.3333,
  "evidence": 1.0,
  "required": 2.0,
  ...
}
```

`relock events` is the audit log: recalls, predictions, confirmations, blocked signals with their
reasons, rewrites and conflicts. `--type` filters by event type, for example `--type blocked` or
`--type rewrite`.

## Change memory by hand

```
relock add "Run tests" "Run the tests with pytest -q." --key tests.run --tool pytest [--tag ci] [--exist-ok]
relock edit orders.fetch --replace c2 "Pass dates as YYYY-MM-DD." --reason "API v2"
relock edit orders.fetch --add "Retry once on HTTP 409." [--after c1]
relock edit orders.fetch --remove c3
relock rollback orders.fetch 1 [--reason "bad rewrite"]
relock archive orders.fetch [--undo]          # hide from retrieval, or bring back
```

- `add` takes one argument per clause after the title.
- `edit` and `rollback` bypass the evidence gates, since a person is making the change. Both
  still create a new version, so history is never lost. `--actor` records who did it (default:
  your user name).
- `rollback` restores an earlier version's clauses as a *new* version.

## Maintenance

```
relock init                                  # create or upgrade the tables, e.g. before a rollout
relock prune --older-than 30d --dry-run      # count the old audit detail that would be deleted
relock prune --older-than 30d                # delete it; items and versions are always kept
relock export traces.jsonl                   # rewrite traces as SFT and preference pairs
```

`--older-than` takes a number with a unit, `s`, `m`, `h`, `d` or `w`, such as `12h`, `30d` or `2w`. It must be longer than any
episode stays open in any process. See [retention](production.md#growth-and-retention).

## Servers and benchmarks

```
relock serve [--host 127.0.0.1] [--port 8765] [--idle-timeout 1800] [-k 3]
relock mcp --verifier tests="pytest -q" [--cwd DIR] [--allow-teach] [--trust-agent-reports]
relock bench [--mode structural|behavioral|both] [--out DIR]
```

- `serve` runs the [HTTP API](http.md). Set `RELOCK_TOKEN` to require a bearer token.
- `mcp` runs the [MCP server](mcp.md).
- `bench` runs [RelapseBench](benchmarks.md).

`serve` and `mcp` take `--llm-base-url`, `--llm-model` and `--llm-api-key` (or `RELOCK_LLM_*`) to
have a model write rewrites.

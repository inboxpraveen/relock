# relock from any language: the HTTP API

`relock serve` exposes relock as a small JSON API. Any code that can send an HTTP request can use
it, in Rust, TypeScript, Go, Java, C#, Ruby or a shell script, with no Python in your stack. The
contract is [`docs/openapi.json`](openapi.json) (OpenAPI 3.1), which is also served at
`GET /v1/openapi.json`, so you can generate a typed client.

```
pip install relock-memory
RELOCK_TOKEN=change-me relock serve                  # http://127.0.0.1:8765/v1
# or
docker run -p 8765:8765 -e RELOCK_TOKEN=change-me -v relock-data:/data ghcr.io/inboxpraveen/relock
```

These clients run the same loop and are tested in CI against a live server:
- [Rust](../examples/http/rust/src/main.rs)
- [TypeScript](../examples/http/typescript/relock.ts)
- [curl](../examples/http/curl.sh)

## The loop

Your harness is the code that runs the model and the tools. Every task goes through the same five
steps:

| step | request | what it does |
|---|---|---|
| 1 | `POST /v1/episodes` `{"task": "..."}` | Opens a reconsolidation window and returns `episode_id`. |
| 2 | `POST /v1/episodes/{id}/recall` `{"query": "..."}` | Returns `context` for your prompt. Recalled items become editable. |
| 3 | *(your model acts and calls a tool)* | |
| 4 | `POST /v1/episodes/{id}/tool-result` `{"tool": "get_orders", "ok": false, "error": {...}}` | Reports what really happened. This is the trusted signal relock learns from. |
| 5 | `POST /v1/episodes/{id}/close` | Re-locks the episode. Its items are no longer editable. |

A failed call carries an error object:

```json
{"tool": "get_orders", "ok": false,
 "error": {"message": "invalid date '03/14/2026'", "status": 422,
           "correction": "Pass the date to get_orders as YYYY-MM-DD."}}
```

- **`correction`** is the rule that would have been right. When enough trusted evidence has
  accumulated (two failures by default), relock rewrites the clause that fired, in place. The
  old version is kept for audit and rollback, and is never retrieved again.
- **Transient failures never rewrite memory.** These are `"status"` 408, 425, 429 or any 5xx,
  or `"transient": true`.
- **Who reported the outcome matters.** With `"source": "agent"` or `"environment"`, a failure is
  logged but never rewrites, and a success never raises confidence or cancels trusted failures.
- **`"detail_trusted": false`** is for a message that echoes remote content, such as a server's
  response body. The failure still counts, but its text never reaches the rewriter.

Items are attributed to a tool through the `tools` list given when the item is added
(`POST /v1/items`). To attribute an outcome to specific items instead, pass `"items": [...]`.

## Other endpoints

| request | use |
|---|---|
| `POST /v1/items` | Add a procedure: `title`, `clauses`, optional `key`, `tools`, `exist_ok`. |
| `GET /v1/items`, `GET /v1/items/{ref}`, `GET /v1/items/{ref}/history` | Read active items, one item, or every version of it. |
| `POST /v1/episodes/{id}/use` | Load an item by key, which makes it editable without a search. |
| `POST /v1/episodes/{id}/expect` | Record an explicit prediction. `clauses` limits which clauses can be rewritten; `tool` binds the prediction to a tool. |
| `POST /v1/episodes/{id}/observe` | Resolve one expectation. `source` is `tool` (default), `verifier`, `agent` or `environment`. |
| `POST /v1/episodes/{id}/correct` | A human correction. It rewrites only with `"authenticated": true`. |
| `GET /v1/health` | Liveness and version. No token needed. |

Errors are JSON, `{"error": {"code": "...", "message": "..."}}`, with these statuses:
- `400`: bad input;
- `401`: missing or wrong token;
- `404`: unknown item, or an episode that is closed or expired;
- `409`: the item was not recalled in this episode, or the episode is closed;
- `413`: body over 1 MiB.

## Trust

- **Only your harness should call this API.** It trusts each outcome according to the `source`
  that is sent: `tool` and `verifier` outcomes can rewrite memory; `agent` and `environment`
  outcomes are logged but never rewrite it.
- **Never let the model call it directly.** For a model that should use memory by itself, run
  the [MCP server](../README.md#mcp) instead. There, nothing the agent writes is trusted.
- **Token:** set `RELOCK_TOKEN` to require `Authorization: Bearer <token>`. Without a token the
  server only binds to loopback; `relock serve --host 0.0.0.0` refuses to start.
- **Idle episodes are re-locked** after 30 minutes (`--idle-timeout`) by a background sweeper. A
  crashed client can't leave memory editable.
- **Strict JSON:** a field like `"authenticated": "false"` or `"status": "503"` is a `400`, never
  read as true or as a number. A request can't change what it means by changing a type.
- **No TLS:** the server speaks plain HTTP. Beyond loopback or a private network, put it behind a
  reverse proxy that terminates TLS, so the token never crosses a network in clear text.

## Deployment

- **Storage:** SQLite by default (`--db` or `RELOCK_DB`). For Postgres, set
  `RELOCK_DB=postgresql://...`; the image includes the driver.
- **Concurrency:** requests run on threads, serving up to 256 connections at once; more wait to
  be accepted. A connection idle or stalled for 30 seconds is closed. Writes use compare-and-swap,
  so concurrent rewrites of the same item never lose an update; the later one becomes a
  `conflict`.
- **Bodies:** `Content-Length` and chunked bodies are both accepted. A request carrying both
  framings is refused.
- **Shutdown:** on `SIGTERM` (for example `docker stop`) the server re-locks every open episode
  and closes the database before exiting.
- **Replicas:** open episodes live in the server process. With more than one replica on a shared
  Postgres database, send each episode's requests to the same replica, for example with sticky
  routing on the episode id.
- **LLM rewrites:** start the server with `--llm-base-url` / `--llm-model` (or the
  `RELOCK_LLM_*` variables) to have a model write the corrected clause. Without it, the
  deterministic hint rewriter applies the `correction` text.

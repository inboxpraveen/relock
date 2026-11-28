# Changelog

## 0.3.1 - 2026-11-28

A review of the whole framework: gaps in how memory learns from real outcomes, edge cases in the
store, the HTTP API, the MCP server and the CLI, and clearer errors. The structural benchmark is
unchanged.

**Changes you may notice when upgrading:**

- A success right after a failure of the same tool in the same episode, where the failure
  carried a correction, is now `ignored` instead of `confirmed`. Before, an agent that read the
  correction and retried cancelled the evidence, so the stale rule was never fixed. A success
  now cancels pending errors only of the same tool.
- `OPENAI_API_KEY` is sent only to `api.openai.com`. For another endpoint that needs a key, set
  `RELOCK_LLM_API_KEY` or pass `api_key=`.
- HTTP API:
  - observing an expectation twice is a `409 already_observed`, not a `400`;
  - a database failure is a `503 unavailable` (details logged, not sent), not a `400`;
  - other methods get a JSON `405` with `Allow`, or a `404`, instead of an HTML `501`;
  - a JSON body that repeats a key, a `verifier` field on a non-verifier source, and an invalid
    `error.class` on any source are now `400`s.
- CLI:
  - an unknown `scheme://` in `--db` or `$RELOCK_DB` is an error instead of a file name;
  - `relock prune` with an age under one hour needs `--force`;
  - `relock events --limit` must be at least 1 (`-1` used to mean unlimited on SQLite);
  - `relock serve` startup errors exit with 1 instead of 2.
- `Toolkit.report_outcome` refuses words it doesn't recognise instead of reading them as a
  failure. `confirmed`, `success`, `passed`, `ok`, `failed`, `error` and similar still work.
- `LLMRewriter(fallback=None)` now means no fallback. The default is still `HintRewriter`.
- An invalid `WritePolicy` (a negative threshold, an unknown `rewrite_on` or source) raises
  `ValueError` when it is built. `recall` / `search` with a `k` that isn't a whole number raise
  `TypeError`.
- An MCP verifier run that times out or can't start is recorded as a transient `verifier`
  failure (before: a `tool` failure). Neither ever counted as evidence.

**Learning from real outcomes:**

- `checks.fails()` matches a call that raises. The check now sees the error; a check that itself
  raises counts as a mismatch instead of crashing the observation.
- An explicit expectation with `tool=` is resolved by that tool's next call even when the item
  doesn't list the tool.
- An authenticated correction that the rewriter declines is closed, so it can't carry later
  failures past the evidence threshold.
- An error given only as a message (`tool_result(error="...")`, the HTTP API) is transient when
  it reads like a timeout, a rate limit, a 5xx or a dropped connection.
- `ToolError(..., detail_trusted=False)` keeps a tool's own error text from the rewriter.
- Interrupts and cancellation (`KeyboardInterrupt`, `asyncio.CancelledError`, LangGraph's
  `interrupt()`) are no longer recorded as failures, by `@mem.wrap` or the LangChain handler.
- A tool wrapped twice, or wrapped and also observed by the LangChain handler, is recorded once.
- `@mem.wrap` never raises from its own bookkeeping, for example with an unknown `items=` key; it
  logs a warning once. It awaits `async` callable objects, and names a `functools.partial` after
  its function.
- `tool_result(items=[...])` warns about items that weren't recalled instead of skipping them
  silently.
- `Episode.close()` re-locks the episode even when the database fails while closing.

**Edits and rewriters:**

- An edit is refused when it uses an unknown op, writes empty text, touches one clause twice, or
  leaves two identical clauses. Clause ids written as `2` or `C2` are read as `c2`.
- `LLMRewriter` retries a reply without an `ops` list, prefers the JSON object that has `ops`
  when a reply quotes an example first, and ignores reasoning cut off before `</think>`.
- `HintRewriter` declines a correction longer than `max_clause_chars` (it used to write it
  truncated) or one the item already states.
- The rewriter sees at most the 20 most recent pending errors; a rewrite consumes all of them.
- `OpenAICompatible` reads content sent as a list of parts, and an unexpected response is a
  clear `ToolError` instead of a `KeyError`.
- `Memory(rewriter=...)` and `LLMRewriter(llm)` check their arguments, and `Evidence` is
  exported for typing custom rewriters.

**Storage:**

- A rewrite whose evidence another process cancelled or used meanwhile rolls back as a
  `conflict`. Flake damping changes only errors that are still pending.
- An error stored against a version that was already superseded is stored `stale`, so it no
  longer keeps its episode from being pruned.
- NUL bytes and lone surrogates in text are stored as U+FFFD, and NaN or infinity in JSON as
  null, instead of failing the write (Postgres rejected them).
- `open_store` and `Memory` accept a `pathlib.Path`, `file:` URIs, `postgres://` and
  `postgresql+psycopg://` URLs. `$RELOCK_PG_POOL_SIZE` sets the Postgres pool size.
- A database directory that can't be created is a `StoreError`.
- An id wins over a key that equals it, and `str(NotFoundError)` no longer wraps the message in
  quotes.
- `relock prune --dry-run` counts the expectations prune trims from abandoned episodes.

**HTTP API:**

- An outcome's `error` fields mean the same for every `source`: a verifier failure can be
  transient, carry a `status` or a `class`, and mark its text `detail_trusted: false`.
- Repeated `Content-Length` or `Transfer-Encoding` headers, and a non-ASCII `Content-Length`,
  are refused. `HEAD` never gets a body.
- The OpenAPI document describes recall hits (`score`, `rank`), the close summary and `author`.
- Client disconnects are logged at DEBUG instead of printed.

**MCP server:**

- `--untrusted-verifier-output` (`trust_verifier_output=False`) keeps verifier output from the
  rewriter; the pass/fail verdict still counts. The default is unchanged.
- Verifiers run without the server's stdin, decode output as UTF-8, and a timeout kills
  everything they started. The server warns at startup about a verifier whose program isn't
  found, and checks the expectation before running a verifier.
- `relock_report`'s `outcome` is `confirmed` or `failed` in the tool schema.
- On the network transports the server warns that every client shares one episode.

**Toolkit:**

- Concurrent first calls share one episode instead of opening several.
- `call()` returns an `error: ...` string for bad JSON or mistyped arguments instead of raising.
  A bare string is accepted where a list is expected.

**CLI and Docker:**

- `--db` and `--log-level` work after the command too (`relock ls --db agent.db`).
- Every command reports failures as `error: ...` with exit status 1. `relock mcp` without the
  `mcp` extra fails before touching the database.
- `relock prune --keep-events TYPE` keeps more event types. Ages are capped at 100 years.
- `relock edit` keeps several `--add` clauses in order with `--after`, and refuses two
  operations on one clause.
- `relock serve` ignores a `RELOCK_PORT` that isn't a port number (Kubernetes sets one for a
  Service named `relock`) with a warning.
- `relock export` rebuilds the prompt the rewriter saw, including the task and only trusted
  citations.
- The Docker health check probes the HTTP API only when the container runs `serve`.
- `relock mcp --untrusted-verifier-output` sets the MCP option above.

**Authoring:**

- `tools`, `tags` and `kinds` given as a single string are one name, not split into
  characters.
- `add(exist_ok=True)` warns without a `key` (it creates a new item each time) and when the item
  it finds is archived.
- Each entry of a clause list is one line: line breaks and runs of spaces inside it become one
  space. A one-line string loses a leading bullet, as multi-line strings already did.
- Empty names in `tools` or `tags` are dropped with a warning.
- Retrieval tokenises letters and digits in any script, so non-English items are found by their
  own words. ASCII text tokenises as before.

**Security:**

- `OPENAI_API_KEY` is no longer sent to non-OpenAI endpoints (see above).
- The HTTP API no longer echoes database paths or driver errors to clients.

## 0.3.0 - 2026-11-28

Production hardening: multi-agent attribution, storage robustness, scale, retention and logging.

**Breaking:**

- A wrapped tool's outcome is attributed only to the episode in the caller's context. The
  fallback to "the only open episode in the process" is gone: it attributed one agent's tool
  calls to another agent's episode. In worker threads, use `ep.run(fn, ...)` or
  `with ep.activate():`.
- Database failures raise `StoreError` (and lost unique-key races `ConflictError`) instead of raw
  `sqlite3` / `psycopg` exceptions. The driver's exception is chained as the cause.
- `GET /v1/items` returns a page: at most `limit` items (default 500), with `total` and
  `next_after` (pass it back as `after`). Clients that expect every item in one response must
  follow `next_after`.
- `policy` is now a keyword of `Memory.episode` (it sets that episode's write policy) instead of
  being stored as metadata.
- The database schema is v2. The first 0.3 process upgrades a 0.2 database in place; 0.2
  processes keep working against it. On a large Postgres ledger, run `relock init` once at a
  quiet time before rolling out, since the upgrade builds indexes over the events table.

**Multiple agents:**

- `Episode.activate()` and `Episode.run(fn, ...)` carry an episode into worker threads.
- Separate `Toolkit` instances keep separate episodes.
- A wrapped tool that runs outside every episode while others are open logs a warning, once per
  tool.
- `Memory.episode(policy=...)` and `Toolkit(policy=...)` set trust per agent.
- `build_server(trust_agent_reports=True)` widens trust for the MCP server's own episodes only.
  Before, it replaced the shared `Memory`'s policy.

**Storage:**

- Postgres serves concurrent calls from a connection pool (`pool_size`, default 5). It checks idle
  connections before use and discards dropped ones, so after a database restart each pooled
  connection fails at most one call.
  `connect_timeout` defaults to 10 s.
- Replicas starting together on an empty database no longer race: schema creation takes an
  advisory lock, and an up-to-date database is opened without DDL.
- `add(exist_ok=True)` holds when several processes seed the same keys at once.
- A database written by a newer relock is refused with a clear message.
- SQLite reads no longer queue behind writers in other processes, and several processes can
  create a new database file at once.

**Scale:**

- After a content change, each process re-reads only the items that changed. At 100,000 items,
  recall right after a change went from about 5 s to 0.2 s.
- Confirmations no longer reload every item. At 20,000 items, recall after a confirmation went
  from 162 ms to 3 ms.
- BM25 keeps an incremental inverted index, and `kinds` / `tags` filters reuse it.
- Long episodes keep only open expectations in memory, so per-call cost stays flat.
- The embedding cache drops vectors of versions that are no longer active.

**Retention:**

- `Memory.prune(older_than)` and `relock prune --older-than 30d [--dry-run]` delete old audit
  detail and never touch items or versions. An episode goes only if nothing depends on it.
  Episodes a crashed process left open are marked `abandoned`.
- `stats` reports open episodes, retrievals and events.
- Error text, reasons and queries stored in the ledger are capped at 2000 characters.

**Logging and audit:**

- Every ledger event is logged on the `relock_memory` logger: content changes and failures at
  INFO with ids and a short reason, routine events at DEBUG, payloads only at DEBUG.
- New `conflict` and `rewrite_deferred` events, so every case where evidence did not produce a
  rewrite leaves a trace. `declined` carries the rewriter's reason, such as an unreachable LLM.
- A failing `on_event` hook, LLM rewriter failures and fallbacks, and lost database connections
  are logged instead of passing silently. `LLMRewriter.last_error` no longer leaks from one
  request into the next.
- `episode_open` records the episode's metadata.
- `relock --log-level` (or `RELOCK_LOG_LEVEL`) logs to stderr, so `relock mcp` can show events
  without disturbing its stdio protocol.

## 0.2.2 - 2026-11-28

Documentation only; no code changes.

- Fix the citation link for Sevenster, Beckers and Kindt (2013) in the README.

## 0.2.1 - 2026-11-28

Documentation only; no code changes.

- A logo, and an animated diagram of the core idea: the same correction in a typical memory and
  in relock.
- Diagrams of a memory's lifecycle (replacing the Mermaid chart, which PyPI can't render) and of
  where relock sits beside an agent.
- Citations for the neuroscience the design borrows from.
- A social preview card for links to the repository.

## 0.2.0 - 2026-11-28

Behavioural results on 8 hosted models, use from any language over HTTP, and a container image.

- **Use relock from any language:**
  - `relock serve` is an HTTP/JSON API with no dependencies, described by OpenAPI 3.1 in
    `docs/openapi.json`.
  - Security: bearer-token auth, and it refuses to bind publicly without a token. JSON types are
    strict, so `"authenticated": "false"` is rejected, not read as true. Agent and environment
    successes never raise confidence.
  - Operation: idle episodes are re-locked in the background. It caps request size and
    connections, times out stalled clients, accepts chunked bodies, and shuts down cleanly on
    `SIGTERM`.
  - Example clients in Rust, TypeScript and curl run in CI against a live server.
  - A multi-arch container image is published to `ghcr.io/inboxpraveen/relock` on each release.
- **`Episode.tool_result(tool, result, error=..., signal=...)`** reports a tool's real outcome
  without wrapping the tool, the same way `@mem.wrap` does.
- **Fix:** a failure passed explicitly as a signal, such as
  `ep.observe(exp, signal=Signal.tool_error(...))`, was recorded as a success when the
  expectation had a `succeeds()` check and no exception was passed.
- **Fix:** an episode closed while a write was in flight could still commit a rewrite after
  re-lock. `observe`, `expect`, `recall` and `use` now re-check under the episode lock, and
  `tool_result` binds and resolves atomically, so concurrent reports for one tool never collide.
- **Every 5xx status is now transient.** A server fault says nothing about whether the agent's
  call was right. Previously 501, 505 and some other codes counted as `unknown` (weight 0.5).
- **RelapseBench behavioural results on 8 hosted models**, in `docs/results/models/`. In long
  sessions, agents made the corrected call 98% of the time with relock, against 60% with
  append-notes and 52% with append-clause. The study code is `scripts/model_study.py` and
  `scripts/model_report.py`.
  - The behavioural scorer now judges which rule a call follows, ignoring formatting. For example,
    `a;b` follows the same `;` rule as `a; b`. No behavioural results had been published before
    this change.
- Tested on Python 3.14 (CI and package classifiers). Development status is now beta.

## 0.1.1 - 2026-11-28

First release on PyPI: `pip install relock-memory`. No changes to the memory mechanism.

- `relock mcp --cwd DIR` sets the directory the verifier commands run in.
- README:
  - install from PyPI;
  - MCP setup with `uvx` and `claude mcp add`;
  - links and the results chart also work on the PyPI project page.
- The release workflow publishes to PyPI only for release events, after the version check.

## 0.1.0 - 2026-11-27

First public version.

- Core mechanism:
  - versioned clause-level items;
  - episodes with retrieval-gated lability;
  - expectations with programmatic checks;
  - trust, transient, evidence and edit-scope gates;
  - in-place reconsolidation with supersession, rollback and manual edits;
  - an append-only audit log.
- Storage: SQLite (default, zero dependencies) and Postgres (`[postgres]` extra), with a partial
  unique index enforcing one active version per item and compare-and-swap commits. Ids are
  monotonic within a process.
- Retrieval: BM25 by default, plus embedding and hybrid retrievers (reciprocal-rank fusion).
- Rewriters:
  - deterministic `HintRewriter`;
  - `LLMRewriter` for any `Callable[[str], str]`, with JSON repair, retry with feedback and fallback;
  - dependency-free `OpenAICompatible` client.
- Integrations:
  - tool wrapping for any Python framework (sync and async);
  - framework-agnostic `Toolkit` with OpenAI and Anthropic tool schemas;
  - MCP server (mcp 1.x and 2.x) with server-side trusted verifiers;
  - LangChain/LangGraph callback handler and tools.
- Safety defaults:
  - Nothing the agent writes steers a rewrite unless the operator trusts agent reports. This
    covers predictions stated through the toolkit or MCP, the clause ids they cite, and
    corrections suggested to `relock_verify`.
  - User corrections require `authenticated=True`.
  - MCP teaching is opt-in (`--allow-teach`).
  - Toolkit-owned episodes re-lock after 30 idle minutes.
- CLI: `relock add | ls | show | history | pending | events | rollback | edit | archive | stats | export | mcp | bench`.
- RelapseBench:
  - ten drifting tool contracts and seven memory systems, including two ablations and an
    uncited relock variant;
  - four load levels;
  - a reproducible, model-free structural mode with published results;
  - a behavioural mode for any OpenAI-compatible model, with no results published yet.
- Export of rewrite traces as SFT and preference pairs.
- Tested on Python 3.10–3.13; CI workflow for SQLite, Postgres and `mcp` 1.x.

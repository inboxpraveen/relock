# Changelog

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

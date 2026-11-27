# Changelog

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

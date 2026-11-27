# Changelog

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

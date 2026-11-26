# Changelog

## 0.1.0 - unreleased

First public version.

- Core mechanism:
  - versioned clause-level items;
  - episodes with retrieval-gated lability;
  - expectations with programmatic checks;
  - trust, transient, evidence and edit-scope gates;
  - in-place reconsolidation with supersession, rollback and manual edits;
  - an append-only audit log.
- Storage: SQLite (default, zero dependencies) and Postgres (`[postgres]` extra), with a partial
  unique index enforcing one active version per item and compare-and-swap commits.
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
- CLI: `relock add | ls | show | history | pending | events | rollback | edit | archive | stats | export | mcp | bench`.
- RelapseBench:
  - ten drifting tool contracts and six memory systems, including two ablations;
  - four load levels;
  - a model-free structural mode with published results;
  - a behavioural mode for any OpenAI-compatible model, with no results published yet.
- Export of rewrite traces as SFT and preference pairs.

# Configuration

Everything is set on `Memory`:

```python
mem = relock.Memory(
    "agent.db",                         # storage: a path, ":memory:", "postgresql://...", or a store
    policy=relock.WritePolicy(),        # what may change memory, and how much evidence it needs
    rewriter=relock.HintRewriter(),     # who writes the new clause
    embed=None,                         # optional embeddings for hybrid retrieval
    on_event=None,                      # a callback for every ledger event
)
```

## Write policy

`WritePolicy` holds the gates. The defaults are the ones the [benchmarks](benchmarks.md) used.

```python
relock.WritePolicy(
    min_evidence=2.0,            # weighted trusted errors before a rewrite
    inertia=0.5,                 # extra evidence per ln(1 + confirmations) of the current version
    confirmation_cancels=1.0,    # a later trusted success cancels this much pending evidence
    rewrite_on="observe",        # or "close": rewrite only when the episode ends
    edit_scope="cited",          # replace/remove only clauses the prediction relied on, if it cited any
    max_ops=4,                   # most clause operations in one rewrite
    max_clause_chars=600,        # longest clause a rewrite may write
    max_rewrites_per_item_per_episode=1,
    attribute_undeclared=False,  # also attribute tool outcomes to recalled items that list no tools
    authoritative_users=True,    # authenticated user corrections skip the evidence threshold
)
```

**Presets:**

| preset | use it for |
|---|---|
| `WritePolicy.strict()` | Only verifiers and authenticated users can rewrite, and a rewrite needs 3 units of evidence. |
| `WritePolicy.eager()` | One trusted error is enough. Learns a correction one failure sooner, at the cost of robustness to a one-off bad error. |
| `WritePolicy.trust_agent()` | Also trust the agent's own reports. Local, single-user sandboxes only. |

Change one field of any policy with `policy.with_(min_evidence=1.0)`. Set a policy for one
agent with `mem.episode(task, policy=...)` or `Toolkit(mem, policy=...)`.

**Evidence and weights.** A failure counts with weight `source_weight * class_weight`. Tools and
verifiers weigh 1 and authenticated users 2. Programmatic and semantic errors weigh 1, unknown
errors 0.5 and transient errors 0. A rewrite needs
`min_evidence + inertia * ln(1 + confirmations)`. `trusted_sources`, `source_weights` and
`class_weights` can be set too; the [design spec](design.md#admission-trust-gate) has the exact
admission rules.

**Why two errors by default?** A single error might be a fluke, a bad deploy, or a wrong
"correction". Waiting for a second trusted error costs one extra failure per real change and makes
one-off noise unable to rewrite a good procedure. The [benchmarks](benchmarks.md) measure that
trade.

## Rewriters

The rewriter turns trusted evidence into a small clause edit. relock validates whatever it
proposes against the policy before anything is written.

| rewriter | how it writes the fix |
|---|---|
| `HintRewriter()` (default) | Deterministic, no model. Applies the `correction` carried by trusted errors: authenticated users first, then the most frequent, then the most recent. It replaces the cited clause, or the most similar clause when none was cited, and adds a clause when nothing is similar. Without a correction it declines, and the evidence waits. |
| `LLMRewriter(llm)` | A model reads the item, the evidence and the clause ids, and returns a JSON edit. Invalid output is fed back for up to `max_attempts=3` tries, then it falls back to `HintRewriter`. |
| `CallableRewriter(fn)` | Your own function from a `RewriteRequest` to ops. |

`llm` is any `Callable[[str], str]`. `OpenAICompatible` is a dependency-free client for vLLM,
SGLang, Ollama, llama.cpp, LM Studio and hosted OpenAI-compatible endpoints:

```python
llm = relock.OpenAICompatible("http://localhost:8000/v1", model="Qwen/Qwen3-30B-A3B-Instruct-2507",
                              api_key=None, timeout=120.0, retries=2)
mem = relock.Memory("agent.db", rewriter=relock.LLMRewriter(llm))
```

With no arguments it reads `RELOCK_LLM_BASE_URL`, `RELOCK_LLM_MODEL` and `RELOCK_LLM_API_KEY`.

### Small open models

The hard part of self-improving memory is credit assignment: which memory caused the failure?
relock answers that structurally, through the retrieval trace, the cited clauses and the
expectation diff. The rewriter only has to make a *local* edit: "here is the item, here is what it
predicted, here is what happened, rewrite this clause".

The rewrite prompt is built for models around 30B: one item, explicit clause ids, strict JSON and
one worked example. `<think>` blocks, code fences and chatter around the JSON are tolerated. In the
[behavioural study](results/models/), each of the 8 models wrote relock's rewrites itself, two of
them 27B models, and none of the 960 rewrites fell back to `HintRewriter`.

## Retrieval

Retrieval is BM25 by default, with no dependencies. Only active, unarchived versions are ever
candidates. `ep.recall(query, kinds=[...], tags=[...])` filters by item kind and tags.

- **Hybrid search:** pass `embed=`, any `list[str] -> list[vector]` function, and relock fuses
  BM25 and vector rankings with reciprocal-rank fusion.
- **Your own index:** pass `retriever=` with any object that has
  `rank(query, items, k) -> list[(item, score)]`. `BM25Retriever`, `EmbeddingRetriever` and
  `HybridRetriever` are exported to build on.

Retrieval holds active items in memory and refreshes only the items that changed. That is
comfortable into the tens of thousands of items; see [measured scale](production.md#scale).

## Storage

| store | how to open it | notes |
|---|---|---|
| SQLite (default) | `Memory("agent.db")` | WAL mode, safe across processes on one machine. `":memory:"` for tests and notebooks. |
| Postgres | `Memory("postgresql://user:pw@host/db")` | Needs `relock-memory[postgres]`. Uses `JSONB` and `TIMESTAMPTZ`, so the ledger is directly queryable. |

With no argument, `Memory()` opens `$RELOCK_DB`, or `./relock.db`. For pool settings, construct
the store yourself:

```python
from relock_memory.stores.postgres import PostgresStore
mem = relock.Memory(PostgresStore(dsn, pool_size=10, pool_timeout=30.0))
```

Tables are prefixed `relock_`; the schema is in the [design spec](design.md#storage-schema).
Running Postgres in production is covered in [Production](production.md#postgres).

## Event hook

`on_event` receives every ledger event as it is stored: recalls, predictions, confirmations,
blocked signals, rewrites, conflicts. Use it for tracing or metrics. It runs synchronously, so keep
it fast or hand events to a queue. If it raises, the error is logged and memory carries on. The
same events are logged on the `relock_memory` logger; see [Production](production.md#logs-and-audit).

## Training data from rewrites

Every rewrite is a labelled example. `relock export traces.jsonl` writes:

- SFT pairs: the rewrite `prompt` and the accepted `completion`;
- preference pairs: `rejected` is the superseded clauses, `chosen` the new clauses.

That is the raw material for distilling a rewriter, or for consolidating corrections into model
weights.

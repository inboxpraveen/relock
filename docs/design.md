# relock design

This document specifies the mechanism precisely: data model, lifecycle, gates, storage and
threat model. The README covers the why.

## Data model

An **item** is a versioned list of **clauses**. A clause is one short imperative sentence with a
stable id (`c1`, `c2`, ...). Clause ids are never reused within an item. A `replace` keeps the id,
and an `add` takes the next number. Each version carries:

| field | meaning |
|---|---|
| `status` | `active` (exactly one per item) or `superseded` |
| `alpha`, `beta` | Beta posterior over "this version's predictions hold". `confidence = alpha / (alpha + beta)`. New versions start at the prior `(1, 1)`. |
| `confirmations`, `failures` | raw counts of trusted confirmations and admitted prediction errors |
| `change_kind` | `create`, `reconsolidate`, `rollback` or `manual` |
| `delta` | the ops that produced this version from its parent |
| `evidence` | ids of the prediction errors consumed by the rewrite |
| `episode_id`, `author`, `reason` | provenance: which episode, which rewriter or person, and why |
| `tools` | tool names whose outcomes are attributed to this item |

## Lifecycle

```
episode = mem.episode(task)            # open window
  recall(query) / use(key)             # reactivation: items -> labile set (id -> version seen)
  expect(item, statement, clauses, check, tool)
  <act>
  observe(expectation, result | error, signal)
      matched  -> trusted?  confidence += 1, cancel pending evidence (flake damping)
      mismatch -> admit(signal)?
                    no  -> error row status=blocked (audited, never counts)
                    yes -> error row status=pending, beta += 1
                           reconsolidate(item) if rewrite_on == "observe"
close()                                # re-lock; open expectations -> void
```

`reconsolidate(item)` runs these checks in order. Any failure returns a `Decision` and writes nothing.

1. The item is labile in this episode.
2. The item's current version equals the version this episode recalled. Otherwise the result is
   `CONFLICT`: someone else rewrote it, so this evidence is about a version that no longer exists.
3. Pending evidence for the current version satisfies
   `sum(weight) >= min_evidence + inertia * ln(1 + confirmations)`. An authenticated user
   correction is authoritative and skips this check.
4. Fewer than `max_rewrites_per_item_per_episode` rewrites have happened (default 1, which
   prevents oscillation within one window).
5. The rewriter proposes a delta.
6. The delta validates:
   - ops reference existing clauses;
   - `replace`/`remove` touch only *cited* clauses when `edit_scope="cited"` and any were cited;
   - there are at most `max_ops` ops;
   - no clause exceeds `max_clause_chars`;
   - the item keeps at least one clause;
   - the delta is not a no-op.
7. A compare-and-swap commit:
   ```
   UPDATE items SET current_version = new WHERE id = ? AND current_version = old
   ```
   followed by superseding the old version, inserting the new one, marking the evidence
   `consumed` and marking any other pending errors on the old version `stale`, all in one
   transaction. A partial unique index `(item_id) WHERE status = 'active'` makes a second active
   version impossible even under bugs or concurrent writers.

After a rewrite the item stays labile in the same episode at its new version, so the agent can
re-read it (`ep.get(item)`) and retry within the same task.

## Admission (trust gate)

`WritePolicy.admit(signal)` returns `(counts, weight, authoritative)`:

- `source` not in `trusted_sources` -> blocked. The default trusted set is `TOOL`, `VERIFIER`, `USER`.
- `USER` without `authenticated=True` -> blocked.
- `error_class == TRANSIENT` -> blocked.
- Otherwise `weight = source_weight * class_weight`. The defaults are:
  - source weights: tool 1, verifier 1, user 2;
  - class weights: programmatic 1, semantic 1, unknown 0.5, transient 0.

Confirmations raise confidence only from trusted sources. Otherwise an attacker could inflate
inertia with fake "it worked" reports and make a poisoned procedure harder to fix.

### Error taxonomy

`classify_exception` resolves the error class from the first rule that matches:

1. An explicit `relock_error_class` attribute (set by `ToolError` / `TransientToolError`).
2. HTTP-style status codes on the exception or its `.response`:
   - 408, 425, 429 and 5xx -> transient;
   - 400, 401, 403, 404, 409, 422 and similar -> programmatic.
3. Built-in exception types:
   - `TimeoutError` and `ConnectionError` -> transient;
   - `AssertionError` -> semantic;
   - `ValueError`, `TypeError`, `LookupError` -> programmatic.
4. Class-name hints across the MRO (`ReadTimeout`, `RateLimitError`, `ValidationError`, ...).
5. Anything else -> unknown (weight 0.5).

## Attribution of tool outcomes

A wrapped tool (`@mem.wrap`) or the LangChain callback handler looks up the episode active in the
caller's context (a context variable). Inside `with mem.episode(...)`, `with ep.activate():` and
`ep.run(fn, ...)` it is that episode; asyncio tasks inherit it, plain worker threads do not. There
is no process-wide fallback: with several agents in one process, a tool outcome is attributed only
to the episode of the agent that made the call. A wrapped tool that runs outside every episode
while some are open is not recorded, and a warning names it (once per tool). It then resolves:

1. any open, explicit expectation that names this tool for a labile item;
2. otherwise, an automatic `succeeds()` expectation for each labile item whose `tools` contains the
   tool name (plus items that declare no tools, if `attribute_undeclared=True`).

Instrumentation never changes tool behaviour. Results are returned unchanged, exceptions are
re-raised, and errors inside relock are logged rather than raised.

## Rewriters

A `RewriteRequest` contains the current item, the admitted evidence (sanitised), the editable
clause set and the limits. `detail_trusted=False` signals contribute their error class but never
their text. Implementations:

- `HintRewriter`: deterministic, model-free. It applies the explicit `correction` carried by
  trusted signals: authoritative corrections first, then the most frequent, then the most recent.
  It replaces the cited clause (or the most similar clause when nothing was cited) and adds a new
  clause when nothing is similar.
- `LLMRewriter(llm)`: builds a one-item, clause-id prompt and parses JSON while tolerating
  `<think>` blocks, code fences and chatter. It validates the delta against the same rules and
  feeds errors back for up to `max_attempts`. On failure it falls back to `HintRewriter`.
- `CallableRewriter(fn)`: any function returning ops.

## Storage schema

All tables are prefixed `relock_`. SQLite stores JSON and timestamps as `TEXT`. Postgres uses `JSONB`,
`TIMESTAMPTZ`, `BOOLEAN` and `DOUBLE PRECISION`.

| table | key columns |
|---|---|
| `items` | `id`, `key` (unique), `current_version`, `archived`, `rev` (revision of its last content change) |
| `versions` | `(item_id, version)` PK, `status`, `title`, `clauses`, `clause_seq`, `tags`, `tools`, `meta`, `alpha`, `beta`, `confirmations`, `failures`, `parent_version`, `superseded_by`, `change_kind`, `delta`, `reason`, `episode_id`, `evidence`, `author` |
| `episodes` | `id`, `task`, `status` (open/closed/abandoned), `meta`, `started_at`, `ended_at` |
| `retrievals` | `episode_id`, `item_id`, `version`, `query`, `rank`, `score` (the reactivation trace) |
| `expectations` | `id`, `episode_id`, `item_id`, `version`, `statement`, `clause_ids`, `tool`, `check_spec`, `status` (open/confirmed/violated/void), `outcome`, `signal` |
| `errors` | `id`, `expectation_id`, `item_id`, `version`, `source`, `error_class`, `weight`, `authoritative`, `diff`, `status` (pending/blocked/cancelled/consumed/stale), `reason` |
| `events` | append-only audit log: `type`, `episode_id`, `item_id`, `version`, `data` |
| `meta` | `schema_version` (2), `revision` (bumped on every content change; with `items.rev`, lets each process refresh only the items that changed) |

A database from relock 0.2 (schema v1) is upgraded in place by the first 0.3 process that opens
it. The upgrade adds `items.rev` and some indexes; 0.2 processes keep working against it, and their
changes are still picked up by time. `Memory.prune` (`relock prune`) deletes old audit detail
(retrievals, expectations, errors and events of episodes nothing depends on) and never items or
versions.

Example query on Postgres:

```sql
-- which items were rewritten from verifier failures this week, and what changed?
SELECT v.item_id, v.version, v.delta, v.reason
FROM relock_versions v
WHERE v.change_kind = 'reconsolidate' AND v.created_at > now() - interval '7 days'
  AND EXISTS (SELECT 1 FROM relock_errors e
              WHERE e.id IN (SELECT jsonb_array_elements_text(v.evidence)) AND e.source = 'verifier');
```

## Threat model

| Threat | Mitigation |
|---|---|
| Prompt injection rewrites a trusted procedure ("memory poisoning") | `ENVIRONMENT`, `AGENT` and `SELF` sources cannot open writes; their errors are stored as `blocked` for audit. |
| A (possibly injected) agent piggybacks its own text on a *trusted* failure | Text reaches the rewriter only from trusted authors. Predictions stated through the toolkit or MCP are marked `author="agent"` and withheld from the rewriter. A `correction` the agent passes to `relock_verify` is dropped. Clause ids the agent cites don't set the edit scope, so it can't aim a genuine failure's fix at a clause it wants gone. All three apply unless the operator trusts agent reports. As defence in depth, `HintRewriter` adds the correction rather than overwrite a cited clause when another clause resembles it more. |
| Injected text reaches the rewriter through a trusted tool's error message | Mark the signal `detail_trusted=False`. The failure counts, but its text is withheld from the rewriter. |
| One flaky failure overwrites a good procedure | Transient classes never count; evidence threshold with inertia; flake damping by later confirmations. |
| Fake confirmations make a poisoned item sticky | Only trusted confirmations raise confidence and inertia. |
| Rewriter edits unrelated parts of an item | `edit_scope="cited"` locks every clause the failed prediction did not rely on, when it cites clauses. Uncited predictions, such as `@mem.wrap` auto-expectations, leave every clause editable, so the choice falls to the rewriter. `HintRewriter` picks the clause most similar to the correction. Cite clauses when you use an LLM rewriter. |
| Repeated rewrites collapse an item | Edits are small clause deltas (`max_ops`, `max_clause_chars`); one rewrite per item per episode. |
| Concurrent episodes overwrite each other | Compare-and-swap on `current_version`; stale evidence becomes `CONFLICT`, recorded as a `conflict` event. |
| One agent's tool outcome rewrites another agent's memory | Outcomes go only to the episode in the caller's context; there is no process-wide fallback. `Toolkit` instances keep separate episodes, and `policy=` sets trust per episode. |
| A bad rewrite slips through | Full version history; `rollback` restores any version as a new version; every decision is in `relock_events`. |
| Late edits after the context is gone | Closed episodes raise `EpisodeClosedError` on any write. Episodes the toolkit or MCP server opened themselves are re-locked after `idle_timeout` seconds (default 1800) without a call, so a client that never ends its task does not keep items editable. |
| "The user said so" | `Signal.user` and `Episode.correct` default to `authenticated=False` and are blocked; the application must vouch for the user explicitly. |
| Someone on the network rewrites memory through the HTTP API | `relock serve` trusts the `source` its caller declares, so callers must be your harness. Without `RELOCK_TOKEN` it binds only to loopback and refuses a public address. With a token, every route except health and the OpenAPI document needs `Authorization: Bearer`, and the token check runs in constant time. JSON types are strict, so `"authenticated": "false"` is rejected rather than read as true. Bodies are capped at 1 MiB, and a request carrying both framings is refused. Connections and open episodes are capped, stalled connections time out, and idle episodes are re-locked in the background. |
| A write races the episode's close | `observe`, `expect`, `recall` and `use` re-check that the episode is open under the lock `close()` takes. A close that lands mid-request wins, and the write fails with `EpisodeClosedError` instead of committing after re-lock. `tool_result` binds and resolves under that lock, so concurrent reports never collide. |

Not mitigated in v0.1:

- An attacker who controls your tool code or verifier commands.
- An authenticated user who gives wrong corrections. Their corrections are authoritative by
  design; use `authoritative_users=False` to put them through the evidence threshold.
- Retrieval-time injection through item *content* written by an agent. Teaching is off by
  default in both the toolkit (`allow_teach`) and the MCP server (`--allow-teach`). When it is on,
  agent-created items carry `meta.origin = "agent"` and author `agent`, and can be reviewed or
  archived. They are not quarantined automatically.

## Limits and non-goals

- Retrieval loads all active items into memory per revision. That is fine for thousands of
  items; beyond that, plug in a `Retriever` backed by a vector index.
- Store access is synchronous, which is fast for local SQLite. For async frameworks this is a
  short blocking call.
- relock does not decide *what* to remember. It manages how remembered procedures change. Pair it
  with whatever creates items (humans, `teach_memory`, distillation from traces).

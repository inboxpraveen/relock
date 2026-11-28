# How relock works

This page explains the mechanism and the reasoning behind it. The [design spec](design.md) has the
exact rules, the storage schema and the threat model.

## The problem: corrections that don't stick

You corrected your agent: the orders API takes ISO dates now. Most memories save that lesson as
something *new*: a note, a vector entry, a line appended to a rules file or playbook. The old rule
is still there. Days later, deep into a long session, the model retrieves both and follows the old
one.

relock edits the rule that fired, in place, under three conditions:

- **Only on trusted evidence.** A typed tool error, a verifier or an authenticated user can
  trigger a rewrite. The model's own claims, a web page, a timeout or a one-off flaky check can't.
- **Only while that memory is recalled** in the current task. The retrieval trace records which
  memory was in play when the failure happened, so the model never has to guess.
- **The old version is kept, but never retrieved again.** Every version and every decision is
  stored for audit and rollback.

## The lifecycle of a memory

Unlock on recall, rewrite on surprise, re-lock.

![The lifecycle of a memory: locked, recalled and editable, through a trust gate, evidence, rewritten in place, re-locked](assets/lifecycle.svg)

1. **Reactivation = retrieval.** Only items recalled in the current episode can be edited. The
   retrieval trace answers "which memory caused this?" structurally.
2. **Explicit prediction.** Before acting on a recalled procedure, the harness (or the agent)
   records what it should produce, optionally citing the clauses it relies on. A wrapped tool
   does this automatically: "this call succeeds".
3. **Prediction error is the write trigger.** A mismatch counts only if the signal is trusted and
   not transient. Evidence accumulates until it reaches a threshold, which grows with how often
   the current version has been confirmed. A later success cancels a one-off flake.
4. **Overwrite in place.** The rewrite is a small clause-level delta (`replace`, `add`,
   `remove`) limited to the clauses that fired. The old version is kept as `superseded` and is
   never retrievable again. The database allows exactly one active version per item, so "do X"
   and "never do X" can't both be retrieved.
5. **Re-lock.** When the episode closes, its editable set closes too, and late writes raise
   `EpisodeClosedError`. Pending evidence carries over to the next episode that recalls the item.

## Where the idea comes from

The shape comes from **memory reconsolidation** in neuroscience:

- [Nader, Schafe and LeDoux (2000)](https://doi.org/10.1038/35021052) showed in rats that a fear
  memory becomes editable again for a short window after it is recalled.
- [Sevenster, Beckers and Kindt (2013)](https://doi.org/10.1126/science.1231357) found, in people,
  that the update needs a prediction error: the memory changes only when what happened differs
  from what it predicted.

relock borrows that shape: a memory can change only while it is recalled, and only when an outcome
contradicts what it predicted. It is a design metaphor, not a biological claim. The evidence comes
mostly from animal work, and results in people are mixed.

## The trust gate

Agents read a lot of text they shouldn't learn from: web pages, documents, their own guesses, a
prompt injection. relock's rule is **the model predicts, the harness judges**. The model can say
what it expects; only real outcomes decide whether memory changes.

![The trust gate: tool errors, verifiers and authenticated users count as evidence; timeouts, the model's own reports, self-critique and retrieved text are logged but never change memory](assets/trust-gate.svg)

| Signal | Source | Can rewrite memory? |
|---|---|---|
| Exception or typed result from a wrapped tool | `TOOL` | yes, if not transient |
| `TimeoutError`, `ConnectionError`, HTTP 408/425/429/5xx | `TOOL` (transient) | **never** |
| Test suite, schema check, grader | `VERIFIER` | yes |
| Human correction your app has verified (`authenticated=True`) | `USER` | yes, immediately (authoritative) |
| Human correction without that flag (the default) | `USER` | no |
| The model's own report | `AGENT` | no (opt-in) |
| LLM-as-judge, self-critique | `SELF` | no |
| Web pages, documents, emails, retrieved text | `ENVIRONMENT` | **never** |

Blocked signals aren't discarded: each one is stored with the reason it was blocked, so you can
see what tried to change memory. On top of the trust gate:

- An **evidence threshold** stops a single noisy failure from rewriting a good procedure. Two
  weighted trusted errors by default.
- **Inertia** makes well-proven versions need more evidence: `+0.5 * ln(1 + confirmations)`.
- **Flake damping** cancels a pending error when the same version is later confirmed.
- **Edit scope** limits a rewrite to the clauses the failed prediction relied on, when it cites
  them (`ep.expect(..., clauses=["c2"])`). Automatic expectations from `@mem.wrap` cite none, so
  there the rewriter picks the clause; `HintRewriter` replaces the clause most similar to the
  correction.
- **Untrusted text never steers a fix.** A prediction the agent wrote, the clause ids it cites
  and a correction it suggests are withheld from the rewriter. A trusted tool whose error message
  echoes remote content can be marked `detail_trusted=False`: the failure still counts, but its
  text never reaches the rewriter.
- **Optimistic concurrency** turns evidence about a version someone else already rewrote into a
  `conflict`, never a lost update.

Every gate decision is written to an append-only event log (`relock events`). The defaults and how
to change them are in [Configuration](configuration.md#write-policy).

## What's different

| | lesson notes, vector memories | rules file, playbook | LLM-managed memory (a model picks add / update / delete) | relock |
|---|---|---|---|---|
| A correction replaces the old rule | no: both are retrievable | no: both are in the file | if the model chooses to | yes: the clause that fired is superseded |
| What can change memory | any write | any write | a model, deciding from what it read | trusted signals only |
| A prompt injection's "fix" | stored | stored | can be stored | logged, never applied |
| A one-off flaky failure | stored | stored | can be stored | needs a second trusted error; the next success cancels it |
| Which memory caused a failure | the model guesses | the model guesses | the model guesses | recorded by the retrieval trace |
| History | varies | git, if any | varies | every version and decision, with rollback |

### How it relates to other work

- **Mem0** maintains memories with add/update/delete operations chosen by an LLM. It can update a
  memory in place, when its model chooses to.
- **ReasoningBank** distils strategies from the agent's own judged successes and failures.
- **ACE** grows playbooks through incremental deltas with helpful/harmful counters.

Each idea in relock exists somewhere, but we have not seen them packaged together:

- writes are allowed only while a memory is **recalled** (lability);
- a write is triggered by an **explicit, trusted prediction error**, never by what the model read;
- the edit is **scoped to the clause that fired**;
- the new version **supersedes** the old one instead of being appended, and every version is kept.

relock is meant to sit under any of these styles, or under your own harness. It doesn't decide
*what* to remember; it manages how remembered procedures change.

## Does it help?

In a study across 8 hosted models and 10 changed tool contracts, agents using relock made the
corrected call 98% of the time in long sessions, against 60% for a memory that appends each lesson
as a note. Gating alone, without replacing the rule, reached 74%. See [Benchmarks](benchmarks.md).

## Is relock for you?

It fits if:

- your agent calls tools or APIs, and corrections to how it calls them don't stick;
- your memory or rules file keeps growing "actually, do X instead" notes;
- your agent learns from its own failures, and a prompt injection or a flaky test must not be able
  to change what it learned.

It is not for chat history, document RAG, or agents with no tool outcomes to learn from.

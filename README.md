<p align="center">
  <img src="docs/assets/logo.svg" alt="relock" width="252" height="64">
</p>

<p align="center">
  <strong>Agent memory that fixes the rule that fired, in place and only on trusted evidence,<br>
  so corrected mistakes don't come back.</strong>
</p>

<p align="center">
  <a href="https://github.com/inboxpraveen/relock/actions/workflows/ci.yml"><img alt="ci" src="https://github.com/inboxpraveen/relock/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/relock-memory/"><img alt="pypi" src="https://img.shields.io/pypi/v/relock-memory"></a>
  <a href="https://pypi.org/project/relock-memory/"><img alt="python" src="https://img.shields.io/pypi/pyversions/relock-memory"></a>
  <a href="https://github.com/inboxpraveen/relock/pkgs/container/relock"><img alt="docker" src="https://img.shields.io/badge/docker-ghcr.io%2Finboxpraveen%2Frelock-blue"></a>
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
</p>

You corrected your agent: the orders API takes ISO dates now. Its memory saved the lesson as a
note. Days later, deep into a long session, it sends `03/14/2026` again. The old rule was still in
memory, next to the note, and the model followed it.

![Same correction, two memories: a typical memory appends a note and the agent repeats the mistake; relock rewrites the rule and the next call succeeds](docs/assets/hero.svg)

Lesson notes, rules files and playbooks all keep both rules and hope the model picks the right
one. relock edits the rule that fired, in place, under three conditions:

- **Only on trusted evidence.** A typed tool error, a verifier or an authenticated user can
  trigger a rewrite. The model's own claims, a web page, a timeout or a one-off flaky check can't.
- **Only while that memory is recalled** in the current task. The retrieval trace records which
  memory caused the failure, so the model never has to guess.
- **The old version is kept, but never retrieved again.** Every version and every decision is
  stored for audit and rollback.

![Correct tool calls per model in a long session: append-notes, append-gated and relock](docs/results/models/by-model.svg)

**The study:** 8 hosted models and 10 changed tool contracts, in long sessions with stale docs and
a prompt injection in context. With relock, agents made the corrected call **98%** of the time.
- A memory that appends each lesson as a note managed **60%**.
- Appending the correction to the procedure, like a rules file, managed **52%**.
- relock was ahead on all 8 models, with the 95% intervals apart for all 8.

**Isolating the mechanism.** The baselines are our own implementations of those patterns, run on
the same events. `append-gated` keeps relock's trust gate but still appends, and it reached
**74%**: the gate keeps the poisoned "fix" out, and replacing the rule is what stops the relapse.

The design was committed before the run, and every call is published. See the
[results](docs/results/models/): 11,520 probe calls, $7.33 in API spend.

## Is this your problem?

relock is for you if:
- your agent calls tools or APIs, and corrections to how it calls them don't stick;
- your memory or rules file keeps growing "actually, do X instead" notes;
- your agent learns from its own failures, and a prompt injection or a flaky test must not be able
  to change what it learned.

It is not for chat history, document RAG, or agents with no tool outcomes to learn from.

## What's different

| | lesson notes, vector memories | rules file, playbook | LLM-managed memory (a model picks add / update / delete) | relock |
|---|---|---|---|---|
| A correction replaces the old rule | no: both are retrievable | no: both are in the file | if the model chooses to | yes: the clause that fired is superseded |
| What can change memory | any write | any write | a model, deciding from what it read | trusted signals only |
| A prompt injection's "fix" | stored | stored | can be stored | logged, never applied |
| A one-off flaky failure | stored | stored | can be stored | needs a second trusted error; the next success cancels it |
| Which memory caused a failure | the model guesses | the model guesses | the model guesses | recorded by the retrieval trace |
| History | varies | git, if any | varies | every version and decision, with rollback |

Each idea exists somewhere, but we have not seen them packaged together:
- writes are allowed only while a memory is recalled;
- a write is triggered by a trusted prediction error;
- the edit is scoped to the clause that fired;
- the new version supersedes the old one instead of being appended.

## Install

```
pip install relock-memory            # core: zero dependencies
pip install "relock-memory[all]"     # + Postgres, MCP, LangChain/LangGraph
docker run -p 8765:8765 -e RELOCK_TOKEN=change-me ghcr.io/inboxpraveen/relock   # HTTP API
```

It works with Python agents, any MCP client, and any language over HTTP/JSON: Rust, TypeScript,
Go and others. See [Plug it into your stack](#plug-it-into-your-stack).

The extras install one at a time too: `[postgres]`, `[mcp]`, `[langchain]`. The import is
`relock_memory`. The PyPI name `relock` belongs to an unrelated SDK.

> **Status:** v0.2, beta. CI tests it on Python 3.10–3.14, on SQLite and Postgres, with `mcp`
> 1.x and 2.x, and from Rust, TypeScript and curl over HTTP. Behavioural results on 8 hosted
> models are in [docs/results/models](docs/results/models/).

---

## The mechanism

Unlock on recall, rewrite on surprise, re-lock.

![The lifecycle of a memory: locked, recalled and editable, through a trust gate, evidence, rewritten in place, re-locked](docs/assets/lifecycle.svg)

The idea comes from **memory reconsolidation** in neuroscience:
- [Nader, Schafe and LeDoux (2000)](https://doi.org/10.1038/35021052) showed in rats that a fear
  memory becomes editable again for a short window after it is recalled.
- [Sevenster, Beckers and Kindt (2013)](https://doi.org/10.1126/science.1231966) found, in people,
  that the update needs a prediction error: the memory changes only when what happened differs from
  what it predicted.

relock borrows that shape: a memory can change only while it is recalled, and only when an outcome
contradicts what it predicted. It is a design metaphor, not a biological claim. The evidence comes
mostly from animal work, and results in people are mixed.

1. **Reactivation = retrieval.** Only items recalled in the current episode can be edited. The
   retrieval trace answers "which memory caused this?" structurally, so the model never has to.
2. **Explicit prediction.** Before acting on a recalled procedure, the harness (or the agent)
   records what it should produce, optionally citing the clauses it relies on.
3. **Prediction error is the write trigger.** A mismatch counts only if the signal is trusted
   (typed tool error, verifier, authenticated user) and not transient (timeouts, 429s, 5xx).
   Evidence accumulates until it reaches a threshold, which grows with how often the
   version has been confirmed. A later success cancels a one-off flake.
4. **Overwrite in place.** The rewrite is a small clause-level delta (`replace` / `add` /
   `remove`) limited to the clauses that fired. The old version is kept as `superseded` for
   audit and rollback and is never retrievable again. A partial unique index guarantees one active
   version per item, so "do X" and "never do X" can never both be retrieved.
5. **Re-lock.** When the episode closes, its editable set closes too. Late writes raise
   `EpisodeClosedError`. Pending evidence carries over to the next episode that recalls the item.

## Quickstart

```python
import relock_memory as relock

mem = relock.Memory("agent.db")                     # or ":memory:", or "postgresql://..."
mem.add("Fetch orders by date",
        ["Authenticate with the orders API key before calling get_orders.",
         "Pass the date to get_orders as MM/DD/YYYY."],   # stale: the API moved to ISO dates
        key="orders.fetch", tools=["get_orders"], exist_ok=True)

@mem.wrap                                           # your real tool; its outcome is the trusted signal
def get_orders(date: str) -> list[dict]:
    if not is_iso(date):
        raise relock.ToolError("invalid date", correction="Pass the date to get_orders as YYYY-MM-DD.")
    return api.orders(date)

with mem.episode("orders for March 14") as ep:
    context = ep.recall("fetch the orders placed on a date").render()   # put this in the prompt
    ...  # your agent runs; calls to get_orders resolve expectations automatically
```

After two trusted failures (configurable), clause `c2` is rewritten in place:

```
$ relock history orders.fetch
v1 [superseded] create by author at 2026-11-26T18:38:29.454852+00:00
    (c1) Authenticate with the orders API key before calling get_orders.
    (c2) Pass the date to get_orders as MM/DD/YYYY.
v2 [active] reconsolidate by hint at 2026-11-26T18:38:29.463781+00:00 (episode ep_01m8cjdsjmv8mbp3d065)
    reason: replace c2 with trusted correction
  - (c2) Pass the date to get_orders as MM/DD/YYYY.
  + (c2) Pass the date to get_orders as YYYY-MM-DD.
```

Run [`examples/01_quickstart.py`](examples/01_quickstart.py) to see the whole loop in about 60 lines.

## Plug it into your stack

![Where relock sits: beside your agent. Recall feeds the prompt; real tool outcomes pass a trust gate; trusted evidence rewrites the clause that fired; every version is stored](docs/assets/architecture.svg)

relock does not replace your agent framework. It sits beside it. The split that keeps memory
safe is **the model predicts, the harness judges**.

| You use | Integrate with | Effort |
|---|---|---|
| Any Python agent (OpenAI Agents SDK, PydanticAI, smolagents, CrewAI, LlamaIndex, AutoGen, your own loop) | `@mem.wrap` on your tools + `with mem.episode()` around a task. Signatures and docstrings are preserved, so framework tool schemas don't change. Sync and async. | 3 lines |
| Raw tool calling (OpenAI-compatible, Anthropic, vLLM, Ollama) | `relock.Toolkit(mem).openai_tools()` / `.anthropic_tools()` + `toolkit.call(name, args)` | a dispatcher branch |
| LangChain / LangGraph | `RelockCallbackHandler(mem)` in `config={"callbacks": [...]}`. Existing tools stay untouched. `langchain_tools(Toolkit(mem))` gives the model `recall_memory` / `state_expectation`. | 1 line |
| Anything that speaks MCP (Claude Code, Cursor, VS Code, OpenAI Agents SDK, LangGraph, PydanticAI...) | `relock mcp --db relock.db --verifier tests="pytest -q"` | config only |
| **Any language**: Rust, TypeScript, Go, Java, C#... | `relock serve` or `docker run ghcr.io/inboxpraveen/relock`, then HTTP/JSON ([OpenAPI](docs/openapi.json), [guide](docs/http.md)) | 4 HTTP calls per task |

### HTTP: any language

```
RELOCK_TOKEN=change-me relock serve        # or: docker run -p 8765:8765 -e RELOCK_TOKEN=change-me ghcr.io/inboxpraveen/relock
```

Each task is four calls from your harness:

| step | request |
|---|---|
| open | `POST /v1/episodes` |
| recall into the prompt | `POST /v1/episodes/{id}/recall` |
| report each tool call's real outcome | `POST /v1/episodes/{id}/tool-result` |
| re-lock | `POST /v1/episodes/{id}/close` |

Clients for [Rust](examples/http/rust/src/main.rs), [TypeScript](examples/http/typescript/relock.ts)
and [curl](examples/http/curl.sh) run this loop against a live server in CI. The full API is in
[docs/http.md](docs/http.md). The server has no dependencies. Without a token it listens only on
loopback, and it refuses a public address without one.

### MCP

```json
{
  "mcpServers": {
    "relock": {
      "command": "uvx",
      "args": ["--from", "relock-memory[mcp]", "relock", "--db", "/path/to/relock.db", "mcp",
               "--verifier", "tests=pytest -q", "--verifier", "lint=ruff check ."]
    }
  }
}
```

[`uvx`](https://docs.astral.sh/uv/) fetches the package the first time the client starts the
server. If you installed `relock-memory[mcp]` with pip instead, use `"command": "relock"` and drop
the first two arguments. In Claude Code the same server is one command:

```
claude mcp add relock -- uvx --from "relock-memory[mcp]" relock --db relock.db mcp --verifier tests="pytest -q"
```

Verifier commands run in the server's working directory, which depends on the client. Pass
`--cwd /path/to/project` to pin it.

Tools exposed: `relock_begin`, `relock_recall`, `relock_expect`, `relock_verify`, `relock_report`,
`relock_history`, `relock_end`, plus `relock_teach` if you start the server with `--allow-teach`.
Items the agent teaches are marked `meta.origin = "agent"`. If a client never calls `relock_end`,
its episode is re-locked after 30 idle minutes (`--idle-timeout`).

Because the agent calls these tools, **nothing the agent writes is trusted by default**:

- its `relock_report`s are logged but never rewrite memory;
- the prediction text it gives `relock_expect` is withheld from the rewriter;
- a `correction` it suggests to `relock_verify` is ignored.

Trusted outcomes come from **verifiers**: commands fixed by whoever starts the server. The agent
can only pick one by name; the server runs it and reads the exit code itself. Two red verifier
runs are therefore enough evidence for a rewrite, but the fix comes from the verifier's output
(with an LLM rewriter) or from a trusted correction, never from the agent's text.

For a single-user local sandbox, `--trust-agent-reports` opts in to all three.

## Safety model

| Signal | Source | Can rewrite memory? |
|---|---|---|
| Exception or typed result from a wrapped tool | `TOOL` | yes, if not transient |
| `TimeoutError`, `ConnectionError`, HTTP 408/425/429/5xx | `TOOL` (transient) | **never** |
| Test suite / schema check / grader | `VERIFIER` | yes |
| Human correction your app has verified (`authenticated=True`) | `USER` | yes, immediately (authoritative) |
| Human correction without that flag (the default) | `USER` | no |
| The model's own report | `AGENT` | no (opt-in) |
| LLM-as-judge / self-critique | `SELF` | no |
| Web pages, documents, emails, retrieved text | `ENVIRONMENT` | **never** |

On top of the trust gate:

- An **evidence threshold** stops a single noisy failure from rewriting a good procedure.
- **Inertia** means well-proven versions need more evidence: `+0.5 * ln(1 + confirmations)`.
- **Flake damping** cancels a pending error when the same version is later confirmed.
- **Edit scope** limits a rewrite to the clauses the failed prediction relied on, when it cites
  them (`ep.expect(..., clauses=["c2"])`, or `clause_ids` from the agent). Automatic expectations
  from `@mem.wrap` cite none, so there the rewriter picks the clause. `HintRewriter` replaces the
  clause most similar to the correction.
- **Optimistic concurrency** turns evidence about a version someone else already rewrote into a `conflict`, never a lost update.
- A trusted tool whose error message echoes remote content can be marked `detail_trusted=False`. The failure still counts, but its text never reaches the rewriter.

Every gate decision is written to an append-only event log (`relock events`).

## Small open models (~30B)

The hard part of self-improving memory is credit assignment: which memory caused the failure?
relock answers that structurally, through the retrieval trace, the cited clauses and the expectation
diff. The rewriter only has to make a *local* edit: "here is the item, here is what it predicted,
here is what happened, rewrite this clause". The rewrite prompt is built for that size class:
one item, explicit clause ids, strict JSON, one worked example. Invalid output is fed back for a
retry, `<think>` blocks are stripped, and the deterministic `HintRewriter` is the fallback. In the
[behavioural study](docs/results/models/), each of the 8 models wrote relock's rewrites
itself, two of them 27B models. None of the 960 rewrites fell back to `HintRewriter`.

```python
llm = relock.OpenAICompatible("http://localhost:8000/v1", model="Qwen/Qwen3-30B-A3B-Instruct-2507")
mem = relock.Memory("agent.db", rewriter=relock.LLMRewriter(llm))
```

`OpenAICompatible` has no dependencies and works with vLLM, SGLang, Ollama, llama.cpp, LM Studio
and hosted endpoints. Any `Callable[[str], str]` also works as `llm`.

Every rewrite is also a labelled training example. `relock export traces.jsonl` writes SFT pairs
(`prompt` / `completion`) and preference pairs (`rejected` = superseded clauses, `chosen` = new
clauses), the raw material for distilling the rewriter or consolidating corrections into weights.

## RelapseBench

**The question:** once an agent has been taught a correction, how often does the old behaviour come
back under load, poisoning and noise?

Ten tool contracts drift: dates, money units, auth schemes, regions, pagination, encodings, SQL
quoting, temperature and distance units, recipient separators. Memory starts with the old rule,
the environment enforces a new one, and a third, wrong rule is used for poisoning. Every system
then gets the **same** stream of events:

1. typed tool errors that carry the correction, until it is learned;
2. successes, two transient timeouts, one prompt injection pushing the wrong rule, and one noisy
   verifier failure that arrives with a wrong "fix".

Each system is then probed at four load levels that shrink the memory budget and top-k, add up to
120 distractor procedures and pad the context with up to 8k tokens of history containing a stale
example and the injected text.

Systems compared:

- `append-notes`: every lesson becomes a new note.
- `append-clause`: lessons are appended to the item.
- `ace-style`: our simplified ACE-like playbook with helpful/harmful counters (not the official ACE code).
- Two ablations of relock: `append-gated`, which keeps relock's gates but appends, and `overwrite-ungated`, which rewrites in place with every gate off.
- `relock` itself, whose predictions cite the clause they rely on.
- `relock-uncited`: relock with uncited predictions, like the plain `@mem.wrap` path.

### Structural mode (measured, model-free)

![RelapseBench structural results](docs/results/structural.svg)

This mode measures which rules reach the agent's memory context. It does not measure what a
model does with them. It is deterministic (repeated runs produce identical reports) and takes a
few seconds: `relock bench`. Full tables are in [docs/results/structural.md](docs/results/structural.md).

What it shows:

- **Each mechanism fixes exactly one failure mode.** Gating alone (`append-gated`) keeps the
  poisoned rule out, but the superseded rule stays in context next to the correction in 100% of
  probes. In-place rewriting alone (`overwrite-ungated`) removes the stale rule, but the injected
  rule reaches context in 100% of probes. In 9 of 10 families it overwrote the correction. In the
  tenth it overwrote the wrong clause, because nothing restricted the edit to the clause that
  fired. relock needs both mechanisms to keep the context clean (100% of probes).
- **Append-style memories carry conflicts.** At the loosest budget (L0), the superseded rule is
  still in context alongside the correction for 90% (`append-notes`) to 100% (`append-clause`,
  `ace-style`) of probes. At the tightest budget (L3), `append-notes` loses the correction entirely
  in 23% of probes.
- **Citing clauses made no difference here.** `relock-uncited` matches `relock` on every metric.
  The trust gate keeps the poisoned fix out, and the hint rewriter finds the drifting clause by
  similarity. Edit scope guards against a rewriter, typically an LLM, that edits the wrong
  clause. This model-free mode doesn't exercise that case.
- **The price:** relock needs 2 trusted errors to learn a correction (its default evidence
  threshold), where append-style systems need 1. That is one extra failure per drift, paid for
  robustness against one-off noise. Set `min_evidence=1` to trade it back.

### Behavioural mode (8 hosted models)

This mode measures what a model actually does after a correction. The agent learns from its own
failures, the same model writes relock's rewrites, and then it is probed under load. The study
compares four systems: `append-notes`, `append-clause`, `append-gated` and `relock`. The other
systems above ran only in structural mode.

![Correct calls by load, pooled over models](docs/results/models/by-load.svg)

| load | append-notes | append-clause | append-gated | relock |
|---|---|---|---|---|
| L0: fresh session | 45% | 5% | 50% | 94% |
| L1: 1.5k-token history | 46% | 10% | 50% | 99% |
| L2: 4k-token history | 46% | 11% | 52% | 98% |
| L3: 8k-token history, tight memory | 60% | 52% | 74% | 98% |

Correct-call rates are pooled over 8 models and 3 seeds. Over all loads, relapse to the old rule
and following the poisoned rule were each 0% with relock, against 22% and 25% with append-notes.
The models wrote all 960 of relock's rewrites themselves, and none fell back to the deterministic
rewriter; the correction text came from the tool error. The per-model and per-task tables, the
costs and the caveats are in [docs/results/models](docs/results/models/).

To run it on your own model:

```
relock bench --mode behavioral --base-url http://localhost:8000/v1 \
             --model Qwen/Qwen3-30B-A3B-Instruct-2507
```

Results on other models are welcome; please include `behavioral.json`.

## CLI

```
relock add "Run tests" "Run the tests with pytest -q." --key tests.run --tool pytest
relock ls                         # active items with confidence and counters
relock show orders.fetch [--version 1]
relock history orders.fetch       # clause-level diffs across versions
relock pending orders.fetch       # pending evidence vs. the rewrite threshold, and what was blocked
relock events --item orders.fetch # audit log: recalls, predictions, gate decisions, rewrites
relock rollback orders.fetch 1    # restores v1 as a new version (history is never destroyed)
relock edit orders.fetch --replace c2 "Pass dates as YYYY-MM-DD."
relock export traces.jsonl        # SFT and preference pairs from every rewrite
relock mcp --verifier tests="pytest -q" [--cwd DIR] [--allow-teach] [--trust-agent-reports]
relock serve [--host H] [--port P]  # HTTP/JSON API for any language (RELOCK_TOKEN for auth)
relock bench [--mode structural|behavioral|both]
```

The database is `--db`, or `$RELOCK_DB`, or `./relock.db`. Postgres URLs work everywhere.

## Configuration

```python
relock.WritePolicy(
    min_evidence=2.0,          # weighted trusted errors before a rewrite
    inertia=0.5,               # extra evidence per ln(1 + confirmations)
    confirmation_cancels=1.0,  # a later success cancels this much pending evidence
    rewrite_on="observe",      # or "close": rewrite only when the episode ends
    edit_scope="cited",        # replace/remove only clauses the prediction relied on
    max_ops=4,
    attribute_undeclared=False,  # attribute tool outcomes to recalled items that declare no tools
)
relock.WritePolicy.strict()       # verifier + user only, higher threshold
relock.WritePolicy.eager()        # one trusted error is enough
relock.WritePolicy.trust_agent()  # also trust the agent's reports (local sandboxes)
```

Retrieval is BM25 by default (no dependencies). Pass `embed=` (any `list[str] -> list[vector]`
function) for hybrid BM25 + vector search with reciprocal-rank fusion, or implement the
one-method `Retriever` protocol. Only active versions are ever candidates.

Storage is SQLite by default (WAL mode, safe across processes) or Postgres with
`pip install "relock-memory[postgres]"`. Postgres uses `JSONB` and `TIMESTAMPTZ`, so the ledger is
directly queryable. The schema is documented in [docs/design.md](docs/design.md).

## How it relates to other work

- **Mem0** maintains memories with add/update/delete operations chosen by an LLM.
- **ReasoningBank** distils strategies from the agent's own judged successes and failures.
- **ACE** grows playbooks through incremental deltas with helpful/harmful counters.

Mem0 can also update a memory in place, when its model chooses to. relock combines four things we
have not seen packaged together:

- writes gated on **retrieval** (lability);
- triggered by an **explicit, trusted prediction error**, never by what the model read;
- **scoped to the clause that fired**;
- applied **in place with supersession** rather than appended, with every version kept.

It is meant to sit under any of these styles, or under your own harness.

## Roadmap

- Behavioural RelapseBench on self-hosted open models, plus a long-horizon variant with real
  context compaction.
- Weight tier: periodically distil accumulated rewrite traces for a skill into adapter weights.
  Fast edits live in the harness, durable ones in the weights.
- pgvector retrieval inside Postgres; published client packages for the HTTP API (npm, crates.io).
- Per-tenant ledgers and signed user corrections.

## Development

```
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                           # SQLite backends
RELOCK_TEST_POSTGRES_URL=postgresql://... pytest # + Postgres
ruff check src tests examples
```

CI runs the suite on Python 3.10–3.14 against SQLite and Postgres. It also runs the MCP tests
on the `mcp` 1.x line. See [CONTRIBUTING.md](CONTRIBUTING.md) for the invariants a change must
keep. Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

Apache-2.0. See [LICENSE](LICENSE).

# Benchmarks: RelapseBench

**The question:** once an agent has been taught a correction, how often does the old behaviour come
back under load, poisoning and noise?

![Correct tool calls per model in a long session: append-notes, append-gated and relock](results/models/by-model.svg)

## Setup

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
- `append-clause`: lessons are appended to the item, like a rules file.
- `ace-style`: our simplified ACE-like playbook with helpful/harmful counters (not the official ACE
  code).
- Two ablations of relock: `append-gated`, which keeps relock's gates but appends, and
  `overwrite-ungated`, which rewrites in place with every gate off.
- `relock` itself, whose predictions cite the clause they rely on.
- `relock-uncited`: relock with uncited predictions, like the plain `@mem.wrap` path.

The baselines are our own implementations of those patterns, run on the same events.

## Behavioural mode: 8 hosted models

This mode measures what a model actually does after a correction. The agent learns from its own
failures, the same model writes relock's rewrites, and then it is probed under load. The study
compares four systems: `append-notes`, `append-clause`, `append-gated` and `relock`.

![Correct calls by load, pooled over models](results/models/by-load.svg)

| load | append-notes | append-clause | append-gated | relock |
|---|---|---|---|---|
| L0: fresh session | 45% | 5% | 50% | 94% |
| L1: 1.5k-token history | 46% | 10% | 50% | 99% |
| L2: 4k-token history | 46% | 11% | 52% | 98% |
| L3: 8k-token history, tight memory | 60% | 52% | 74% | 98% |

Correct-call rates are pooled over 8 models and 3 seeds. relock was ahead on all 8 models, with
the 95% intervals apart for all 8. Over all loads, relapse to the old rule and following the
poisoned rule were each 0% with relock, against 22% and 25% with append-notes.

**Isolating the mechanism.** `append-gated` keeps relock's trust gate but still appends, and it
reached 74% at L3: the gate keeps the poisoned "fix" out, and replacing the rule is what stops the
relapse.

The models wrote all 960 of relock's rewrites themselves, and none fell back to the deterministic
rewriter; the correction text came from the tool error. The design was committed before the run,
and every call is published: 11,520 probe calls, $7.33 in API spend. The per-model and per-task
tables, the costs and the caveats are in [results/models](results/models/), and the
pre-registered design in [DESIGN.md](results/models/DESIGN.md).

## Structural mode: model-free

![RelapseBench structural results](results/structural.svg)

This mode measures which rules reach the agent's memory context. It does not measure what a model
does with them. It is deterministic (repeated runs produce identical reports) and takes a few
seconds. Full tables are in [results/structural.md](results/structural.md).

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

## Run it yourself

```
relock bench                                     # structural mode, a few seconds, no model
relock bench --mode behavioral --base-url http://localhost:8000/v1 \
             --model Qwen/Qwen3-30B-A3B-Instruct-2507
```

Behavioural mode works with any OpenAI-compatible endpoint: vLLM, SGLang, Ollama, llama.cpp, LM
Studio or a hosted API. `--loads`, `--seeds`, `--probes` and `--systems` narrow a run, and
`--out DIR` sets where the reports go.

Results on other models are welcome. Please open an issue or pull request with the generated
`behavioral.json` and `behavioral.md`, and the exact model, quantisation and serving stack.

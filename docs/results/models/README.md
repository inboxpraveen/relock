# relock vs. "append the lesson": 8 models, 10 changed tool contracts

An agent is corrected: the API now wants ISO dates, amounts in cents, the EU region. Later it
works through a long session, with stale docs and an injected web page in its context. Does it
keep making the corrected call?

**Long sessions:** with relock, agents made the corrected call **97%** of the
time, pooled over 8 models.
- A memory that appends each lesson as a note managed **60%**.
- Appending the correction to the procedure, like a rules file, managed **51%**.
- relock beat append-notes for **8 of 8 models**. The 95% intervals don't
  overlap for 7 of them.
- That is 405 wrong tool calls per 1,000 with append-notes, against 25 with
  relock.

![Correct tool calls per model, long session](by-model.svg)

The design was committed before the run: see [DESIGN.md](DESIGN.md). Every number below is
computed by [`scripts/model_report.py`](../../../scripts/model_report.py) from
[`records.csv`](records.csv), which holds every probe call.

## Why the usual memories fail

Across all loads and models, the share of calls in each outcome was:

| memory | correct | relapsed to the old rule | followed the poisoned rule | other |
|---|---|---|---|---|
| append-notes | 51% | 22% | 24% | 4% |
| append-clause | 22% | 11% | 64% | 3% |
| append-gated | 58% | 37% | 1% | 4% |
| relock | 97% | 0% | 0% | 3% |

Two separate failures account for the gap.

- **Relapse.** The appended note sits next to the old rule, and models follow the old one. The
  relapse rate was 22% for append-notes and 11% for append-clause. relock rewrites the clause in place,
  so the old rule can't be retrieved and the relapse rate was 0%.
- **Poisoning.** Without a trust gate, a flaky check's wrong "fix" goes into memory too. Models
  followed the poisoned rule 24% of the time with append-notes and
  0% with relock.

**Gating alone isn't enough.** `append-gated` keeps relock's trust and evidence gates but still
appends. It cuts poisoning to 1%, but the old rule stays
retrievable, and its relapse rate was 37%. In long sessions it
made 74% correct calls, against 97% for relock. Most of the gain
comes from superseding the rule in place, and the gate keeps that rewrite safe.

![Outcomes by memory system](outcomes.svg)

## Every model, every load

Correct-call rate with 95% Wilson intervals, n ≈ 59 per cell:

**L0: fresh session**

| model | append-notes | append-clause | append-gated | relock |
|---|---|---|---|---|
| `anthropic/claude-haiku-5.5` | 38% (27%–51%) | 2% (0%–9%) | 65% (52%–76%) | 100% (94%–100%) |
| `deepseek/deepseek-v4.1-flash` | 52% (39%–64%) | 2% (0%–9%) | 47% (35%–59%) | 98% (91%–100%) |
| `moonshotai/kimi-k3` | 42% (30%–54%) | 2% (0%–9%) | 43% (32%–56%) | 100% (94%–100%) |
| `openai/gpt-6-luna` | 47% (35%–60%) | 13% (6%–24%) | 38% (27%–51%) | 85% (73%–92%) |
| `qwen/qwen3.8-27b` | 50% (38%–62%) | 0% (0%–6%) | 47% (35%–59%) | 90% (80%–95%) |
| `qwen/qwen3.8-flash` | 52% (39%–64%) | 0% (0%–6%) | 53% (41%–65%) | 100% (94%–100%) |
| `radixark/daisyrose-27b` | 38% (27%–51%) | 0% (0%–6%) | 43% (32%–56%) | 90% (80%–95%) |
| `thinkingmachines/inkling-2-mini` | 43% (32%–56%) | 22% (13%–34%) | 63% (51%–74%) | 90% (80%–95%) |
| **all 8 models** | 45% (41%–50%) | 5% (3%–7%) | 50% (46%–54%) | 94% (92%–96%) |

**L1: 1.5k-token history**

| model | append-notes | append-clause | append-gated | relock |
|---|---|---|---|---|
| `anthropic/claude-haiku-5.5` | 52% (39%–64%) | 30% (20%–43%) | 50% (38%–62%) | 100% (94%–100%) |
| `deepseek/deepseek-v4.1-flash` | 53% (41%–65%) | 7% (3%–16%) | 50% (38%–62%) | 100% (94%–100%) |
| `moonshotai/kimi-k3` | 42% (30%–54%) | 23% (14%–35%) | 42% (30%–54%) | 100% (94%–100%) |
| `openai/gpt-6-luna` | 60% (46%–72%) | 7% (3%–17%) | 40% (28%–55%) | 94% (84%–98%) |
| `qwen/qwen3.8-27b` | 45% (33%–58%) | 0% (0%–6%) | 65% (52%–76%) | 98% (91%–100%) |
| `qwen/qwen3.8-flash` | 33% (23%–46%) | 2% (0%–9%) | 50% (38%–62%) | 98% (91%–100%) |
| `radixark/daisyrose-27b` | 48% (36%–61%) | 0% (0%–6%) | 70% (57%–80%) | 98% (91%–100%) |
| `thinkingmachines/inkling-2-mini` | 62% (49%–73%) | 18% (11%–30%) | 38% (27%–51%) | 100% (94%–100%) |
| **all 8 models** | 49% (45%–54%) | 11% (8%–14%) | 51% (46%–55%) | 99% (97%–99%) |

**L3: 8k-token history, tight memory**

| model | append-notes | append-clause | append-gated | relock |
|---|---|---|---|---|
| `anthropic/claude-haiku-5.5` | 63% (51%–74%) | 57% (44%–68%) | 73% (61%–83%) | 98% (91%–100%) |
| `deepseek/deepseek-v4.1-flash` | 58% (46%–70%) | 50% (38%–62%) | 63% (51%–74%) | 100% (94%–100%) |
| `moonshotai/kimi-k3` | 48% (36%–61%) | 53% (41%–65%) | 65% (52%–76%) | 100% (94%–100%) |
| `openai/gpt-6-luna` | 69% (56%–80%) | 47% (35%–60%) | 78% (65%–88%) | 89% (79%–95%) |
| `qwen/qwen3.8-27b` | 73% (61%–83%) | 60% (47%–71%) | 98% (91%–100%) | 100% (94%–100%) |
| `qwen/qwen3.8-flash` | 43% (32%–56%) | 40% (29%–53%) | 63% (51%–74%) | 92% (82%–96%) |
| `radixark/daisyrose-27b` | 78% (66%–87%) | 50% (38%–62%) | 95% (86%–98%) | 100% (94%–100%) |
| `thinkingmachines/inkling-2-mini` | 43% (32%–56%) | 53% (41%–65%) | 55% (42%–67%) | 100% (94%–100%) |
| **all 8 models** | 60% (55%–64%) | 51% (47%–56%) | 74% (70%–78%) | 97% (96%–99%) |

![Correct calls by load, pooled over models](by-load.svg)

## Where it bites

This table pools the long-session (`L3`) calls across all models. The consequence column is our
description of what the wrong call would do; it was not measured.

| task | append-notes | relock | what a wrong call does |
|---|---|---|---|
| `orders.fetch` | 94% | 100% | MM/DD/YYYY sent to an ISO-date API: rejected or wrong day |
| `payments.charge` | 53% | 94% | dollars sent to a cents API: wrong amount charged, or a failed charge |
| `hvac.thermostat` | 58% | 100% | Fahrenheit sent as Celsius: a 72 °C setpoint |
| `reports.auth` | 69% | 96% | old auth scheme: every call is a 401 |
| `email.send` | 88% | 100% | wrong recipient separator: mail goes to a malformed address |
| `deploy.service` | 100% | 98% | deploys to the old region: data residency breach |
| `users.list` | 0% | 90% | 1-based page on a 0-based API: silently skips a page |
| `files.write` | 0% | 100% | Latin-1 instead of UTF-8: corrupted text |
| `sql.count` | 33% | 98% | wrong identifier quoting: query fails |
| `routes.limit` | 100% | 100% | miles sent as kilometres: routes capped 38% short |

## What it costs and what it saves

| memory | errors before learning a correction | prompt tokens per call | model $ per 1,000 correct calls |
|---|---|---|---|
| append-notes | 2.22 | 3,646 | $1.31 |
| append-clause | 1.33 | 3,715 | $3.08 |
| append-gated | 3.31 | 3,625 | $1.18 |
| relock | 2.18 | 3,627 | $0.83 |

- **Learning:** these are the errors before the model first followed the correction (mean per
  task):
  append-notes 2.22, append-clause 1.33, append-gated 3.31, relock 2.18.
  relock waits for 2 trusted errors before it rewrites (`min_evidence=2`; set it to 1 to learn
  from the first error).
  `append-clause` learned fastest, because the correction lands inside the procedure. The
  poisoned "fix" lands there too, which is why it followed the poisoned rule in
  64% of calls.
- **Rewrite overhead:** relock made 480 rewrites, done by the same model, for
  $0.196. That is 15.1% of relock's model spend in this study.
  The LLM rewriter fell back to the deterministic one 0 times.
- **Tokens:** prompts averaged 3,627 tokens with relock and 3,646 with append-notes.
  relock keeps one current rule instead of a growing pile of notes, so the fix adds no prompt
  cost.
- **Per 1,000 calls in long sessions:** about 379 fewer wrong calls than
  append-notes. Each one is a failed request, a retry, or, for the silent cases above, a wrong
  charge, region or file.

## Cost of this study

- **Recorded spend:** $4.71 for 5,760 probe calls, plus
  learning and rewrite calls, across 8 models. The run used OpenRouter's default provider
  routing.
- **Reproduction:** the probe calls cost $3.86 here and would cost about
  $2.67 at each model's cheapest provider. `scripts/model_study.py` now routes there
  by default.

## Caveats

- **API errors:** 113 calls failed at the API after retries, all on `openai/gpt-6-luna`.
  Failed probes are excluded from the rates above. A failure during learning counted as a wrong
  answer for every memory system alike.
- **Refusals:** `gpt-6-luna` sometimes declines the payment task ("can't charge without looking up
  the card"). These count as `other` for every memory.
- **Exact reruns differ slightly.** OpenRouter routes calls to different providers; the provider
  of every call is in `records.csv`.
- **Synthetic tasks.** The 10 tasks are short tool-contract changes, built to have one
  unambiguous right answer. Real agents face messier memories, and we expect the old-rule
  conflict to matter more there, but that is not measured here.

Reproduce:

```
pip install -e .
OPENROUTER_API_KEY=... python scripts/model_study.py run --budget 5
python scripts/model_report.py --prices
```

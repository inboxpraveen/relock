# relock vs. "append the lesson": 8 models, 10 changed tool contracts

An agent is corrected: the API now wants ISO dates, amounts in cents, the EU region. Later it
works through a long session, with stale docs and an injected web page in its context. Does it
keep making the corrected call?

**Long sessions:** with relock, agents made the corrected call **98%** of the
time, pooled over 8 models.
- A memory that appends each lesson as a note managed **60%**.
- Appending the correction to the procedure, like a rules file, managed **52%**.
- relock beat append-notes for **8 of 8 models**. The 95% intervals don't
  overlap for 8 of them.
- That is 396 wrong tool calls per 1,000 with append-notes, against 21 with
  relock.

![Correct tool calls per model, long session](by-model.svg)

The design was committed before the run: see [DESIGN.md](DESIGN.md). Every number below is
computed by [`scripts/model_report.py`](../../../scripts/model_report.py) from
[`records.csv`](records.csv), which holds every probe call.

## Why the usual memories fail

Across all loads and models, the share of calls in each outcome was:

| memory | correct | relapsed to the old rule | followed the poisoned rule | other |
|---|---|---|---|---|
| append-notes | 49% | 22% | 25% | 4% |
| append-clause | 19% | 9% | 68% | 4% |
| append-gated | 56% | 38% | 2% | 4% |
| relock | 97% | 0% | 0% | 2% |

Two separate failures account for the gap.

- **Relapse.** The appended note sits next to the old rule, and models follow the old one. The
  relapse rate was 22% for append-notes and 9% for append-clause. relock rewrites the clause in place,
  so the old rule can't be retrieved and the relapse rate was 0%.
- **Poisoning.** Without a trust gate, a flaky check's wrong "fix" goes into memory too. Models
  followed the poisoned rule 25% of the time with append-notes and
  0% with relock.

**Gating alone isn't enough.** `append-gated` keeps relock's trust and evidence gates but still
appends. It cuts poisoning to 2%, but the old rule stays
retrievable, and its relapse rate was 38%. In long sessions it
made 74% correct calls, against 98% for relock. Most of the gain
comes from superseding the rule in place, and the gate keeps that rewrite safe.

![Outcomes by memory system](outcomes.svg)

## Every model, every load

Correct-call rate with 95% Wilson intervals, n ≈ 90 per cell:

**L0: fresh session**

| model | append-notes | append-clause | append-gated | relock |
|---|---|---|---|---|
| `anthropic/claude-haiku-5.5` | 38% (28%–48%) | 4% (2%–11%) | 64% (54%–74%) | 100% (96%–100%) |
| `deepseek/deepseek-v4.1-flash` | 53% (43%–63%) | 2% (1%–8%) | 46% (36%–56%) | 99% (94%–100%) |
| `moonshotai/kimi-k3` | 41% (32%–51%) | 1% (0%–6%) | 44% (35%–55%) | 100% (96%–100%) |
| `openai/gpt-6-luna` | 49% (39%–60%) | 12% (7%–21%) | 35% (26%–46%) | 84% (76%–90%) |
| `qwen/qwen3.8-27b` | 47% (37%–57%) | 0% (0%–4%) | 48% (38%–58%) | 90% (82%–95%) |
| `qwen/qwen3.8-flash` | 52% (42%–62%) | 0% (0%–4%) | 51% (41%–61%) | 99% (94%–100%) |
| `radixark/daisyrose-27b` | 40% (30%–50%) | 0% (0%–4%) | 42% (33%–53%) | 90% (82%–95%) |
| `thinkingmachines/inkling-2-mini` | 41% (32%–51%) | 20% (13%–29%) | 67% (56%–76%) | 90% (82%–95%) |
| **all 8 models** | 45% (42%–49%) | 5% (4%–7%) | 50% (46%–53%) | 94% (92%–96%) |

**L1: 1.5k-token history**

| model | append-notes | append-clause | append-gated | relock |
|---|---|---|---|---|
| `anthropic/claude-haiku-5.5` | 53% (43%–63%) | 30% (22%–40%) | 50% (40%–60%) | 100% (96%–100%) |
| `deepseek/deepseek-v4.1-flash` | 53% (43%–63%) | 6% (2%–12%) | 46% (36%–56%) | 100% (96%–100%) |
| `moonshotai/kimi-k3` | 38% (28%–48%) | 24% (17%–34%) | 42% (33%–53%) | 100% (96%–100%) |
| `openai/gpt-6-luna` | 53% (43%–63%) | 8% (4%–16%) | 36% (26%–46%) | 95% (89%–98%) |
| `qwen/qwen3.8-27b` | 39% (29%–49%) | 0% (0%–4%) | 61% (51%–71%) | 99% (94%–100%) |
| `qwen/qwen3.8-flash` | 36% (26%–46%) | 2% (1%–8%) | 47% (37%–57%) | 99% (94%–100%) |
| `radixark/daisyrose-27b` | 48% (38%–58%) | 0% (0%–4%) | 71% (61%–79%) | 98% (92%–99%) |
| `thinkingmachines/inkling-2-mini` | 48% (38%–58%) | 12% (7%–21%) | 46% (36%–56%) | 99% (94%–100%) |
| **all 8 models** | 46% (42%–50%) | 10% (8%–13%) | 50% (46%–53%) | 99% (98%–99%) |

**L2: 4k-token history**

| model | append-notes | append-clause | append-gated | relock |
|---|---|---|---|---|
| `anthropic/claude-haiku-5.5` | 49% (39%–59%) | 36% (26%–46%) | 52% (42%–62%) | 100% (96%–100%) |
| `deepseek/deepseek-v4.1-flash` | 53% (43%–63%) | 6% (2%–12%) | 44% (35%–55%) | 100% (96%–100%) |
| `moonshotai/kimi-k3` | 26% (18%–35%) | 32% (23%–42%) | 39% (29%–49%) | 99% (94%–100%) |
| `openai/gpt-6-luna` | 56% (46%–66%) | 7% (3%–14%) | 44% (34%–55%) | 95% (89%–98%) |
| `qwen/qwen3.8-27b` | 46% (36%–56%) | 1% (0%–6%) | 56% (45%–65%) | 99% (94%–100%) |
| `qwen/qwen3.8-flash` | 50% (40%–60%) | 3% (1%–9%) | 42% (33%–53%) | 94% (88%–98%) |
| `radixark/daisyrose-27b` | 48% (38%–58%) | 0% (0%–4%) | 74% (65%–82%) | 99% (94%–100%) |
| `thinkingmachines/inkling-2-mini` | 42% (33%–53%) | 0% (0%–4%) | 60% (50%–70%) | 99% (94%–100%) |
| **all 8 models** | 46% (43%–50%) | 11% (9%–13%) | 52% (48%–55%) | 98% (97%–99%) |

**L3: 8k-token history, tight memory**

| model | append-notes | append-clause | append-gated | relock |
|---|---|---|---|---|
| `anthropic/claude-haiku-5.5` | 62% (52%–72%) | 61% (51%–71%) | 76% (66%–83%) | 99% (94%–100%) |
| `deepseek/deepseek-v4.1-flash` | 60% (50%–70%) | 48% (38%–58%) | 63% (53%–73%) | 100% (96%–100%) |
| `moonshotai/kimi-k3` | 49% (39%–59%) | 56% (45%–65%) | 62% (52%–72%) | 100% (96%–100%) |
| `openai/gpt-6-luna` | 66% (56%–75%) | 46% (36%–56%) | 77% (67%–85%) | 92% (84%–96%) |
| `qwen/qwen3.8-27b` | 71% (61%–79%) | 60% (50%–70%) | 94% (88%–98%) | 100% (96%–100%) |
| `qwen/qwen3.8-flash` | 47% (37%–57%) | 40% (30%–50%) | 60% (50%–70%) | 92% (85%–96%) |
| `radixark/daisyrose-27b` | 77% (67%–84%) | 50% (40%–60%) | 96% (89%–98%) | 100% (96%–100%) |
| `thinkingmachines/inkling-2-mini` | 52% (42%–62%) | 56% (45%–65%) | 62% (52%–72%) | 100% (96%–100%) |
| **all 8 models** | 60% (57%–64%) | 52% (48%–56%) | 74% (70%–77%) | 98% (97%–99%) |

![Correct calls by load, pooled over models](by-load.svg)

## Where it bites

This table pools the long-session (`L3`) calls across all models. It shows the correct-call rate
and how append-notes went wrong: relapsed to the old rule, or followed the poisoned "fix".
The last column is our description of what the old rule does; it was not measured.

| task | append-notes | its relapses / poisoned | relock | what the old rule does |
|---|---|---|---|---|
| `orders.fetch` | 96% | 0% / 4% | 99% | MM/DD/YYYY sent to an ISO-date API: rejected or wrong day |
| `payments.charge` | 47% | 46% / 1% | 96% | dollars sent to a cents API: wrong amount charged, or a failed charge |
| `hvac.thermostat` | 65% | 32% / 1% | 100% | Fahrenheit sent as Celsius: a 72 °C setpoint |
| `reports.auth` | 68% | 32% / 0% | 97% | old auth scheme: every call is a 401 |
| `email.send` | 90% | 0% / 10% | 100% | wrong recipient separator: mail goes to a malformed address |
| `deploy.service` | 100% | 0% / 0% | 99% | deploys to the old region: data residency breach |
| `users.list` | 0% | 0% / 100% | 92% | 1-based page on a 0-based API: silently skips a page |
| `files.write` | 0% | 0% / 100% | 100% | Latin-1 instead of UTF-8: corrupted text |
| `sql.count` | 38% | 54% / 3% | 99% | wrong identifier quoting: query fails |
| `routes.limit` | 100% | 0% / 0% | 99% | miles sent as kilometres: routes capped 38% short |

On `users.list` and `files.write`, a flaky check's wrong "fix" went into append-notes'
memory, and the models followed it even in fresh sessions. The trust gate keeps that fix out of
memory. relock and append-gated still meet the poisoned value in the injected page of a long
session's context, but follow it far less often.

## What it costs and what it saves

| memory | errors before learning a correction | prompt tokens per call | model $ per 1,000 correct calls |
|---|---|---|---|
| append-notes | 2.19 | 3,912 | $1.16 |
| append-clause | 1.31 | 3,971 | $3.00 |
| append-gated | 3.35 | 3,895 | $1.03 |
| relock | 2.16 | 3,870 | $0.72 |

- **Learning:** these are the errors before the model first followed the correction (mean per
  task):
  append-notes 2.19, append-clause 1.31, append-gated 3.35, relock 2.16.
  relock waits for 2 trusted errors before it rewrites (`min_evidence=2`; set it to 1 to learn
  from the first error).
  `append-clause` learned fastest, because the correction lands inside the procedure. The
  poisoned "fix" lands there too, which is why it followed the poisoned rule in
  68% of calls.
- **Rewrite overhead:** relock made 960 rewrites, done by the same model, for
  $0.391. That is 16.9% of relock's model spend in this study.
  The LLM rewriter fell back to the deterministic one 0 times.
- **Tokens:** prompts averaged 3,870 tokens with relock and 3,912 with append-notes.
  relock keeps one current rule instead of a growing pile of notes, so the fix adds no prompt
  cost.
- **Per 1,000 calls in long sessions:** about 375 fewer wrong calls than
  append-notes. Each one is a failed request and a retry or, in the silent cases above, a
  wrong charge, a skipped page or a corrupted file.

## Cost of this study

- **Calls in `records.csv`:** 11,520 probe calls across 8 models, plus the
  learning and rewrite calls. Priced call by call they come to $8.24, of which probes
  were $6.61. Identical prompts were answered from a local cache, so the real cost
  was lower.
- **Actual spend:** $7.33 on the OpenRouter key, including pilots and the re-run of
  failed calls.
- **Routing:** the first run (seeds 0-1) used OpenRouter's default provider routing. The
  extension used each model's cheapest provider, which `scripts/model_study.py` now does by
  default.
- **Reproduction:** the same probe calls cost about $5.74 at each model's listed
  OpenRouter price.

## Caveats

- **API errors:** 62 calls failed at the API after retries, all on `openai/gpt-6-luna`.
  Of the 40 failed probe calls, 23 were on `payments.charge`. A fresh call reproduced
  the cause: an HTTP 400 from the upstream provider's usage-policy filter, not a network
  failure.
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
OPENROUTER_API_KEY=... python scripts/model_study.py run --seeds 0 1 2 --loads L0 L1 L2 L3
python scripts/model_report.py --prices
```

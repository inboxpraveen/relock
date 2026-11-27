# Behavioural study across hosted models: design

This design was fixed and committed before the run. The results are in [README.md](README.md).

## Question

After an agent has been corrected, how often does it go back to the old rule, or follow a
poisoned one, when its memory works the usual way ("append the lesson")? And how often does it
do so with relock?

## Setup

- **Tasks:** the 10 RelapseBench tool contracts. Each is a real-world rule that changed:
  - date format;
  - money in cents;
  - Celsius;
  - Bearer auth;
  - `;` between recipients;
  - the EU region;
  - 0-based pages;
  - UTF-8;
  - ANSI SQL quoting;
  - kilometres.
- **Models:** 8, through OpenRouter, with reasoning switched off, temperature 0 and at most 512
  output tokens. The same model also performs relock's rewrites (`LLMRewriter`).
  - `qwen/qwen3.8-27b`
  - `qwen/qwen3.8-flash`
  - `deepseek/deepseek-v4.1-flash`
  - `moonshotai/kimi-k3`
  - `anthropic/claude-haiku-5.5`
  - `openai/gpt-6-luna`
  - `thinkingmachines/inkling-2-mini`
  - `radixark/daisyrose-27b`
- **Memory systems:**
  - `append-notes`: every correction becomes a new memory note. This is what most memory layers do.
  - `append-clause`: the correction is appended to the procedure, like adding a line to a rules
    file.
  - `append-gated`: append-notes behind relock's trust and evidence gates. This ablation separates
    gating from in-place rewriting.
  - `relock`: gated, in-place rewriting.
- **Loop for every model, system and task:**
  1. Memory starts with the old rule.
  2. The agent acts. Each wrong call returns a typed tool error that carries the correct rule.
     This repeats until the agent gets it right, at most 5 errors.
  3. The same noise reaches every system: successes, timeouts, one prompt injection and one flaky
     check that comes with a wrong "fix".
  4. The agent is probed 3 times under a load.
- **Loads:**
  - `L0`: fresh session.
  - `L1`: 1.5k tokens of history with a stale doc example and the injected text; smaller memory
    budget.
  - `L3`: 8k tokens of history with the same stale and injected text; tight memory budget;
    120 distractor procedures.
- **Seeds:** 2.
- **Sample size:** n = 60 probes per model, system and load.

## Metrics

Each probe is scored by which rule the call follows. Formatting that doesn't change the rule is
ignored (`a;b` and `a; b` both follow the `;` rule).

- **Primary:** correct-call rate at `L3` (long session), `relock` against `append-notes`, per model.
- **Secondary:**
  - relapse rate (old rule) and poisoned rate (injected rule);
  - correct-call rate at `L0` and `L1`;
  - errors needed to learn;
  - unparseable replies, reported as their own category;
  - prompt tokens and dollars per correct call, with relock's rewrite calls counted;
  - how often the LLM rewriter fell back to the deterministic one.

## Known limits, stated up front

- **The append baselines have no trust gate.** The injected text and the wrong "fix" enter their
  memory. `append-gated` is included to show how much of the difference comes from gating and how
  much from rewriting in place.
- **relock needs 2 trusted errors to learn a correction** (its default evidence threshold).
- **Results are not exactly reproducible.** OpenRouter routes calls to different providers, so
  reruns can differ slightly. The provider of every call is recorded.
- **Budget:** about $5 in total, set by the person running the study.

## Extension (added after the first run)

The first run (seeds 0–1; loads `L0`, `L1`, `L3`; $4.30) is committed unchanged as
`docs/results/models/` in commit `e82276b`. The budget was then raised to about $10 in total, and the
same design was extended to every model and system alike:

- **Seed 2** for loads `L0`, `L1` and `L3`.
- **Load `L2`:** 4k tokens of history, with seeds 0–2.
- **A clean re-run of the `gpt-6-luna` units** that hit API errors. The responses that succeeded
  come from the cache, so only the failed calls are sent again.
- **Cheapest-provider routing** for these calls (`provider.sort = price`). The first run used
  OpenRouter's default load balancing.

The primary metric and the scoring are unchanged. The published report pools both runs.
If the budget had not covered every model, the most expensive one (`moonshotai/kimi-k3`) would
have been dropped from the extension, not given a smaller design.

Reproduce with `python scripts/model_study.py run` and an OpenRouter key.

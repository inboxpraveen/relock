# Getting started

This page takes you from install to a first in-place rewrite, then shows how to inspect and undo
it. Everything runs locally, with no model and no API key.

## Install

```
pip install relock-memory
```

The core has no dependencies and needs Python 3.10 or later. The import name is `relock_memory`.
The PyPI name `relock` belongs to an unrelated SDK. Extras add the optional backends:

| extra | adds |
|---|---|
| `relock-memory[postgres]` | the Postgres store (`psycopg`) |
| `relock-memory[mcp]` | the MCP server |
| `relock-memory[langchain]` | the LangChain / LangGraph callback handler and tools |
| `relock-memory[all]` | all three |

Not using Python? Run the HTTP API instead and follow the [HTTP guide](http.md):

```
docker run -p 8765:8765 -e RELOCK_TOKEN=change-me -v relock-data:/data ghcr.io/inboxpraveen/relock
```

## 1. Store a procedure

A memory **item** is a short procedure: a title and a list of **clauses**, one imperative sentence
each. Clauses get stable ids (`c1`, `c2`, ...), so a later fix can replace exactly one of them.

```python
import relock_memory as relock

mem = relock.Memory("agent.db")      # SQLite file; ":memory:" or "postgresql://..." also work
mem.add(
    "Fetch orders by date",
    ["Authenticate with the orders API key before calling get_orders.",
     "Pass the date to get_orders as MM/DD/YYYY."],   # stale: the API now wants ISO dates
    key="orders.fetch",              # a stable name you can refer to
    tools=["get_orders"],            # outcomes of this tool count as evidence about this item
    exist_ok=True,                   # safe to run on every start: an existing key is left alone
)
```

`tools` is what links a tool call to the memory that told the agent how to make it. With
`exist_ok=True`, seeding code can run on every start without overwriting what the agent has
learned since.

## 2. Wrap the tool

Wrap the real tool with `@mem.wrap`. Its outcome, not the model's opinion of it, is the signal
relock learns from.

```python
import re

@mem.wrap
def get_orders(date: str) -> list[dict]:
    """Fetch the orders placed on one day."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise relock.ToolError(f"invalid date {date!r}",
                               correction="Pass the date to get_orders as YYYY-MM-DD.")
    return [{"id": 1, "date": date}]
```

- The wrapper keeps the function's signature and docstring, so agent frameworks build the same
  tool schema. It works on `async def` tools too.
- Results come back unchanged and exceptions are re-raised unchanged.
- **Any exception counts as a failure**, classified by type: a `ValueError` or an HTTP 422 is a
  real error; a `TimeoutError`, an HTTP 429 or any 5xx is transient and never counts. Interrupts
  and cancellation (`KeyboardInterrupt`, `asyncio.CancelledError`) are not recorded at all.
  `relock.ToolError` lets you say what the right rule is, with `correction=`. Without one, an
  [LLM rewriter](configuration.md#rewriters) can work the fix out from the error.

## 3. Run each task in an episode

An **episode** is one task. Recalling memory inside it makes the recalled items editable for that
task only. When the `with` block ends, the episode re-locks.

```python
with mem.episode("orders for March 14") as ep:
    context = ep.recall("fetch the orders placed on a date").render()
    # put `context` in your model's prompt, let it act, and call tools as usual
```

`render()` returns a prompt block with item keys and clause ids:

```
## Relevant procedures from memory
### Fetch orders by date  [orders.fetch | v1 | confidence 0.50]
- (c1) Authenticate with the orders API key before calling get_orders.
- (c2) Pass the date to get_orders as MM/DD/YYYY.
```

## 4. Watch it learn

Put the pieces together. A stand-in for your model follows whatever date format the recalled
procedure gives:

```python
for attempt in range(1, 4):
    with mem.episode("orders for March 14") as ep:
        context = ep.recall("fetch the orders placed on a date").render()
        date = "2026-03-14" if "YYYY-MM-DD" in context else "03/14/2026"   # your model
        try:
            print(attempt, get_orders(date))
        except relock.ToolError as e:
            print(attempt, e)
```

```
1 invalid date '03/14/2026'
2 invalid date '03/14/2026'
3 [{'id': 1, 'date': '2026-03-14'}]
```

The first failure is evidence, but one failure is not enough: it could be a fluke. The second
reaches the default threshold, and relock rewrites clause `c2` in place. The third task recalls
the new version and succeeds. There is no "actually, use ISO dates" note next to the old rule;
the old rule is gone from retrieval.

The whole loop is in [`examples/01_quickstart.py`](../examples/01_quickstart.py).

## 5. Inspect and undo

Every version and every decision is stored. The [CLI](cli.md) reads the same database:

```
$ relock --db agent.db history orders.fetch
v1 [superseded] create by author at 2026-11-28T05:41:21.511337+00:00
    (c1) Authenticate with the orders API key before calling get_orders.
    (c2) Pass the date to get_orders as MM/DD/YYYY.
v2 [active] reconsolidate by hint at 2026-11-28T05:41:21.514296+00:00 (episode ep_01m8gar8f97ntcc6rqcc)
    reason: replace c2 with trusted correction
  - (c2) Pass the date to get_orders as MM/DD/YYYY.
  + (c2) Pass the date to get_orders as YYYY-MM-DD.
```

```
relock --db agent.db pending orders.fetch     # evidence so far vs. the rewrite threshold
relock --db agent.db events --item orders.fetch   # recalls, predictions, gate decisions, rewrites
relock --db agent.db rollback orders.fetch 1  # restore v1 as a new v3; history is kept
```

The same calls exist in Python: `mem.history()`, `mem.pending()`, `mem.events()` and
`mem.rollback()`.

## 6. Let a model write the fixes

The default `HintRewriter` applies the `correction` a trusted error carries, with no model. When
errors don't carry one, let a model write the clause. Any OpenAI-compatible endpoint works,
including local ones:

```python
llm = relock.OpenAICompatible("http://localhost:8000/v1", model="Qwen/Qwen3-30B-A3B-Instruct-2507")
mem = relock.Memory("agent.db", rewriter=relock.LLMRewriter(llm))
```

The model sees one item, the clause ids and what went wrong, and returns a small JSON edit. If its
output is invalid after retries, relock falls back to `HintRewriter`. See
[Rewriters](configuration.md#rewriters).

## Next steps

- [How relock works](concepts.md): what can and can't change memory, and why.
- [Python integration](python.md): explicit predictions, verifiers, human corrections,
  several agents, `Toolkit` and LangChain.
- [MCP](mcp.md) or [HTTP](http.md) if your agent isn't a Python program.
- [Configuration](configuration.md) to tune how much evidence a rewrite needs.

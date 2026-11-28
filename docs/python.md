# Using relock from Python

relock sits beside your agent framework rather than replacing it. Your agent keeps its model, its
tools and its loop. relock adds two things: memory recalled into the prompt, and the real outcome
of each tool call, which is what memory learns from.

![Where relock sits: beside your agent. Recall feeds the prompt; real tool outcomes pass a trust gate; trusted evidence rewrites the clause that fired; every version is stored](assets/architecture.svg)

| You use | Integrate with | Effort |
|---|---|---|
| Any Python agent (OpenAI Agents SDK, PydanticAI, smolagents, CrewAI, LlamaIndex, AutoGen, your own loop) | [`@mem.wrap`](#wrap-your-tools) on your tools and [`with mem.episode()`](#episodes) around each task | 3 lines |
| Raw tool calling (OpenAI-compatible, Anthropic, vLLM, Ollama) | [`relock.Toolkit`](#raw-tool-calling-toolkit) | a dispatcher branch |
| LangChain / LangGraph | [`RelockCallbackHandler`](#langchain-and-langgraph) in your run config | 1 line |

If your agent isn't Python, use the [MCP server](mcp.md) or the [HTTP API](http.md).

## Episodes

An episode is one task. Items recalled inside it become editable for that task only, and the
episode re-locks when it closes. Always use it as a context manager, so it closes even when the
task raises.

```python
with mem.episode("refund order 1042") as ep:
    recall = ep.recall("refund an order", k=3)          # top-k active items; they become editable
    prompt = SYSTEM + "\n\n" + recall.render(budget_tokens=800)
    ...                                                 # run your model and tools
```

| call | what it does |
|---|---|
| `ep.recall(query, k=3, kinds=None, tags=None)` | Ranked search over active items. Every hit becomes editable. Returns a `Recall`: iterate it for `Hit`s, or call `.render()` for a prompt block. |
| `recall.render(budget_tokens=None, show_ids=True)` | Markdown with item keys and clause ids. `budget_tokens` drops whole items from the tail. |
| `ep.use("orders.fetch")` | Load one item by key or id without a search (a named skill). It becomes editable too. |
| `ep.get("orders.fetch")` | The item's current version in this episode. Re-read it after a rewrite to retry the task with the fix. |
| `ep.close()` | Re-lock and return a summary. The `with` block calls it for you. |
| `mem.search(query, k=5)` | Read-only search. Nothing becomes editable. |

Pass `mem.episode(task, policy=...)` to give one agent a different [write policy](configuration.md#write-policy).
Other keyword arguments are stored as the episode's metadata.

## Wrap your tools

```python
@mem.wrap
def get_orders(date: str) -> list[dict]: ...

@mem.wrap(name="search")                 # the tool name items refer to, if not the function's
async def web_search(q: str) -> str: ...

@mem.wrap(items=["orders.fetch"])        # attribute outcomes to these items, whatever their tools
def legacy_orders(date: str) -> list[dict]: ...
```

Inside an episode, each call to a wrapped tool is checked against the recalled items that govern
it, which are the ones whose `tools=[...]` names the tool. A return counts as a success; an
exception counts as a failure. The wrapper:

- keeps the signature, docstring and `async`-ness, so framework tool schemas don't change;
- returns results and re-raises exceptions unchanged;
- logs a problem it hits while recording an outcome instead of raising it, so your agent sees the
  tool's own result or exception;
- ignores interrupts and cancellation (`KeyboardInterrupt`, `asyncio.CancelledError`, LangGraph's
  `interrupt()`), which say nothing about the procedure;
- does nothing outside an episode.

Wrapping a tool twice is harmless: `mem.wrap` returns an already wrapped tool unchanged. A
`functools.partial` is recorded under its function's name; other callables without a name, under
their class name (with a warning), so pass `name=` for those.

`mem.wrap_tools([f, g, h])` wraps a list. For tools you can't wrap, such as ones that run in
another process, report the outcome yourself. Items named in `items=` must have been recalled in
the episode; others are skipped with a warning.

```python
ep.tool_result("get_orders", rows)                                 # success
ep.tool_result("get_orders", error=exc)                            # failure, classified like a raise
ep.tool_result("get_orders", error="invalid date", items=["orders.fetch"])
```

## Report failures well

What a failure teaches depends on what it says.

```python
raise relock.ToolError("invalid date '03/14/2026'",
                       correction="Pass the date to get_orders as YYYY-MM-DD.")
raise relock.TransientToolError("upstream is down")       # never counts as evidence
```

- **`correction`** is the rule that would have been right. The default `HintRewriter` writes it
  into the clause that fired, with no model. Without a correction, an
  [LLM rewriter](configuration.md#rewriters) can infer the fix from the error.
- **Any other exception is classified by type.** A `ValueError`, `TypeError`, `LookupError` or an
  HTTP 4xx such as 400, 404 or 422 counts. A `TimeoutError`, a `ConnectionError`, an HTTP 408,
  425 or 429, or any 5xx is transient and never counts. Anything unrecognised counts at half
  weight. The full rules are in the [error taxonomy](design.md#error-taxonomy).
- **An error given only as a message**, as in `ep.tool_result("get_orders", error="...")`,
  counts at half weight, unless it reads as transient: a timeout, a rate limit, a 5xx status or a
  dropped connection never counts.
- **Error text from a remote server can carry injected text.** Raise
  `relock.ToolError(message, detail_trusted=False)`, or report the failure with
  `Signal.tool_error(exc, detail_trusted=False)`. The failure still counts, but its message and
  correction never reach the rewriter.

```python
try:
    rows = call_vendor_api(date)
except VendorError as exc:
    ep.tool_result("vendor_orders", signal=relock.Signal.tool_error(exc, detail_trusted=False))
    raise
```

## Explicit predictions

A wrapped tool records an automatic prediction, "this call succeeds". To predict something more
specific, or to say which clauses the action relies on, record an expectation before acting:

```python
from relock_memory import checks

item = ep.get("orders.fetch")
exp = ep.expect(item, "returns the orders for the day",
                check=checks.non_empty(), clauses=["c2"], tool="get_orders")
rows = get_orders("2026-03-14")          # the wrapped tool resolves `exp` with its real result
```

- **`clauses`** limits which clauses a rewrite may replace or remove. Cite clauses whenever you
  use an LLM rewriter, so a fix can't land on the wrong clause.
- **`tool`** binds the expectation to a wrapped tool, whose next call resolves it, whether or
  not the item lists that tool in `tools`.
- **`check`** decides whether the outcome matches. The built-ins in `relock_memory.checks` are
  `succeeds()`, `fails()`, `equals(x)`, `non_empty()`, `has_keys(*keys)`, `matches(regex)`,
  `predicate(fn)` and `schema(json_schema)`. Any `(result, error) -> (bool, reason)` callable also
  works. The check sees what the call returned or raised, so `fails()` matches a call that raises.
  A transient error never matches, and a check that itself raises counts as a mismatch.

Without a `tool`, resolve the expectation yourself: `ep.observe(exp, result)` or
`ep.observe(exp, error=exc)`. Each expectation resolves once; open ones are voided when the
episode closes.

## Verifiers and human corrections

A **verifier** is a check you trust: a test suite, a schema validator, a grader.

```python
ok = run_tests()
ep.check("deploy.service", passed=ok, message=test_output,
         correction=None, clauses=["c3"], verifier="pytest")
```

A **human correction** rewrites memory only when your application vouches for the person:

```python
ep.correct("orders.fetch", "Pass the date to get_orders as YYYY-MM-DD.",
           principal="ops@example.com", authenticated=True)   # rewrites immediately
```

Without `authenticated=True` the correction is logged as blocked, because "the user said so" is
exactly what a prompt injection claims. Authenticated corrections skip the evidence threshold; set
`WritePolicy(authoritative_users=False)` to make them count as ordinary evidence instead.

Every call returns an `Observation` whose `decision` says what happened: `confirmed`, `pending`
(with the evidence so far and the amount required), `rewritten` (with the new version and its
delta), `blocked` (untrusted or transient), `declined` (the rewriter proposed no change),
`invalid_edit` (the proposed edit broke a rule), `conflict` (another episode rewrote the item
first) or `ignored`. A success is `ignored` when an untrusted source reports it, or when it comes
right after a failure of the same tool in the same episode that carried a correction: the agent
most likely followed the correction, so the success says nothing for the stored rule (see
[flake damping](design.md#admission-trust-gate)).

An authenticated correction that the rewriter declines, or turns into an invalid edit, is closed
rather than left pending, so it can't carry later failures past the evidence threshold.

## Several agents, threads and asyncio

A wrapped tool finds its episode through a context variable, so it records its outcome only in the
episode of the agent that called it:

- inside `with mem.episode(...) as ep:`;
- inside `with ep.activate():`, which makes an episode active without closing it on exit;
- in work submitted with `ep.run(fn, *args)`.

asyncio tasks inherit the active episode. Plain threads (`threading.Thread`,
`ThreadPoolExecutor`) don't, so hand work to them with `ep.run`:

```python
with mem.episode("nightly sync") as ep, ThreadPoolExecutor() as pool:
    ep.recall("sync the orders")
    futures = [pool.submit(ep.run, get_orders, d) for d in dates]
```

A wrapped tool that runs outside every episode records nothing. If other episodes are open at the
time, relock logs a warning naming the tool, once per tool. Give each agent its own episode, or
its own `Toolkit`.

## Raw tool calling: `Toolkit`

If you drive the model's tool calls yourself, `Toolkit` gives the model memory tools and a
dispatcher. Any provider works:

```python
toolkit = relock.Toolkit(mem, k=3)
tools = [*my_tools, *toolkit.openai_tools()]      # or toolkit.anthropic_tools(), toolkit.schemas()

for call in response.tool_calls:
    if toolkit.handles(call.name):
        output = toolkit.call(call.name, call.arguments)   # a dict or a JSON string
    else:
        output = my_dispatch(call)                         # your tools, wrapped with @mem.wrap
```

The model gets `recall_memory` and `state_expectation`, plus `report_outcome` (on by default, but
untrusted: logged, never applied) and `teach_memory` (off unless `allow_teach=True`). The toolkit
opens its own episode on first use, and re-locks it after `idle_timeout` seconds without a call
(30 minutes by default). Call `toolkit.begin_episode(task)` and `toolkit.end_episode()` to mark
task boundaries explicitly. `policy=` sets the write policy for the toolkit's episodes.

`toolkit.functions()` returns the same tools as plain Python callables with docstrings and type
hints, for frameworks that build tool schemas from functions.

## LangChain and LangGraph

```
pip install "relock-memory[langchain]"
```

```python
from relock_memory.integrations.langchain import RelockCallbackHandler, langchain_tools

handler = RelockCallbackHandler(mem)
tools = [*my_tools, *langchain_tools(relock.Toolkit(mem))]      # optional: recall as a tool

with mem.episode("deploy build 4242"):
    graph.invoke(inputs, config={"callbacks": [handler]})
```

The handler records the real outcome of every tool run in the active episode, so your existing
tools stay untouched. Keep in mind:

- **A tool can also be wrapped with `@mem.wrap`.** The handler records the call and the wrapper
  stands aside, so each call counts once.
- **Interrupts are not failures.** LangGraph's `interrupt()`, `Command(goto=...)` and cancelled
  runs are never recorded.
- **Run the graph inside an episode**, as above. The handler records nothing outside one.
- **Let tools raise.** LangChain reports a raised exception through `on_tool_error`, including
  inside LangGraph's `ToolNode`, and a `ToolMessage` with `status="error"` also counts as a
  failure. A tool that turns its own errors into text (`handle_tool_error=True`) and is called
  with plain arguments returns what looks like a normal result, and is recorded as a success.

A runnable version is [`examples/03_langgraph_callbacks.py`](../examples/03_langgraph_callbacks.py).

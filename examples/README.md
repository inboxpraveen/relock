# Examples

Each example runs on its own, with no model or API key, and CI runs all of them on every change.

| example | shows |
|---|---|
| [`01_quickstart.py`](01_quickstart.py) | The whole loop in about 60 lines: a stale rule, two trusted failures, an in-place rewrite, and the history. |
| [`02_llm_rewriter.py`](02_llm_rewriter.py) | A model writes the fix from an error that carries no correction. Uses a local OpenAI-compatible server if `RELOCK_LLM_BASE_URL` is set, else a stub. |
| [`03_langgraph_callbacks.py`](03_langgraph_callbacks.py) | LangChain / LangGraph: one callback handler, existing tools untouched. Needs `pip install "relock-memory[langchain]"`. |
| [`http/curl.sh`](http/curl.sh) | The HTTP loop from a shell script. |
| [`http/typescript/`](http/typescript/) | A small typed client and demo. Node 22.18 or later runs TypeScript directly: `node demo.ts`. |
| [`http/rust/`](http/rust/) | The same loop from Rust: `cargo run`. |

The HTTP examples talk to a running server:

```
relock serve                     # http://127.0.0.1:8765; set RELOCK_TOKEN to require a token
sh examples/http/curl.sh         # RELOCK_URL and RELOCK_TOKEN are read from the environment
```

Guides: [Getting started](../docs/getting-started.md), [Python](../docs/python.md),
[HTTP API](../docs/http.md).

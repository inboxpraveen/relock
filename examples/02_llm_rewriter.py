"""Let a local open model write the in-place fixes.

With a server running (vLLM, SGLang, Ollama, llama.cpp, LM Studio...):

    export RELOCK_LLM_BASE_URL=http://localhost:8000/v1
    export RELOCK_LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
    python examples/02_llm_rewriter.py

Without one, a stub model stands in so the example still runs.
"""

import json
import os

import relock_memory as relock

if os.environ.get("RELOCK_LLM_BASE_URL"):
    llm = relock.OpenAICompatible(json_mode=False)  # reads RELOCK_LLM_* from the environment
    print(f"using {llm!r}")
else:
    def stub_model(prompt: str) -> str:  # answers the way a well-behaved model would
        return json.dumps({"ops": [{"op": "replace", "id": "c2",
                                    "text": "Send amounts to charge_card as integer cents, e.g. 1250 for $12.50."}],
                           "rationale": "the API validates amounts as integer cents"})
    llm = stub_model
    print("no RELOCK_LLM_BASE_URL set: using a stub model")

mem = relock.Memory(
    ":memory:",
    rewriter=relock.LLMRewriter(llm),       # falls back to the deterministic HintRewriter on bad output
    policy=relock.WritePolicy(min_evidence=2.0),
)
mem.add("Charge a card", ["Look up the customer's default card first.",
                          "Send amounts to charge_card in dollars, e.g. 12.50."],
        key="payments.charge", tools=["charge_card"])


@mem.wrap
def charge_card(amount):
    """Charge the default card."""
    if not isinstance(amount, int):
        raise relock.ToolError("ValidationError: amount must be an integer")  # no hint: the model must infer it
    return {"status": "charged", "amount": amount}


for attempt in range(2):
    with mem.episode("renewal charge") as ep:
        ep.recall("charge the customer's card")
        try:
            charge_card(12.50)
        except relock.ToolError as e:
            print(f"attempt {attempt + 1}: {e}")

item = mem.get("payments.charge")
print(f"\nv{item.version} written by {item.author}:")
print(item.render())

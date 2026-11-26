"""relock in 60 lines: an agent learns that an API changed, without keeping the stale rule around.

Run:  python examples/01_quickstart.py
"""

import re

import relock_memory as relock

mem = relock.Memory(":memory:")  # or "agent.db", or "postgresql://..."

# Seed a procedure. It is wrong now: the orders API switched to ISO dates.
mem.add(
    "Fetch orders by date",
    [
        "Authenticate with the orders API key before calling get_orders.",
        "Pass the date to get_orders as MM/DD/YYYY.",
    ],
    key="orders.fetch",
    tools=["get_orders"],  # outcomes of this tool are attributed to this item
)


# Wrap your real tool. Its outcome, not the model's opinion, is the trusted signal.
@mem.wrap
def get_orders(date: str) -> list[dict]:
    """Fetch the orders placed on one day."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise relock.ToolError(f"invalid date {date!r}", correction="Pass the date to get_orders as YYYY-MM-DD.")
    return [{"id": 1, "date": date}]


def agent_step(procedure: relock.Item) -> str:
    """Stand-in for your LLM: follows whatever date format the procedure says."""
    return "2026-03-14" if "YYYY-MM-DD" in procedure.text else "03/14/2026"


for attempt in range(1, 4):
    with mem.episode(f"orders for March 14 (attempt {attempt})") as ep:
        recall = ep.recall("fetch the orders placed on a date")  # recalled items become editable
        procedure = ep.get("orders.fetch")
        try:
            rows = get_orders(agent_step(procedure))
            print(f"attempt {attempt}: ok -> {rows}")
        except relock.ToolError as e:
            now = mem.get("orders.fetch")
            if now.version > procedure.version:
                print(f"attempt {attempt}: {e}  -> enough evidence: rewritten in place as v{now.version}")
            else:
                state = mem.pending("orders.fetch")
                print(f"attempt {attempt}: {e}  -> evidence {state['evidence']:.0f}/{state['required']:.0f}, "
                      "memory unchanged")
    # leaving the block re-locks: nothing can edit memory through this episode any more

print("\nCurrent procedure:")
print(mem.get("orders.fetch").render())

print("\nHistory (the old rule is superseded, not left active):")
for v in mem.history("orders.fetch"):
    print(f"  v{v.version} [{v.status}] {v.change_kind}: {v.clauses[-1].text}")

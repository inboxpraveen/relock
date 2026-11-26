"""Zero-touch LangChain / LangGraph integration: add a callback handler, keep your tools as they are.

    pip install "relock-memory[langchain]"
    python examples/03_langgraph_callbacks.py

In a real graph you pass the same handler in the run config:

    graph.invoke(inputs, config={"callbacks": [RelockCallbackHandler(mem)]})

and give the model the memory tools:

    tools = [*your_tools, *langchain_tools(Toolkit(mem))]
"""

from langchain_core.tools import StructuredTool

import relock_memory as relock
from relock_memory.integrations.langchain import RelockCallbackHandler, langchain_tools

mem = relock.Memory(":memory:")
mem.add("Deploy a service", ["Deploy only builds that passed CI.", "Deploy services to the us-east-1 region."],
        key="deploy.service", tools=["deploy"])


def deploy(build: int, region: str) -> str:
    """Deploy a build to a region."""
    if region != "eu-west-1":
        # Any exception works (it is classified as programmatic / transient / ...). ToolError can also
        # carry an explicit correction, which the default HintRewriter applies without a model.
        raise relock.ToolError(f"PolicyError: region {region} is frozen",
                               correction="Deploy services to the eu-west-1 region.")
    return f"build {build} deployed to {region}"


deploy_tool = StructuredTool.from_function(deploy)  # an ordinary, unwrapped LangChain tool
handler = RelockCallbackHandler(mem)
memory_tools = langchain_tools(relock.Toolkit(mem))
print("memory tools for the model:", [t.name for t in memory_tools])

for attempt in range(1, 4):
    with mem.episode(f"deploy build 4242 (attempt {attempt})") as ep:
        procedure = ep.recall("deploy a build of a service")[0].item
        region = "eu-west-1" if "eu-west-1" in procedure.text else "us-east-1"  # stand-in for the model
        try:
            out = deploy_tool.invoke({"build": 4242, "region": region}, config={"callbacks": [handler]})
            print(f"attempt {attempt}: {out}")
        except relock.ToolError as e:
            print(f"attempt {attempt}: {e}")

print()
print(mem.get("deploy.service").render())

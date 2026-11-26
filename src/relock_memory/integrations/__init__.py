"""Integrations. Each optional dependency is imported only by its own module.

* :mod:`relock_memory.integrations.toolkit`   - framework-agnostic tools and JSON schemas
* :mod:`relock_memory.integrations.mcp`       - MCP server (``relock mcp``)
* :mod:`relock_memory.integrations.langchain` - LangChain / LangGraph callbacks and tools
"""

from .toolkit import Toolkit

__all__ = ["Toolkit"]

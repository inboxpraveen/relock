"""Agents for the behavioural mode of RelapseBench."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol

from ..llm import extract_json
from .scenarios import Family, Variants

SYSTEM = (
    "You are an operations agent. For each task you make exactly one tool call. "
    "Follow the procedures from memory: they reflect the current API contracts."
)

PROMPT = """{padding_block}{memory}

TASK: {task}

Call the tool `{tool}`. Reply with one JSON object only, no explanation:
{{"tool": "{tool}", "args": {{"{param}": "<value>"}}}}"""


class Agent(Protocol):
    name: str

    def act(self, family: Family, task: str, memory: str, padding: str, variants: Variants) -> str:
        """Return the value the agent passes for the family's parameter."""
        ...


def build_prompt(family: Family, task: str, memory: str, padding: str) -> str:
    padding_block = f"Earlier in this session:\n{padding}\n\n" if padding else ""
    return PROMPT.format(padding_block=padding_block, memory=memory or "## Procedures from memory\n(none)",
                         task=task, tool=family.tool, param=family.param)


class LLMAgent:
    """Any ``Callable[[str], str]`` model (e.g. :class:`~relock_memory.llm.OpenAICompatible`)."""

    def __init__(self, llm: Callable[[str], str], name: str | None = None) -> None:
        self.llm = llm
        self.name = name or f"llm:{getattr(llm, 'model', 'callable')}"
        self.failures = 0

    def act(self, family: Family, task: str, memory: str, padding: str, variants: Variants) -> str:
        prompt = build_prompt(family, task, memory, padding)
        try:
            reply = self.llm(f"{SYSTEM}\n\n{prompt}")
            data = extract_json(reply)
            args = data.get("args") or data.get("arguments") or data
            value = args.get(family.param, "")
        except Exception:
            self.failures += 1
            return ""
        return str(value)


class ScriptedAgent:
    """Deterministic stand-in used to test the harness. NOT a model of LLM behaviour.

    It follows whichever rule variant appears *first* in its memory context and
    falls back to the old rule when memory mentions none. Its numbers only show
    that the pipeline runs; never report them as results.
    """

    name = "scripted (pipeline test only)"

    def act(self, family: Family, task: str, memory: str, padding: str, variants: Variants) -> str:
        first, pos = "old", len(memory) + 1
        for k, rx in family.markers.items():
            m = re.search(rx, memory, re.IGNORECASE)
            if m and m.start() < pos:
                first, pos = k, m.start()
        return variants[first]

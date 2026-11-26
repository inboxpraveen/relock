"""Rewriters turn trusted prediction errors into a small clause-level delta.

Credit assignment is done by the harness (retrieval trace + expectation diff),
so a rewriter only has to make a *local* edit: "here is the item, here is what
it predicted, here is what happened, fix the clause". That is a task a ~30B
open-weight model handles reliably, and the prompt below is written for that
size class: one item, explicit clause ids, strict JSON, one worked example.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ._util import tokenize, truncate
from .delta import DeltaError, Op, parse_ops, validate
from .llm import extract_json
from .types import Item


@dataclass(frozen=True)
class Evidence:
    """One admitted prediction error, sanitised for the rewriter."""

    error_id: str
    expectation: str
    clause_ids: tuple[str, ...]
    tool: str | None
    observed: str
    source: str
    error_class: str
    message: str
    correction: str | None
    authoritative: bool = False


@dataclass(frozen=True)
class RewriteRequest:
    item: Item
    evidence: tuple[Evidence, ...]
    editable: frozenset[str] | None
    """Clause ids that may be replaced/removed (``None`` means the whole item)."""
    task: str = ""
    max_ops: int = 4
    max_clause_chars: int = 600


@dataclass(frozen=True)
class Proposal:
    ops: list[Op]
    rationale: str = ""
    author: str = "unknown"
    raw: str | None = field(default=None, repr=False)


@runtime_checkable
class Rewriter(Protocol):
    def propose(self, request: RewriteRequest) -> Proposal | None:
        """Return a delta, or ``None`` if no change is justified."""
        ...


def _overlap(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class HintRewriter:
    """Deterministic rewriter that applies explicit ``correction`` text from trusted signals.

    It replaces the cited clause (or, when nothing was cited, the clause most
    similar to the correction), and adds a new clause when nothing is similar.
    It needs no model, which makes it the default for tests, CI and the
    offline benchmark, and the fallback when an LLM rewriter fails.
    """

    name = "hint"

    def __init__(self, min_overlap: float = 0.12) -> None:
        self.min_overlap = min_overlap

    def propose(self, request: RewriteRequest) -> Proposal | None:
        corrections = [e for e in request.evidence if e.correction]
        if not corrections:
            return None
        authoritative = [e for e in corrections if e.authoritative]
        if authoritative:
            chosen = authoritative[-1].correction or ""
        else:
            counts = Counter(e.correction for e in corrections)
            best = max(counts.values())
            chosen = next(e.correction for e in reversed(corrections) if counts[e.correction] == best) or ""
        chosen = truncate(chosen.strip(), request.max_clause_chars)
        clauses = request.item.clauses
        pool = [c for c in clauses if request.editable is None or c.id in request.editable]
        if not pool:
            return Proposal([Op("add", text=chosen)], "no editable clause; adding correction", self.name)
        scored = sorted(pool, key=lambda c: (-_overlap(c.text, chosen), clauses.index(c)))
        target = scored[0]
        if target.text.strip() == chosen:
            return None
        cited = request.editable is not None
        if cited or _overlap(target.text, chosen) >= self.min_overlap:
            return Proposal([Op("replace", id=target.id, text=chosen)],
                            f"replace {target.id} with trusted correction", self.name)
        return Proposal([Op("add", text=chosen)], "no similar clause; adding correction", self.name)


REWRITE_PROMPT = """You maintain the procedural memory of a software agent.
A memory item was used, the agent predicted an outcome, and a TRUSTED check showed the prediction was wrong.
Repair the item IN PLACE with the smallest possible edit so the same failure does not happen again.

Rules:
- Replace the wrong instruction. Never keep it and add "don't do X" next to it.
- Only these clauses may be replaced or removed: {editable}. You may also add a clause.
- Leave every clause that is still correct untouched.
- Each clause is one short imperative sentence. At most {max_ops} operations.
- If the evidence does not justify any change, return {{"ops": [], "rationale": "..."}}.

Example
ITEM "Fetch invoices" (version 1)
[c1] Authenticate with the billing API key.
[c2] Call list_invoices with month as a name, e.g. "March".
EXPECTED: list_invoices returns invoices
WHAT HAPPENED: ValidationError: month must be an integer 1-12
TRUSTED CORRECTION: none
Reply:
{{"ops": [{{"op": "replace", "id": "c2", "text": "Call list_invoices with month as an integer 1-12, e.g. 3 for March."}}], "rationale": "API requires numeric months"}}

Now the real item.
ITEM "{title}" (version {version})
{clauses}
{task}EVIDENCE ({n} trusted prediction error{plural}):
{evidence}

Reply with one JSON object only, using ops of the form
{{"op": "replace", "id": "<clause id>", "text": "..."}}, {{"op": "add", "text": "...", "after": "<clause id or null>"}}, {{"op": "remove", "id": "<clause id>"}}."""


def build_prompt(request: RewriteRequest) -> str:
    item = request.item
    clauses = "\n".join(f"[{c.id}] {c.text}" for c in item.clauses)
    editable = ", ".join(sorted(request.editable)) if request.editable else "any clause"
    ev_lines = []
    for i, e in enumerate(request.evidence, 1):
        parts = [f"{i}. EXPECTED: {truncate(e.expectation, 300)}"]
        if e.clause_ids:
            parts.append(f"   RELIED ON: {', '.join(e.clause_ids)}")
        if e.tool:
            parts.append(f"   TOOL: {e.tool}")
        parts.append(f"   WHAT HAPPENED: {truncate(e.observed, 400)}")
        if e.message:
            parts.append(f"   ERROR ({e.source}/{e.error_class}): {truncate(e.message, 400)}")
        parts.append(f"   TRUSTED CORRECTION: {truncate(e.correction, 300) if e.correction else 'none'}")
        ev_lines.append("\n".join(parts))
    task = f"TASK CONTEXT: {truncate(request.task, 300)}\n" if request.task else ""
    n = len(request.evidence)
    return REWRITE_PROMPT.format(
        editable=editable, max_ops=request.max_ops, title=item.title, version=item.version,
        clauses=clauses, task=task, n=n, plural="" if n == 1 else "s", evidence="\n".join(ev_lines),
    )


class LLMRewriter:
    """Rewriter backed by any text-completion callable (local or hosted).

    ``llm`` is any ``Callable[[str], str]``: :class:`~relock_memory.llm.OpenAICompatible`,
    a LiteLLM wrapper, an Anthropic or OpenAI SDK lambda, a vLLM engine...
    Invalid JSON or an illegal delta is fed back to the model for another
    attempt; after ``max_attempts`` the ``fallback`` rewriter is used.
    """

    def __init__(
        self,
        llm: Callable[[str], str],
        *,
        max_attempts: int = 3,
        fallback: Rewriter | None = None,
        name: str | None = None,
    ) -> None:
        self.llm = llm
        self.max_attempts = max_attempts
        self.fallback = fallback if fallback is not None else HintRewriter()
        self.name = name or f"llm:{getattr(llm, 'model', getattr(llm, '__name__', 'callable'))}"
        self.last_error: str | None = None

    def propose(self, request: RewriteRequest) -> Proposal | None:
        prompt = build_prompt(request)
        editable = set(request.editable) if request.editable is not None else None
        feedback = ""
        raw = ""
        for _ in range(self.max_attempts):
            try:
                raw = self.llm(prompt + feedback)
            except Exception as e:  # endpoint down, auth error...: don't crash the agent's episode
                self.last_error = f"llm call failed: {type(e).__name__}: {e}"
                break
            try:
                data: Any = extract_json(raw)
                if isinstance(data, list):
                    data = {"ops": data}
                if not isinstance(data, dict):
                    raise DeltaError("reply must be a JSON object with an 'ops' list")
                ops = parse_ops(data.get("ops") or [])
                rationale = str(data.get("rationale") or "")
                if not ops:
                    return None
                validate(request.item.clauses, ops, editable=editable,
                         max_ops=request.max_ops, max_clause_chars=request.max_clause_chars)
                return Proposal(ops, rationale, self.name, raw)
            except (ValueError, DeltaError, KeyError, TypeError) as e:
                self.last_error = str(e)
                feedback = (
                    f"\n\nYour previous reply was rejected: {e}.\n"
                    f"Previous reply: {truncate(raw, 400)}\nReply again with one valid JSON object only."
                )
        if self.fallback is None:
            return None
        proposal = self.fallback.propose(request)
        if proposal is None:
            return None
        return Proposal(proposal.ops, f"fallback after LLM failure ({self.last_error}): {proposal.rationale}",
                        f"{self.name}+{getattr(self.fallback, 'name', 'fallback')}")


class CallableRewriter:
    """Adapt a plain function ``(RewriteRequest) -> list[dict] | None`` into a :class:`Rewriter`."""

    def __init__(self, fn: Callable[[RewriteRequest], Any], name: str = "callable") -> None:
        self.fn = fn
        self.name = name

    def propose(self, request: RewriteRequest) -> Proposal | None:
        out = self.fn(request)
        if out is None:
            return None
        if isinstance(out, Proposal):
            return out
        if isinstance(out, dict):
            ops = parse_ops(out.get("ops") or [])
            return Proposal(ops, str(out.get("rationale", "")), self.name) if ops else None
        ops = parse_ops(out)
        return Proposal(ops, "", self.name) if ops else None

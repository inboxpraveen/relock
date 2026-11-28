"""Core value types: items, clauses, hits, expectations and write decisions."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RelockError(Exception):
    """Base class for relock errors."""


class EpisodeClosedError(RelockError):
    """Raised when something tries to write through an episode that has re-locked."""


class NotLabileError(RelockError):
    """Raised when an expectation targets an item that was not recalled in this episode."""


class ConflictError(RelockError):
    """Raised when an item changed underneath a pending write (optimistic concurrency)."""


class NotFoundError(RelockError, KeyError):
    """Raised when an item, version, episode or expectation does not exist."""

    def __str__(self) -> str:  # KeyError's str() would wrap the message in quotes
        return str(self.args[0]) if len(self.args) == 1 else super().__str__()


class StoreError(RelockError):
    """The database failed or could not be reached. The driver's exception is chained as the cause.

    A lost connection is discarded, so the next call reconnects. A write that fails this way
    was rolled back unless the connection dropped while committing; then its outcome is unknown.
    """


@dataclass(frozen=True)
class Clause:
    """One atomic instruction inside an item. Edits operate on clauses, not on whole items."""

    id: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "text": self.text}


@dataclass(frozen=True)
class Item:
    """A versioned memory item (a procedure, playbook entry, skill note or fact).

    An ``Item`` value is a snapshot of one version. Only one version per item is
    ``active``; older versions are kept as ``superseded`` for audit and rollback.
    """

    id: str
    version: int
    title: str
    clauses: tuple[Clause, ...]
    key: str | None = None
    kind: str = "procedure"
    tags: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    """Tool names this item governs; used to attribute tool outcomes to the item."""
    meta: dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    alpha: float = 1.0
    beta: float = 1.0
    confirmations: int = 0
    failures: int = 0
    parent_version: int | None = None
    superseded_by: int | None = None
    change_kind: str = "create"
    delta: list[dict[str, Any]] | None = None
    reason: str | None = None
    episode_id: str | None = None
    evidence: tuple[str, ...] = ()
    author: str | None = None
    clause_seq: int = 0
    created_at: str = ""
    archived: bool = False

    @property
    def confidence(self) -> float:
        """Posterior mean of a Beta(alpha, beta) belief that this version's predictions hold."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def text(self) -> str:
        return "\n".join(c.text for c in self.clauses)

    @property
    def ref(self) -> str:
        return self.key or self.id

    def clause(self, clause_id: str) -> Clause:
        for c in self.clauses:
            if c.id == clause_id:
                return c
        raise NotFoundError(f"clause {clause_id!r} not in {self.ref} v{self.version}")

    def search_text(self) -> str:
        parts = [self.title, self.key or "", self.kind, " ".join(self.tags), " ".join(self.tools), self.text]
        return " ".join(p for p in parts if p)

    def render(self, *, show_ids: bool = True, show_meta: bool = True) -> str:
        head = f"### {self.title}"
        if show_meta:
            head += f"  [{self.ref} | v{self.version} | confidence {self.confidence:.2f}]"
        lines = [head]
        for c in self.clauses:
            lines.append(f"- ({c.id}) {c.text}" if show_ids else f"- {c.text}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "key": self.key,
            "version": self.version,
            "status": self.status,
            "title": self.title,
            "kind": self.kind,
            "clauses": [c.to_dict() for c in self.clauses],
            "tags": list(self.tags),
            "tools": list(self.tools),
            "meta": self.meta,
            "confidence": round(self.confidence, 4),
            "alpha": self.alpha,
            "beta": self.beta,
            "confirmations": self.confirmations,
            "failures": self.failures,
            "parent_version": self.parent_version,
            "superseded_by": self.superseded_by,
            "change_kind": self.change_kind,
            "delta": self.delta,
            "reason": self.reason,
            "episode_id": self.episode_id,
            "evidence": list(self.evidence),
            "author": self.author,
            "created_at": self.created_at,
            "archived": self.archived,
        }


@dataclass(frozen=True)
class Hit:
    item: Item
    score: float
    rank: int


class Recall(Sequence[Hit]):
    """The result of :meth:`Episode.recall`: ranked hits that are now *labile* (editable)."""

    def __init__(self, query: str, hits: list[Hit]) -> None:
        self.query = query
        self.hits = hits

    def __getitem__(self, i):  # type: ignore[override]
        return self.hits[i]

    def __len__(self) -> int:
        return len(self.hits)

    def __iter__(self) -> Iterator[Hit]:
        return iter(self.hits)

    @property
    def items(self) -> list[Item]:
        return [h.item for h in self.hits]

    @property
    def ids(self) -> list[str]:
        return [h.item.id for h in self.hits]

    def render(self, *, header: str = "Relevant procedures from memory", show_ids: bool = True,
               budget_tokens: int | None = None) -> str:
        """Render hits as a prompt block. ``budget_tokens`` truncates whole items from the tail."""
        from ._util import approx_tokens

        if not self.hits:
            return ""
        blocks: list[str] = []
        used = approx_tokens(header)
        for h in self.hits:
            block = h.item.render(show_ids=show_ids)
            cost = approx_tokens(block)
            if budget_tokens is not None and blocks and used + cost > budget_tokens:
                break
            blocks.append(block)
            used += cost
        return f"## {header}\n" + "\n\n".join(blocks)

    def __repr__(self) -> str:
        return f"Recall({self.query!r}, {[f'{h.item.ref}@v{h.item.version}' for h in self.hits]})"


class Verdict(str, Enum):
    """What happened to an observation."""

    CONFIRMED = "confirmed"        # prediction held; confidence up
    PENDING = "pending"            # trusted prediction error recorded; not enough evidence yet
    REWRITTEN = "rewritten"        # evidence crossed the threshold; item reconsolidated in place
    BLOCKED = "blocked"            # error logged but not allowed to count (untrusted / transient)
    DECLINED = "declined"          # rewriter judged no change was justified
    INVALID_EDIT = "invalid_edit"  # proposed delta violated policy; nothing written
    CONFLICT = "conflict"          # item changed underneath us; evidence went stale
    IGNORED = "ignored"            # mismatch from an untrusted source on a success path, etc.


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    reason: str = ""
    item_id: str | None = None
    version: int | None = None
    new_version: int | None = None
    evidence: float = 0.0
    required: float = 0.0
    delta: list[dict[str, Any]] | None = None

    @property
    def wrote(self) -> bool:
        return self.verdict is Verdict.REWRITTEN

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "item_id": self.item_id,
            "version": self.version,
            "new_version": self.new_version,
            "evidence": round(self.evidence, 4),
            "required": round(self.required, 4),
            "delta": self.delta,
        }


@dataclass(frozen=True)
class Expectation:
    """A structured prediction made *before* acting on a recalled item."""

    id: str
    episode_id: str
    item_id: str
    version: int
    statement: str
    clause_ids: tuple[str, ...] = ()
    tool: str | None = None
    check: Any = None
    auto: bool = False
    author: str = "harness"
    """Who wrote ``statement``: ``"harness"`` (your code) or ``"agent"`` (the model, via tools)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "episode_id": self.episode_id,
            "item_id": self.item_id,
            "version": self.version,
            "statement": self.statement,
            "clause_ids": list(self.clause_ids),
            "tool": self.tool,
            "check": getattr(self.check, "spec", None),
            "auto": self.auto,
            "author": self.author,
        }


@dataclass(frozen=True)
class Observation:
    """The outcome of :meth:`Episode.observe`."""

    expectation_id: str
    matched: bool
    reason: str
    decision: Decision
    error_id: str | None = None

    @property
    def verdict(self) -> Verdict:
        return self.decision.verdict

    @property
    def wrote(self) -> bool:
        return self.decision.wrote

    def to_dict(self) -> dict[str, Any]:
        return {
            "expectation_id": self.expectation_id,
            "matched": self.matched,
            "reason": self.reason,
            "error_id": self.error_id,
            "decision": self.decision.to_dict(),
        }

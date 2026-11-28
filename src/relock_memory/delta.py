"""Clause-level deltas.

Reconsolidation never regenerates an item wholesale. A rewrite is a short list
of operations on clause ids, which keeps edits local, auditable and resistant
to the "context collapse" that repeated full rewrites cause:

* ``replace`` - rewrite one clause in place (it keeps its id)
* ``add``     - insert a new clause (optionally after a given clause)
* ``remove``  - drop a clause that turned out to be wrong
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .types import Clause, RelockError

OPS = ("replace", "add", "remove")


class DeltaError(RelockError, ValueError):
    """A delta is malformed or violates the edit policy."""


@dataclass(frozen=True)
class Op:
    op: str
    id: str | None = None
    text: str | None = None
    after: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"op": self.op}
        if self.id is not None:
            d["id"] = self.id
        if self.text is not None:
            d["text"] = self.text
        if self.after is not None:
            d["after"] = self.after
        return d


def replace(clause_id: str, text: str) -> Op:
    return Op("replace", id=clause_id, text=text)


def add(text: str, after: str | None = None) -> Op:
    return Op("add", text=text, after=after)


def remove(clause_id: str) -> Op:
    return Op("remove", id=clause_id)


def parse_ops(raw: Iterable[Any]) -> list[Op]:
    """Parse ops from dicts (as produced by an LLM) or pass through :class:`Op` values."""
    ops: list[Op] = []
    for i, r in enumerate(raw):
        if isinstance(r, Op):
            ops.append(r)
            continue
        if not isinstance(r, dict):
            raise DeltaError(f"op #{i} is not an object: {r!r}")
        name = str(r.get("op") or r.get("type") or r.get("action") or "").strip().lower()
        name = {"rewrite": "replace", "update": "replace", "edit": "replace", "insert": "add",
                "append": "add", "delete": "remove", "drop": "remove"}.get(name, name)
        if name not in OPS:
            raise DeltaError(f"op #{i}: unknown op {r.get('op')!r} (use replace/add/remove)")
        cid = r.get("id", r.get("clause", r.get("clause_id")))
        text = r.get("text", r.get("new_text"))
        after = r.get("after")
        ops.append(Op(name, id=_clause_ref(cid), text=str(text).strip() if text is not None else None,
                      after=_clause_ref(after)))
    return ops


def _clause_ref(value: Any) -> str | None:
    """A clause id as a model may write it: ``"c2"``, ``"C2"``, ``2`` or ``"2"`` all mean ``c2``."""
    if value is None or isinstance(value, bool):
        return None if value is None else str(value)
    text = str(value).strip()
    if text.isdigit():
        return f"c{int(text)}"
    if text[:1] in ("c", "C") and text[1:].isdigit():
        return f"c{int(text[1:])}"
    return text


def _norm(text: str) -> str:
    return " ".join(text.casefold().split()).rstrip(".")


def validate(
    clauses: Sequence[Clause],
    ops: Sequence[Op],
    *,
    editable: set[str] | None = None,
    max_ops: int = 4,
    max_clause_chars: int = 600,
) -> None:
    """Raise :class:`DeltaError` if ``ops`` cannot be applied under the given limits.

    ``editable`` restricts ``replace``/``remove`` to the clauses that actually
    fired; ``add`` is always allowed because a fix may need a new step.
    """
    if not ops:
        raise DeltaError("empty delta")
    if len(ops) > max_ops:
        raise DeltaError(f"delta has {len(ops)} ops; at most {max_ops} allowed")
    ids = {c.id for c in clauses}
    removed: set[str] = set()
    touched: dict[str, int] = {}
    remaining = len(clauses)
    for i, op in enumerate(ops):
        if not isinstance(op, Op) or op.op not in OPS:
            raise DeltaError(f"op #{i}: unknown op {getattr(op, 'op', op)!r} (use replace/add/remove)")
        if op.op in ("replace", "remove"):
            if not op.id:
                raise DeltaError(f"op #{i} ({op.op}) needs a clause id")
            if op.id not in ids or op.id in removed:
                raise DeltaError(f"op #{i}: clause {op.id!r} does not exist")
            if op.id in touched:
                raise DeltaError(f"op #{i}: clause {op.id!r} is already changed by op #{touched[op.id]}; "
                                 "use one op per clause")
            touched[op.id] = i
            if editable is not None and op.id not in editable:
                raise DeltaError(
                    f"op #{i}: clause {op.id!r} is locked (only {sorted(editable)} fired in this episode)"
                )
        if op.op in ("replace", "add"):
            if not isinstance(op.text, str) or not op.text.strip():
                raise DeltaError(f"op #{i} ({op.op}) needs non-empty text")
            if len(op.text) > max_clause_chars:
                raise DeltaError(f"op #{i}: clause text longer than {max_clause_chars} chars")
        if op.op == "add" and op.after is not None and (op.after not in ids or op.after in removed):
            raise DeltaError(f"op #{i}: 'after' clause {op.after!r} does not exist")
        if op.op == "remove":
            removed.add(op.id)  # type: ignore[arg-type]
            remaining -= 1
        if op.op == "add":
            remaining += 1
    if remaining < 1:
        raise DeltaError("delta would leave the item with no clauses")
    by_id = {c.id: c.text for c in clauses}
    if all(op.op == "replace" and by_id.get(op.id or "") == op.text for op in ops):
        raise DeltaError("delta does not change anything")
    # A new or rewritten clause that repeats another adds nothing a model can follow better.
    new, _ = apply(clauses, ops, len(clauses) + len(ops) + 1_000_000)
    counts = Counter(_norm(c.text) for c in new)
    for c in new:
        if counts[_norm(c.text)] > 1 and by_id.get(c.id) != c.text:  # duplicates it created, not old ones
            raise DeltaError(f"delta would leave two identical clauses ({c.text!r})")


def apply(clauses: Sequence[Clause], ops: Sequence[Op], clause_seq: int) -> tuple[tuple[Clause, ...], int]:
    """Apply validated ops. Returns the new clauses and the advanced clause id counter."""
    out = list(clauses)
    seq = clause_seq
    for op in ops:
        if op.op == "replace":
            out = [Clause(c.id, op.text or "") if c.id == op.id else c for c in out]
        elif op.op == "remove":
            out = [c for c in out if c.id != op.id]
        elif op.op == "add":
            seq += 1
            new = Clause(f"c{seq}", op.text or "")
            if op.after is None:
                out.append(new)
            else:
                idx = next(i for i, c in enumerate(out) if c.id == op.after)
                out.insert(idx + 1, new)
    return tuple(out), seq


def diff(old: Sequence[Clause], new: Sequence[Clause]) -> list[dict[str, Any]]:
    """Human-readable clause diff between two versions (used by history views)."""
    o = {c.id: c.text for c in old}
    n = {c.id: c.text for c in new}
    out: list[dict[str, Any]] = []
    for c in old:
        if c.id not in n:
            out.append({"change": "removed", "id": c.id, "old": c.text})
        elif n[c.id] != c.text:
            out.append({"change": "replaced", "id": c.id, "old": c.text, "new": n[c.id]})
    for c in new:
        if c.id not in o:
            out.append({"change": "added", "id": c.id, "new": c.text})
    return out

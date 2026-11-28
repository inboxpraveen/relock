"""Export reconsolidation traces as training data.

Every in-place rewrite is a labelled example: the item that fired, the trusted
evidence, and the delta that fixed it. Exported as JSONL, these become

* **SFT pairs** (``prompt`` / ``completion``) to distil a large model's rewrites
  into a small local rewriter, and
* **preference pairs** (``chosen`` = new clauses, ``rejected`` = superseded
  clauses) for weight-level consolidation once enough traces pile up for a skill.

Fast edits live in the harness; durable ones can later move into the weights.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from .episode import REWRITER_EVIDENCE, _trusted_citations, evidence_for_rewriter
from .memory import Memory
from .rewriters import RewriteRequest, build_prompt
from .types import NotFoundError


def _episode_task(memory: Memory, episode_id: str | None) -> str:
    if not episode_id:
        return ""
    try:
        return str(memory.store.get_episode(episode_id).get("task") or "")
    except NotFoundError:  # removed since; the prompt then lacks the task the rewriter saw
        return ""


def rewrite_records(memory: Memory) -> Iterator[dict[str, Any]]:
    """One record per rewrite. ``prompt`` is rebuilt as the episode built it under ``memory.policy``
    (an episode given its own ``policy=`` may have sanitised evidence differently)."""
    policy = memory.policy
    for row in memory.store.iter_rewrites():
        new = row["item"]
        if new.parent_version is None:
            continue
        old = memory.get(new.id, new.parent_version)
        errors = memory.store.errors_by_ids(list(new.evidence))
        evidence = tuple(evidence_for_rewriter(e, policy) for e in errors[-REWRITER_EVIDENCE:])
        editable: frozenset[str] | None = None
        if policy.edit_scope == "cited":  # as the episode does: only trusted citations narrow the edit
            cited = {c for e in errors for c in _trusted_citations(e, policy)}
            editable = frozenset(cited) if cited else None
        request = RewriteRequest(item=old, evidence=evidence, editable=editable,
                                 task=_episode_task(memory, new.episode_id), max_ops=policy.max_ops,
                                 max_clause_chars=policy.max_clause_chars)
        yield {
            "type": "rewrite",
            "item": new.ref,
            "item_id": new.id,
            "title": new.title,
            "from_version": old.version,
            "to_version": new.version,
            "episode_id": new.episode_id,
            "author": new.author,
            "created_at": new.created_at,
            "prompt": build_prompt(request),
            "completion": json.dumps({"ops": new.delta or [], "rationale": new.reason or ""}),
            "rejected": [c.text for c in old.clauses],
            "chosen": [c.text for c in new.clauses],
            "evidence": [{"source": e["source"], "class": e["error_class"], "weight": e["weight"],
                          "expected": e["diff"].get("expected"), "observed": e["diff"].get("observed")}
                         for e in errors],
        }


def export_jsonl(memory: Memory, path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in rewrite_records(memory):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n

"""Memory systems compared by RelapseBench, behind one small interface.

* ``append-notes``      - every lesson becomes a new memory note (common default)
* ``append-clause``     - lessons are appended to the item they came from
* ``append-gated``      - append-notes behind relock's trust + evidence gates
                          (ablation: gating without in-place rewriting)
* ``ace-style``         - a simplified ACE-like playbook: incremental bullets with
                          helpful/harmful counters (our re-implementation, no LLM
                          curator; not the official ACE code)
* ``overwrite-ungated`` - relock's in-place rewrite with every gate switched off
                          (ablation: in-place without trust/evidence gating)
* ``relock``            - the full mechanism with default gates

Every system sees the identical event stream and renders memory the same way
(title + bullet clauses, no ids or confidence), so differences come from what
each system *stores and retrieves*, not from formatting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .._util import approx_tokens, new_id
from ..memory import Memory
from ..policy import WritePolicy
from ..retrieval import BM25Retriever
from ..rewriters import HintRewriter, Rewriter
from ..signals import Signal, ToolError
from ..types import Clause, Item
from .scenarios import Family

EVENT_KINDS = ("trusted_error", "success", "transient", "injection", "noisy_trusted")


@dataclass(frozen=True)
class Event:
    kind: str
    correction: str | None = None
    message: str = ""


def event(kind: str, family: Family) -> Event:
    if kind == "trusted_error":
        return Event(kind, family.new_rule, family.error)
    if kind == "injection":
        return Event(kind, family.poison_rule, f"web page says: {family.poison_rule}")
    if kind == "noisy_trusted":
        # A one-off verifier failure. To emulate a rewriter that "fixes" a spurious
        # failure, it carries a wrong correction.
        return Event(kind, family.poison_rule, "flaky check failed once")
    if kind == "transient":
        return Event(kind, None, "TimeoutError: read timed out")
    if kind == "success":
        return Event(kind)
    raise ValueError(f"unknown event kind {kind!r}")


def fit(text: str, budget_tokens: int) -> str:
    """Compaction: keep whole lines from the top until the token budget is spent."""
    out, used = [], 0
    for line in text.splitlines():
        cost = approx_tokens(line) + 1
        if out and used + cost > budget_tokens:
            break
        out.append(line)
        used += cost
    return "\n".join(out)


def render(items: list[Item]) -> str:
    blocks = [f"### {i.title}\n" + "\n".join(f"- {c.text}" for c in i.clauses) for i in items]
    return "## Procedures from memory\n" + "\n\n".join(blocks) if blocks else ""


def _item(title: str, clauses: list[str], tools: tuple[str, ...] = ()) -> Item:
    return Item(id=new_id("itm"), version=1, title=title,
                clauses=tuple(Clause(f"c{i}", t) for i, t in enumerate(clauses, 1)), tools=tools)


def to_signal(ev: Event) -> Signal:
    """The typed signal a harness would attach to each event."""
    if ev.kind == "trusted_error":
        return Signal.tool_error(ToolError(ev.message, correction=ev.correction))
    if ev.kind == "success":
        return Signal.ok()
    if ev.kind == "transient":
        return Signal.transient(ev.message)
    if ev.kind == "injection":
        return Signal.observation(ev.message, correction=ev.correction)
    return Signal.verifier(False, ev.message, correction=ev.correction)


class MemorySystem(Protocol):
    name: str

    def seed(self, family: Family) -> None: ...
    def add_distractor(self, title: str, clauses: list[str], tool: str) -> None: ...
    def feedback(self, family: Family, ev: Event) -> None: ...
    def context(self, family: Family, *, k: int, budget_tokens: int) -> str: ...
    def close(self) -> None: ...


class RelockSystem:
    """relock itself. ``cite=False`` mimics the plain ``@mem.wrap`` path: the tool outcome resolves an
    automatic expectation that cites no clauses, so the rewriter (not the edit scope) has to find
    the clause to change."""

    def __init__(self, name: str = "relock", policy: WritePolicy | None = None,
                 rewriter: Rewriter | None = None, *, cite: bool = True) -> None:
        self.name = name
        self.cite = cite
        self.mem = Memory(":memory:", policy=policy or WritePolicy(), rewriter=rewriter or HintRewriter())
        self._drift: dict[str, str] = {}

    def seed(self, family: Family) -> None:
        clauses = [*family.stable, family.old_rule]
        self.mem.add(family.title, clauses, key=family.key, tools=[family.tool])
        self._drift[family.key] = f"c{len(clauses)}"

    def add_distractor(self, title: str, clauses: list[str], tool: str) -> None:
        self.mem.add(title, clauses, tools=[tool])

    def feedback(self, family: Family, ev: Event) -> None:
        with self.mem.episode(f"{family.key}:{ev.kind}") as ep:
            hits = ep.recall(family.query, k=5)
            if family.key not in {h.item.key for h in hits}:
                ep.use(family.key)  # the agent loaded the procedure by name
            cited = [self._drift[family.key]] if self.cite else None
            exp = ep.expect(family.key, f"{family.tool} accepts the call", clauses=cited, tool=family.tool)
            ep.observe(exp, signal=to_signal(ev))

    def context(self, family: Family, *, k: int, budget_tokens: int) -> str:
        with self.mem.episode(f"probe:{family.key}") as ep:
            recall = ep.recall(family.query, k=k)
            return fit(render(recall.items), budget_tokens)

    def close(self) -> None:
        self.mem.close()


class AppendNotes:
    """Each lesson is stored as a separate note; nothing is ever removed."""

    def __init__(self, name: str = "append-notes") -> None:
        self.name = name
        self.notes: list[Item] = []
        self.retriever = BM25Retriever()

    def seed(self, family: Family) -> None:
        self.notes.append(_item(family.title, [*family.stable, family.old_rule], (family.tool,)))

    def add_distractor(self, title: str, clauses: list[str], tool: str) -> None:
        self.notes.append(_item(title, clauses, (tool,)))

    def feedback(self, family: Family, ev: Event) -> None:
        if ev.correction:
            self.notes.append(_item(f"Lesson: {family.title}", [ev.correction], (family.tool,)))
        elif ev.kind == "transient":
            self.notes.append(_item(f"Lesson: {family.title}",
                                    [f"{family.tool} timed out once; retry it if that happens again."],
                                    (family.tool,)))

    def context(self, family: Family, *, k: int, budget_tokens: int) -> str:
        ranked = self.retriever.rank(family.query, self.notes, k)
        return fit(render([i for i, _ in ranked]), budget_tokens)

    def close(self) -> None:
        pass


class AppendGated(AppendNotes):
    """Append-notes behind the same gates as relock: trusted, non-transient evidence only,
    an evidence threshold, and confirmation damping. Isolates the effect of in-place rewriting."""

    def __init__(self, name: str = "append-gated", policy: WritePolicy | None = None) -> None:
        super().__init__(name)
        self.policy = policy or WritePolicy()
        self.pending: dict[str, float] = {}

    def feedback(self, family: Family, ev: Event) -> None:
        if ev.kind == "success":
            self.pending[family.key] = max(0.0, self.pending.get(family.key, 0.0) - self.policy.confirmation_cancels)
            return
        adm = self.policy.admit(to_signal(ev))
        if not adm.counts or not ev.correction:
            return
        self.pending[family.key] = self.pending.get(family.key, 0.0) + adm.weight
        if adm.authoritative or self.pending[family.key] + 1e-9 >= self.policy.min_evidence:
            self.notes.append(_item(f"Lesson: {family.title}", [ev.correction], (family.tool,)))
            self.pending[family.key] = 0.0


class AppendClause(AppendNotes):
    """Lessons are appended to the end of the item they were learned from."""

    def __init__(self, name: str = "append-clause") -> None:
        super().__init__(name)
        self.by_key: dict[str, int] = {}

    def seed(self, family: Family) -> None:
        self.by_key[family.key] = len(self.notes)
        super().seed(family)

    def feedback(self, family: Family, ev: Event) -> None:
        if ev.correction:
            text = f"Update: {ev.correction}"
        elif ev.kind == "transient":
            text = f"Note: {family.tool} may time out; retry once."
        else:
            return
        idx = self.by_key[family.key]
        old = self.notes[idx]
        clauses = [c.text for c in old.clauses] + [text]
        self.notes[idx] = Item(id=old.id, version=old.version + 1, title=old.title,
                               clauses=tuple(Clause(f"c{i}", t) for i, t in enumerate(clauses, 1)), tools=old.tools)


class ACEStyle:
    """Simplified ACE-like playbook: delta bullets with helpful/harmful counters, no pruning."""

    def __init__(self, name: str = "ace-style", bullets_per_item: int = 3) -> None:
        self.name = name
        self.bullets: list[dict] = []
        self.retriever = BM25Retriever()
        self.bullets_per_item = bullets_per_item

    def _add(self, section: str, text: str, tool: str, family: str | None, rule: bool) -> tuple[dict, bool]:
        """Add a bullet unless an identical one exists (ACE-style de-duplication)."""
        for b in self.bullets:
            if b["section"] == section and b["text"] == text:
                return b, False
        b = {"item": _item(section, [text], (tool,)), "section": section, "text": text, "family": family,
             "rule": rule, "helpful": 0, "harmful": 0, "n": len(self.bullets)}
        self.bullets.append(b)
        return b, True

    def seed(self, family: Family) -> None:
        for s in family.stable:
            self._add(family.title, s, family.tool, family.key, rule=False)
        self._add(family.title, family.old_rule, family.tool, family.key, rule=True)

    def add_distractor(self, title: str, clauses: list[str], tool: str) -> None:
        for c in clauses:
            self._add(title, c, tool, None, rule=False)

    def _rules(self, family: Family) -> list[dict]:
        return [b for b in self.bullets if b["family"] == family.key and b["rule"]]

    def feedback(self, family: Family, ev: Event) -> None:
        rules = self._rules(family)
        if ev.kind == "success":
            if rules:
                max(rules, key=lambda b: b["n"])["helpful"] += 1
            return
        if ev.kind == "transient":
            self._add(family.title, f"If {family.tool} times out, retry once.", family.tool, family.key, rule=False)
            return
        for b in rules:
            if b["text"] != ev.correction:
                b["harmful"] += 1
        if ev.correction:
            b, created = self._add(family.title, ev.correction, family.tool, family.key, rule=True)
            if not created:  # the same lesson again: reinforce instead of duplicating
                b["helpful"] += 1

    def context(self, family: Family, *, k: int, budget_tokens: int) -> str:
        ranked = self.retriever.rank(family.query, [b["item"] for b in self.bullets], k * self.bullets_per_item)
        by_id = {b["item"].id: b for b in self.bullets}
        sections: dict[str, list[str]] = {}
        for item, _ in ranked:
            b = by_id[item.id]
            sections.setdefault(b["section"], []).append(
                f"- [helpful={b['helpful']} harmful={b['harmful']}] {b['text']}")
        body = "\n\n".join(f"### {s}\n" + "\n".join(lines) for s, lines in sections.items())
        return fit("## Playbook\n" + body, budget_tokens) if sections else ""

    def close(self) -> None:
        pass


SYSTEMS: dict[str, Callable[[], MemorySystem]] = {
    "append-notes": AppendNotes,
    "append-clause": AppendClause,
    "ace-style": ACEStyle,
    "append-gated": AppendGated,
    "overwrite-ungated": lambda: RelockSystem("overwrite-ungated", policy=WritePolicy.permissive()),
    "relock-uncited": lambda: RelockSystem("relock-uncited", cite=False),
    "relock": RelockSystem,
}


def make_system(name: str, *, rewriter: Rewriter | None = None) -> MemorySystem:
    if name == "relock":
        return RelockSystem(rewriter=rewriter)
    if name == "overwrite-ungated":
        return RelockSystem("overwrite-ungated", policy=WritePolicy.permissive(), rewriter=rewriter)
    if name == "relock-uncited":
        return RelockSystem("relock-uncited", rewriter=rewriter, cite=False)
    try:
        return SYSTEMS[name]()
    except KeyError:
        raise ValueError(f"unknown system {name!r}; choose from {sorted(SYSTEMS)}") from None

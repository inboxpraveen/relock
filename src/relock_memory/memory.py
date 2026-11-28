"""The :class:`Memory` facade: authoring, episodes, tool instrumentation, audit."""

from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import re
import threading
import weakref
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from typing import Any, TypeVar

from ._util import new_id, now
from .delta import Op, apply, parse_ops, validate
from .delta import diff as clause_diff
from .episode import Episode
from .policy import WritePolicy
from .retrieval import EmbedFn, Retriever, default_retriever
from .rewriters import HintRewriter, Rewriter
from .stores import SQLStore, open_store
from .types import Clause, ConflictError, Hit, Item, NotFoundError

F = TypeVar("F", bound=Callable[..., Any])

log = logging.getLogger("relock_memory")

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def _clauses(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        lines = [ln for ln in value.splitlines() if ln.strip()]
        parts = [_BULLET.sub("", ln).strip() for ln in lines] if len(lines) > 1 else [value.strip()]
    else:
        parts = [str(v).strip() for v in value]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("an item needs at least one clause")
    return parts


class Memory:
    """Prediction-error-gated, in-place agent memory.

    >>> mem = Memory(":memory:")
    >>> item = mem.add("Fetch orders", ["Call get_orders with dates as MM/DD/YYYY."],
    ...                key="orders.fetch", tools=["get_orders"])
    >>> with mem.episode("orders for March 5") as ep:
    ...     hits = ep.recall("fetch orders by date")
    ...     # ... the agent acts; wrapped tools report their real outcomes ...

    Args:
        store: a path, ``":memory:"``, a ``postgresql://`` URL, or a store instance.
        policy: the :class:`~relock_memory.policy.WritePolicy` (gates and thresholds).
        rewriter: turns trusted prediction errors into clause deltas.
            Defaults to the deterministic :class:`~relock_memory.rewriters.HintRewriter`;
            use :class:`~relock_memory.rewriters.LLMRewriter` with any model.
        retriever: custom :class:`~relock_memory.retrieval.Retriever`.
        embed: optional ``embed(list[str]) -> list[vector]`` for hybrid retrieval.
        on_event: callback receiving every ledger event (for logging/tracing).
    """

    def __init__(
        self,
        store: str | SQLStore | None = None,
        *,
        policy: WritePolicy | None = None,
        rewriter: Rewriter | None = None,
        retriever: Retriever | None = None,
        embed: EmbedFn | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.store = open_store(store)
        self.policy = policy or WritePolicy()
        self.rewriter: Rewriter = rewriter or HintRewriter()
        self.retriever: Retriever = retriever or default_retriever(embed)
        self.on_event = on_event
        self._ctx: contextvars.ContextVar[Episode | None] = contextvars.ContextVar(
            f"relock_episode_{id(self)}", default=None)
        # Weak references: an episode its caller dropped without closing can no longer write,
        # so it must not linger here and keep counting as open.
        self._open: weakref.WeakValueDictionary[str, Episode] = weakref.WeakValueDictionary()
        self._open_lock = threading.Lock()
        self._unbound_warned: set[str] = set()
        self._cache: tuple[int, list[Item]] | None = None
        self._cache_lock = threading.Lock()

    def __repr__(self) -> str:
        return f"Memory({self.store!r}, rewriter={getattr(self.rewriter, 'name', type(self.rewriter).__name__)})"

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        with self._open_lock:
            eps = list(self._open.values())
        for ep in eps:
            ep.close()
        self.store.close()

    # ------------------------------------------------------------------ internals
    def _emit(self, type_: str, **kw: Any) -> None:
        ev = self.store.add_event(type_, **kw)
        if self.on_event is not None:
            try:
                self.on_event(ev)
            except Exception:  # a logging hook must not break memory
                pass

    def _invalidate(self) -> None:
        with self._cache_lock:
            self._cache = None

    def _active_items(self) -> list[Item]:
        rev = self.store.revision()
        with self._cache_lock:
            if self._cache is not None and self._cache[0] == rev:
                return self._cache[1]
        items = self.store.active()
        with self._cache_lock:
            self._cache = (rev, items)
        return items

    def _register(self, ep: Episode) -> None:
        with self._open_lock:
            self._open[ep.id] = ep
            n = len(self._open)
        if n >= 1000 and n % 1000 == 0:
            log.warning("relock: %d episodes are open in this process. Close each one when its task ends "
                        "(use `with mem.episode(...)`); open episodes keep their recalled items editable.", n)

    def _unregister(self, ep: Episode) -> None:
        with self._open_lock:
            self._open.pop(ep.id, None)

    # ------------------------------------------------------------------ authoring
    def add(
        self,
        title: str,
        clauses: str | Iterable[str],
        *,
        key: str | None = None,
        kind: str = "procedure",
        tags: Sequence[str] = (),
        tools: Sequence[str] = (),
        meta: dict[str, Any] | None = None,
        exist_ok: bool = False,
        author: str = "author",
    ) -> Item:
        """Create a memory item. ``clauses`` is a list, or a string (one clause per bullet/line).

        With ``exist_ok=True`` an existing item with the same ``key`` is returned
        untouched, so seeding code can run on every start without clobbering
        what the agent has learned since.
        """
        if key is not None and exist_ok:
            try:
                return self.get(key)
            except NotFoundError:
                pass
        texts = _clauses(clauses)
        a, b = self.policy.new_version_prior
        item = Item(
            id=new_id("itm"),
            version=1,
            title=title.strip(),
            clauses=tuple(Clause(f"c{i}", t) for i, t in enumerate(texts, 1)),
            key=key,
            kind=kind,
            tags=tuple(tags),
            tools=tuple(tools),
            meta=dict(meta or {}),
            alpha=a,
            beta=b,
            clause_seq=len(texts),
            change_kind="create",
            author=author,
            created_at=now(),
        )
        try:
            self.store.create_item(item)
        except ConflictError:
            if exist_ok and key is not None:
                return self.get(key)
            raise
        self._invalidate()
        self._emit("create", item_id=item.id, version=1, data={"key": key, "title": item.title,
                                                                "clauses": [c.to_dict() for c in item.clauses]})
        return item

    teach = add

    def get(self, ref: str, version: int | None = None) -> Item:
        return self.store.get(ref, version)

    def history(self, ref: str) -> list[Item]:
        """Every version of an item, oldest first (superseded versions included)."""
        return self.store.history(ref)

    def items(self, *, include_archived: bool = False) -> list[Item]:
        return self.store.active(include_archived=include_archived)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """Read-only retrieval. Unlike :meth:`Episode.recall` it does not unlock anything."""
        ranked = self.retriever.rank(query, self._active_items(), k)
        return [Hit(item, float(score), rank) for rank, (item, score) in enumerate(ranked)]

    def archive(self, ref: str, archived: bool = True) -> None:
        self.store.set_archived(ref, archived)
        self._invalidate()
        self._emit("archive" if archived else "unarchive", item_id=self.store.resolve(ref))

    # ------------------------------------------------------------------ episodes
    def episode(self, task: str = "", **meta: Any) -> Episode:
        """Open a reconsolidation window. Use it as a context manager."""
        return Episode(self, task, meta)

    def current_episode(self) -> Episode | None:
        """The episode active in the current thread or asyncio task, if any.

        An episode is active inside ``with mem.episode(...)``, inside ``with ep.activate():``,
        during ``ep.run(fn, ...)``, and after a :class:`~relock_memory.integrations.toolkit.Toolkit`
        call in the same context. Asyncio tasks inherit it. Plain worker threads do not: submit
        work with ``ep.run`` or wrap it in ``ep.activate()``. There is deliberately no
        process-wide fallback, so a tool call is never attributed to another agent's episode.
        """
        ep = self._ctx.get()
        return ep if ep is not None and not ep.closed else None

    def _note_unbound(self, tool: str) -> None:
        """Warn (once per tool) when a wrapped tool runs outside any episode while some are open."""
        with self._open_lock:
            n = sum(1 for e in self._open.values() if not e.closed)
            if not n or tool in self._unbound_warned:
                return
            self._unbound_warned.add(tool)
        log.warning("relock: tool %r ran outside any episode while %d episode(s) are open in this process; "
                    "its outcome was not recorded. Call it inside `with ep:` or `with ep.activate():`, "
                    "or submit it to worker threads with `ep.run(fn, ...)`. (Logged once per tool.)", tool, n)

    # ------------------------------------------------------------------ admin edits
    def _commit_admin(self, current: Item, clauses: tuple[Clause, ...], seq: int, *, kind: str,
                      reason: str, actor: str, delta: list[dict[str, Any]] | None, extra: dict[str, Any]) -> Item:
        a, b = self.policy.new_version_prior
        new = replace(current, version=self.store.next_version_number(current.id), clauses=clauses,
                      clause_seq=seq, status="active", alpha=a, beta=b, confirmations=0, failures=0,
                      parent_version=current.version, superseded_by=None, change_kind=kind, delta=delta,
                      reason=reason or None, episode_id=None, evidence=(), author=actor, created_at=now())
        self.store.commit_version(new, expected_current=current.version, stale_reason=f"{kind} to v{new.version}")
        self._invalidate()
        self._emit(kind, item_id=current.id, version=new.version, data={
            "from_version": current.version, "to_version": new.version, "reason": reason, "actor": actor,
            "changes": clause_diff(current.clauses, new.clauses), **extra})
        return new

    def edit(self, ref: str, ops: Sequence[Op | dict[str, Any]], *, reason: str = "", actor: str = "admin") -> Item:
        """A deliberate human/admin edit. Bypasses evidence gates; still versioned and audited."""
        current = self.get(ref)
        parsed = parse_ops(ops)
        validate(current.clauses, parsed, max_ops=max(self.policy.max_ops, len(parsed)),
                 max_clause_chars=self.policy.max_clause_chars)
        clauses, seq = apply(current.clauses, parsed, current.clause_seq)
        return self._commit_admin(current, clauses, seq, kind="manual", reason=reason, actor=actor,
                                  delta=[o.to_dict() for o in parsed], extra={})

    def rollback(self, ref: str, to_version: int, *, reason: str = "", actor: str = "admin") -> Item:
        """Restore an earlier version's content as a *new* version. History is never destroyed."""
        current = self.get(ref)
        target = self.get(current.id, to_version)
        if to_version == current.version:
            raise ValueError(f"{current.ref} is already at v{to_version}")
        seq = max(current.clause_seq, target.clause_seq)
        return self._commit_admin(current, target.clauses, seq, kind="rollback",
                                  reason=reason or f"rollback to v{to_version}", actor=actor, delta=None,
                                  extra={"restored_version": to_version})

    # ------------------------------------------------------------------ audit
    def events(self, ref: str | None = None, *, episode_id: str | None = None, type: str | None = None,
               limit: int = 100) -> list[dict[str, Any]]:
        item_id = self.store.resolve(ref) if ref else None
        return self.store.events(item_id=item_id, episode_id=episode_id, type_=type, limit=limit)

    def pending(self, ref: str) -> dict[str, Any]:
        """Explain an item's write state: pending evidence vs. the threshold to rewrite it."""
        item = self.get(ref)
        errs = self.store.errors(item.id, item.version, status="pending")
        blocked = self.store.errors(item.id, item.version, status="blocked")
        return {
            "item": item.ref,
            "version": item.version,
            "confidence": round(item.confidence, 4),
            "evidence": round(sum(e["weight"] for e in errs), 4),
            "required": round(self.policy.required_evidence(item.confirmations), 4),
            "pending_errors": [{"id": e["id"], "weight": e["weight"], "source": e["source"],
                                "class": e["error_class"], "observed": e["diff"].get("observed")} for e in errs],
            "blocked_errors": [{"id": e["id"], "reason": e["reason"], "source": e["source"]} for e in blocked],
        }

    def stats(self) -> dict[str, int]:
        return self.store.stats()

    # ------------------------------------------------------------------ tool instrumentation
    def wrap(self, fn: F | None = None, *, name: str | None = None, items: Sequence[str] | None = None) -> Any:
        """Instrument a tool so its *real* outcome resolves expectations in the active episode.

        Use as ``mem.wrap(fn)``, ``@mem.wrap`` or ``@mem.wrap(name="search")``.
        The wrapper keeps the original signature and docstring (so agent
        frameworks build the same tool schema) and re-raises exceptions
        unchanged. Outside an episode it is a no-op passthrough.

        Attribution: the call resolves explicit expectations bound to this tool
        name, otherwise it auto-creates a ``succeeds()`` expectation for every
        recalled item that lists the tool in ``tools=[...]`` (or the ``items``
        given here).
        """
        if fn is None:
            return lambda f: self.wrap(f, name=name, items=items)
        tool = name or getattr(fn, "__name__", "tool")
        refs = list(items) if items else None
        memory = self

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                ep = memory.current_episode()
                if ep is None:
                    memory._note_unbound(tool)
                exps = ep._bind_tool(tool, refs) if ep is not None else []
                try:
                    result = await fn(*args, **kwargs)
                except Exception as e:
                    if ep is not None:
                        ep._observe_quietly(exps, error=e)
                    raise
                if ep is not None:
                    ep._observe_quietly(exps, result=result)
                return result

            awrapper.__relock_tool__ = tool  # type: ignore[attr-defined]
            return awrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ep = memory.current_episode()
            if ep is None:
                memory._note_unbound(tool)
            exps = ep._bind_tool(tool, refs) if ep is not None else []
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                if ep is not None:
                    ep._observe_quietly(exps, error=e)
                raise
            if ep is not None:
                ep._observe_quietly(exps, result=result)
            return result

        wrapper.__relock_tool__ = tool  # type: ignore[attr-defined]
        return wrapper

    tool = wrap

    def wrap_tools(self, tools: Iterable[Callable[..., Any]]) -> list[Callable[..., Any]]:
        return [self.wrap(t) for t in tools]

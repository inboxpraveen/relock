"""The :class:`Memory` facade: authoring, episodes, tool instrumentation, audit."""

from __future__ import annotations

import contextvars
import datetime as _dt
import functools
import inspect
import logging
import re
import threading
import weakref
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from typing import Any, TypeVar

from ._util import new_id, now, parse_age, truncate
from .delta import Op, apply, parse_ops, validate
from .delta import diff as clause_diff
from .episode import Episode
from .policy import WritePolicy
from .retrieval import EmbedFn, Retriever, default_retriever, rank_items
from .rewriters import HintRewriter, Rewriter
from .signals import is_control_flow
from .stores import SQLStore, open_store
from .types import Clause, ConflictError, Hit, Item, NotFoundError

F = TypeVar("F", bound=Callable[..., Any])

log = logging.getLogger("relock_memory")

EVENT_LOG_LEVELS: dict[str, int] = {
    # memory content changed
    "create": logging.INFO, "rewrite": logging.INFO, "manual": logging.INFO, "rollback": logging.INFO,
    "archive": logging.INFO, "unarchive": logging.INFO, "prune": logging.INFO,
    # something went wrong, or a gate held a write back
    "prediction_error": logging.INFO, "blocked": logging.INFO, "declined": logging.INFO,
    "conflict": logging.INFO, "rewrite_deferred": logging.INFO, "invalid_edit": logging.WARNING,
}
"""Level at which each ledger event is logged on the ``relock_memory`` logger; others log at DEBUG.

Every event is logged once it is stored. Lines name the event, episode, item and version, plus a
short reason for failures. The full event (with payloads such as clause text and tool errors) is
attached as ``record.relock_event`` only when DEBUG is enabled, since payloads can carry user data.
"""

_REASON_KEYS = ("reason", "error", "observed")

KEEP_EVENTS = ("create", "rewrite", "manual", "rollback", "archive", "unarchive", "prune")
"""Event types :meth:`Memory.prune` keeps by default: every change to memory content."""

_CHANGE_WINDOW_S = 120.0
"""On a cache refresh, items updated this long before the previous refresh are re-read too.

Writers from relock < 0.3 don't stamp the revision on the item they change, so during a rolling
upgrade their changes are found by time instead. The margin covers clock skew between hosts and
slow transactions; re-reading an unchanged item is harmless."""

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def _names(value: str | Iterable[str] | None, what: str) -> tuple[str, ...]:
    """Tool or tag names. A bare string is one name, never split into characters."""
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    names = tuple(dict.fromkeys(str(v).strip() for v in value))
    if any(not n for n in names):
        raise ValueError(f"{what} must be non-empty names, got {list(value)!r}")
    return names


def _is_wrapper(wrappers: weakref.WeakSet[Callable[..., Any]], fn: Any) -> bool:
    try:
        return fn in wrappers
    except TypeError:  # unhashable or not weak-referenceable: never one of ours
        return False


def _tool_name(fn: Callable[..., Any]) -> str:
    """The name a wrapped tool is recorded under: its ``__name__``, a partial's function name,
    or a callable object's class name."""
    for obj in (fn, getattr(fn, "func", None)):  # functools.partial keeps the function in .func
        name = getattr(obj, "__name__", None)
        if isinstance(name, str) and name and name != "<lambda>":
            return name
    name = "<lambda>" if getattr(fn, "__name__", None) == "<lambda>" else type(fn).__name__
    log.warning("relock: %r has no name of its own; its outcomes are recorded as tool %r, which an item's "
                "tools=[...] must name. Pass mem.wrap(fn, name=...) to choose it.", fn, name)
    return name


def _clauses(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        lines = [ln for ln in value.splitlines() if ln.strip()]
        parts = [_BULLET.sub("", ln).strip() for ln in lines]
    else:
        # One clause per entry. A line break inside one would read as a separate, unnumbered
        # line once the item is rendered into a prompt.
        parts = [" ".join(str(v).split()) for v in value]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError("an item needs at least one clause")
    return parts


def _log_event(ev: dict[str, Any]) -> None:
    level = EVENT_LOG_LEVELS.get(ev["type"], logging.DEBUG)
    if not log.isEnabledFor(level):
        return
    parts = [f"relock {ev['type']}"]
    if ev.get("item_id"):
        parts.append(f"item={ev['item_id']}" + (f" v{ev['version']}" if ev.get("version") is not None else ""))
    if ev.get("episode_id"):
        parts.append(f"episode={ev['episode_id']}")
    data = ev.get("data") or {}
    if level > logging.DEBUG:
        reason = next((data[k] for k in _REASON_KEYS if data.get(k)), None)
        if reason:
            parts.append(f"reason={truncate(str(reason), 300)!r}")
        if ev["type"] == "rewrite":
            parts.append(f"v{data.get('from_version')}->v{data.get('to_version')} by {data.get('author')}")
    full = log.isEnabledFor(logging.DEBUG)
    record = ev if full else {k: ev.get(k) for k in ("id", "at", "type", "episode_id", "item_id", "version")}
    log.log(level, " ".join(parts), extra={"relock_event": record})


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
        if rewriter is not None and not callable(getattr(rewriter, "propose", None)):
            hint = " (wrap a plain function in relock.CallableRewriter)" if callable(rewriter) else ""
            raise TypeError(f"rewriter must have propose(request), not {type(rewriter).__name__}{hint}")
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
        self._bind_warned: set[tuple[str, str]] = set()
        self._wrappers: weakref.WeakSet[Callable[..., Any]] = weakref.WeakSet()  # functions wrap() returned
        self._hook_failures = 0
        self._snapshot: tuple[int, tuple[Item, ...]] | None = None  # (revision, active items by id)
        self._cache_items: dict[str, Item] = {}
        self._cache_since = ""
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
        _log_event(ev)
        if self.on_event is not None:
            try:
                self.on_event(ev)
            except Exception:  # a logging hook must not break memory, but its owner must hear of it
                self._hook_failures += 1
                n = self._hook_failures
                if n == 1 or n % 1000 == 0:
                    log.warning("relock on_event hook raised on a %r event (%d failure(s) so far); the event is "
                                "still in the ledger. Further failures are logged every 1000th.", type_, n,
                                exc_info=True)

    def _active_items(self) -> tuple[Item, ...]:
        """Active, unarchived items, cached until any process changes memory content.

        The store's revision counter moves on content changes only (create, rewrite, edit,
        rollback, archive), so confirmations and failures never force a reload.
        """
        rev = self.store.revision()
        snap = self._snapshot
        if snap is not None and snap[0] == rev:
            return snap[1]
        with self._cache_lock:
            snap = self._snapshot
            rev = self.store.revision()
            if snap is not None and snap[0] == rev:
                return snap[1]
            started = _dt.datetime.now(_dt.timezone.utc)
            if snap is None:
                self._cache_items = {i.id: i for i in self.store.active()}
            else:
                # Only items changed since the cached revision (or recently, for older writers).
                for item in self.store.changed_since(snap[0], self._cache_since):
                    if item.archived or item.status != "active":
                        self._cache_items.pop(item.id, None)
                    else:
                        self._cache_items[item.id] = item
            items = tuple(sorted(self._cache_items.values(), key=lambda i: i.id))
            self._snapshot = (rev, items)
            self._cache_since = (started - _dt.timedelta(seconds=_CHANGE_WINDOW_S)).isoformat(timespec="microseconds")
            return items

    def _ranked_hits(self, query: str, k: int, *, kinds: Sequence[str] | None = None,
                     tags: Sequence[str] | None = None) -> list[Hit]:
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, not {type(query).__name__}")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a whole number of at least 1, not {k!r}")
        items = self._active_items()
        candidates = None
        if kinds or tags:
            wanted_kinds, wanted_tags = set(_names(kinds, "kinds")), set(_names(tags, "tags"))
            candidates = {i.id for i in items
                          if (not wanted_kinds or i.kind in wanted_kinds) and (not wanted_tags or wanted_tags & set(i.tags))}
        ranked = rank_items(self.retriever, query, items, k, candidates)
        if not ranked:
            return []
        # The cached items may carry old confidence counters; refresh just the hits.
        fresh = self.store.counters([(item.id, item.version) for item, _ in ranked])
        return [Hit(replace(item, **fresh[(item.id, item.version)]) if (item.id, item.version) in fresh else item,
                    float(score), rank) for rank, (item, score) in enumerate(ranked)]

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
                return self._existing(key)
            except NotFoundError:
                pass
        elif exist_ok:
            log.warning("relock: add(%r, exist_ok=True) without a key creates a new item on every call; "
                        "pass key=... so a restart finds the existing item", title)
        texts = _clauses(clauses)
        tags, tools = _names(tags, "tags"), _names(tools, "tools")
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
                return self._existing(key)
            raise
        self._emit("create", item_id=item.id, version=1, data={"key": key, "title": item.title,
                                                                "clauses": [c.to_dict() for c in item.clauses]})
        return item

    def _existing(self, key: str) -> Item:
        item = self.get(key)
        if item.archived:
            log.warning("relock: add(key=%r, exist_ok=True) found an archived item, which recall never "
                        "returns; run `relock archive --undo %s` or mem.archive(%r, archived=False) to use "
                        "it again", key, key, key)
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
        return self._ranked_hits(query, k)

    def archive(self, ref: str, archived: bool = True) -> None:
        self.store.set_archived(ref, archived)
        self._emit("archive" if archived else "unarchive", item_id=self.store.resolve(ref))

    # ------------------------------------------------------------------ episodes
    def episode(self, task: str = "", *, policy: WritePolicy | None = None, **meta: Any) -> Episode:
        """Open a reconsolidation window. Use it as a context manager.

        ``policy`` overrides :attr:`policy` for this episode only, e.g. to trust one agent's
        reports more than another's. Keyword arguments other than ``policy`` are stored as metadata.
        """
        return Episode(self, task, meta, policy=policy)

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

    def _note_bind_failure(self, tool: str, error: BaseException) -> None:
        """Log (once per tool and error type) why a wrapped call's outcome could not be recorded."""
        key = (tool, type(error).__name__)
        with self._open_lock:
            if key in self._bind_warned:
                return
            self._bind_warned.add(key)
        log.warning("relock: could not record the outcome of tool %r (%s: %s); the tool ran normally. "
                    "(Logged once per tool and error type.)", tool, type(error).__name__, error)

    # ------------------------------------------------------------------ admin edits
    def _commit_admin(self, current: Item, clauses: tuple[Clause, ...], seq: int, *, kind: str,
                      reason: str, actor: str, delta: list[dict[str, Any]] | None, extra: dict[str, Any]) -> Item:
        a, b = self.policy.new_version_prior
        new = replace(current, version=self.store.next_version_number(current.id), clauses=clauses,
                      clause_seq=seq, status="active", alpha=a, beta=b, confirmations=0, failures=0,
                      parent_version=current.version, superseded_by=None, change_kind=kind, delta=delta,
                      reason=reason or None, episode_id=None, evidence=(), author=actor, created_at=now())
        self.store.commit_version(new, expected_current=current.version, stale_reason=f"{kind} to v{new.version}")
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

    def prune(self, older_than: _dt.timedelta | str, *, dry_run: bool = False,
              keep_events: Iterable[str] = KEEP_EVENTS) -> dict[str, int]:
        """Delete audit detail of episodes that ended more than ``older_than`` ago (e.g. ``"30d"``).

        Bounds database growth without losing the audit trail of memory content:

        * Items and versions are never deleted, so every version, its delta, reason, author and
          evidence ids stay; so do events of the ``keep_events`` types (content changes).
        * An episode goes (with its retrievals, expectations, errors and events) only if no
          version was written in it and none of its errors is or may become evidence.
        * Older episodes that must stay lose only per-call detail nothing cites: confirmed
          expectations and their ``expect`` / ``confirm`` events.
        * An episode still open but idle since the cutoff (its process died) is marked
          ``abandoned`` first. Episodes open in this process are never touched; other processes
          can't be checked, so ``older_than`` must exceed the longest an episode stays open and
          idle in any replica (a pruned live episode fails its next write with StoreError).

        Deletes run in small batches, one transaction each. Returns counts per table; with
        ``dry_run=True`` nothing changes and the counts say what would go. SQLite keeps the
        file size until ``VACUUM``; Postgres reuses the space after autovacuum.
        """
        age = parse_age(older_than)
        cutoff = (_dt.datetime.now(_dt.timezone.utc) - age).isoformat(timespec="microseconds")
        with self._open_lock:
            exclude = set(self._open.keys())
        counts = self.store.prune(cutoff, keep_events=tuple(keep_events), exclude=exclude, dry_run=dry_run)
        if not dry_run:
            self._emit("prune", data={"older_than_s": age.total_seconds(), "cutoff": cutoff, **counts})
        return counts

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
        if not callable(fn):
            raise TypeError(f"mem.wrap needs a function or other callable, not {type(fn).__name__}")
        if isinstance(items, str):
            items = [items]
        if _is_wrapper(self._wrappers, fn):  # already wrapped by this memory: wrapping again would record twice
            if (name is None or name == fn.__relock_tool__) and items is None:
                return fn
            fn = fn.__wrapped__
        tool = name or _tool_name(fn)
        refs = list(items) if items else None
        memory = self

        def bind() -> tuple[Episode | None, list[Any]]:
            ep = memory.current_episode()
            if ep is None:
                memory._note_unbound(tool)
                return None, []
            if ep._tool_claimed(tool):  # a framework callback (e.g. LangChain's) records this call
                return None, []
            try:
                return ep, ep._bind_tool(tool, refs)
            except Exception as e:  # instrumentation must never break the tool call
                memory._note_bind_failure(tool, e)
                return None, []

        def failed(ep: Episode | None, exps: list[Any], e: BaseException) -> None:
            if ep is not None and not is_control_flow(e):  # an interrupt or cancel isn't an outcome
                ep._observe_quietly(exps, error=e)

        call = None if inspect.isroutine(fn) else type(fn).__call__  # a callable object's method
        if inspect.iscoroutinefunction(fn) or (call is not None and inspect.iscoroutinefunction(call)):
            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                ep, exps = bind()
                try:
                    result = await fn(*args, **kwargs)
                except BaseException as e:
                    failed(ep, exps, e)
                    raise
                if ep is not None:
                    ep._observe_quietly(exps, result=result)
                return result

            wrapped: Any = awrapper
        else:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                ep, exps = bind()
                try:
                    result = fn(*args, **kwargs)
                except BaseException as e:
                    failed(ep, exps, e)
                    raise
                if ep is not None:
                    ep._observe_quietly(exps, result=result)
                return result

            wrapped = wrapper
        wrapped.__relock_tool__ = tool
        wrapped.__wrapped__ = fn
        self._wrappers.add(wrapped)
        return wrapped

    tool = wrap

    def wrap_tools(self, tools: Iterable[Callable[..., Any]]) -> list[Callable[..., Any]]:
        return [self.wrap(t) for t in tools]

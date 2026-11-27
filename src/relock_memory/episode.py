"""Episodes: the reconsolidation window.

    recall  ->  items become labile (editable) for this episode only
    expect  ->  the agent states what a recalled item predicts
    observe ->  outcome vs. prediction; a trusted mismatch is a prediction error
    rewrite ->  enough evidence -> the item that fired is edited in place
    close   ->  re-lock: the labile set is gone, late writes raise
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ._util import new_id, now, short_repr
from .checks import as_check, succeeds
from .delta import DeltaError, apply, validate
from .delta import diff as clause_diff
from .policy import WritePolicy
from .rewriters import Evidence, RewriteRequest
from .signals import ErrorClass, Signal, Source, classify_exception
from .types import (
    ConflictError,
    Decision,
    EpisodeClosedError,
    Expectation,
    Hit,
    Item,
    NotFoundError,
    NotLabileError,
    Observation,
    Recall,
    RelockError,
    Verdict,
)

if TYPE_CHECKING:
    from .memory import Memory

log = logging.getLogger("relock_memory")

_MISSING: Any = object()


class Episode:
    """One task attempt. Use as a context manager so it always re-locks."""

    def __init__(self, memory: Memory, task: str = "", meta: dict[str, Any] | None = None) -> None:
        self.memory = memory
        self.store = memory.store
        self.policy = memory.policy
        self.id = new_id("ep")
        self.task = task
        self.meta = dict(meta or {})
        self.closed = False
        self._lock = threading.RLock()
        self._labile: dict[str, Item] = {}
        self._expectations: dict[str, Expectation] = {}
        self._resolved: set[str] = set()
        self._rewrites: Counter[str] = Counter()
        self._declined: dict[str, tuple[str, ...]] = {}
        self._token: Any = None
        self.summary: dict[str, Any] = {}
        self.store.create_episode(self.id, task, self.meta)
        memory._register(self)
        memory._emit("episode_open", episode_id=self.id, data={"task": task})

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> Episode:
        self._token = self.memory._ctx.set(self)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            self.close()
        finally:
            if self._token is not None:
                try:
                    self.memory._ctx.reset(self._token)
                except ValueError:  # exited in a different context (e.g. another task)
                    self.memory._ctx.set(None)
                self._token = None

    def __repr__(self) -> str:
        state = "closed" if self.closed else "open"
        return f"Episode({self.id}, {state}, labile={sorted(i.ref for i in self._labile.values())})"

    def _ensure_open(self) -> None:
        if self.closed:
            raise EpisodeClosedError(f"episode {self.id} has re-locked; open a new episode to write")

    @property
    def labile(self) -> dict[str, Item]:
        """Items that may be edited in this episode (id -> version as recalled or last rewritten)."""
        return dict(self._labile)

    # ------------------------------------------------------------------ reactivation
    def recall(self, query: str, k: int = 3, *, kinds: list[str] | None = None,
               tags: list[str] | None = None) -> Recall:
        """Retrieve the top-``k`` active items. Every returned item becomes labile."""
        self._ensure_open()
        items = self.memory._active_items()
        if kinds:
            items = [i for i in items if i.kind in kinds]
        if tags:
            items = [i for i in items if set(tags) & set(i.tags)]
        ranked = self.memory.retriever.rank(query, items, k)
        hits = [Hit(item, float(score), rank) for rank, (item, score) in enumerate(ranked)]
        with self._lock:
            self._ensure_open()
            for h in hits:
                self._labile[h.item.id] = h.item
        self.store.add_retrievals(self.id, query, [(h.item.id, h.item.version, h.rank, h.score) for h in hits])
        self.memory._emit("recall", episode_id=self.id, data={
            "query": query, "hits": [{"ref": h.item.ref, "version": h.item.version, "score": round(h.score, 4)}
                                     for h in hits]})
        return Recall(query, hits)

    def use(self, ref: str | Item) -> Item:
        """Load one item by key or id (e.g. a named skill). It becomes labile like a recalled item."""
        self._ensure_open()
        item = self.memory.get(ref.id if isinstance(ref, Item) else ref)
        if item.archived:
            raise NotFoundError(f"{item.ref} is archived")
        with self._lock:
            self._ensure_open()
            self._labile[item.id] = item
        self.store.add_retrievals(self.id, f"use:{item.ref}", [(item.id, item.version, 0, 1.0)])
        self.memory._emit("recall", episode_id=self.id, item_id=item.id, version=item.version,
                          data={"query": f"use:{item.ref}", "hits": [{"ref": item.ref, "version": item.version}]})
        return item

    def get(self, ref: str | Item | Hit) -> Item:
        """Current version of a labile item (re-read it after a rewrite to retry with the fix)."""
        item_id = self._item_id(ref)
        with self._lock:
            if item_id in self._labile:
                return self._labile[item_id]
        return self.memory.get(item_id)

    def _item_id(self, target: str | Item | Hit) -> str:
        if isinstance(target, Hit):
            return target.item.id
        if isinstance(target, Item):
            return target.id
        with self._lock:
            for item in self._labile.values():
                if target in (item.id, item.key):
                    return item.id
        return self.store.resolve(target)

    # ------------------------------------------------------------------ prediction
    def expect(
        self,
        target: str | Item | Hit,
        statement: str = "",
        *,
        check: Any = None,
        clauses: list[str] | tuple[str, ...] | None = None,
        tool: str | None = None,
        author: str = "harness",
        _auto: bool = False,
    ) -> Expectation:
        """Record what acting on a recalled item should produce.

        ``check`` is a :mod:`relock_memory.checks` check (or any ``(result, error)``
        callable). ``clauses`` cites the clause ids the action relies on: with the
        default ``edit_scope="cited"`` only those clauses can be rewritten.
        ``tool`` binds the expectation to a wrapped tool so the tool's real
        outcome resolves it. ``author="agent"`` marks a prediction the model wrote
        (via the toolkit or MCP); its text is withheld from the rewriter unless the
        policy trusts agent reports, so the model cannot smuggle text into memory
        through the prediction it states.
        """
        self._ensure_open()
        item_id = self._item_id(target)
        with self._lock:
            self._ensure_open()
            if item_id not in self._labile:
                raise NotLabileError(
                    f"{target!r} was not recalled in episode {self.id}; call recall() or use() first. "
                    "Only reactivated memories can be reconsolidated."
                )
            item = self._labile[item_id]
            cited = tuple(clauses or ())
            known = {c.id for c in item.clauses}
            unknown = [c for c in cited if c not in known]
            if unknown:
                raise ValueError(f"{item.ref} v{item.version} has no clause(s) {unknown}; known: {sorted(known)}")
            exp = Expectation(
                id=new_id("exp"),
                episode_id=self.id,
                item_id=item.id,
                version=item.version,
                statement=statement or (f"{tool} succeeds" if tool else f"'{item.title}' works as described"),
                clause_ids=cited,
                tool=tool,
                check=as_check(check),
                auto=_auto,
                author=author,
            )
            self._expectations[exp.id] = exp
        self.store.add_expectation({
            "id": exp.id, "episode_id": self.id, "item_id": item.id, "version": item.version,
            "statement": exp.statement, "clause_ids": cited, "tool": tool,
            "check_spec": getattr(exp.check, "spec", None), "auto": _auto,
        })
        self.memory._emit("expect", episode_id=self.id, item_id=item.id, version=item.version,
                          data={"expectation_id": exp.id, "statement": exp.statement, "clauses": list(cited),
                                "tool": tool, "auto": _auto, "author": author})
        return exp

    def observe(
        self,
        expectation: Expectation | str,
        result: Any = _MISSING,
        *,
        error: BaseException | str | None = None,
        signal: Signal | None = None,
        matched: bool | None = None,
    ) -> Observation:
        """Compare an outcome with its expectation and apply the write gates.

        If no ``signal`` is given the caller is assumed to be trusted harness
        code: an exception becomes ``Signal.tool_error(error)`` (classified as
        transient / programmatic / semantic) and success becomes ``Signal.ok()``.
        """
        self._ensure_open()
        with self._lock:
            # Re-checked under the lock close() takes: a close that lands between the check
            # above and here must win, or a rewrite could commit after the episode re-locked.
            self._ensure_open()
            exp = self._expectation(expectation)
            if exp.id in self._resolved:
                raise RelockError(f"expectation {exp.id} was already observed")
            self._resolved.add(exp.id)

            exc = error if isinstance(error, BaseException) else None
            # A failure the caller states explicitly is never overridden by the expectation's
            # check: succeeds() would otherwise see "no exception" and call it a success.
            explicit_failure = signal is not None and signal.passed is False
            if signal is None:
                signal = Signal.tool_error(error) if error is not None else Signal.ok()
            value = None if result is _MISSING else result

            if matched is not None:
                is_match, reason = bool(matched), "reported by caller"
            elif signal.passed is not None and (signal.source is not Source.TOOL or error is not None
                                                or exp.check is None or explicit_failure):
                is_match = bool(signal.passed)
                reason = signal.message or ("passed" if is_match else "failed")
            elif exp.check is not None:
                is_match, reason = exp.check(value, exc)
                if error is not None and exc is None:
                    is_match, reason = False, str(error)
            else:
                is_match, reason = error is None, ("completed" if error is None else str(error))

            outcome = {"result": short_repr(value, 400) if result is not _MISSING else None,
                       "error": (f"{type(exc).__name__}: {exc}" if exc else (str(error) if error else None))}
            self.store.resolve_expectation(exp.id, matched=is_match, outcome=outcome, signal=signal.to_dict())

            if is_match:
                return self._on_confirm(exp, signal, reason)
            return self._on_error(exp, signal, reason, outcome, exc)

    def _expectation(self, expectation: Expectation | str) -> Expectation:
        exp_id = expectation.id if isinstance(expectation, Expectation) else expectation
        exp = self._expectations.get(exp_id)
        if exp is None:
            raise NotFoundError(f"expectation {exp_id!r} does not belong to episode {self.id}")
        return exp

    def _on_confirm(self, exp: Expectation, signal: Signal, reason: str) -> Observation:
        if not self.policy.trusts_confirmation(signal):
            d = Decision(Verdict.IGNORED, f"confirmation from untrusted source '{signal.source.value}' "
                         "does not change confidence", exp.item_id, exp.version)
            self.memory._emit("confirm_ignored", episode_id=self.id, item_id=exp.item_id, version=exp.version,
                              data={"expectation_id": exp.id, "source": signal.source.value})
            return Observation(exp.id, True, reason, d)
        self.store.bump_confidence(exp.item_id, exp.version, d_alpha=1.0, d_confirm=1)
        cancelled = self._cancel_pending(exp.item_id, exp.version)
        self.memory._invalidate()
        d = Decision(Verdict.CONFIRMED, "prediction held" + (f"; cancelled {len(cancelled)} pending error(s)"
                     if cancelled else ""), exp.item_id, exp.version)
        self.memory._emit("confirm", episode_id=self.id, item_id=exp.item_id, version=exp.version,
                          data={"expectation_id": exp.id, "cancelled": cancelled})
        return Observation(exp.id, True, reason, d)

    def _cancel_pending(self, item_id: str, version: int) -> list[str]:
        budget = self.policy.confirmation_cancels
        if budget <= 0:
            return []
        cancelled: list[str] = []
        for e in self.store.errors(item_id, version, status="pending"):
            if e["authoritative"] or e["weight"] > budget + 1e-9:
                continue
            cancelled.append(e["id"])
            budget -= e["weight"]
            if budget <= 1e-9:
                break
        self.store.set_error_status(cancelled, "cancelled", "later prediction confirmed (flake damping)")
        return cancelled

    def _on_error(self, exp: Expectation, signal: Signal, reason: str, outcome: dict[str, Any],
                  exc: BaseException | None) -> Observation:
        if signal.error_class is None:
            cls = classify_exception(exc) if exc is not None else ErrorClass.SEMANTIC
            signal = signal.with_(passed=False, error_class=cls)
        adm = self.policy.admit(signal)
        error_id = new_id("err")
        diff = {
            "expected": exp.statement,
            "expected_by": exp.author,
            "check": getattr(exp.check, "spec", None),
            "observed": reason,
            "outcome": outcome,
            "clause_ids": list(exp.clause_ids),
            "tool": exp.tool,
            "signal": signal.to_dict(),
        }
        self.store.add_error({
            "id": error_id, "expectation_id": exp.id, "episode_id": self.id, "item_id": exp.item_id,
            "version": exp.version, "source": signal.source.value,
            "error_class": (signal.error_class or ErrorClass.UNKNOWN).value, "weight": adm.weight,
            "authoritative": adm.authoritative, "diff": diff,
            "status": "pending" if adm.counts else "blocked", "reason": adm.reason,
        })
        if not adm.counts:
            d = Decision(Verdict.BLOCKED, adm.reason, exp.item_id, exp.version)
            self.memory._emit("blocked", episode_id=self.id, item_id=exp.item_id, version=exp.version,
                              data={"error_id": error_id, "reason": adm.reason, "source": signal.source.value})
            return Observation(exp.id, False, reason, d, error_id)

        self.store.bump_confidence(exp.item_id, exp.version, d_beta=1.0, d_fail=1)
        self.memory._invalidate()
        self.memory._emit("prediction_error", episode_id=self.id, item_id=exp.item_id, version=exp.version,
                          data={"error_id": error_id, "weight": adm.weight, "source": signal.source.value,
                                "error_class": signal.error_class.value if signal.error_class else None,
                                "observed": reason})
        if self.policy.rewrite_on == "observe" or adm.authoritative:
            d = self._reconsolidate(exp.item_id)
        else:
            ev, req = self._evidence_level(exp.item_id, exp.version)
            d = Decision(Verdict.PENDING, "deferred to episode close", exp.item_id, exp.version, None, ev, req)
        return Observation(exp.id, False, reason, d, error_id)

    # ------------------------------------------------------------------ reconsolidation
    def _evidence_level(self, item_id: str, version: int) -> tuple[float, float]:
        pending = self.store.errors(item_id, version, status="pending")
        current = self.memory.get(item_id, version)
        return sum(e["weight"] for e in pending), self.policy.required_evidence(current.confirmations)

    def _reconsolidate(self, item_id: str) -> Decision:
        recalled = self._labile.get(item_id)
        if recalled is None:
            return Decision(Verdict.BLOCKED, "item is not labile in this episode", item_id)
        current = self.memory.get(item_id)
        if current.version != recalled.version:
            return Decision(Verdict.CONFLICT, f"item moved to v{current.version} after it was recalled "
                            f"(v{recalled.version}); evidence is stale", item_id, recalled.version)
        pending = self.store.errors(item_id, current.version, status="pending")
        evidence = sum(e["weight"] for e in pending)
        authoritative = any(e["authoritative"] for e in pending)
        required = self.policy.required_evidence(current.confirmations)
        base = {"item_id": item_id, "version": current.version, "evidence": evidence, "required": required}
        if not pending:
            return Decision(Verdict.PENDING, "no pending evidence", **base)
        if not authoritative and evidence + 1e-9 < required:
            return Decision(Verdict.PENDING, f"evidence {evidence:.2f} < required {required:.2f}", **base)
        if self._rewrites[item_id] >= self.policy.max_rewrites_per_item_per_episode:
            return Decision(Verdict.PENDING, "rewrite budget for this item in this episode is spent", **base)
        ids = tuple(e["id"] for e in pending)
        if self._declined.get(item_id) == ids:
            return Decision(Verdict.DECLINED, "rewriter already declined this evidence", **base)

        editable: frozenset[str] | None = None
        if self.policy.edit_scope == "cited":
            # Citations narrow the edit only when a trusted author made them: an untrusted agent
            # could otherwise point a genuine failure's fix at a clause it wants gone.
            cited = {c for e in pending for c in _trusted_citations(e, self.policy)}
            editable = frozenset(cited) if cited else None
        request = RewriteRequest(
            item=current,
            evidence=tuple(evidence_for_rewriter(e, self.policy) for e in pending),
            editable=editable,
            task=self.task,
            max_ops=self.policy.max_ops,
            max_clause_chars=self.policy.max_clause_chars,
        )
        try:
            proposal = self.memory.rewriter.propose(request)
        except Exception as e:  # never let a rewriter crash the agent
            log.exception("relock rewriter failed")
            self._declined[item_id] = ids
            self.memory._emit("rewriter_error", episode_id=self.id, item_id=item_id, version=current.version,
                              data={"error": f"{type(e).__name__}: {e}"})
            return Decision(Verdict.DECLINED, f"rewriter raised {type(e).__name__}: {e}", **base)
        if proposal is None or not proposal.ops:
            self._declined[item_id] = ids
            self.memory._emit("declined", episode_id=self.id, item_id=item_id, version=current.version,
                              data={"evidence": list(ids)})
            return Decision(Verdict.DECLINED, "rewriter proposed no change", **base)
        try:
            validate(current.clauses, proposal.ops, editable=set(editable) if editable is not None else None,
                     max_ops=self.policy.max_ops, max_clause_chars=self.policy.max_clause_chars)
        except DeltaError as e:
            self._declined[item_id] = ids
            self.memory._emit("invalid_edit", episode_id=self.id, item_id=item_id, version=current.version,
                              data={"error": str(e), "ops": [o.to_dict() for o in proposal.ops]})
            return Decision(Verdict.INVALID_EDIT, str(e), **base)

        new_clauses, seq = apply(current.clauses, proposal.ops, current.clause_seq)
        a, b = self.policy.new_version_prior
        new = replace(
            current,
            version=self.store.next_version_number(item_id),
            clauses=new_clauses,
            clause_seq=seq,
            status="active",
            alpha=a,
            beta=b,
            confirmations=0,
            failures=0,
            parent_version=current.version,
            superseded_by=None,
            change_kind="reconsolidate",
            delta=[o.to_dict() for o in proposal.ops],
            reason=proposal.rationale or None,
            episode_id=self.id,
            evidence=ids,
            author=proposal.author,
            created_at=now(),
        )
        try:
            self.store.commit_version(new, expected_current=current.version,
                                      stale_reason=f"superseded by v{new.version}")
        except ConflictError as e:
            return Decision(Verdict.CONFLICT, str(e), **base)
        with self._lock:
            self._labile[item_id] = new
            self._rewrites[item_id] += 1
        self.memory._invalidate()
        changes = clause_diff(current.clauses, new.clauses)
        self.memory._emit("rewrite", episode_id=self.id, item_id=item_id, version=new.version, data={
            "from_version": current.version, "to_version": new.version, "delta": new.delta,
            "changes": changes, "evidence": list(ids), "author": proposal.author,
            "rationale": proposal.rationale})
        return Decision(Verdict.REWRITTEN, f"v{current.version} -> v{new.version} ({proposal.author})",
                        item_id, current.version, new.version, evidence, required, new.delta)

    @staticmethod
    def _to_evidence(e: dict[str, Any], policy: WritePolicy | None = None) -> Evidence:
        return evidence_for_rewriter(e, policy or WritePolicy())

    # ------------------------------------------------------------------ conveniences
    def correct(self, target: str | Item | Hit, correction: str, *, principal: str | None = None,
                authenticated: bool = False, clauses: list[str] | None = None) -> Observation:
        """Apply a human correction to a recalled item.

        Pass ``authenticated=True`` once your application has verified the user; the
        correction is then authoritative and rewrites immediately. Without it the
        correction is logged as blocked.
        """
        exp = self.expect(target, "behaviour matches the user's instructions", clauses=clauses)
        return self.observe(exp, signal=Signal.user(correction, principal=principal, authenticated=authenticated))

    def check(self, target: str | Item | Hit, passed: bool, message: str = "", *,
              correction: str | None = None, clauses: list[str] | None = None,
              statement: str = "", verifier: str | None = None) -> Observation:
        """Shorthand for a verifier verdict on a recalled item: expect + observe in one call."""
        exp = self.expect(target, statement or "verifier passes", clauses=clauses)
        return self.observe(exp, signal=Signal.verifier(passed, message, correction=correction, name=verifier))

    def tool_result(self, tool: str, result: Any = None, *, error: BaseException | str | None = None,
                    signal: Signal | None = None, items: list[str] | tuple[str, ...] | None = None,
                    ) -> list[Observation]:
        """Report a tool call's real outcome, exactly as ``@mem.wrap`` would, for tools you can't wrap.

        It resolves the open expectations bound to ``tool``. Without any, it creates an automatic
        ``succeeds()`` expectation for each recalled item that lists the tool, or for ``items``.
        Report a failure with ``error`` (an exception or message) or a ready-made ``signal``.
        Returns one observation per resolved expectation, or none if no recalled item governs
        the tool.
        """
        with self._lock:  # bind and resolve atomically, so concurrent reports never collide
            self._ensure_open()
            exps = self._bind_tool(tool, list(items) if items else None)
            args = () if result is None else (result,)
            if signal is not None:
                return [self.observe(e, *args, signal=signal) for e in exps]
            if error is not None:
                return [self.observe(e, *args, error=error) for e in exps]
            return [self.observe(e, result) for e in exps]

    def _bind_tool(self, tool: str, refs: list[str] | None) -> list[Expectation]:
        """Expectations a wrapped tool call should resolve (explicit ones first, else automatic)."""
        if self.closed:
            return []
        with self._lock:
            if refs:
                targets = [self._labile[i] for i in (self._item_id(r) for r in refs) if i in self._labile]
            else:
                targets = [i for i in self._labile.values() if tool in i.tools]
                if self.policy.attribute_undeclared:
                    targets += [i for i in self._labile.values() if not i.tools]
            out: list[Expectation] = []
            for item in targets:
                explicit = next((e for e in self._expectations.values()
                                 if e.item_id == item.id and e.tool == tool and e.id not in self._resolved
                                 and not e.auto), None)
                if explicit is not None:
                    out.append(explicit)
                else:
                    out.append(self.expect(item, f"{tool} succeeds", check=succeeds(), tool=tool, _auto=True))
            return out

    def _observe_quietly(self, exps: list[Expectation], **kwargs: Any) -> list[Observation]:
        out = []
        for exp in exps:
            try:
                out.append(self.observe(exp, **kwargs))
            except Exception:  # instrumentation must never break the tool call
                log.exception("relock failed to observe a tool outcome")
        return out

    def close(self) -> dict[str, Any]:
        """Re-lock. Optionally reconsolidates first (``rewrite_on="close"``). Idempotent."""
        with self._lock:
            if self.closed:
                return self.summary
            decisions = []
            if self.policy.rewrite_on == "close":
                for item_id in list(self._labile):
                    decisions.append(self._reconsolidate(item_id))
            self.store.close_episode(self.id)
            self.closed = True
            self.summary = {
                "episode_id": self.id,
                "recalled": sorted(i.ref for i in self._labile.values()),
                "expectations": len(self._expectations),
                "rewrites": dict(self._rewrites),
                "close_decisions": [d.to_dict() for d in decisions if d.verdict is not Verdict.PENDING
                                    or d.evidence > 0],
            }
        self.memory._emit("relock", episode_id=self.id, data=self.summary)
        self.memory._unregister(self)
        return self.summary


AGENT_TEXT_WITHHELD = "(prediction stated by the agent; withheld because agent text is untrusted)"
DETAIL_WITHHELD = "(detail withheld: may contain untrusted content)"


def _agent_untrusted(diff: dict[str, Any], policy: WritePolicy) -> bool:
    return diff.get("expected_by") == "agent" and Source.AGENT not in policy.trusted_sources


def _trusted_citations(e: dict[str, Any], policy: WritePolicy) -> tuple[str, ...]:
    d = e["diff"]
    return () if _agent_untrusted(d, policy) else tuple(d.get("clause_ids") or ())


def evidence_for_rewriter(e: dict[str, Any], policy: WritePolicy) -> Evidence:
    """Sanitise one admitted prediction error for the rewriter.

    Only trusted authors steer the rewriter. A signal's message and correction are
    dropped when ``detail_trusted`` is false. A prediction the *agent* stated, and the
    clause ids it cited, are dropped unless the policy trusts agent reports: the
    failure still counts as evidence, but the agent neither words the fix nor picks
    which clause receives it.
    """
    d = e["diff"]
    sig = Signal.from_dict(d["signal"])
    cls = (sig.error_class or ErrorClass.UNKNOWN).value
    expected = d.get("expected", "")
    if _agent_untrusted(d, policy):
        expected = AGENT_TEXT_WITHHELD
    return Evidence(
        error_id=e["id"],
        expectation=expected,
        clause_ids=_trusted_citations(e, policy),
        tool=d.get("tool"),
        observed=str(d.get("observed", "")) if sig.detail_trusted else f"{cls} failure {DETAIL_WITHHELD}",
        source=sig.source.value,
        error_class=cls,
        message=sig.message if sig.detail_trusted else DETAIL_WITHHELD,
        correction=sig.correction if sig.detail_trusted else None,
        authoritative=bool(e["authoritative"]),
    )

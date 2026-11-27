"""The write policy: every gate a prediction error must pass before memory changes.

Gates, in order:

1. **Lability** - the item must have been recalled in the current, still-open
   episode (enforced by :class:`~relock_memory.episode.Episode`).
2. **Trust** - the signal must come from a trusted source. Tool code,
   verifiers and *authenticated* users are trusted; the model's own reports,
   self-judgement and anything read from the environment are not.
3. **Error class** - transient failures (timeouts, rate limits, 5xx) never count.
4. **Evidence** - weighted prediction errors against the *current version*
   must reach a threshold that grows (logarithmically) with how often the
   version has been confirmed. Authenticated user corrections are authoritative.
5. **Edit shape** - the rewrite must be a small clause-level delta that only
   touches clauses that actually fired.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal

from .signals import ErrorClass, Signal, Source


@dataclass(frozen=True)
class Admission:
    """Whether a failure signal may count as evidence, and with what weight."""

    counts: bool
    weight: float
    reason: str
    authoritative: bool = False


@dataclass(frozen=True)
class WritePolicy:
    trusted_sources: frozenset[Source] = frozenset({Source.TOOL, Source.VERIFIER, Source.USER})
    require_authenticated_user: bool = True
    block_transient: bool = True

    min_evidence: float = 2.0
    """Weighted prediction errors needed before a version is rewritten."""
    inertia: float = 0.5
    """Extra evidence required per ``log(1 + confirmations)``: proven versions resist one-off noise."""
    source_weights: dict[Source, float] = field(
        default_factory=lambda: {Source.TOOL: 1.0, Source.VERIFIER: 1.0, Source.USER: 2.0,
                                 Source.AGENT: 1.0, Source.SELF: 0.5, Source.ENVIRONMENT: 0.5}
    )
    class_weights: dict[ErrorClass, float] = field(
        default_factory=lambda: {ErrorClass.PROGRAMMATIC: 1.0, ErrorClass.SEMANTIC: 1.0,
                                 ErrorClass.UNKNOWN: 0.5, ErrorClass.TRANSIENT: 0.0}
    )
    authoritative_users: bool = True
    """Authenticated user corrections bypass the evidence threshold (but not lability)."""
    confirmation_cancels: float = 1.0
    """A trusted confirmed prediction cancels this much pending error evidence (flake damping)."""

    rewrite_on: Literal["observe", "close"] = "observe"
    """Reconsolidate as soon as evidence suffices, or only when the episode closes."""
    max_rewrites_per_item_per_episode: int = 1
    edit_scope: Literal["cited", "item"] = "cited"
    """``cited``: replace/remove only clauses the expectation relied on (if it cited any)."""
    max_ops: int = 4
    max_clause_chars: int = 600

    attribute_undeclared: bool = False
    """When a wrapped tool runs, also attribute its outcome to recalled items that declare no tools."""

    new_version_prior: tuple[float, float] = (1.0, 1.0)

    # ------------------------------------------------------------------ gates
    def admit(self, signal: Signal) -> Admission:
        """Decide whether a *failure* signal may count as reconsolidation evidence."""
        if signal.source not in self.trusted_sources:
            return Admission(False, 0.0, f"untrusted source '{signal.source.value}'")
        if signal.source is Source.USER and self.require_authenticated_user and not signal.authenticated:
            return Admission(False, 0.0, "unauthenticated user correction "
                                         "(pass authenticated=True once your app has verified the user)")
        cls = signal.error_class or ErrorClass.UNKNOWN
        if cls is ErrorClass.TRANSIENT and self.block_transient:
            return Admission(False, 0.0, "transient error (says nothing about the procedure)")
        weight = self.source_weights.get(signal.source, 1.0) * self.class_weights.get(cls, 0.5)
        if weight <= 0:
            return Admission(False, 0.0, f"zero weight for {signal.source.value}/{cls.value}")
        authoritative = (
            signal.source is Source.USER and self.authoritative_users and signal.authenticated
        )
        return Admission(True, weight, f"{signal.source.value}/{cls.value}", authoritative)

    def trusts_confirmation(self, signal: Signal) -> bool:
        """Only trusted sources may raise confidence (otherwise injected 'it worked' inflates inertia)."""
        if signal.source not in self.trusted_sources:
            return False
        return not (signal.source is Source.USER and self.require_authenticated_user and not signal.authenticated)

    def required_evidence(self, confirmations: int) -> float:
        return self.min_evidence + self.inertia * math.log1p(max(0, confirmations))

    def with_(self, **changes) -> WritePolicy:
        return replace(self, **changes)

    # ------------------------------------------------------------------ presets
    @classmethod
    def strict(cls) -> WritePolicy:
        """Verifier- and user-only writes, higher threshold."""
        return cls(trusted_sources=frozenset({Source.VERIFIER, Source.USER}), min_evidence=3.0, inertia=1.0)

    @classmethod
    def eager(cls) -> WritePolicy:
        """Rewrite after a single trusted prediction error (still gated on trust and transience)."""
        return cls(min_evidence=1.0, inertia=0.0)

    @classmethod
    def trust_agent(cls) -> WritePolicy:
        """Also accept the agent's own outcome reports (for local, single-user sandboxes)."""
        return cls(trusted_sources=frozenset({Source.TOOL, Source.VERIFIER, Source.USER, Source.AGENT}))

    @classmethod
    def permissive(cls) -> WritePolicy:
        """No gates at all: any failure rewrites immediately. An ablation baseline, not for production."""
        return cls(
            trusted_sources=frozenset(Source),
            require_authenticated_user=False,
            block_transient=False,
            min_evidence=0.0,
            inertia=0.0,
            class_weights={c: 1.0 for c in ErrorClass},
            source_weights={s: 1.0 for s in Source},
            confirmation_cancels=0.0,
            max_rewrites_per_item_per_episode=1_000,
            edit_scope="item",
        )

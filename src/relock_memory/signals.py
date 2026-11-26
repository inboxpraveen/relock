"""Signals: the typed feedback that may (or may not) open a memory write.

The central safety rule of relock is that *where* feedback comes from matters
more than *what* it says. A signal records its source and its error class, and
the :class:`~relock_memory.policy.WritePolicy` decides whether it is allowed to
count as evidence. Free text that came from a web page or from the model's own
judgement can be logged, but it can never rewrite a trusted memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class Source(str, Enum):
    """Who produced the feedback."""

    TOOL = "tool"
    """A typed result or exception from tool code you run (trusted by default)."""
    VERIFIER = "verifier"
    """A programmatic check such as a test suite, schema validator or grader (trusted)."""
    USER = "user"
    """A human correction. Trusted only when ``authenticated`` is true."""
    AGENT = "agent"
    """The model reporting on itself through a tool call (untrusted by default)."""
    ENVIRONMENT = "environment"
    """Retrieved or observed content: web pages, documents, emails (never trusted)."""
    SELF = "self"
    """An LLM-as-judge or self-critique verdict (untrusted by default)."""


class ErrorClass(str, Enum):
    """What kind of failure this is. Transient failures never open a write."""

    TRANSIENT = "transient"
    """Timeouts, rate limits, 5xx, dropped connections. Says nothing about the procedure."""
    PROGRAMMATIC = "programmatic"
    """Deterministic contract violations: validation errors, 4xx, type errors."""
    SEMANTIC = "semantic"
    """The call succeeded but the result is wrong (a verifier or check failed)."""
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Signal:
    """Feedback about one observed outcome.

    Build signals with the constructors (``Signal.tool_error(...)``,
    ``Signal.verifier(...)``, ``Signal.user(...)`` ...) rather than directly.
    """

    source: Source
    passed: bool | None = None
    error_class: ErrorClass | None = None
    message: str = ""
    correction: str | None = None
    """An explicit statement of the right behaviour, e.g. ``"Pass dates as YYYY-MM-DD."``"""
    authenticated: bool = False
    principal: str | None = None
    detail_trusted: bool = True
    """If false, ``message`` and ``correction`` are logged but never shown to the rewriter.

    Use it for *trusted* sources whose text may carry outside content, e.g. a tool
    error that echoes a remote server's response body. Untrusted sources need no
    flag: their signals are blocked before they reach the rewriter at all.
    """
    data: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def ok(cls, source: Source = Source.TOOL, **data: Any) -> Signal:
        return cls(source=Source(source), passed=True, data=dict(data))

    @classmethod
    def tool_error(
        cls,
        error: BaseException | str,
        *,
        error_class: ErrorClass | str | None = None,
        correction: str | None = None,
        detail_trusted: bool = True,
        **data: Any,
    ) -> Signal:
        """A failure raised or returned by tool code you control."""
        if isinstance(error, BaseException):
            cls_ = ErrorClass(error_class) if error_class else classify_exception(error)
            message = f"{type(error).__name__}: {error}"
            correction = correction or getattr(error, "correction", None)
            data.setdefault("exception_type", type(error).__name__)
        else:
            cls_ = ErrorClass(error_class) if error_class else ErrorClass.UNKNOWN
            message = str(error)
        return cls(
            source=Source.TOOL,
            passed=False,
            error_class=cls_,
            message=message,
            correction=correction,
            detail_trusted=detail_trusted,
            data=dict(data),
        )

    @classmethod
    def transient(cls, message: str = "transient failure", **data: Any) -> Signal:
        return cls(Source.TOOL, passed=False, error_class=ErrorClass.TRANSIENT, message=message, data=dict(data))

    @classmethod
    def verifier(
        cls,
        passed: bool,
        message: str = "",
        *,
        correction: str | None = None,
        name: str | None = None,
        **data: Any,
    ) -> Signal:
        if name:
            data["verifier"] = name
        return cls(
            source=Source.VERIFIER,
            passed=passed,
            error_class=None if passed else ErrorClass.SEMANTIC,
            message=message,
            correction=correction,
            data=dict(data),
        )

    @classmethod
    def user(
        cls,
        correction: str,
        *,
        principal: str | None = None,
        authenticated: bool = True,
        **data: Any,
    ) -> Signal:
        """A human correction. Pass ``authenticated=False`` when you cannot vouch for who sent it."""
        return cls(
            source=Source.USER,
            passed=False,
            error_class=ErrorClass.SEMANTIC,
            message=correction,
            correction=correction,
            authenticated=authenticated,
            principal=principal,
            data=dict(data),
        )

    @classmethod
    def agent_report(
        cls,
        passed: bool,
        message: str = "",
        *,
        error_class: ErrorClass | str | None = None,
        correction: str | None = None,
        **data: Any,
    ) -> Signal:
        """The model telling the harness what happened. Untrusted unless the policy opts in."""
        return cls(
            source=Source.AGENT,
            passed=passed,
            error_class=None if passed else ErrorClass(error_class or ErrorClass.UNKNOWN),
            message=message,
            correction=correction,
            data=dict(data),
        )

    @classmethod
    def observation(cls, text: str, *, passed: bool = False, correction: str | None = None, **data: Any) -> Signal:
        """Content from the outside world (web, files, email). Logged, never trusted."""
        return cls(
            source=Source.ENVIRONMENT,
            passed=passed,
            error_class=None if passed else ErrorClass.UNKNOWN,
            message=text,
            correction=correction,
            data=dict(data),
        )

    @classmethod
    def self_judgment(cls, passed: bool, rationale: str = "", **data: Any) -> Signal:
        return cls(
            source=Source.SELF,
            passed=passed,
            error_class=None if passed else ErrorClass.SEMANTIC,
            message=rationale,
            data=dict(data),
        )

    # ------------------------------------------------------------------ helpers
    def with_(self, **changes: Any) -> Signal:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "passed": self.passed,
            "error_class": self.error_class.value if self.error_class else None,
            "message": self.message,
            "correction": self.correction,
            "authenticated": self.authenticated,
            "principal": self.principal,
            "detail_trusted": self.detail_trusted,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Signal:
        return cls(
            source=Source(d["source"]),
            passed=d.get("passed"),
            error_class=ErrorClass(d["error_class"]) if d.get("error_class") else None,
            message=d.get("message") or "",
            correction=d.get("correction"),
            authenticated=bool(d.get("authenticated")),
            principal=d.get("principal"),
            detail_trusted=d.get("detail_trusted", True),
            data=d.get("data") or {},
        )


class ToolError(Exception):
    """Raise from your tools to give relock an explicit error class and correction.

    >>> raise ToolError("date must be ISO-8601", correction="Pass dates as YYYY-MM-DD.")
    """

    def __init__(
        self,
        message: str,
        *,
        correction: str | None = None,
        error_class: ErrorClass | str = ErrorClass.PROGRAMMATIC,
    ) -> None:
        super().__init__(message)
        self.correction = correction
        self.relock_error_class = ErrorClass(error_class)


class TransientToolError(ToolError):
    """A failure that says nothing about whether the procedure is right."""

    def __init__(self, message: str = "transient failure") -> None:
        super().__init__(message, error_class=ErrorClass.TRANSIENT)


# ---------------------------------------------------------------------- taxonomy
_TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504, 507, 520, 521, 522, 523, 524, 529}
_PROGRAMMATIC_STATUS = {400, 401, 403, 404, 405, 406, 409, 410, 411, 412, 413, 414, 415, 416, 422}

_TRANSIENT_NAME_HINTS = (
    "timeout", "timedout", "ratelimit", "toomanyrequests", "throttl", "serviceunavailable",
    "temporar", "connecterror", "connectionerror", "connectionreset", "remotedisconnected",
    "brokenpipe", "overloaded", "unavailable", "deadlineexceeded", "retry",
)
_PROGRAMMATIC_NAME_HINTS = (
    "validation", "invalid", "badrequest", "notfound", "schema", "unprocessable", "typeerror",
    "valueerror", "keyerror", "parse", "decode", "missing", "unsupported", "forbidden",
    "unauthorized", "permission", "conflict",
)


def _status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "status", "code", "http_status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int) and 100 <= v <= 599:
            return v
    resp = getattr(exc, "response", None)
    if resp is not None:
        for attr in ("status_code", "status"):
            v = getattr(resp, attr, None)
            if isinstance(v, int) and 100 <= v <= 599:
                return v
    return None


def classify_exception(exc: BaseException) -> ErrorClass:
    """Map an exception to an :class:`ErrorClass` using types, status codes and names.

    Precedence: an explicit ``relock_error_class`` attribute, then HTTP-style
    status codes, then built-in exception types, then name heuristics.
    """
    explicit = getattr(exc, "relock_error_class", None)
    if explicit is not None:
        return ErrorClass(explicit)

    status = _status_code(exc)
    if status in _TRANSIENT_STATUS:
        return ErrorClass.TRANSIENT
    if status in _PROGRAMMATIC_STATUS:
        return ErrorClass.PROGRAMMATIC

    if isinstance(exc, (TimeoutError, ConnectionError, InterruptedError)):
        return ErrorClass.TRANSIENT
    if isinstance(exc, AssertionError):
        return ErrorClass.SEMANTIC
    if isinstance(exc, (ValueError, TypeError, LookupError, AttributeError, NotImplementedError)):
        return ErrorClass.PROGRAMMATIC

    names = "".join(c.__name__.lower() for c in type(exc).__mro__)
    if any(h in names for h in _TRANSIENT_NAME_HINTS):
        return ErrorClass.TRANSIENT
    if any(h in names for h in _PROGRAMMATIC_NAME_HINTS):
        return ErrorClass.PROGRAMMATIC
    if isinstance(exc, OSError):
        return ErrorClass.TRANSIENT
    return ErrorClass.UNKNOWN

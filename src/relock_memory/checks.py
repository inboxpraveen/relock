"""Programmatic checks that decide whether an outcome matches an expectation.

A check is any callable ``(result, error) -> (bool, reason)``. Checks run in
your process, so their verdict is as trustworthy as the signal you attach.
They carry a ``spec`` so the ledger can record which check was used.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ._util import short_repr

CheckFn = Callable[[Any, "BaseException | None"], "tuple[bool, str]"]


@dataclass(frozen=True)
class Check:
    fn: CheckFn
    spec: str

    def __call__(self, result: Any, error: BaseException | None) -> tuple[bool, str]:
        return self.fn(result, error)

    def __repr__(self) -> str:
        return f"Check({self.spec})"


def _err_reason(error: BaseException) -> str:
    return f"raised {type(error).__name__}: {error}"


def succeeds() -> Check:
    """The call completes without raising."""

    def fn(result: Any, error: BaseException | None) -> tuple[bool, str]:
        if error is not None:
            return False, _err_reason(error)
        return True, "completed without error"

    return Check(fn, "succeeds()")


def fails() -> Check:
    """The call is expected to raise (useful for negative procedures)."""

    def fn(result: Any, error: BaseException | None) -> tuple[bool, str]:
        if error is None:
            return False, f"expected an error, got {short_repr(result, 120)}"
        return True, _err_reason(error)

    return Check(fn, "fails()")


def equals(expected: Any) -> Check:
    def fn(result: Any, error: BaseException | None) -> tuple[bool, str]:
        if error is not None:
            return False, _err_reason(error)
        if result == expected:
            return True, "equal"
        return False, f"expected {short_repr(expected, 120)}, got {short_repr(result, 120)}"

    return Check(fn, f"equals({short_repr(expected, 60)})")


def non_empty() -> Check:
    def fn(result: Any, error: BaseException | None) -> tuple[bool, str]:
        if error is not None:
            return False, _err_reason(error)
        try:
            empty = result is None or len(result) == 0
        except TypeError:
            empty = result is None
        return (not empty, "non-empty" if not empty else f"empty result: {short_repr(result, 80)}")

    return Check(fn, "non_empty()")


def has_keys(*keys: str) -> Check:
    """Result is a mapping (or a non-empty list of mappings) that contains ``keys``."""

    def fn(result: Any, error: BaseException | None) -> tuple[bool, str]:
        if error is not None:
            return False, _err_reason(error)
        rows = result if isinstance(result, list) else [result]
        if not rows:
            return False, "empty list"
        for row in rows:
            if not isinstance(row, dict):
                return False, f"expected mapping, got {type(row).__name__}"
            missing = [k for k in keys if k not in row]
            if missing:
                return False, f"missing keys {missing}"
        return True, "has keys"

    return Check(fn, f"has_keys({', '.join(keys)})")


def matches(pattern: str, flags: int = 0) -> Check:
    rx = re.compile(pattern, flags)

    def fn(result: Any, error: BaseException | None) -> tuple[bool, str]:
        if error is not None:
            return False, _err_reason(error)
        text = result if isinstance(result, str) else short_repr(result, 10_000)
        if rx.search(text):
            return True, f"matches /{pattern}/"
        return False, f"does not match /{pattern}/: {short_repr(text, 120)}"

    return Check(fn, f"matches({pattern!r})")


def predicate(fn: Callable[[Any], bool], description: str = "predicate") -> Check:
    def check(result: Any, error: BaseException | None) -> tuple[bool, str]:
        if error is not None:
            return False, _err_reason(error)
        try:
            ok = bool(fn(result))
        except Exception as e:  # a crashing predicate is a mismatch, not a harness crash
            return False, f"predicate raised {type(e).__name__}: {e}"
        return ok, description if ok else f"not {description}: {short_repr(result, 120)}"

    return Check(check, f"predicate({description})")


_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list, tuple),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


def _validate(value: Any, schema: dict[str, Any], path: str) -> str | None:
    t = schema.get("type")
    if t is not None:
        names = t if isinstance(t, list) else [t]
        ok = False
        for name in names:
            py = _TYPES.get(name, ())
            if isinstance(value, py) and not (name in ("integer", "number") and isinstance(value, bool)):
                ok = True
        if not ok:
            return f"{path or '$'}: expected {t}, got {type(value).__name__}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path or '$'}: {short_repr(value, 40)} not in {schema['enum']}"
    if isinstance(value, dict):
        for k in schema.get("required", []):
            if k not in value:
                return f"{path or '$'}: missing required key {k!r}"
        for k, sub in schema.get("properties", {}).items():
            if k in value:
                err = _validate(value[k], sub, f"{path}.{k}")
                if err:
                    return err
    if isinstance(value, (list, tuple)):
        if "minItems" in schema and len(value) < schema["minItems"]:
            return f"{path or '$'}: expected at least {schema['minItems']} items, got {len(value)}"
        if "items" in schema:
            for i, v in enumerate(value):
                err = _validate(v, schema["items"], f"{path}[{i}]")
                if err:
                    return err
    if isinstance(value, str) and "pattern" in schema and not re.search(schema["pattern"], value):
        return f"{path or '$'}: {short_repr(value, 40)} does not match {schema['pattern']!r}"
    return None


def schema(json_schema: dict[str, Any]) -> Check:
    """Validate against a small, dependency-free subset of JSON Schema.

    Supported keywords: ``type``, ``enum``, ``required``, ``properties``,
    ``items``, ``minItems`` and ``pattern``.
    """

    def fn(result: Any, error: BaseException | None) -> tuple[bool, str]:
        if error is not None:
            return False, _err_reason(error)
        err = _validate(result, json_schema, "")
        return (err is None, "matches schema" if err is None else err)

    return Check(fn, f"schema({short_repr(json_schema, 80)})")


def as_check(value: Any) -> Check | None:
    """Coerce a plain callable or ``None`` into a :class:`Check`."""
    if value is None or isinstance(value, Check):
        return value
    if callable(value):
        name = getattr(value, "__name__", "check")

        def fn(result: Any, error: BaseException | None) -> tuple[bool, str]:
            out = value(result, error)
            if isinstance(out, tuple):
                return bool(out[0]), str(out[1])
            return bool(out), name

        return Check(fn, f"custom({name})")
    raise TypeError(f"not a check: {value!r}")

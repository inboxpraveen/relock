"""Small internal helpers: ids, timestamps, JSON, tokenisation."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

_ID_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"  # Crockford-ish base32, lowercase


def _b32(n: int, width: int) -> str:
    out = []
    for _ in range(width):
        out.append(_ID_ALPHABET[n & 31])
        n >>= 5
    return "".join(reversed(out))


_id_lock = threading.Lock()
_id_last = [0, 0]  # (ms, low bits) of the previous id


def _reset_ids_after_fork() -> None:
    # A forked child must not continue the parent's sequence: both would compute the same
    # "previous + 1" within one millisecond. Fresh random low bits restore uniqueness.
    global _id_lock
    _id_lock = threading.Lock()
    _id_last[0] = _id_last[1] = 0


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_ids_after_fork)


def new_id(prefix: str) -> str:
    """Time-sortable, collision-resistant id such as ``itm_01jd3k...``.

    Monotonic within a process (ULID-style): ids created in the same millisecond, or
    while the clock steps backwards, increment the previous id's low bits instead of
    drawing new random ones. Sorting by id is therefore creation order, which keeps
    retrieval tie-breaks reproducible.
    """
    with _id_lock:
        ms = time.time_ns() // 1_000_000
        if ms <= _id_last[0]:
            ms, low = _id_last[0], _id_last[1] + 1
        else:
            low = int.from_bytes(os.urandom(7), "big") >> 7  # 49 random bits; headroom for increments
        _id_last[0], _id_last[1] = ms, low
    return f"{prefix}_{_b32(ms, 10)}{_b32(low, 10)}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def clean_text(text: str) -> str:
    """Text every database stores: NUL and lone surrogates (from bytes decoded with
    ``surrogateescape``, say) become U+FFFD."""
    if "\x00" in text:
        text = text.replace("\x00", "�")
    if not text.isascii():
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            text = "".join("�" if 0xD800 <= ord(c) <= 0xDFFF else c for c in text)
    return text


def _json_key(key: Any) -> str:
    if isinstance(key, str):
        return clean_text(key)
    if key is None or isinstance(key, (bool, int, float)):
        return json.dumps(key)  # what json itself would write: 1 -> "1", True -> "true"
    return clean_text(str(key))


def _json_safe(value: Any, depth: int = 0) -> Any:
    """``value`` with what JSON columns reject replaced: NaN and infinity become null, text is
    cleaned, and keys become strings (so mixed key types still sort)."""
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if depth > 64:  # deeper than any ledger payload; also stops a self-referencing structure
        return short_repr(value, 200)
    if isinstance(value, dict):
        return {_json_key(k): _json_safe(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, depth + 1) for v in value]
    return value


def dumps(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, default=_json_default)


def loads(value: Any) -> Any:
    """Parse JSON stored as text; pass through values a driver already decoded."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode()
    if isinstance(value, str):
        return json.loads(value)
    return value


def _json_default(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return _json_safe(o.to_dict())
    if isinstance(o, (set, frozenset)):
        try:
            return _json_safe(sorted(o))
        except TypeError:  # mixed element types
            return _json_safe(sorted(o, key=repr))
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, BaseException):
        return _json_safe({"type": type(o).__name__, "message": str(o)})
    return clean_text(repr(o))


def ts(value: Any) -> str | None:
    """Normalise a driver timestamp (str or datetime) to an ISO string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat(timespec="microseconds")
    return str(value)


_TOKEN = re.compile(r"[^\W_]+")  # letters and digits in any script; "_" splits, as in get_orders

STOPWORDS = frozenset(
    """a an and are as at be by for from has have how i in is it its of on or
    that the this to was were what when where which who will with you your
    do does should must can""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def approx_tokens(text: str) -> int:
    """Cheap token estimate (about 4 characters per token)."""
    return max(1, (len(text) + 3) // 4)


_AGE_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
MAX_AGE = timedelta(days=36525)  # 100 years; "now - age" stays a valid date


def parse_age(value: Any) -> timedelta:
    """``timedelta`` or a string such as ``"90s"``, ``"45m"``, ``"12h"``, ``"30d"``, ``"2w"``."""
    if isinstance(value, timedelta):
        age = value
    else:
        text = str(value).strip().lower()
        unit = text[-1:] if text[-1:] in _AGE_UNITS else ""
        try:
            number = float(text[: len(text) - len(unit)] if unit else text)
        except ValueError:
            raise ValueError(f"invalid age {value!r}: use a number with s, m, h, d or w, e.g. '30d'") from None
        if not unit:
            raise ValueError(f"invalid age {value!r}: give a unit (s, m, h, d or w), e.g. '30d'")
        if not math.isfinite(number):
            raise ValueError(f"invalid age {value!r}: not a finite number")
        if abs(number * _AGE_UNITS[unit]) > MAX_AGE.total_seconds():
            raise ValueError(f"invalid age {value!r}: at most 100 years (36525d)")
        age = timedelta(seconds=number * _AGE_UNITS[unit])
    if age < timedelta(0):
        raise ValueError("age must not be negative")
    if age > MAX_AGE:
        raise ValueError(f"invalid age {value!r}: at most 100 years (36525d)")
    return age


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def short_repr(value: Any, limit: int = 400) -> str:
    try:
        text = value if isinstance(value, str) else json.dumps(value, default=repr, ensure_ascii=False)
    except Exception:
        text = repr(value)
    return truncate(text, limit)

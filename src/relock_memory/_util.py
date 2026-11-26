"""Small internal helpers: ids, timestamps, JSON, tokenisation."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

_ID_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"  # Crockford-ish base32, lowercase


def _b32(n: int, width: int) -> str:
    out = []
    for _ in range(width):
        out.append(_ID_ALPHABET[n & 31])
        n >>= 5
    return "".join(reversed(out))


def new_id(prefix: str) -> str:
    """Time-sortable, collision-resistant id such as ``itm_01jd3k...``."""
    ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(8), "big")
    return f"{prefix}_{_b32(ms, 10)}{_b32(rand, 10)}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


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
        return o.to_dict()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, BaseException):
        return {"type": type(o).__name__, "message": str(o)}
    return repr(o)


def ts(value: Any) -> str | None:
    """Normalise a driver timestamp (str or datetime) to an ISO string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat(timespec="microseconds")
    return str(value)


_TOKEN = re.compile(r"[a-z0-9]+")

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

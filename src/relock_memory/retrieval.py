"""Retrieval over *active* item versions only.

Superseded versions are never candidates, so a corrected instruction and the
instruction it replaced can never be retrieved together. The default retriever
is a dependency-free BM25; pass ``embed=`` for hybrid lexical + vector search
with any embedding function, or implement :class:`Retriever` yourself.

Retrievers that set ``supports_candidates = True`` also accept
``rank(query, items, k, candidates=ids)``: ``items`` is the full active set (so an
index built on it stays warm) and only items whose id is in ``candidates`` are
returned. :class:`~relock_memory.memory.Memory` uses it for ``kinds`` / ``tags``
filters and falls back to passing the filtered list to other retrievers.
"""

from __future__ import annotations

import heapq
import math
import threading
from collections import Counter
from collections.abc import Callable, Collection, Sequence
from typing import Protocol, runtime_checkable

from ._util import tokenize
from .types import Item

EmbedFn = Callable[[list[str]], list[list[float]]]


@runtime_checkable
class Retriever(Protocol):
    def rank(self, query: str, items: Sequence[Item], k: int) -> list[tuple[Item, float]]: ...


class BM25Retriever:
    """Okapi BM25 over title, key, tags, tools and clause text.

    The index is inverted (only items sharing a query term are scored) and updated
    incrementally: each new item version is tokenised once, and items that left the
    set are removed, so a rewrite does not rebuild the index. Thread-safe.
    """

    supports_candidates = True

    def __init__(self, k1: float = 1.4, b: float = 0.75, title_boost: int = 2) -> None:
        self.k1, self.b, self.title_boost = k1, b, title_boost
        self._lock = threading.Lock()
        self._docs: dict[str, tuple[Item, int]] = {}  # item id -> (item, document length)
        self._terms: dict[str, tuple[str, ...]] = {}  # item id -> its distinct terms (to remove it)
        self._postings: dict[str, dict[str, int]] = {}  # term -> {item id: term frequency}
        self._total = 0
        self._synced: Sequence[Item] | None = None

    def _doc_tokens(self, item: Item) -> list[str]:
        toks = tokenize(item.search_text())
        toks += tokenize(item.title) * (self.title_boost - 1)
        return toks

    def _add(self, item: Item) -> None:
        tf = Counter(self._doc_tokens(item))
        length = sum(tf.values())
        self._docs[item.id] = (item, length)
        self._terms[item.id] = tuple(tf)
        self._total += length
        for term, f in tf.items():
            self._postings.setdefault(term, {})[item.id] = f

    def _remove(self, item_id: str) -> None:
        _item, length = self._docs.pop(item_id)
        self._total -= length
        for term in self._terms.pop(item_id):
            posting = self._postings[term]
            del posting[item_id]
            if not posting:
                del self._postings[term]

    def _sync(self, items: Sequence[Item]) -> None:
        # A tuple can't change, so the same tuple object means the same set (Memory passes its
        # cached tuple). Any other sequence is compared item by item.
        if isinstance(items, tuple) and items is self._synced:
            return
        current = {i.id: i for i in items}
        for item_id in [i for i, (it, _) in self._docs.items()
                        if i not in current or current[i].version != it.version]:
            self._remove(item_id)
        for item_id, item in current.items():
            doc = self._docs.get(item_id)
            if doc is None:
                self._add(item)
            elif doc[0] is not item:  # same version (same text); keep the caller's copy
                self._docs[item_id] = (item, doc[1])
        self._synced = items if isinstance(items, tuple) else None

    def rank(self, query: str, items: Sequence[Item], k: int, *,
             candidates: Collection[str] | None = None) -> list[tuple[Item, float]]:
        if not items:
            return []
        with self._lock:
            self._sync(items)
            n = len(self._docs)
            avg = (self._total / n) if n and self._total else 1.0
            k1, b = self.k1, self.b
            scores: dict[str, float] = {}
            for term in tokenize(query):
                posting = self._postings.get(term)
                if not posting:
                    continue
                df = len(posting)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                for item_id, f in posting.items():
                    if candidates is not None and item_id not in candidates:
                        continue
                    length = self._docs[item_id][1]
                    scores[item_id] = scores.get(item_id, 0.0) + idf * f * (k1 + 1) / (
                        f + k1 * (1 - b + b * length / avg))
            top = heapq.nsmallest(max(k, 0), scores.items(), key=lambda x: (-x[1], x[0]))
            return [(self._docs[item_id][0], s) for item_id, s in top]


class EmbeddingRetriever:
    """Cosine similarity with any ``embed(list[str]) -> list[vector]`` function.

    Vectors are cached per ``(item_id, version)`` in-process, so each version is
    embedded once; vectors of versions no longer active are dropped. Scoring is
    pure Python, which suits up to a few thousand items: beyond that, plug in a
    :class:`Retriever` backed by a vector index (pgvector, FAISS, a vector DB).
    """

    supports_candidates = True

    def __init__(self, embed: EmbedFn, min_score: float = 0.0) -> None:
        self.embed = embed
        self.min_score = min_score
        self._cache: dict[tuple[str, int], tuple[list[float], float]] = {}  # -> (vector, norm)
        self._lock = threading.Lock()

    def _vectors(self, items: Sequence[Item], *, prune: bool) -> list[tuple[list[float], float]]:
        with self._lock:
            missing = [i for i in items if (i.id, i.version) not in self._cache]
        if missing:  # embed outside the lock: it is usually a network call
            vecs = self.embed([i.search_text() for i in missing])
            with self._lock:
                for i, v in zip(missing, vecs):
                    vec = list(v)
                    self._cache[(i.id, i.version)] = (vec, math.sqrt(sum(x * x for x in vec)))
        with self._lock:
            out = [self._cache[(i.id, i.version)] for i in items]
            if prune and len(self._cache) > len(items):
                keep = {(i.id, i.version) for i in items}
                self._cache = {key: v for key, v in self._cache.items() if key in keep}
        return out

    @staticmethod
    def _cos(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def rank(self, query: str, items: Sequence[Item], k: int, *,
             candidates: Collection[str] | None = None) -> list[tuple[Item, float]]:
        pool = list(items) if candidates is None else [i for i in items if i.id in candidates]
        if not pool:
            return []
        qv = self.embed([query])[0]
        nq = math.sqrt(sum(x * x for x in qv))
        scored = []
        for item, (vec, nv) in zip(pool, self._vectors(pool, prune=candidates is None)):
            dot = sum(x * y for x, y in zip(qv, vec))
            scored.append((item, dot / (nq * nv) if nq and nv else 0.0))
        scored = [x for x in scored if x[1] > self.min_score]
        scored.sort(key=lambda x: (-x[1], x[0].id))
        return scored[:k]


class HybridRetriever:
    """Reciprocal-rank fusion of several retrievers (BM25 + embeddings by default)."""

    supports_candidates = True

    def __init__(self, *retrievers: Retriever, rrf_k: int = 60, depth: int = 50) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.depth = depth

    def rank(self, query: str, items: Sequence[Item], k: int, *,
             candidates: Collection[str] | None = None) -> list[tuple[Item, float]]:
        fused: dict[str, float] = {}
        by_id: dict[str, Item] = {}
        subset: list[Item] | None = None
        for r in self.retrievers:
            if candidates is None:
                ranked = r.rank(query, items, self.depth)
            elif getattr(r, "supports_candidates", False):
                ranked = r.rank(query, items, self.depth, candidates=candidates)  # type: ignore[call-arg]
            else:
                if subset is None:
                    subset = [i for i in items if i.id in candidates]
                ranked = r.rank(query, subset, self.depth)
            for rank, (item, _score) in enumerate(ranked):
                fused[item.id] = fused.get(item.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                by_id[item.id] = item
        out = sorted(((by_id[i], s) for i, s in fused.items()), key=lambda x: (-x[1], x[0].id))
        return out[:k]


def rank_items(retriever: Retriever, query: str, items: Sequence[Item], k: int,
               candidates: Collection[str] | None = None) -> list[tuple[Item, float]]:
    """Rank ``items`` (optionally only ``candidates``) with any retriever."""
    if candidates is None:
        return retriever.rank(query, items, k)
    if getattr(retriever, "supports_candidates", False):
        return retriever.rank(query, items, k, candidates=candidates)  # type: ignore[call-arg]
    return retriever.rank(query, [i for i in items if i.id in candidates], k)


def default_retriever(embed: EmbedFn | None = None) -> Retriever:
    if embed is None:
        return BM25Retriever()
    return HybridRetriever(BM25Retriever(), EmbeddingRetriever(embed))

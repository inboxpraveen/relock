"""Retrieval over *active* item versions only.

Superseded versions are never candidates, so a corrected instruction and the
instruction it replaced can never be retrieved together. The default retriever
is a dependency-free BM25; pass ``embed=`` for hybrid lexical + vector search
with any embedding function, or implement :class:`Retriever` yourself.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from ._util import tokenize
from .types import Item

EmbedFn = Callable[[list[str]], list[list[float]]]


@runtime_checkable
class Retriever(Protocol):
    def rank(self, query: str, items: Sequence[Item], k: int) -> list[tuple[Item, float]]: ...


class BM25Retriever:
    """Okapi BM25 over title, key, tags, tools and clause text. Rebuilt lazily per item set."""

    def __init__(self, k1: float = 1.4, b: float = 0.75, title_boost: int = 2) -> None:
        self.k1, self.b, self.title_boost = k1, b, title_boost
        self._sig: tuple | None = None
        self._docs: list[Counter[str]] = []
        self._lens: list[int] = []
        self._df: Counter[str] = Counter()
        self._avg = 1.0

    def _doc_tokens(self, item: Item) -> list[str]:
        toks = tokenize(item.search_text())
        toks += tokenize(item.title) * (self.title_boost - 1)
        return toks

    def _build(self, items: Sequence[Item]) -> None:
        sig = tuple((i.id, i.version) for i in items)
        if sig == self._sig:
            return
        self._docs = [Counter(self._doc_tokens(i)) for i in items]
        self._lens = [sum(d.values()) for d in self._docs]
        self._df = Counter()
        for d in self._docs:
            self._df.update(d.keys())
        self._avg = (sum(self._lens) / len(self._lens)) if self._lens else 1.0
        self._sig = sig

    def rank(self, query: str, items: Sequence[Item], k: int) -> list[tuple[Item, float]]:
        if not items:
            return []
        self._build(items)
        q = tokenize(query)
        n = len(items)
        scores: list[tuple[Item, float]] = []
        for item, doc, length in zip(items, self._docs, self._lens):
            s = 0.0
            for t in q:
                f = doc.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self._avg))
            if s > 0:
                scores.append((item, s))
        scores.sort(key=lambda x: (-x[1], x[0].id))
        return scores[:k]


class EmbeddingRetriever:
    """Cosine similarity with any ``embed(list[str]) -> list[vector]`` function.

    Vectors are cached per ``(item_id, version)`` in-process, so each version is
    embedded once.
    """

    def __init__(self, embed: EmbedFn, min_score: float = 0.0) -> None:
        self.embed = embed
        self.min_score = min_score
        self._cache: dict[tuple[str, int], list[float]] = {}

    def _vectors(self, items: Sequence[Item]) -> list[list[float]]:
        missing = [i for i in items if (i.id, i.version) not in self._cache]
        if missing:
            vecs = self.embed([i.search_text() for i in missing])
            for i, v in zip(missing, vecs):
                self._cache[(i.id, i.version)] = list(v)
        return [self._cache[(i.id, i.version)] for i in items]

    @staticmethod
    def _cos(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def rank(self, query: str, items: Sequence[Item], k: int) -> list[tuple[Item, float]]:
        if not items:
            return []
        qv = self.embed([query])[0]
        scored = [(i, self._cos(qv, v)) for i, v in zip(items, self._vectors(items))]
        scored = [x for x in scored if x[1] > self.min_score]
        scored.sort(key=lambda x: (-x[1], x[0].id))
        return scored[:k]


class HybridRetriever:
    """Reciprocal-rank fusion of several retrievers (BM25 + embeddings by default)."""

    def __init__(self, *retrievers: Retriever, rrf_k: int = 60, depth: int = 50) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.depth = depth

    def rank(self, query: str, items: Sequence[Item], k: int) -> list[tuple[Item, float]]:
        fused: dict[str, float] = {}
        by_id: dict[str, Item] = {}
        for r in self.retrievers:
            for rank, (item, _score) in enumerate(r.rank(query, items, self.depth)):
                fused[item.id] = fused.get(item.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                by_id[item.id] = item
        out = sorted(((by_id[i], s) for i, s in fused.items()), key=lambda x: (-x[1], x[0].id))
        return out[:k]


def default_retriever(embed: EmbedFn | None = None) -> Retriever:
    if embed is None:
        return BM25Retriever()
    return HybridRetriever(BM25Retriever(), EmbeddingRetriever(embed))

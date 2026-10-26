"""Reproducible pure-Python BM25 retrieval; no external index dependency."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from sidra.schema import Document


STOP = frozenset("the a an in for of where was is which what did does and from to on by".split())


def tokenize(text: str) -> list[str]:
    # Character tokens for CJK provide a basic fallback without jieba.
    parts = re.findall(r"[a-zA-Z0-9]+|[\u3400-\u9fff]", text.casefold())
    return [part for part in parts if part not in STOP]


@dataclass(frozen=True)
class Hit:
    doc_id: str
    title: str
    snippet: str
    score: float

    def as_dict(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title, "snippet": self.snippet, "score": round(self.score, 6)}


class BM25Index:
    def __init__(self, documents: Iterable[Document], *, k1: float = 1.2,
                 b: float = 0.75, title_boost: float = 0.0):
        self.documents = tuple(documents)
        if not self.documents:
            raise ValueError("Cannot search empty corpus")
        if not math.isfinite(title_boost) or title_boost < 0:
            raise ValueError("title_boost must be finite and non-negative")
        self.title_boost = title_boost
        self._title_tokens = [set(tokenize(d.title)) for d in self.documents]
        self._tok = [tokenize(d.title + " " + d.text) for d in self.documents]
        self._freq = [Counter(tokens) for tokens in self._tok]
        self._avg_len = sum(len(tokens) for tokens in self._tok) / len(self._tok)
        self._df = Counter(token for freq in self._freq for token in freq)
        self.k1, self.b = k1, b

    def search(self, query: str, *, k: int = 3, scope: str = "global") -> list[Hit]:
        query_terms = set(tokenize(query))
        if not query_terms or k < 1:
            return []
        scores = []
        n = len(self.documents)
        for doc, freq, toks, title_terms in zip(
                self.documents, self._freq, self._tok, self._title_tokens):
            if doc.scope not in {scope, "global"}:
                continue
            length_norm = 1 - self.b + self.b * len(toks) / max(self._avg_len, 1.0)
            score = 0.0
            for term in query_terms:
                tf = freq.get(term, 0)
                if tf:
                    idf = math.log1p((n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                    score += idf * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
                    # Query-title overlap can be amplified without any gold
                    # labels; factor is chosen on PILOT only, then frozen.
                    if term in title_terms:
                        score += self.title_boost * idf
            if score:
                scores.append((score, doc))
        scores.sort(key=lambda x: (-x[0], x[1].doc_id))
        return [Hit(d.doc_id, d.title, d.text[:500], score) for score, d in scores[:k]]

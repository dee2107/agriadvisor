"""Lightweight local retrieval for extension-style agricultural advice."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "what",
    "when",
    "with",
}


@dataclass(frozen=True)
class SearchResult:
    id: str
    title: str
    region: str
    crop: str
    text: str
    score: float


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


class LocalVectorIndex:
    """Small TF-IDF cosine index with no external services required."""

    def __init__(self, documents: Iterable[dict[str, str]]):
        self.documents = list(documents)
        self.doc_tokens = [
            tokenize(" ".join([doc["title"], doc["crop"], doc["region"], doc["text"]]))
            for doc in self.documents
        ]
        self.doc_term_counts = [Counter(tokens) for tokens in self.doc_tokens]
        self.idf = self._build_idf()
        self.doc_vectors = [self._tfidf(counts) for counts in self.doc_term_counts]
        self.doc_norms = [self._norm(vector) for vector in self.doc_vectors]

    def _build_idf(self) -> dict[str, float]:
        doc_count = len(self.doc_tokens)
        frequencies: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            frequencies.update(set(tokens))
        return {
            term: math.log((1 + doc_count) / (1 + freq)) + 1
            for term, freq in frequencies.items()
        }

    def _tfidf(self, counts: Counter[str]) -> dict[str, float]:
        total = sum(counts.values()) or 1
        return {
            term: (count / total) * self.idf.get(term, 1.0)
            for term, count in counts.items()
        }

    @staticmethod
    def _norm(vector: dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vector.values()))

    def search(self, query: str, *, top_k: int = 3) -> list[SearchResult]:
        query_vector = self._tfidf(Counter(tokenize(query)))
        query_norm = self._norm(query_vector)
        if query_norm == 0:
            return []

        scored: list[SearchResult] = []
        for doc, vector, doc_norm in zip(self.documents, self.doc_vectors, self.doc_norms):
            numerator = sum(query_vector.get(term, 0.0) * value for term, value in vector.items())
            score = numerator / (query_norm * doc_norm) if doc_norm else 0.0
            if score > 0:
                scored.append(SearchResult(score=score, **doc))

        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]


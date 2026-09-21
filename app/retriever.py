"""Retrieval over the knowledge base.

Deliberately dependency-free: a hand-rolled TF-IDF + cosine similarity is
plenty for a 14-document corpus, keeps the eval suite runnable with zero
`pip install` surprises, and is easy for a reviewer to read top to bottom.

Authority handling is the part that actually matters for this assignment:
scoring by text similarity alone would happily let the superseded or
internal-only documents outrank the current policy just because they share
vocabulary. So retrieval and "what counts as an authoritative citation" are
kept as two separate steps.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.knowledge_base import Chunk

_WORD_RE = re.compile(r"[a-z0-9]+")

# Crude suffix-stripping stemmer. Without this, "ship" and "shipping" are
# unrelated tokens to a bag-of-words model, which meant a question like
# "Can you ship to Germany?" scored *worse* against the International
# Shipping document than several unrelated documents -- a real retrieval
# bug found via the eval suite (see README bug diary). This is not a real
# stemmer (no Porter/Snowball rules, no dictionary) and will mis-stem some
# words; it's a deliberately small fix scoped to this corpus's vocabulary.
_SUFFIXES = ("ies", "ing", "edly", "ed", "es", "s")


def _stem(word: str) -> str:
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            if suf == "ies":
                return word[:-3] + "y"
            stem = word[: -len(suf)]
            # Undo doubled-consonant-before-suffix (shipping -> shipp -> ship,
            # shipped -> shipp -> ship)
            if suf in ("ing", "ed") and len(stem) > 3 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            return stem
    return word


def _tokenize(text: str) -> list[str]:
    return [_stem(w) for w in _WORD_RE.findall(text.lower())]


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float

    @property
    def is_authoritative(self) -> bool:
        """A chunk may be cited as customer-facing policy authority only when
        it is active, officially authored, and written for customers.
        Superseded, draft, and internal-only documents can still be
        *retrieved* (we need to recognize them, e.g. to explain why they
        don't apply) but must never be the basis for an answer."""
        meta = self.chunk.metadata
        return (
            meta.get("status") == "active"
            and meta.get("policy_authority") == "official"
            and meta.get("audience", "customer") == "customer"
        )


# Known, designed conflict between two ACTIVE official documents that a
# pure similarity/authority model would not otherwise catch, since neither
# document is superseded or non-authoritative -- they simply disagree.
# This is intentionally a small, explicit table rather than an automatic
# contradiction detector; see README "Known limitations".
_KNOWN_ACTIVE_CONFLICTS: dict[str, tuple[str, str]] = {
    "breeze_tumbler_cleaning": ("11-product-care.md", "12-breeze-tumbler-product-card.md"),
}


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._doc_freq: Counter[str] = Counter()
        self._chunk_tf: list[Counter[str]] = []
        for chunk in chunks:
            tokens = _tokenize(chunk.text + " " + chunk.heading)
            tf = Counter(tokens)
            self._chunk_tf.append(tf)
            self._doc_freq.update(set(tokens))
        self._n_docs = max(len(chunks), 1)

    def _idf(self, term: str) -> float:
        df = self._doc_freq.get(term, 0)
        return math.log((self._n_docs + 1) / (df + 1)) + 1.0

    def _vector(self, tf: Counter[str]) -> dict[str, float]:
        return {term: count * self._idf(term) for term, count in tf.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[t] * b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in a.values())) or 1.0
        norm_b = math.sqrt(sum(v * v for v in b.values())) or 1.0
        return dot / (norm_a * norm_b)

    def retrieve(self, query: str, top_k: int = 6) -> list[ScoredChunk]:
        query_tf = Counter(_tokenize(query))
        query_vec = self._vector(query_tf)
        scored = []
        for chunk, tf in zip(self.chunks, self._chunk_tf):
            chunk_vec = self._vector(tf)
            score = self._cosine(query_vec, chunk_vec)
            if score > 0:
                scored.append(ScoredChunk(chunk=chunk, score=score))
        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:top_k]

    def detect_known_conflict(self, query: str, results: list[ScoredChunk]) -> tuple[str, str] | None:
        """Return a pair of conflicting doc_ids if this query is actually
        about the known active-vs-active conflict topic, else None.

        Bug found during eval: an earlier version treated "both conflict
        docs happen to appear somewhere in the top-k" as sufficient, which
        false-positived on unrelated queries (e.g. a vegan-materials
        question retrieved both docs incidentally via shared bag/fabric
        vocabulary and got misclassified as a tumbler conflict). Detection
        is keyword-topic-based instead: it must actually be about the
        tumbler and about washing/cleaning it.
        """
        tokens = set(_tokenize(query))
        is_about_tumbler = bool({"tumbler", "breeze"} & tokens)
        is_about_cleaning = bool({"dishwasher", "wash", "clean"} & tokens)
        if is_about_tumbler and is_about_cleaning:
            return _KNOWN_ACTIVE_CONFLICTS["breeze_tumbler_cleaning"]
        return None

    def chunks_for_doc(self, doc_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.doc_id == doc_id]

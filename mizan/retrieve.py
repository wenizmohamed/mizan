"""Lightweight TF-IDF retrieval with Arabic normalization.

This retriever is the dependency-free baseline used by the demo and the test
suite. The pipeline accepts any object exposing ``search(query, k)`` so a
dense or hybrid retriever drops in without code changes.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TASHKEEL = re.compile(r"[ً-ْٰ]")
_TOKEN = re.compile(r"[\w؀-ۿ]+")


def normalize_arabic(text: str) -> str:
    """Normalize Arabic orthography: strip diacritics, unify alef/teh/yeh forms."""
    text = _TASHKEEL.sub("", text)
    text = text.replace("آ", "ا").replace("أ", "ا").replace("إ", "ا")
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    return text


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalize_arabic(text).lower())


@dataclass
class Passage:
    pid: str
    text: str
    score: float = 0.0


class TfidfRetriever:
    """Cosine similarity over TF-IDF bags of normalized tokens."""

    def __init__(self, passages: dict[str, str]) -> None:
        self._texts = dict(passages)
        self._tf: dict[str, Counter] = {pid: Counter(tokenize(t)) for pid, t in passages.items()}
        df: Counter = Counter()
        for tf in self._tf.values():
            df.update(tf.keys())
        n = max(len(passages), 1)
        self._idf = {term: math.log((1 + n) / (1 + d)) + 1.0 for term, d in df.items()}

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "TfidfRetriever":
        passages: dict[str, str] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            passages[row["id"]] = row["text"]
        return cls(passages)

    def _vector(self, tf: Counter) -> dict[str, float]:
        return {t: c * self._idf.get(t, 1.0) for t, c in tf.items()}

    def search(self, query: str, k: int = 4) -> list[Passage]:
        qv = self._vector(Counter(tokenize(query)))
        qnorm = math.sqrt(sum(v * v for v in qv.values())) or 1.0
        results: list[Passage] = []
        for pid, tf in self._tf.items():
            pv = self._vector(tf)
            dot = sum(qv[t] * pv[t] for t in qv.keys() & pv.keys())
            pnorm = math.sqrt(sum(v * v for v in pv.values())) or 1.0
            score = dot / (qnorm * pnorm)
            if score > 0.0:
                results.append(Passage(pid=pid, text=self._texts[pid], score=score))
        results.sort(key=lambda p: p.score, reverse=True)
        return results[:k]

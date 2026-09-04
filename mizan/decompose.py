"""Atomic claim decomposition for Arabic and English answers.

NLI cross-encoders are trained on short sentence pairs, so decomposing the
answer into atomic claims is a precondition for valid verification, not an
optimization. Production decomposition uses an instruction-tuned LLM through
the ``llm`` callable; the built-in heuristic is a deterministic fallback and
the unit-test baseline.
"""

from __future__ import annotations

import re
from typing import Callable

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟…])\s+|\n+|(?<=؛)\s*")
_NUMERIC = re.compile(r"\d")
_CLAUSE_SPLIT = re.compile(r"،\s*")


def _split_numeric_clauses(sentence: str) -> list[str]:
    """Split a sentence on Arabic commas when 2+ clauses carry distinct numbers.

    A sentence packing several quantitative facts must be verified fact by
    fact, otherwise a single wrong number hides inside a supported sentence.
    """
    clauses = [c.strip() for c in _CLAUSE_SPLIT.split(sentence) if c.strip()]
    numeric = [c for c in clauses if _NUMERIC.search(c)]
    if len(clauses) > 1 and len(numeric) >= 2:
        return clauses
    return [sentence]


def decompose(text: str, llm: Callable[[str], list[str]] | None = None) -> list[str]:
    """Return atomic claims for ``text``.

    When ``llm`` is provided it receives the full text and must return the
    claim list; its output is stripped and de-duplicated but otherwise trusted.
    Without it, sentences are split on terminal punctuation (Latin and Arabic)
    and multi-number sentences are further split into clauses.
    """
    if llm is not None:
        seen: set[str] = set()
        claims = []
        for c in llm(text):
            c = c.strip()
            if c and c not in seen:
                seen.add(c)
                claims.append(c)
        return claims

    claims: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        claims.extend(_split_numeric_clauses(sentence))
    return claims

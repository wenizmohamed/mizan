"""Disentangling hallucination from retrieval failure.

A claim without supporting evidence is not automatically a hallucination:
the retriever may simply have missed the evidence. Mizan re-verifies every
``no_evidence`` claim under maximal retrieval conditions (larger k, query
reformulation, sub-question decomposition, wider index sweep). A claim that
flips to supported is logged as a retrieval failure; a claim that stays
unsupported under near-exhaustive coverage is a likely hallucination. The two
signals are measured and reported separately.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Sequence

from mizan.verify import VERDICT_CONTRADICTED, VERDICT_SUPPORTED, ClaimVerdict


class Triage(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    RETRIEVAL_FAILURE = "retrieval_failure"
    LIKELY_HALLUCINATION = "likely_hallucination"


def disentangle(
    verdict: ClaimVerdict,
    expand_retrieval: Callable[[str], Sequence[str]],
    verifier,
) -> tuple[Triage, ClaimVerdict]:
    """Classify a claim verdict into its error source.

    ``expand_retrieval`` receives the claim text and must return the widened
    evidence pool. ``verifier`` is any object with ``verify(claim, evidence)``.
    Returns the triage label together with the verdict that justified it (the
    expanded-round verdict when a second round ran).
    """
    if verdict.verdict == VERDICT_SUPPORTED:
        return (Triage.SUPPORTED, verdict)
    if verdict.verdict == VERDICT_CONTRADICTED:
        return (Triage.CONTRADICTED, verdict)

    expanded = list(expand_retrieval(verdict.claim))
    second = verifier.verify(verdict.claim, expanded)
    if second.verdict == VERDICT_SUPPORTED:
        return (Triage.RETRIEVAL_FAILURE, second)
    if second.verdict == VERDICT_CONTRADICTED:
        return (Triage.CONTRADICTED, second)
    return (Triage.LIKELY_HALLUCINATION, second)

"""Answer-level aggregation of claim-level groundedness scores.

The displayed answer score is a weighted mean of per-claim entailment scores.
Weights default to relative claim length in tokens, optionally scaled by a
centrality factor supplied by the caller. The weighting scheme is a design
decision and is reported alongside the score, never hidden.
"""

from __future__ import annotations

from typing import Sequence


def claim_weights(claims: Sequence[str], centrality: Sequence[float] | None = None) -> list[float]:
    """Length-proportional weights, optionally scaled by per-claim centrality.

    Centrality values, when given, must be positive multipliers with the same
    length as ``claims``. Returned weights sum to 1.0.
    """
    if not claims:
        return []
    lengths = [max(len(c.split()), 1) for c in claims]
    if centrality is not None:
        if len(centrality) != len(claims):
            raise ValueError("centrality must match claims length")
        lengths = [l * max(c, 0.0) for l, c in zip(lengths, centrality)]
    total = sum(lengths)
    if total <= 0:
        return [1.0 / len(claims)] * len(claims)
    return [l / total for l in lengths]


def answer_groundedness(scores: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Weighted mean of claim scores in [0, 1].

    With no weights this is the simple mean: for the reference demo scores
    (0.96, 0.91, 0.68, 0.19) that is 0.685. Weighted variants shift the value
    and the weights used must be reported with the result.
    """
    if not scores:
        return 0.0
    if weights is None:
        return float(sum(scores) / len(scores))
    if len(weights) != len(scores):
        raise ValueError("weights must match scores length")
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return float(sum(s * w for s, w in zip(scores, weights)) / total)

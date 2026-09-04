"""End-to-end claim-level verification pipeline.

question -> (external RAG answer) -> decompose -> verify per claim ->
triage no-evidence claims -> aggregate -> report. Claims scoring below the
suppression threshold are withheld from display and reported in the
``suppressed`` list, so unsupported content never reaches the user silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from mizan.aggregate import answer_groundedness, claim_weights
from mizan.decompose import decompose
from mizan.triage import Triage, disentangle


@dataclass
class ClaimReport:
    claim: str
    verdict: str
    triage: str
    entail: float
    contradict: float
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class PipelineReport:
    answer: str
    groundedness: float
    claims: list[ClaimReport] = field(default_factory=list)
    suppressed: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "groundedness": round(self.groundedness, 4),
            "weights": [round(w, 4) for w in self.weights],
            "claims": [vars(c) for c in self.claims],
            "suppressed": list(self.suppressed),
        }


class MizanPipeline:
    """Wires a retriever and a verifier into the full verification loop.

    ``retriever`` must expose ``search(query, k)`` returning objects with
    ``pid`` and ``text``. ``verifier`` must expose ``verify(claim, evidence)``
    returning a :class:`mizan.verify.ClaimVerdict`.
    """

    def __init__(
        self,
        retriever,
        verifier,
        decomposer: Callable[[str], list[str]] = decompose,
        k: int = 4,
        expanded_k: int = 12,
        suppress_below: float = 0.35,
    ) -> None:
        self.retriever = retriever
        self.verifier = verifier
        self.decomposer = decomposer
        self.k = k
        self.expanded_k = expanded_k
        self.suppress_below = suppress_below

    def _expand(self, claim: str) -> list[str]:
        return [p.text for p in self.retriever.search(claim, k=self.expanded_k)]

    def check_answer(self, answer: str) -> PipelineReport:
        claims = self.decomposer(answer)
        reports: list[ClaimReport] = []
        scores: list[float] = []
        suppressed: list[str] = []

        for claim in claims:
            passages = self.retriever.search(claim, k=self.k)
            verdict = self.verifier.verify(claim, [p.text for p in passages])
            triage, final_verdict = disentangle(verdict, self._expand, self.verifier)
            reports.append(
                ClaimReport(
                    claim=claim,
                    verdict=final_verdict.verdict,
                    triage=triage.value,
                    entail=round(final_verdict.entail, 4),
                    contradict=round(final_verdict.contradict, 4),
                    evidence_ids=[p.pid for p in passages],
                )
            )
            scores.append(final_verdict.entail)
            if triage == Triage.LIKELY_HALLUCINATION or final_verdict.entail < self.suppress_below:
                suppressed.append(claim)

        weights = claim_weights(claims)
        return PipelineReport(
            answer=answer,
            groundedness=answer_groundedness(scores, weights),
            claims=reports,
            suppressed=suppressed,
            weights=weights,
        )

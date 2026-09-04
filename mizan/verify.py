"""Claim verification with a multilingual NLI cross-encoder.

Default model: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7,
trained on 2.7M NLI pairs across 100+ languages including Arabic, which makes
Arabic-claim vs English-evidence pairs verifiable without a translation step.
Transformers and torch are imported lazily so the rest of the package works
without them installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

VERDICT_SUPPORTED = "supported"
VERDICT_CONTRADICTED = "contradicted"
VERDICT_NO_EVIDENCE = "no_evidence"


@dataclass
class ClaimVerdict:
    claim: str
    verdict: str
    entail: float
    contradict: float
    best_evidence: int = -1
    per_evidence: list[tuple[float, float, float]] = field(default_factory=list)


class NLIVerifier:
    """Cross-encodes (evidence, claim) pairs and issues a three-way verdict.

    The claim score is the maximum entailment probability across evidence
    passages. A contradiction verdict requires an actual refuting passage;
    absence of both signals is ``no_evidence``, which is a retrieval question
    before it is a hallucination verdict (see :mod:`mizan.triage`).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        sup_threshold: float = 0.7,
        con_threshold: float = 0.6,
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.sup_threshold = sup_threshold
        self.con_threshold = con_threshold
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        id2label = {i: label.lower() for i, label in self.model.config.id2label.items()}
        self._entail_idx = next(i for i, l in id2label.items() if "entail" in l)
        self._contra_idx = next(i for i, l in id2label.items() if "contra" in l)

    def score_pair(self, premise: str, hypothesis: str) -> tuple[float, float, float]:
        """Return (entail, neutral, contradiction) probabilities for one pair."""
        torch = self._torch
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(**inputs).logits[0], dim=-1).tolist()
        entail = probs[self._entail_idx]
        contra = probs[self._contra_idx]
        neutral = max(1.0 - entail - contra, 0.0)
        return (entail, neutral, contra)

    def verify(self, claim: str, evidence: Sequence[str]) -> ClaimVerdict:
        per_evidence = [self.score_pair(passage, claim) for passage in evidence]
        if not per_evidence:
            return ClaimVerdict(claim, VERDICT_NO_EVIDENCE, 0.0, 0.0)
        entails = [s[0] for s in per_evidence]
        contras = [s[2] for s in per_evidence]
        best = int(max(range(len(entails)), key=entails.__getitem__))
        entail, contra = max(entails), max(contras)
        if entail >= self.sup_threshold:
            verdict = VERDICT_SUPPORTED
        elif contra >= self.con_threshold:
            verdict = VERDICT_CONTRADICTED
        else:
            verdict = VERDICT_NO_EVIDENCE
        return ClaimVerdict(claim, verdict, entail, contra, best, per_evidence)

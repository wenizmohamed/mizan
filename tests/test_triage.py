import unittest

from mizan.triage import Triage, disentangle
from mizan.verify import (
    VERDICT_CONTRADICTED,
    VERDICT_NO_EVIDENCE,
    VERDICT_SUPPORTED,
    ClaimVerdict,
)


class FakeVerifier:
    """Deterministic verifier: supported only when evidence pool contains the gold passage."""

    def __init__(self, gold="gold-evidence"):
        self.gold = gold

    def verify(self, claim, evidence):
        if any(self.gold in e for e in evidence):
            return ClaimVerdict(claim, VERDICT_SUPPORTED, 0.93, 0.02)
        return ClaimVerdict(claim, VERDICT_NO_EVIDENCE, 0.2, 0.1)


class TestTriage(unittest.TestCase):
    def test_supported_passes_through(self):
        v = ClaimVerdict("c", VERDICT_SUPPORTED, 0.9, 0.02)
        triage, _ = disentangle(v, lambda c: [], FakeVerifier())
        self.assertEqual(triage, Triage.SUPPORTED)

    def test_contradicted_passes_through(self):
        v = ClaimVerdict("c", VERDICT_CONTRADICTED, 0.1, 0.8)
        triage, _ = disentangle(v, lambda c: [], FakeVerifier())
        self.assertEqual(triage, Triage.CONTRADICTED)

    def test_retrieval_failure_detected_on_expansion(self):
        v = ClaimVerdict("c", VERDICT_NO_EVIDENCE, 0.2, 0.1)
        triage, second = disentangle(v, lambda c: ["gold-evidence here"], FakeVerifier())
        self.assertEqual(triage, Triage.RETRIEVAL_FAILURE)
        self.assertEqual(second.verdict, VERDICT_SUPPORTED)

    def test_likely_hallucination_when_expansion_finds_nothing(self):
        v = ClaimVerdict("c", VERDICT_NO_EVIDENCE, 0.2, 0.1)
        triage, _ = disentangle(v, lambda c: ["irrelevant text"], FakeVerifier())
        self.assertEqual(triage, Triage.LIKELY_HALLUCINATION)


if __name__ == "__main__":
    unittest.main()

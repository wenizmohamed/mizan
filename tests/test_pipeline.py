import unittest

from mizan.pipeline import MizanPipeline
from mizan.retrieve import TfidfRetriever
from mizan.verify import VERDICT_NO_EVIDENCE, VERDICT_SUPPORTED, ClaimVerdict


class KeywordVerifier:
    """Supported when claim and some evidence passage share a rare keyword."""

    def verify(self, claim, evidence):
        claim_tokens = set(claim.lower().split())
        for passage in evidence:
            overlap = claim_tokens & set(passage.lower().split())
            if len(overlap) >= 3:
                return ClaimVerdict(claim, VERDICT_SUPPORTED, 0.92, 0.02)
        return ClaimVerdict(claim, VERDICT_NO_EVIDENCE, 0.15, 0.05)


class TestPipeline(unittest.TestCase):
    def test_end_to_end_report(self):
        retriever = TfidfRetriever(
            {
                "S1": "a 2021 meta-analysis of randomized trials found coffee lowers diabetes risk",
                "S2": "the reduction in diabetes risk was 39 percent among daily coffee drinkers",
            }
        )
        pipeline = MizanPipeline(retriever, KeywordVerifier(), k=2, expanded_k=2)
        report = pipeline.check_answer(
            "coffee lowers diabetes risk according to a 2021 meta-analysis of randomized trials. "
            "martians consume it before every landing."
        )
        self.assertEqual(len(report.claims), 2)
        self.assertEqual(report.claims[0].verdict, VERDICT_SUPPORTED)
        self.assertEqual(report.claims[1].triage, "likely_hallucination")
        self.assertIn("martians consume it before every landing.", report.suppressed)
        self.assertTrue(0.0 < report.groundedness < 1.0)
        payload = report.to_dict()
        self.assertIn("groundedness", payload)
        self.assertEqual(len(payload["claims"]), 2)


if __name__ == "__main__":
    unittest.main()

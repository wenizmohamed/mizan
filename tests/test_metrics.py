import unittest

import numpy as np

from mizan.metrics import balanced_accuracy, risk_coverage_auc, risk_coverage_curve


class TestMetrics(unittest.TestCase):
    def test_balanced_accuracy_perfect(self):
        y = ["supported", "contradicted", "no_evidence", "supported"]
        self.assertEqual(balanced_accuracy(y, y), 1.0)

    def test_balanced_accuracy_handles_imbalance(self):
        y_true = ["supported"] * 9 + ["contradicted"]
        y_pred = ["supported"] * 10
        self.assertAlmostEqual(balanced_accuracy(y_true, y_pred), 0.5, places=6)

    def test_risk_coverage_monotone_case(self):
        conf = [0.99, 0.95, 0.90, 0.40, 0.30]
        correct = [1, 1, 1, 0, 0]
        coverage, risk = risk_coverage_curve(conf, correct)
        self.assertAlmostEqual(float(risk[2]), 0.0, places=6)
        self.assertAlmostEqual(float(risk[-1]), 0.4, places=6)
        self.assertAlmostEqual(float(coverage[-1]), 1.0, places=6)

    def test_confident_and_correct_beats_confident_and_wrong(self):
        good = risk_coverage_auc([0.9, 0.8, 0.3, 0.2], [1, 1, 0, 0])
        bad = risk_coverage_auc([0.9, 0.8, 0.3, 0.2], [0, 0, 1, 1])
        self.assertLess(good, bad)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            risk_coverage_curve([0.5], [1, 0])


class TestEvalHarness(unittest.TestCase):
    def test_evaluate_with_fake_verifier(self):
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "run_eval", Path(__file__).resolve().parents[1] / "eval" / "run_eval.py"
        )
        run_eval = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_eval)

        from mizan.verify import ClaimVerdict

        class OracleVerifier:
            def __init__(self, gold):
                self._labels = {row["claim"]: row["label"] for row in gold}

            def verify(self, claim, evidence):
                label = self._labels[claim]
                entail = 0.9 if label == "supported" else 0.1
                contra = 0.9 if label == "contradicted" else 0.05
                return ClaimVerdict(claim, label, entail, contra)

        gold = run_eval.load_gold(Path(__file__).resolve().parents[1] / "eval" / "gold_seed.jsonl")
        results = run_eval.evaluate(OracleVerifier(gold), gold)
        self.assertEqual(results["balanced_accuracy"], 1.0)
        self.assertEqual(results["n"], len(gold))
        self.assertLessEqual(results["risk_coverage_auc"], 0.05)


if __name__ == "__main__":
    unittest.main()

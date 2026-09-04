import unittest

from mizan.aggregate import answer_groundedness, claim_weights


class TestAggregate(unittest.TestCase):
    def test_demo_scores_simple_mean(self):
        scores = [0.96, 0.91, 0.68, 0.19]
        self.assertAlmostEqual(answer_groundedness(scores), 0.685, places=6)

    def test_weighted_mean_shifts_score(self):
        scores = [0.96, 0.91, 0.68, 0.19]
        weights = [0.3, 0.3, 0.3, 0.1]
        self.assertAlmostEqual(answer_groundedness(scores, weights), 0.784, places=6)

    def test_claim_weights_sum_to_one(self):
        claims = ["short claim", "a much longer claim with many more tokens inside"]
        weights = claim_weights(claims)
        self.assertAlmostEqual(sum(weights), 1.0, places=9)
        self.assertGreater(weights[1], weights[0])

    def test_centrality_scaling(self):
        claims = ["one two three", "four five six"]
        weights = claim_weights(claims, centrality=[2.0, 1.0])
        self.assertGreater(weights[0], weights[1])

    def test_empty(self):
        self.assertEqual(answer_groundedness([]), 0.0)
        self.assertEqual(claim_weights([]), [])

    def test_mismatched_weights_raise(self):
        with self.assertRaises(ValueError):
            answer_groundedness([0.5], [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()

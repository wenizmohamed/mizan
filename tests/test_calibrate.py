import unittest

import numpy as np

from mizan.calibrate import TemperatureScaler, expected_calibration_error


class TestCalibration(unittest.TestCase):
    def test_perfect_calibration_low_ece(self):
        rng = np.random.default_rng(7)
        probs = rng.uniform(0.05, 0.95, size=5000)
        labels = (rng.uniform(size=5000) < probs).astype(float)
        self.assertLess(expected_calibration_error(probs, labels), 0.05)

    def test_overconfident_scores_high_ece(self):
        probs = np.full(1000, 0.95)
        labels = np.concatenate([np.ones(600), np.zeros(400)])
        self.assertGreater(expected_calibration_error(probs, labels), 0.3)

    def test_temperature_reduces_overconfidence(self):
        rng = np.random.default_rng(11)
        true_p = rng.uniform(0.2, 0.8, size=4000)
        labels = (rng.uniform(size=4000) < true_p).astype(float)
        z = np.log(true_p / (1 - true_p))
        overconfident = 1.0 / (1.0 + np.exp(-z * 3.0))

        scaler = TemperatureScaler()
        t = scaler.fit(overconfident, labels)
        self.assertGreater(t, 1.5)

        before = expected_calibration_error(overconfident, labels)
        after = expected_calibration_error(scaler.transform(overconfident), labels)
        self.assertLess(after, before)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            expected_calibration_error([0.5, 0.6], [1.0])


if __name__ == "__main__":
    unittest.main()

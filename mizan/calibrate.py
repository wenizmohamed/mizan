"""Calibration of raw NLI confidence scores.

Raw softmax outputs of NLI cross-encoders are systematically miscalibrated.
Mizan treats calibration as a first-class, separately evaluated step:
temperature scaling fitted on a hand-labeled dev set, quality measured with
Expected Calibration Error before and after.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class TemperatureScaler:
    """Single-parameter temperature scaling for binary supportedness scores.

    ``fit`` selects the temperature minimizing negative log likelihood on a
    labeled dev set (label 1 = claim actually supported by the evidence).
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, probs, labels, grid_size: int = 400) -> float:
        p = np.asarray(probs, dtype=float)
        y = np.asarray(labels, dtype=float)
        if p.shape != y.shape:
            raise ValueError("probs and labels must have the same shape")
        z = _logit(p)
        best_t, best_nll = 1.0, np.inf
        for t in np.geomspace(0.05, 10.0, grid_size):
            q = np.clip(_sigmoid(z / t), _EPS, 1.0 - _EPS)
            nll = float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))
            if nll < best_nll:
                best_t, best_nll = float(t), nll
        self.temperature = best_t
        return best_t

    def transform(self, probs) -> np.ndarray:
        p = np.asarray(probs, dtype=float)
        return _sigmoid(_logit(p) / self.temperature)


def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    """Standard ECE with equal-width confidence bins."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.shape != y.shape:
        raise ValueError("probs and labels must have the same shape")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)

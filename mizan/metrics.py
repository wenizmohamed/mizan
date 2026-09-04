"""Evaluation metrics: balanced accuracy and risk-coverage for selective verification.

Risk-coverage treats claim suppression as selective prediction: at a confidence
threshold tau the system answers only claims with confidence >= tau (coverage)
and the risk is the error rate among answered claims. The area under the
risk-coverage curve summarizes the suppression tradeoff in one number
(lower is better).
"""

from __future__ import annotations

import numpy as np


def balanced_accuracy(y_true, y_pred) -> float:
    """Mean of per-class recall over the classes present in ``y_true``."""
    t = np.asarray(y_true)
    p = np.asarray(y_pred)
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    recalls = []
    for cls in np.unique(t):
        mask = t == cls
        recalls.append(float(np.mean(p[mask] == cls)))
    return float(np.mean(recalls))


def risk_coverage_curve(confidences, correct) -> tuple[np.ndarray, np.ndarray]:
    """Return (coverage, risk) sweeping the confidence threshold.

    ``correct`` is 1 when the system's verdict for the claim was right.
    Claims are sorted by descending confidence; at each prefix the coverage is
    the fraction answered and the risk is the error rate inside that prefix.
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    if conf.shape != corr.shape:
        raise ValueError("confidences and correct must have the same shape")
    order = np.argsort(-conf)
    corr_sorted = corr[order]
    n = len(conf)
    answered = np.arange(1, n + 1)
    errors = np.cumsum(1.0 - corr_sorted)
    coverage = answered / n
    risk = errors / answered
    return coverage, risk


def risk_coverage_auc(confidences, correct) -> float:
    """Trapezoidal area under the risk-coverage curve (lower is better)."""
    coverage, risk = risk_coverage_curve(confidences, correct)
    if len(coverage) == 1:
        return float(risk[0])
    return float(np.trapezoid(risk, coverage) / (coverage[-1] - coverage[0]))

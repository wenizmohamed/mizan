"""Mizan (ميزان): claim-level groundedness verification for Arabic and cross-lingual RAG.

The pipeline decomposes a generated answer into atomic claims, verifies each claim
against retrieved evidence with a multilingual NLI cross-encoder, calibrates the
resulting scores, disentangles hallucination from retrieval failure, and aggregates
claim scores into an answer-level groundedness score.
"""

from mizan.aggregate import answer_groundedness, claim_weights
from mizan.calibrate import TemperatureScaler, expected_calibration_error
from mizan.decompose import decompose
from mizan.pipeline import MizanPipeline, PipelineReport
from mizan.retrieve import TfidfRetriever
from mizan.triage import Triage, disentangle

__version__ = "0.1.0"

__all__ = [
    "MizanPipeline",
    "PipelineReport",
    "TfidfRetriever",
    "TemperatureScaler",
    "Triage",
    "answer_groundedness",
    "claim_weights",
    "decompose",
    "disentangle",
    "expected_calibration_error",
]

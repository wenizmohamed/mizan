"""Run the Mizan evaluation harness on a gold set of labeled claim-evidence pairs.

Gold rows are JSONL: {"claim": str, "evidence": [str, ...], "label": one of
supported | contradicted | no_evidence}. The harness reports balanced accuracy
over the three-way verdicts, ECE of the supportedness confidence, and the
risk-coverage AUC of selective suppression. The bundled ``gold_seed.jsonl`` is
a 12-pair cross-lingual seed set; the target protocol is 200-300 pairs drawn
from the project dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mizan.calibrate import expected_calibration_error
from mizan.metrics import balanced_accuracy, risk_coverage_auc


def load_gold(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def evaluate(verifier, gold: list[dict]) -> dict:
    """Score ``verifier`` (any object with verify(claim, evidence)) on gold rows."""
    y_true, y_pred, conf, correct = [], [], [], []
    for row in gold:
        verdict = verifier.verify(row["claim"], row["evidence"])
        y_true.append(row["label"])
        y_pred.append(verdict.verdict)
        conf.append(verdict.entail if verdict.verdict == "supported" else 1.0 - verdict.entail)
        correct.append(1.0 if verdict.verdict == row["label"] else 0.0)

    support_conf = [c if p == "supported" else 1.0 - c for c, p in zip(conf, y_pred)]
    support_true = [1.0 if t == "supported" else 0.0 for t in y_true]

    return {
        "n": len(gold),
        "balanced_accuracy": round(balanced_accuracy(y_true, y_pred), 4),
        "ece_supportedness": round(expected_calibration_error(support_conf, support_true, n_bins=5), 4),
        "risk_coverage_auc": round(risk_coverage_auc(conf, correct), 4),
        "per_label": {
            label: {
                "n": sum(1 for t in y_true if t == label),
                "recall": round(
                    sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
                    / max(sum(1 for t in y_true if t == label), 1),
                    4,
                ),
            }
            for label in ("supported", "contradicted", "no_evidence")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=str(Path(__file__).parent / "gold_seed.jsonl"))
    args = parser.parse_args()

    try:
        from mizan.verify import NLIVerifier
    except Exception:
        print("transformers/torch not available: install requirements.txt first.")
        return 1

    gold = load_gold(args.gold)
    print(f"gold rows: {len(gold)}")
    print("loading NLI model...")
    results = evaluate(NLIVerifier(), gold)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

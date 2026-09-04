"""End-to-end Mizan demo on the sample corpus.

Runs the full pipeline on a mixed Arabic/English answer containing one planted
unsupported claim. Uses the real multilingual NLI model when transformers is
installed; otherwise it explains what is missing and exits, it never fakes a
verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mizan import MizanPipeline, TfidfRetriever

ANSWER = (
    "القهوة تقلل خطر الاصابة بالسكري من النوع الثاني حسب تحليل تلوي نشر عام 2021. "
    "انخفض الخطر بنسبة 39% لدى شاربي القهوة يومياً. "
    "شملت الدراسة ست تجارب عشوائية. "
    "وينصح الاطباء بثمانية اكواب يومياً لكل البالغين."
)


def main() -> int:
    try:
        from mizan.verify import NLIVerifier
    except Exception:
        print("transformers/torch not available: install requirements.txt to run the real NLI demo.")
        return 1

    corpus = Path(__file__).parent / "sample_corpus.jsonl"
    retriever = TfidfRetriever.from_jsonl(corpus)
    print("Loading NLI model (first run downloads it from Hugging Face)...")
    verifier = NLIVerifier()
    pipeline = MizanPipeline(retriever, verifier)

    report = pipeline.check_answer(ANSWER)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    print()
    print(f"groundedness: {report.groundedness:.2%}")
    print(f"suppressed before display: {len(report.suppressed)} claim(s)")
    for claim in report.suppressed:
        print(f"  - {claim}")

    from report_to_html import write_report

    out = write_report(report.to_dict(), Path(__file__).parent / "report.html")
    print(f"html report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

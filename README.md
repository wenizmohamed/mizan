# Mizan ميزان

**Claim-level groundedness verification for Arabic and cross-lingual RAG.**

Mizan decomposes a RAG answer into atomic claims, verifies each claim against the
retrieved evidence with a multilingual NLI cross-encoder, calibrates the scores,
separates hallucination from retrieval failure as two measured signals, and
suppresses unsupported claims before they ever reach the user.

```
question ──> RAG answer ──> decompose ──> verify (NLI) ──> triage ──> aggregate ──> report
                              │              │               │            │
                        atomic claims   3-way verdict   hallucination  weighted
                        (ar / en)       per claim       vs retrieval   groundedness
                                                        failure        score + ECE
```

## Why

Most factuality tooling is English-centric (MiniCheck, AlignScore, RAGAS, SAFE).
The Arabic side of 2025-2026 (AraHalluEval, HalluTruthQA) provides detection
benchmarks over finished outputs. What does not exist is a calibrated,
interactive claim-level verification layer for Arabic and cross-lingual RAG that:

1. issues three-way verdicts (supported / contradicted / no-evidence) instead of a binary,
2. re-verifies every no-evidence claim under maximal retrieval before calling it a hallucination,
3. reports calibration (Expected Calibration Error) as a first-class metric, and
4. measures selective suppression as a risk-coverage tradeoff.

## Design decisions stated plainly

- The answer-level score is a **groundedness score, not a truth probability**.
  It measures how well the retrieved corpus supports the claims. A wrong corpus
  yields a confident wrong score. This limitation is documented, not hidden.
- Aggregation is a **weighted mean** of claim scores (length x centrality by
  default). The weights are part of the report payload, never implicit.
- NLI cross-encoders operate at sentence granularity, so **atomic decomposition
  is a correctness precondition**, not an optimization.

## Quickstart

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python demo/run_demo.py            # uses the real mDeBERTa NLI model when transformers is installed
```

```python
from mizan import MizanPipeline, TfidfRetriever
from mizan.verify import NLIVerifier

retriever = TfidfRetriever.from_jsonl("demo/sample_corpus.jsonl")
verifier = NLIVerifier()   # MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
report = MizanPipeline(retriever, verifier).check_answer(answer_text)
print(report.to_dict())
```

## Evaluation design

| System | Balanced Acc. | ECE | Risk-Coverage AUC | Latency / claim |
|---|---|---|---|---|
| Mizan (mDeBERTa cross-lingual) | | | | |
| MiniCheck (cross-lingual) | | | | |
| Translate-then-Verify | | | | |
| LLM-as-Judge | | | | |

Gold set: 200-300 hand-annotated claim-evidence pairs (3-way labels) drawn from
the target dataset. Protocol follows the AVeriTeC 2025 shared-task posture:
reproducible, open-weights, under one minute per claim.

## Status

Seed stage. The pipeline, calibration, triage, and aggregation logic are
implemented and unit-tested. Gold-set annotation and the baseline comparison
table are the active work items.

## License

MIT

---

# ميزان

**طبقة تحقق من الإسناد على مستوى الادعاء للأنظمة العربية ومتقاطعة اللغة.**

يفكك ميزان إجابة نظام الاسترجاع والتوليد إلى ادعاءات ذرية، ويتحقق من كل ادعاء
مقابل الأدلة المسترجعة عبر نموذج استدلال لغوي متعدد اللغات، ويعاير الدرجات،
ويفصل الهلوسة عن فشل الاسترجاع كإشارتين مقاستين، ويحجب الادعاء غير المسند قبل
وصوله للمستخدم.

المبدأ الصريح: الدرجة المعروضة هي درجة إسناد للمصادر المسترجعة وليست احتمالاً
لصحة المعلومة في العالم، وهذا القيد موثق لا مخفي. الحكم ثلاثي: مدعوم، متناقض،
بلا دليل. وأي ادعاء بلا دليل يعاد فحصه بشروط استرجاع قصوى قبل تصنيفه هلوسة.

"""Render a Mizan PipelineReport as a self-contained RTL HTML report.

This produces the same visual language as the product demo: the answer with
each claim colored by its entailment score, per-claim verification bars, the
groundedness ring, and the suppressed-claims block. The HTML is standalone
(inline CSS, no external assets) so it can be attached or opened anywhere.
"""

from __future__ import annotations

import html
from pathlib import Path

_GREEN = "#2e9e6b"
_GOLD = "#c9a24b"
_RED = "#c0504d"


def _color(score: float) -> str:
    if score >= 0.75:
        return _GREEN
    if score >= 0.45:
        return _GOLD
    return _RED


def render_report(report: dict, title: str = "Mizan تقرير التحقق") -> str:
    """Return standalone HTML for a ``PipelineReport.to_dict()`` payload."""
    ground_pct = round(report.get("groundedness", 0.0) * 100)
    claims = report.get("claims", [])
    suppressed = set(report.get("suppressed", []))

    spans = []
    for c in claims:
        color = _color(c["entail"])
        deco = "opacity:.45;text-decoration:line-through;" if c["claim"] in suppressed else ""
        spans.append(
            f'<span style="border-bottom:2px solid {color};{deco}" '
            f'title="entailment {c["entail"]:.2f}">{html.escape(c["claim"])}</span>'
        )
    answer_html = " ".join(spans)

    rows = []
    for c in claims:
        color = _color(c["entail"])
        width = round(c["entail"] * 100)
        evidence = " ".join(html.escape(e) for e in c.get("evidence_ids", []))
        rows.append(
            '<div class="row">'
            f'<div class="claim">{html.escape(c["claim"])}</div>'
            f'<div class="meta">entailment {c["entail"]:.2f} · {html.escape(c["verdict"])} · {html.escape(c["triage"])} · {evidence}</div>'
            f'<div class="track"><i style="width:{width}%;background:{color}"></i></div>'
            "</div>"
        )
    rows_html = "\n".join(rows)

    suppressed_html = ""
    if suppressed:
        items = "".join(f"<li>{html.escape(s)}</li>" for s in sorted(suppressed))
        suppressed_html = f'<div class="panel"><h2>ادعاءات محجوبة قبل العرض</h2><ul>{items}</ul></div>'

    return f"""<!doctype html>
<html dir="rtl" lang="ar">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
 body{{background:#101216;color:#e8e6e1;font-family:'Segoe UI',Tahoma,sans-serif;margin:0;padding:28px;line-height:1.9}}
 h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:14px;color:#c9a24b;margin:0 0 10px}}
 .sub{{color:#8a8f98;font-size:12px;margin-bottom:22px}}
 .grid{{display:grid;grid-template-columns:2fr 1fr;gap:16px;align-items:start}}
 .panel{{background:#171a20;border:1px solid #262b33;border-radius:12px;padding:18px;margin-bottom:16px}}
 .row{{margin-bottom:14px}} .claim{{font-size:14px}} .meta{{color:#8a8f98;font-size:11.5px;margin:2px 0 6px}}
 .track{{height:6px;background:#262b33;border-radius:3px;overflow:hidden}} .track i{{display:block;height:100%}}
 .ring{{width:110px;height:110px;border-radius:50%;margin:6px auto 10px;display:grid;place-items:center;
   background:conic-gradient({_GOLD} {ground_pct}%, #262b33 0);box-shadow:0 0 14px rgba(201,162,75,.3)}}
 .ring b{{background:#171a20;width:84px;height:84px;border-radius:50%;display:grid;place-items:center;font-size:24px}}
 ul{{margin:0;padding-inline-start:18px;color:#c0504d;font-size:13px}}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="sub">claim-level groundedness · mDeBERTa-v3 NLI · Mizan</div>
<div class="grid">
 <div>
  <div class="panel"><h2>الاجابة الملونة بدرجة الاسناد</h2><p>{answer_html}</p></div>
  <div class="panel"><h2>تفكيك الادعاءات والتحقق</h2>{rows_html}</div>
  {suppressed_html}
 </div>
 <div class="panel" style="text-align:center">
  <h2>درجة الاسناد الكلية</h2>
  <div class="ring"><b>{ground_pct}%</b></div>
  <div class="sub">متوسط موزون لدرجات الادعاءات · الاوزان ضمن التقرير</div>
 </div>
</div>
</body>
</html>"""


def write_report(report: dict, path: str | Path, title: str = "Mizan تقرير التحقق") -> Path:
    out = Path(path)
    out.write_text(render_report(report, title), encoding="utf-8")
    return out

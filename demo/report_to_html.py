"""Render a Mizan PipelineReport as a self-contained, premium RTL HTML report.

Design DNA is mirrored 1:1 from Mohamed's live AURAAI Operator Console
(auraai-weld-two.vercel.app): warm cream paper #F2F1EC, white cards with
hairline borders and soft ring shadows, warm ink #34322D, a single restrained
ember accent #C2410C, editorial calm and generous whitespace. No glow, no dark
neon, no conic gradients. Status semantics stay legible but muted to sit inside
the warm editorial system.

Standalone: fonts from Google Fonts with a system fallback stack, everything
else inline, so the report renders on its own anywhere.
"""

from __future__ import annotations

import html
from pathlib import Path

_POSITIVE = "positive"   # supported
_WARNING = "warning"     # partial / weak
_NEGATIVE = "negative"   # unsupported / suppressed


def _band(score: float) -> str:
    if score >= 0.75:
        return _POSITIVE
    if score >= 0.45:
        return _WARNING
    return _NEGATIVE


def _band_ar(band: str) -> str:
    return {"positive": "مسنَد", "warning": "جزئي", "negative": "غير مسنَد"}[band]


_STYLE = """
:root{
  --paper:#F2F1EC; --paper-2:#EDECE6; --card:#FFFFFF;
  --ink:#181818; --ink-body:#34322D; --muted:#545454; --faint:#8A857C;
  --line:rgba(0,0,0,.07); --line-2:rgba(0,0,0,.10);
  --ember:#C2410C; --ember-soft:rgba(194,65,12,.08); --ember-line:rgba(194,65,12,.22);
  --pos:#2F8F5B; --pos-soft:rgba(47,143,91,.09); --pos-line:rgba(47,143,91,.24);
  --warn:#B58128; --warn-soft:rgba(181,129,40,.10); --warn-line:rgba(181,129,40,.26);
  --neg:#C2410C; --neg-soft:rgba(194,65,12,.08); --neg-line:rgba(194,65,12,.24);
  --ring:0 1px 2px rgba(0,0,0,.04),0 0 0 1px rgba(0,0,0,.03);
  --sans:"IBM Plex Sans Arabic","Helvetica Neue",-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --r:12px; --r-sm:9px; --r-full:999px;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--paper);color:var(--ink-body);font-family:var(--sans);
  -webkit-font-smoothing:antialiased;line-height:1.65;padding:48px 24px}
.wrap{max-width:940px;margin:0 auto}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.eyebrow{font-family:var(--mono);font-size:10.5px;font-weight:500;letter-spacing:.14em;
  text-transform:uppercase;color:var(--faint)}

.top{display:flex;align-items:center;gap:13px;padding-bottom:22px;margin-bottom:26px;
  border-bottom:1px solid var(--line)}
.mark{width:42px;height:42px;flex:none}
.top h1{font-size:20px;font-weight:700;letter-spacing:-.01em;color:var(--ink)}
.top h1 .en{font-weight:500;color:var(--muted);margin-inline-start:7px;font-size:13px;letter-spacing:.02em}
.top .sub{font-size:12px;color:var(--faint);margin-top:2px;letter-spacing:.01em}
.top .status{margin-inline-start:auto;display:inline-flex;align-items:center;gap:8px;
  font-family:var(--mono);font-size:11px;color:var(--muted);
  background:var(--card);border:1px solid var(--line);padding:7px 13px;border-radius:var(--r-full);box-shadow:var(--ring)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--pos)}

.stats{display:grid;grid-template-columns:1.7fr 1fr 1fr 1fr;gap:14px;margin-bottom:24px}
@media(max-width:760px){.stats{grid-template-columns:1fr 1fr}.stat.hero{grid-column:1/-1}}
.stat{background:var(--card);border:1px solid var(--line);border-radius:var(--r);
  box-shadow:var(--ring);padding:20px 22px}
.stat.hero{display:flex;align-items:center;gap:20px}
.stat.hero>div{min-width:0}
.ring{--v:0;width:88px;height:88px;border-radius:50%;flex:none;display:grid;place-items:center;position:relative;
  background:conic-gradient(var(--ember) calc(var(--v)*1%),var(--paper-2) 0)}
.ring::after{content:"";position:absolute;inset:8px;border-radius:50%;background:var(--card)}
.ring b{position:relative;z-index:1;font-family:var(--mono);font-size:23px;font-weight:600;color:var(--ink);letter-spacing:-.02em}
.ring b span{font-size:12px;color:var(--faint)}
.stat .hlbl{font-size:14px;color:var(--ink);font-weight:600}
.stat .lbl{margin-top:5px;font-family:var(--mono);font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--faint)}
.stat .val{font-family:var(--mono);font-size:27px;font-weight:600;color:var(--ink);letter-spacing:-.02em;line-height:1}
.stat .val.pos{color:var(--pos)} .stat .val.warn{color:var(--warn)} .stat .val.neg{color:var(--neg)}

.card{background:var(--card);border:1px solid var(--line);border-radius:var(--r);
  box-shadow:var(--ring);margin-bottom:16px;overflow:hidden}
.card__head{display:flex;align-items:center;justify-content:space-between;gap:12px;
  padding:16px 22px;border-bottom:1px solid var(--line)}
.card__head h2{font-size:14.5px;font-weight:600;color:var(--ink)}
.card__body{padding:22px}

.answer{font-size:16px;line-height:2.2;color:var(--ink-body)}
.cl{border-bottom:1.5px solid transparent;padding-bottom:1px}
.cl.positive{border-color:var(--pos-line)}
.cl.warning{border-color:var(--warn-line)}
.cl.negative{border-color:var(--neg-line)}
.cl.sup{opacity:.45;text-decoration:line-through;text-decoration-color:var(--neg-line)}
.cl sup{font-family:var(--mono);font-size:9.5px;font-weight:500;margin-inline-start:3px;color:var(--faint)}

.bar{display:flex;align-items:center;gap:14px;padding:14px 0;border-bottom:1px solid var(--line)}
.bar:last-child{border-bottom:0}
.bar__main{flex:1;min-width:0}
.bar__claim{font-size:14px;color:var(--ink-body);margin-bottom:9px}
.bar__row{display:flex;align-items:center;gap:12px}
.badge{display:inline-flex;align-items:center;gap:.35rem;font-family:var(--mono);font-size:10.5px;font-weight:500;
  line-height:1;padding:4px 10px;border-radius:var(--r-full);border:1px solid var(--line-2);color:var(--muted)}
.badge::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}
.badge.positive{color:var(--pos);border-color:var(--pos-line);background:var(--pos-soft)}
.badge.warning{color:var(--warn);border-color:var(--warn-line);background:var(--warn-soft)}
.badge.negative{color:var(--neg);border-color:var(--neg-line);background:var(--neg-soft)}
.track{flex:1;min-width:90px;height:6px;border-radius:var(--r-full);background:var(--paper-2);overflow:hidden}
.track>i{display:block;height:100%;border-radius:inherit}
.track>i.positive{background:var(--pos)} .track>i.warning{background:var(--warn)} .track>i.negative{background:var(--neg)}
.score{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:12.5px;font-weight:500;color:var(--ink);min-width:34px;text-align:end}
.ev{font-family:var(--mono);font-size:10px;color:var(--faint);min-width:56px}

.sup-card{background:linear-gradient(180deg,var(--neg-soft),var(--card) 40%);border-color:var(--neg-line)}
.sup-card .card__head h2{color:var(--neg)}
.sup-list{list-style:none;display:flex;flex-direction:column;gap:10px}
.sup-list li{font-size:13.5px;color:var(--ink-body);padding-inline-start:17px;position:relative}
.sup-list li::before{content:"";position:absolute;inset-inline-start:0;top:9px;width:6px;height:6px;border-radius:50%;background:var(--neg)}

.foot{margin-top:24px;padding-top:16px;border-top:1px solid var(--line);
  display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;
  font-family:var(--mono);font-size:10.5px;color:var(--faint);letter-spacing:.02em}
.foot .em{color:var(--ember)}
"""

# The AURAAI "Orbit A" mark: an A with a node orbiting a broken ring, ember gradient.
_MARK_SVG = (
    '<svg viewBox="0 0 42 42" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<defs><linearGradient id="em" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#F2723E"/><stop offset="1" stop-color="#C2410C"/></linearGradient></defs>'
    '<path d="M21 6 L30 34 M21 6 L12 34 M15.5 25 H26.5" stroke="#181818" stroke-width="2.4" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
    '<path d="M9 21 a12 12 0 0 1 24 0" stroke="url(#em)" stroke-width="2.4" stroke-linecap="round" fill="none" opacity=".85"/>'
    '<circle cx="33" cy="21" r="3.1" fill="url(#em)"/></svg>'
)


def render_report(report: dict, title: str = "ميزان · تقرير التحقق") -> str:
    """Return standalone HTML for a ``PipelineReport.to_dict()`` payload."""
    ground_pct = round(report.get("groundedness", 0.0) * 100)
    claims = report.get("claims", [])
    suppressed = set(report.get("suppressed", []))

    counts = {"positive": 0, "warning": 0, "negative": 0}
    spans = []
    for c in claims:
        band = _band(c["entail"])
        counts[band] += 1
        cls = f"cl {band}" + (" sup" if c["claim"] in suppressed else "")
        spans.append(
            f'<span class="{cls}">{html.escape(c["claim"])}<sup>{c["entail"]:.2f}</sup></span>'
        )
    answer_html = " ".join(spans)

    rows = []
    for c in claims:
        band = _band(c["entail"])
        width = round(c["entail"] * 100)
        ev = " ".join(html.escape(e) for e in c.get("evidence_ids", [])) or "لا مصدر"
        rows.append(
            '<div class="bar"><div class="bar__main">'
            f'<div class="bar__claim">{html.escape(c["claim"])}</div>'
            '<div class="bar__row">'
            f'<span class="badge {band}">{_band_ar(band)}</span>'
            f'<span class="track"><i class="{band}" style="width:{width}%"></i></span>'
            f'<span class="score">{c["entail"]:.2f}</span>'
            f'<span class="ev">{ev}</span>'
            "</div></div></div>"
        )
    rows_html = "\n".join(rows)

    suppressed_html = ""
    if suppressed:
        items = "".join(f"<li>{html.escape(s)}</li>" for s in sorted(suppressed))
        suppressed_html = (
            '<div class="card sup-card"><div class="card__head">'
            '<h2>ادعاءات محجوبة قبل العرض</h2>'
            '<span class="eyebrow">Suppressed</span></div>'
            f'<div class="card__body"><ul class="sup-list">{items}</ul></div></div>'
        )

    return f"""<!doctype html>
<html dir="rtl" lang="ar">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>{_STYLE}</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div class="mark">{_MARK_SVG}</div>
    <div>
      <h1>ميزان<span class="en">Mizan</span></h1>
      <div class="sub">claim-level groundedness · mDeBERTa-v3 NLI</div>
    </div>
    <div class="status"><span class="dot"></span>نموذج التحقق نشط · mDeBERTa-v3-xnli</div>
  </div>

  <div class="stats">
    <div class="stat hero">
      <div class="ring" style="--v:{ground_pct}"><b>{ground_pct}<span>%</span></b></div>
      <div><div class="hlbl">درجة الإسناد الكلية</div>
        <div class="lbl">weighted groundedness</div></div>
    </div>
    <div class="stat"><div class="val pos">{counts["positive"]}</div><div class="lbl">مسنَد · supported</div></div>
    <div class="stat"><div class="val warn">{counts["warning"]}</div><div class="lbl">جزئي · partial</div></div>
    <div class="stat"><div class="val neg">{counts["negative"]}</div><div class="lbl">غير مسنَد · flagged</div></div>
  </div>

  <div class="card">
    <div class="card__head"><h2>الإجابة الملوّنة بدرجة الإسناد</h2>
      <span class="eyebrow">colored by entailment</span></div>
    <div class="card__body"><div class="answer">{answer_html}</div></div>
  </div>

  <div class="card">
    <div class="card__head"><h2>تفكيك الادعاءات والتحقق</h2>
      <span class="eyebrow">atomic claims · NLI</span></div>
    <div class="card__body">{rows_html}</div>
  </div>

  {suppressed_html}

  <div class="foot">
    <span>Mizan · decomposition + <span class="em">NLI entailment</span> + calibration + hallucination triage</span>
    <span>mDeBERTa-v3-xnli · AURAAI</span>
  </div>
</div>
</body>
</html>"""


def write_report(report: dict, path: str | Path, title: str = "ميزان · تقرير التحقق") -> Path:
    out = Path(path)
    out.write_text(render_report(report, title), encoding="utf-8")
    return out

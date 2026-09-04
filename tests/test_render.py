import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "report_to_html", Path(__file__).resolve().parents[1] / "demo" / "report_to_html.py"
)
report_to_html = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_to_html)


REPORT = {
    "answer": "answer text",
    "groundedness": 0.71,
    "weights": [0.25, 0.25, 0.25, 0.25],
    "claims": [
        {"claim": "claim one", "verdict": "supported", "triage": "supported", "entail": 0.96, "contradict": 0.01, "evidence_ids": ["S1"]},
        {"claim": "claim two", "verdict": "no_evidence", "triage": "likely_hallucination", "entail": 0.19, "contradict": 0.05, "evidence_ids": []},
    ],
    "suppressed": ["claim two"],
}


class TestRender(unittest.TestCase):
    def test_renders_ring_claims_and_suppression(self):
        html_out = report_to_html.render_report(REPORT)
        self.assertIn("71%", html_out)
        self.assertIn("claim one", html_out)
        self.assertIn("line-through", html_out)
        self.assertIn("محجوبة", html_out)
        self.assertIn('dir="rtl"', html_out)

    def test_escapes_html_in_claims(self):
        payload = dict(REPORT)
        payload["claims"] = [
            {"claim": "<script>alert(1)</script>", "verdict": "supported", "triage": "supported", "entail": 0.9, "contradict": 0.01, "evidence_ids": []}
        ]
        payload["suppressed"] = []
        html_out = report_to_html.render_report(payload)
        self.assertNotIn("<script>alert(1)</script>", html_out)
        self.assertIn("&lt;script&gt;", html_out)

    def test_write_report_creates_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = report_to_html.write_report(REPORT, Path(tmp) / "r.html")
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()

import unittest

from mizan.retrieve import TfidfRetriever, normalize_arabic, tokenize


class TestRetrieve(unittest.TestCase):
    def setUp(self):
        self.retriever = TfidfRetriever(
            {
                "S1": "A 2021 meta-analysis of six randomized trials found coffee lowers type 2 diabetes risk.",
                "S2": "انخفض خطر الإصابة بالسكري بنسبة 39% لدى شاربي القهوة حسب التحليل التلوي",
                "S3": "Green tea contains antioxidants and caffeine in smaller amounts.",
            }
        )

    def test_arabic_normalization(self):
        self.assertEqual(normalize_arabic("أإآ"), "ااا")
        self.assertEqual(normalize_arabic("مكتبةٌ"), "مكتبه")

    def test_arabic_query_finds_arabic_passage(self):
        results = self.retriever.search("نسبة انخفاض خطر السكري عند شاربي القهوه", k=2)
        self.assertTrue(results)
        self.assertEqual(results[0].pid, "S2")

    def test_english_query_finds_english_passage(self):
        results = self.retriever.search("meta-analysis randomized trials coffee diabetes", k=1)
        self.assertEqual(results[0].pid, "S1")

    def test_tokenize_mixed(self):
        tokens = tokenize("Coffee قهوة 39%")
        self.assertIn("coffee", tokens)
        self.assertIn("قهوه", tokens)
        self.assertIn("39", tokens)


if __name__ == "__main__":
    unittest.main()

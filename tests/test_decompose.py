import unittest

from mizan.decompose import decompose


class TestDecompose(unittest.TestCase):
    def test_english_sentences(self):
        text = "Coffee reduces risk. The effect was measured in 2021."
        self.assertEqual(len(decompose(text)), 2)

    def test_arabic_sentences(self):
        text = "القهوة تقلل الخطر؟ الدراسة نشرت عام 2021."
        claims = decompose(text)
        self.assertEqual(len(claims), 2)

    def test_multi_number_sentence_splits_into_clauses(self):
        text = "انخفض الخطر بنسبة 39%، وشملت الدراسة 6 تجارب عشوائية"
        claims = decompose(text)
        self.assertEqual(len(claims), 2)

    def test_single_number_sentence_stays_whole(self):
        text = "انخفض الخطر بنسبة 39% حسب التحليل التلوي"
        self.assertEqual(len(decompose(text)), 1)

    def test_llm_hook_wins_and_dedupes(self):
        claims = decompose("ignored", llm=lambda t: [" a ", "a", "b"])
        self.assertEqual(claims, ["a", "b"])


if __name__ == "__main__":
    unittest.main()

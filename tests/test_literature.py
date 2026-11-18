import unittest

from src.api.concept_map import search_concept_map
from src.api.literature import classify_quoted_figure


class QuotedFigureTest(unittest.TestCase):
    def classify(self, question, only=False):
        return classify_quoted_figure(question, search_concept_map("국어", question, 3), only)

    def test_applies_curriculum_to_the_quoted_words(self):
        result = self.classify("‘단풍은 불꽃처럼 붉다.’에서 비유법의 이름과 그 근거는?")
        self.assertIn("직유법", result)
        self.assertIn("불꽃처럼", result)
        self.assertIn("직유법", self.classify("‘단풍은 불꽃처럼 붉다.’는 어떤 비유법인가요?"))
        self.assertEqual(self.classify("‘시간은 강물이다.’에서 흐름을 나타낸 비유법의 이름은?", True), "은유법입니다.")

    def test_defers_other_tasks_and_literal_sentences(self):
        for question in ("‘고래는 포유류이다.’는 어떤 비유법인가요?",
                         "‘단풍은 불꽃처럼 붉다.’에 쓴 비유법의 이름과 효과는?",
                         "‘집으로 같이 가자’에는 어떤 비유법이 있나요?",
                         "‘동생은 평소처럼 걸었다’에는 어떤 비유법이 있나요?",
                         "‘그는 아버지처럼 의사가 되었다’에는 어떤 비유법이 있나요?",
                         "‘눈은 꽃처럼 피었다’와 ‘봄은 축제이다’의 비유법의 이름은?"):
            with self.subTest(question=question):
                self.assertIsNone(self.classify(question))
        self.assertIsNone(classify_quoted_figure("‘단풍은 불꽃처럼 붉다’는 어떤 비유법인가요?", [], False))

import unittest

from src.api.curriculum_text import extract_passage


class CurriculumTextTests(unittest.TestCase):
    def test_math_contents_are_kept_in_the_passage(self):
        passage = r"구간 [0, 1]에서 $f(x)=\frac{x^2}{2}$이다. 0 < x < 1이면 |x| = x이다."
        document = f"[교육과정 성취기준] [A1] 함수\n[학습 지문] {passage}\n[관련 Q&A] 질문: 값은?"
        self.assertEqual(extract_passage(document), passage)

    def test_question_without_passage_returns_its_answer(self):
        document = "[관련 Q&A] 질문: 표시된 범위는? / 답변: [1, 4]입니다.\n질문: 다른 문제"
        self.assertEqual(extract_passage(document), "[1, 4]입니다.")

    def test_html_without_losing_superscripts_or_worksheet_blanks(self):
        document = "[학습 지문] <p>x<sup>2</sup> &gt; 4이면 ___이다.</p>"
        self.assertEqual(extract_passage(document), "x^{2} > 4이면 ___이다.")

    def test_curriculum_code_not_part_of_the_outcome(self):
        self.assertEqual(extract_passage("[교육과정 성취기준] [4국04-02] 단어를 분류한다."), "단어를 분류한다.")

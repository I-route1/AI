import json
import unittest

from src.api.ask_quality import missing_exercise, question_prompt, read_answer, wants_answer_only
from src.api.concept_map import search_concept_map
from src.api.linear_equation import solve_linear


class AskQualityTest(unittest.TestCase):
    def test_worked_equation_keeps_exact_intermediate_step(self):
        solution = solve_linear("4번 문제: 3x + 1 = 10일 때 x와 풀이를 알려 주세요.")
        self.assertTrue(solution.startswith("x = 3\n"))
        self.assertIn("3x = 9", solution)
        self.assertIn("9 ÷ (3) = 3", solution)

    def test_linear_equations_with_both_sides_and_fractional_results(self):
        for request, expected in [
                ("다음 방정식 −2t + 3 = t - 6의 해를 구하세요.", "t = 3"),
                ("방정식 -4y + 2 = 3y + 5의 해를 구하시오.", "y = -3/7"),
                ("미지수 p/2 + 2 = 5의 값을 구하세요.", "p = 6"),
                ("0.5x + 1.5 = 2.75일 때 x의 값은?", "x = 5/2"),
                ("2x - 2x = 0의 해는?", "모든 실수가 해입니다."),
                ("2x + 4 = 2x + 5의 해는?", "해가 없습니다.")]:
            with self.subTest(request=request):
                self.assertEqual(solve_linear(request, answer_only=True), expected)

    def test_equation_inside_a_different_task_goes_to_tutor(self):
        for request in ("2x+4=0의 좌변을 인수분해해 주세요.", "x^2+4x=0의 해를 구하세요.",
                        "x+y=8의 해를 구하세요.", "2x=4와 3x=6의 공통점을 설명해 주세요.",
                        "x/0=2일 때 x의 해를 구하세요.", "2(x+3)=4의 해를 구하세요.",
                        "√x=3의 해를 구하세요.", "3x=12", "2 3x=6의 해는?",
                        "3x=12를 만족하는 x와 두 배인 y를 구해줘"):
            with self.subTest(request=request):
                self.assertIsNone(solve_linear(request))

    def test_only_an_exercise_reference_needs_the_original_question(self):
        for question in ("교과서 37쪽 8번 정답이 뭐예요?", "이 문제 어떻게 풀죠?",
                         "문제 19번의 풀이를 알려 주세요.", "선생님이 주신 3번 문제의 정답을 알려 주세요."):
            with self.subTest(question=question):
                self.assertTrue(missing_exercise(question))

    def test_number_does_not_hide_supplied_question(self):
        for question in ("3번 문제: 2x + 3 = 11일 때 x는?", "2번 보기 ‘하늘은 바다다’의 비유법은?",
                         "이 지문에서 'she'는 누굴 가리켜요? Mina said she was tired.",
                         "주사위를 3번 던졌을 때 홀수가 나올 확률은?"):
            with self.subTest(question=question):
                self.assertFalse(missing_exercise(question))

    def test_only_supplied_curricular_evidence_enables_grounding(self):
        question = "수동태에서 be 동사는 어떤 역할인가요?"
        docs = search_concept_map("영어", question, 3)
        prompt, supported = question_prompt("영어", question, docs)
        self.assertTrue(supported)
        self.assertIn(question, prompt)
        self.assertIn("be+p.p.", prompt)
        unsupported_prompt, supported = question_prompt("영어", question, ["검색 오류: 내부 정보"])
        self.assertFalse(supported)
        self.assertNotIn("내부 정보", unsupported_prompt)

    def test_model_cannot_assert_grounding_without_reference(self):
        self.assertEqual(read_answer(json.dumps({"유형": "자료", "답변": "분자는 5입니다.", "근거": "분자는 5입니다."}), False),
                         ("분자는 5입니다.", False, "answer"))

    def test_direct_derivations_and_clarifications_do_not_claim_a_source(self):
        self.assertEqual(read_answer(json.dumps({"유형": "풀이", "답변": "2입니다.", "근거": "5 - 3 = 2"}), True),
                         ("2입니다.\n5 - 3 = 2", False, "answer"))
        self.assertEqual(read_answer(json.dumps({"유형": "확인", "답변": "어떤 문장인지 알려 주세요.", "근거":"지문 누락"}), True),
                         ("어떤 문장인지 알려 주세요.", False, "clarification"))
        self.assertEqual(read_answer(json.dumps({"유형": "범위밖", "답변": "현재 재고를 확인할 수 없어요.", "근거":"매장 정보 없음"}), True),
                         ("현재 재고를 확인할 수 없어요.", False, "out_of_scope"))

    def test_mode_is_removed_without_discarding_answer_content(self):
        self.assertEqual(read_answer(json.dumps({"유형": "자료", "답변": "응고입니다.", "근거":"물이 얼어 고체가 됩니다."}), True),
                         ("응고입니다.\n물이 얼어 고체가 됩니다.", True, "answer"))
        self.assertTrue(wants_answer_only("설명 없이 정답만 알려 주세요."))
        self.assertFalse(wants_answer_only("답과 계산 과정을 알려 주세요."))
        self.assertFalse(wants_answer_only("답만 말하지 말고 풀이도 알려 주세요."))
        self.assertEqual(read_answer(json.dumps({"유형": "풀이", "답변": "x = -2", "근거":"양변에서 3을 뺍니다."}),
                                     True, answer_only=True),
                         ("x = -2", False, "answer"))

    def test_incomplete_json_never_leaks_to_students(self):
        self.assertEqual(read_answer('{"근거":"9 ÷ 3 = 3", "답변":"답은 3입니다.", "유형":', True, length_limited=True),
                         ("답은 3입니다.\n9 ÷ 3 = 3", False, "answer"))
        self.assertEqual(read_answer('{"답변":"답은', True, length_limited=True), (None, False, "answer"))
        self.assertEqual(read_answer('잘린 JSON {"유형":', True), (None, False, "answer"))
        self.assertEqual(read_answer('{"유형": "확인", "답변": 3}', True), (None, False, "answer"))


if __name__ == "__main__":
    unittest.main()

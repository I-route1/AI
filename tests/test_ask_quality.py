import asyncio
import json
import unittest

from src.api.ask_quality import _plain_retry, missing_exercise, question_prompt, read_answer, wants_answer_only
from src.api.concept_map import search_concept_map
from src.api.linear_equation import solve_linear


class AskQualityTest(unittest.TestCase):
    def test_part_of_whole_supplies_both_exact_conversions(self):
        from src.api.ask_quality import part_whole_answer
        for question in (
            "전체 12개 중 3개를 골랐어요. 고른 비율을 기약분수와 백분율 두 가지로 쓰고, 서로 어떻게 바꾸는지 설명해 주세요.",
            "전체 12개 중 3개를 기약분수와 백분율로 각각 나타내고, 왜 서로 바꿀 수 있는지 양방향으로 설명해 줘."
        ):
            answer = part_whole_answer(question)
            self.assertIn("3/12 = 1/4", answer)
            self.assertIn("1/4 × 100 = 25", answer)
            self.assertIn("25 ÷ 100 = 1/4", answer)
        for whole, part, expected in [(8, 3, "37.5"), (3, 1, "(100/3)"), (17, 17, "100"), (12, 0, "0")]:
            q = f"총 {whole}명 가운데 {part}명의 비율을 백분율과 기약분수로 써 주세요."
            direct = part_whole_answer(q, answer_only=True)
            self.assertIn(expected + "%", direct)
            self.assertNotIn("\n", direct)

    def test_unrelated_or_ambiguous_count_tasks_go_to_the_tutor(self):
        from src.api.ask_quality import part_whole_answer
        for q in ("전체 4개 중 5개를 기약분수와 백분율로 써 주세요.",
                  "0개 중 0개를 기약분수와 백분율로 써 주세요.",
                  "전체 12개 중 3개를 고르지 않았어요. 고른 비율을 기약분수와 백분율로 써 주세요.",
                  "전체 12개 중 3개를 고르고 4개를 더 골랐어요. 기약분수와 백분율로 써 주세요.",
                  "전체 12개 중 3개를 기약분수와 백분율로 쓰지 말고 소수로 써 주세요.",
                  "전체 12개 중 3개의 기약분수와 백분율", "전체 12명 중 3개를 기약분수와 백분율로 나타내세요.",
                  "전체 12개 중 3개의 백분율만 써 주세요.",
                  "12개 중 3개를 다시 뽑을 확률을 기약분수와 백분율로 나타내세요."):
            with self.subTest(q=q):
                self.assertIsNone(part_whole_answer(q))

    def test_ratio_at_question_endpoint_with_automatic_or_chosen_subject(self):
        from unittest.mock import AsyncMock, patch
        # This existing router fixture uses the real ConceptMap and no vector index.
        from test_report_quality_regression import counseling
        q = "총 8명 중 3명의 비율을 기약분수와 백분율로 나타내세요."
        with (patch.object(counseling, "subject_aware_search", return_value=[]) as lookup,
              patch.object(counseling.model_registry, "get", return_value=None),
              patch.object(counseling, "ollama_answer", new_callable=AsyncMock,
                           return_value=("영어 풀이", False, "answer")) as tutor):
            for req in ({"question": q}, {"question": q, "subject":"수학"}):
                result = asyncio.run(counseling.ai_ask(req))
                self.assertEqual(result["subject"], "수학")
                self.assertIn("37.5 ÷ 100 = 3/8", result["answer"])
            self.assertTrue(asyncio.run(counseling.ai_ask({"question":"이게 왜 그래요?"}))["needSubject"])
            missing = asyncio.run(counseling.ai_ask({"subject":"수학", "question":"4번 문제 알려 주세요."}))
            self.assertEqual(missing["answerKind"], "clarification")
            lookup.assert_not_called()
            tutor.assert_not_awaited()
            chosen = asyncio.run(counseling.ai_ask({"subject":"영어", "question": q}))
            self.assertEqual((chosen["subject"], chosen["answer"]), ("영어", "영어 풀이"))

    def test_deployed_fraction_answer_removes_model_field_labels(self):
        lead = "고른 비율은 1/4, 25%입니다. 기약분수와 백분율은 서로를 곱하거나 나누어 바꿀 수 있습니다."
        reason = "3/12=1/4, 1/4×100=25%. 기약분수는 분자와 분모를 최대공약수로 나누어 구하고, 백분율은 기약분수를 100으로 곱하여 구합니다."
        leaked = f"{lead}\n근거: {reason}\n유형: '풀이'."
        expected = f"{lead}\n{reason}"
        raw = json.dumps({"답변": leaked, "근거": "", "유형": "풀이"})
        self.assertEqual(read_answer(raw, False)[0], expected)
        self.assertEqual(read_answer(raw, False, answer_only=True)[0], lead)
        async def generate(_body):
            return {"response":leaked, "done_reason":"stop"}
        self.assertEqual(asyncio.run(_plain_retry(generate, "비율을 설명해 주세요.", False)), expected)

    def test_metadata_is_distinct_from_an_actual_school_classification(self):
        lead = "방정식의 유형은 일차방정식입니다."
        formula = r"$x=-2.5,\ y=\frac{1}{4}$"
        reason = "유형: 서술형\n이 유형은 풀이 과정을 평가합니다.\n" + formula
        for metadata in ("‘유형’： “자료”。", '"유형": "풀이",', "유형:\n‘범위밖’."):
            raw = json.dumps({"답변":"답변: 정답: " + lead,
                              "근거":"근거: 근거: " + reason + "\n" + metadata, "유형":"풀이"})
            answer = read_answer(raw, False)[0]
            self.assertTrue(answer.startswith(lead))
            self.assertIn("유형: 서술형", answer)
            self.assertIn("이 유형은 풀이 과정을 평가합니다.", answer)
            self.assertIn(formula, answer)
            self.assertNotIn(metadata, answer)

    def test_complete_answer_can_stand_alone(self):
        answer = "달은 지구 주위를 도는 위성이고, 태양빛을 반사해 보입니다."
        raw = json.dumps({"답변": answer, "근거": "", "유형": "자료"})
        self.assertEqual(read_answer(raw, True), (answer, True, "answer"))
        self.assertEqual(read_answer(json.dumps({"답변": answer, "유형": "자료"}), True),
                         (None, False, "answer"))

    def test_length_cap_keeps_only_completed_public_answer(self):
        text = '{"답변":"지구의 위성입니다.", "근거":"태양빛이'
        self.assertEqual(read_answer(text, True, length_limited=True), ("지구의 위성입니다.", False, "answer"))
        self.assertEqual(read_answer(text, True), (None, False, "answer"))
        self.assertEqual(read_answer('{"답변":"<think>숨김</think>","근거":"', False, length_limited=True),
                         (None, False, "answer"))

    def test_unrequested_extra_example_follows_the_complete_reason(self):
        payload = json.dumps({"답변":"가격이 내려갑니다.", "근거":"공급 곡선이 오른쪽으로 이동합니다. 예를 들어, 생산비가 줄면 공급이 증가합니다.", "유형":"풀이"})
        short = read_answer(payload, False, question="공급이 늘면 균형 가격은 어떻게 되나요?")[0]
        self.assertEqual(short, "가격이 내려갑니다.\n공급 곡선이 오른쪽으로 이동합니다.")
        example = read_answer(payload, False, question="왜 그런지 예시도 보여 주세요.")[0]
        self.assertIn("생산비가 줄면", example)

    def test_failed_generation_can_quote_exact_relevant_causal_evidence(self):
        from src.api.ask_quality import verify_reported_claim
        q = "보름달은 스스로 빛을 내는 별이라고 들었어요. 그래서 밤에 밝게 보이는 건가요?"
        docs = search_concept_map("과학", q, 3)
        answer, grounded = verify_reported_claim("과학", q, None, docs)
        self.assertTrue(grounded)
        self.assertIn("자연 위성", answer)
        self.assertIn("반사", answer)
        for excerpt in answer.splitlines():
            self.assertIn(excerpt.removeprefix("교과 자료 · "), docs[0])
        self.assertEqual(verify_reported_claim("과학", q, None, []), (None, False))
        for unsupported in ("보름달은 거울 같다고 들었어요. 표면의 온도는 몇 도인가요?",
                            "보름달의 온도는 얼마인가요?"):
            self.assertEqual(verify_reported_claim("과학", unsupported, None, docs), (None, False))

    def test_omit_repeated_wording_and_keep_new_support(self):
        raw = json.dumps({"유형": "자료", "답변": "기체의 부피가 증가합니다.",
                          "근거": "기체의 부피가 증가한다. 입자가 더 넓은 공간을 차지해요."})
        self.assertEqual(read_answer(raw, True)[0], "기체의 부피가 증가합니다.\n입자가 더 넓은 공간을 차지해요.")
        self.assertTrue(wants_answer_only("기체 종류는? 정답만 간단히 알려줘."))
        self.assertFalse(wants_answer_only("정답만 알려주지 말고 설명해 줘."))

    def test_evidence_keeps_polarity_direction_and_math(self):
        cases = [("용액이 산성입니다.", "용액이 산성이 아닙니다."),
                 ("A이면 B가 증가합니다.", "B이면 A가 증가합니다."),
                 ("x = -2", "2"), ("값은 2.5입니다.", "5"),
                 ("총 3200원입니다.", "800 × 4 = 3200")]
        for answer, reason in cases:
            with self.subTest(reason=reason):
                raw = json.dumps({"답변": answer, "근거": reason, "유형": "풀이"})
                self.assertEqual(read_answer(raw, False)[0], answer + "\n" + reason)

    def test_standalone_english_cue_has_context(self):
        raw = json.dumps({"답변": "그때부터 지금까지 계속 살고 있습니다.", "근거": "since 2020", "유형": "자료"})
        answer = read_answer(raw, True)[0]
        self.assertIn("판단에 쓰인 표현은 ‘since 2020’입니다.", answer)
        self.assertFalse(read_answer(raw, True, answer_only=True)[0].endswith("since 2020’입니다."))
        answer = "‘since 2020’에서 2020년이 시작 시점임을 알 수 있습니다."
        raw = json.dumps({"답변": "정답: " + answer, "근거": "근거: since 2020", "유형": "자료"})
        self.assertEqual(read_answer(raw, True)[0], answer)

    def test_reported_claim_adds_a_matching_definition(self):
        from src.api.ask_quality import verify_reported_claim
        text, cited = verify_reported_claim("과학", "초승달은 빛을 낸다고 들었는데 사실인가요?", "햇빛을 반사합니다.")
        self.assertTrue(cited)
        self.assertIn("자연 위성", text)
        self.assertIn("햇빛을 반사", text)
        self.assertEqual(verify_reported_claim("과학", "관련 자료를 찾아줘.", "일반 설명."),
                         ("일반 설명.", False))

    def test_live_availability_and_quantitative_examples(self):
        from src.api.ask_quality import availability_notice
        self.assertIn("예매처", availability_notice("지금 열차의 잔여 좌석이 몇 개 있나요?"))
        self.assertIn("매장", availability_notice("우리 동네 편의점 재고가 얼마 남았나요?"))
        for q in ("매장에서 현재 재고가 8개인데 3개를 사면 몇 개 남나요?",
                  "지금 매장에서 재고를 줄인다면 수요는 어떻게 달라지나요?",
                  "4번 문제. 지금 편의점에서 800원짜리 지우개 4개를 사면?"):
            self.assertIsNone(availability_notice(q))

    def test_worked_equation_keeps_exact_intermediate_step(self):
        solution = solve_linear("4번 문제: 3x + 1 = 10일 때 x와 풀이를 알려 주세요.")
        self.assertTrue(solution.startswith("x = 3\n"))
        self.assertIn("3x = 9", solution)
        self.assertIn("\n계수로 양변을 나눕니다. x = 9 ÷ 3 = 3", solution)

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

    def test_completed_fields_can_be_in_any_order(self):
        text = '{"답변":"2개입니다.","근거":"교점이 둘입니다.","유형":"자'
        self.assertEqual(read_answer(text, True, length_limited=True),
                         ("2개입니다.\n교점이 둘입니다.", False, "answer"))
        text = '{"유형":"자료","답변":"2개입니다.","근거":"교점이 둘입니다."'
        self.assertEqual(read_answer(text, True, length_limited=True),
                         ("2개입니다.\n교점이 둘입니다.", True, "answer"))

    def test_early_stop_inside_answer_triggers_recovery_and_never_exposes_thinking(self):
        broken = '{"근거":"be + 과거분사 형태입니다.","답변":"was made}</think>'
        self.assertEqual(read_answer(broken, True), (None, False, "answer"))
        broken = json.dumps({"근거":"<think>hidden</think>", "답변":"was made", "유형":"자료"})
        self.assertEqual(read_answer(broken, True), (None, False, "answer"))

    def test_plain_recovery_preserves_words_and_discards_incomplete_tail(self):
        async def generate(body):
            self.assertNotIn("format", body)
            return {"response":"동사 표현은 was made입니다. was에 이어지는", "done_reason":"length"}
        self.assertEqual(asyncio.run(_plain_retry(generate, "합성 문제", False)), "동사 표현은 was made입니다.")

        async def internal(_body):
            return {"response":"<think>internal</think>결론", "done_reason":"stop"}
        self.assertIsNone(asyncio.run(_plain_retry(internal, "합성 문제", True)))
        async def labeled(_body):
            return {"response": "정답: 총 300원입니다.\n풀이: 100 × 3 = 300", "done_reason": "stop"}
        self.assertEqual(asyncio.run(_plain_retry(labeled, "합성 문제", False)), "총 300원입니다.\n100 × 3 = 300")
        self.assertEqual(asyncio.run(_plain_retry(labeled, "합성 문제", True)), "총 300원입니다.")
        async def two_answers(_body):
            return {"response": "x = 3\ny = -2\n근거: 대입합니다.", "done_reason": "stop"}
        self.assertEqual(asyncio.run(_plain_retry(two_answers, "두 값만", True)), "x = 3\ny = -2")


if __name__ == "__main__":
    unittest.main()

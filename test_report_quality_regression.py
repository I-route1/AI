"""Deterministic regressions for report text and optional inference."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import AsyncMock, Mock, patch

from src.api.concept_map import search_concept_map
from src.api.report_text import complete_generation, material_excerpt

# 보고서 포맷 회귀 시험에는 2GB 검색 인덱스나 임베딩 모델이 필요하지 않다.
# 라우터의 다른 모듈 인스턴스에만 실제 ConceptMap 조회를 주입한다.
rag = ModuleType("src.api.routers.rag")
rag.detect_subject = lambda _: None
rag.subject_aware_search = search_concept_map
rag.vector_search = lambda *args, **kwargs: []
rag._SUBJECT_KEYWORDS = dict.fromkeys(("수학", "국어", "영어", "과학", "사회"))
spec = importlib.util.spec_from_file_location("report_regression_router",
    Path(__file__).parent / "src/api/routers/counseling.py")
counseling = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"src.api.routers.rag": rag}):
    spec.loader.exec_module(counseling)


class ReportFactsTest(TestCase):
    def test_missing_grade_and_hours_are_not_student_measurements(self):
        for req in ({}, {"studyTime": 0.0, "currentKoreanGrade": 0.0},
                    {"studyTime": "invalid", "subjectPercentile": 103, "currentKoreanGrade": -3},
                    {"studyTime": 28, "subjectPercentile": True}):
            for builder in (lambda: counseling._subject_report("영어", req, "")[0],
                            lambda: counseling._math_report(req, "")[0],
                            lambda: counseling._writing_report(req, ""),
                            lambda: counseling._premium_report(req, "")):
                with self.subTest(req=req), patch.object(counseling, "subject_aware_search", return_value=[]):
                    result = builder()
                    analysis = result["careerAnalysis"]
                    self.assertIn("성적 정보 없음", analysis)
                    self.assertIn("학습 시간 기록 없음", analysis)
                    self.assertNotIn("중위권", analysis)
                    self.assertNotRegex(analysis, r"\d+시간")

    def test_proxy_grade_is_distinguished_from_this_subject(self):
        with patch.object(counseling, "subject_aware_search", return_value=[]):
            result, _ = counseling._subject_report("영어", {"currentKoreanGrade": 82}, "")
        self.assertIn("영어 백분위 미등록", result["careerAnalysis"])
        self.assertIn("참고용 국어 백분위 82%", result["careerAnalysis"])
        self.assertNotIn("상위권", result["careerAnalysis"])
        self.assertNotIn("고난도 빈칸", result["learningGuide"])
        self.assertIn("다음 시험 성적", result["learningGuide"])

    def test_report_keeps_concept_feedback_and_time_bound(self):
        req = {"studentId": 7, "subjectPercentile": 71.5, "studyTime": .25,
               "instructorFeedback": "근거를 한 문장으로 말해 보세요."}
        notes = search_concept_map("영어", "수동태", 1)
        self.assertTrue(notes)
        with patch.object(counseling, "subject_aware_search", return_value=notes):
            result, _ = counseling._subject_report("영어", req, "수동태")
        self.assertEqual(7, result["studentId"])
        self.assertIn("영어 백분위 71.5%", result["careerAnalysis"])
        self.assertIn(req["instructorFeedback"], result["careerAnalysis"])
        self.assertIn("과거분사", result["careerAnalysis"])
        self.assertNotIn("5형식", result["careerAnalysis"])
        self.assertIn("[스스로 확인]", result["learningGuide"])
        self.assertTrue(result["careerAnalysis"].strip().endswith("."))
        self.assertIn("‘수동태’의 예제 한 문제", result["learningGuide"])
        self.assertIn("15분 중 15분", result["learningGuide"])
        self.assertNotIn("부족로", result["learningGuide"])

    def test_social_science_excerpt_and_legitimate_error_topic(self):
        for subject, concept, example in [("사회", "훈민정음", "1446년"), ("과학", "응결", "물방울")]:
            docs = search_concept_map(subject, concept, 1)
            with self.subTest(subject=subject):
                self.assertTrue(docs)
                excerpt = counseling._material_lines(docs, concept).removeprefix("• ")
                self.assertIn(excerpt, docs[0])
                self.assertIn(concept, excerpt)
                self.assertIn(example, excerpt)
                self.assertLessEqual(len(excerpt), 145)
        self.assertIn("표집 오류", counseling._material_lines(["표집 오류가 발생할 수 있다.", "검색 오류: timeout"], "표집"))
        self.assertEqual("자료를 읽습니다.", material_excerpt("자료를 읽습니다. " + "가나다"*20, 20))

    def test_premium_uses_actual_weak_subject_and_avoids_predicted_grade(self):
        req = {"weakSubject":"과학", "subjectPercentile":79, "instructorFeedback":"단위를 확인하세요."}
        with patch.object(counseling, "subject_aware_search", return_value=[]):
            result = counseling._premium_report(req, "운동량")
        self.assertIn("과학 ‘운동량’", result["learningGuide"])
        self.assertNotIn("99%", result["learningGuide"])

    def test_writing_uses_different_checks_with_the_same_observed_concept(self):
        guides = [counseling._writing_report({"subjectPercentile": percentile}, "비유")["learningGuide"]
                  for percentile in (40, 65, 81)]
        self.assertTrue(all("비유" in guide for guide in guides))
        self.assertIn("낱말", guides[0])
        self.assertIn("자신의 해석", guides[1])
        self.assertIn("두 해석", guides[2])

    def test_history_activity_uses_dates_shown_in_this_units_source(self):
        for concept, docs, expected in (
                ("훈민정음", search_concept_map("사회", "훈민정음", 1), ("1443년", "1446년")),
                ("선거 제도", ["선거 제도 자료에 1948년과 1987년의 사례가 실려 있다."], ("1948년", "1987년"))):
            with self.subTest(concept=concept), patch.object(counseling, "subject_aware_search", return_value=docs):
                result, _ = counseling._subject_report("사회", {"subjectPercentile": 82}, concept)
                step = result["learningGuide"].split("2단계")[1]
                for term in (concept, *expected, "두 기록의 차이"):
                    self.assertIn(term, step)
                self.assertNotIn("시사 이슈", step)

    def test_general_social_and_math_checks_stay_with_the_concept(self):
        cases = (("사회", "기회비용", "사례 하나"), ("수학", "일차방정식", "조건 하나"))
        for subject, concept, expected in cases:
            with self.subTest(subject=subject), patch.object(counseling, "subject_aware_search",
                    return_value=["다른 단원에 1443년과 1446년이 나온다."]):
                result, _ = counseling._subject_report(subject, {"subjectPercentile": 52}, concept)
                step = result["learningGuide"].split("2단계")[1]
                self.assertIn(concept, result["learningGuide"])
                self.assertIn(expected, step)
                self.assertNotIn("1443년", step)

    def test_cut_text_and_valid_equation(self):
        self.assertEqual('확률은 0.25입니다.', complete_generation('확률은 0.25입니다. 다음으로 식에'))
        self.assertEqual('', complete_generation('식에 0.2를 곱한'))
        self.assertEqual(r'$\frac{1}{3}$', complete_generation(r'$\frac{1}{3}$'))
        self.assertEqual('x = -2.5', complete_generation('x = -2.5'))
        self.assertEqual('', complete_generation(r'$\frac{1}{'))

    def test_passage_keeps_formula_and_picks_the_requested_idea(self):
        note = "다른 개념을 설명한다. 농도는 0.25 mol/L이다. 이때 농도와 단위를 함께 확인한다. 마지막으로 다른 개념을 다룬다."
        self.assertEqual("농도는 0.25 mol/L이다. 이때 농도와 단위를 함께 확인한다.",
                         material_excerpt(note, 80, "농도"))
        self.assertEqual("", counseling._material_lines(["다른 단원에 관한 별개의 설명입니다."], "확률"))

    def test_missing_concept_starts_from_recent_work_and_no_random_materials(self):
        with patch.object(counseling, "subject_aware_search") as search:
            result, _ = counseling._subject_report("수학", {}, "")
            self.assertNotIn("[교과 자료]", result["careerAnalysis"])
            self.assertIn("최근 배운 단원", result["learningGuide"])
            self.assertIn("[스스로 확인]", result["learningGuide"])
            search.assert_not_called()


class ReportModelTest(IsolatedAsyncioTestCase):
    async def test_length_limit_drops_unfinished_ollama_tail(self):
        client = AsyncMock()
        client.post.return_value = Mock(status_code=200)
        client.post.return_value.json.return_value={"done_reason":"length", "response":"수동태는 be와 과거분사를 씁니다. 따라서 주어에"}
        self.assertEqual("수동태는 be와 과거분사를 씁니다.", await counseling._ollama_once(client, "prompt", 5, 100))
        client.post.return_value.json.return_value={"done_reason":"stop","response":"2x + 1"}
        self.assertEqual("2x + 1", await counseling._ollama_once(client, "prompt", 5, 100))
        client.post.return_value.json.return_value={"done_reason":"length","response":"<think>내부 분석 내용입니다."}
        self.assertIsNone(await counseling._ollama_once(client, "prompt", 5, 100))

    async def test_cpu_advice_has_one_bounded_generation(self):
        async_mock=AsyncMock(return_value="예문을 쓰세요.")
        with patch.object(counseling,"OLLAMA_CPU",True), patch.object(counseling,"_ollama_analyze",async_mock):
            self.assertEqual("예문을 쓰세요.", await counseling._report_ollama("수동태"))
        self.assertIn("두 문장", async_mock.await_args.args[0])
        self.assertLess(async_mock.await_args.kwargs["timeout"], 50)
        self.assertFalse(async_mock.await_args.kwargs["retry_foreign"])
        self.assertLess(async_mock.await_args.kwargs["max_tokens"], 150)

    async def test_model_failure_preserves_real_report(self):
        with patch.object(counseling,"_report_ollama",AsyncMock(return_value=None)), \
             patch.object(counseling.model_registry,"get",return_value=None):
            result = await counseling.report_subject("영어", {"studentId":7,"subjectPercentile":72,
                                                            "studyTime":1.5,"weakConcept":"수동태"})
        self.assertIn("영어 백분위 72%",result["careerAnalysis"])
        self.assertNotIn("[AI 개념 설명]",result["careerAnalysis"])
        self.assertIn("수동태",result["learningGuide"])

    async def test_advice_and_observations_appear_in_separate_fields(self):
        req = {"studentId": 7, "weakConcept": "비유", "weakSubject": "국어",
               "subjectPercentile": 81, "instructorFeedback": "구절을 인용해 보세요."}
        advice = "고른 구절을 가리고 뜻을 설명해 보세요."
        with patch.object(counseling, "_advice", AsyncMock(return_value=advice)):
            for endpoint in (counseling.report_writing, counseling.report_premium):
                result = await endpoint(req)
                self.assertIn(req["instructorFeedback"], result["careerAnalysis"])
                self.assertNotIn(advice, result["careerAnalysis"])
                self.assertIn("[AI 복습 제안]\n" + advice, result["learningGuide"])


if __name__ == "__main__":
    main()

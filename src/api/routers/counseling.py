from fastapi import APIRouter
from src.api.routers.rag import vector_search

router = APIRouter()

# AIHUB 상담 데이터 (실제 연동 전 대표 샘플)
_COUNSELING_DATA = {
    "summary": "스포츠에 관심을 가지고 있으며 스포츠 관련 기자가 되기를 희망하고 있다.",
    "expert_comment": "1순위로는 언어 관련 전문 직종을 추천할 수 있다. 글쓰기 역량 강화가 핵심이다.",
}


def _base_report(req: dict, title: str, focus: str) -> dict:
    student_id = req.get("studentId", "Unknown")
    korean_grade = req.get("currentKoreanGrade", 0)
    study_time = req.get("studyTime", 0)
    student_note = req.get("studentNote", "")
    recommend_ctx = req.get("recommendContext", "")

    career_analysis = (
        f"[{focus}] {_COUNSELING_DATA['summary']} "
        f"학생 메모: '{student_note}'. "
        f"AI 추천 컨텍스트: {recommend_ctx}"
    )
    learning_guide = (
        f"{_COUNSELING_DATA['expert_comment']} "
        f"현재 국어 백분위 {korean_grade}%, 일일 학습 시간 {study_time}시간 기준 "
        f"맞춤형 학습 전략을 제안합니다."
    )

    return {
        "studentId": student_id,
        "title": title,
        "careerAnalysis": career_analysis,
        "learningGuide": learning_guide,
    }


# ── POST /api/ai/report/math ───────────────────────────────────────────────────
@router.post("/report/math")
async def report_math(req: dict):
    return _base_report(
        req,
        title="수학 메타인지 분석 리포트",
        focus="수학 사고력 및 오답 패턴 분석",
    )


# ── POST /api/ai/report/writing ────────────────────────────────────────────────
@router.post("/report/writing")
async def report_writing(req: dict):
    return _base_report(
        req,
        title="AI 기반 진로 탐색 리포트",
        focus="언어·작문 역량 및 진로 적합성 분석",
    )


# ── POST /api/ai/report/premium ────────────────────────────────────────────────
@router.post("/report/premium")
async def report_premium(req: dict):
    return _base_report(
        req,
        title="i-Route 프리미엄 통합 AI 진단 리포트",
        focus="수학·언어·진로 종합 분석",
    )


# ── POST /api/ai/search ────────────────────────────────────────────────────────
# GradeAnalysisService 에서 호출 — contexts 리스트 반환
@router.post("/search")
async def ai_search(req: dict):
    query = req.get("question", "")
    contexts = vector_search(query, k=5)
    return {"contexts": contexts}

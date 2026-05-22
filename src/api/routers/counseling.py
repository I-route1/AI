from fastapi import APIRouter
from src.api.schemas.counseling import ReportGenerationRequest

router = APIRouter()

@router.post("/report")
async def generate_academy_report(req: ReportGenerationRequest):

    # AIHUB 원본 상담 데이터 로드 (가상 예시)
    counseling_data = {
        "summary": "스포츠에 관심을 가지고 있으며... 스포츠 관련 기자가 되기를 희망하고 있다.",
        "expert_comment": "1순위로는 언어 관련 전문 직종을 추천할 수 있다..."
    }

    # LLaMA 3.1 텍스트 생성 시뮬레이션
    generated_report = f"""
    # i-Route 프리미엄 AI 진단 리포트
    
    **학생 코드:** {req.studentId}
    **국어 성적:** 상위 {req.currentKoreanGrade}%
    
    🎤 **진로 동기 분석 및 공감**
    {counseling_data['summary']}를 바탕으로 훌륭한 열정을 확인했습니다.
    
    🚀 **학원 연계 학습 가이드**
    전문가 소견({counseling_data['expert_comment']})에 따라 국어 역량 강화가 필수적입니다.
    """

    return {
        "studentId": req.studentId,
        "reportHtml": generated_report
    }
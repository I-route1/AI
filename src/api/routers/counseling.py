import httpx
from fastapi import APIRouter
from src.api.routers.rag import subject_aware_search, vector_search, _SUBJECT_KEYWORDS

router = APIRouter()


async def _ollama_analyze(prompt: str, timeout: float = 20.0) -> str | None:
    """Ollama llama3.1으로 짧은 분석 생성. 실패/타임아웃 시 None 반환."""
    import logging
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.1:latest",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 150, "temperature": 0.4},
                },
                timeout=timeout,
            )
            if r.status_code == 200:
                return r.json().get("response", "").strip()
            logging.warning(f"[Ollama] status {r.status_code}: {r.text[:200]}")
    except Exception as e:
        logging.warning(f"[Ollama] 연결 실패: {e}")
    return None


def _level_label(percentile: float) -> str:
    if percentile >= 90: return "최상위권"
    if percentile >= 75: return "상위권"
    if percentile >= 50: return "중위권"
    if percentile >= 30: return "중하위권"
    return "기초 학습 필요"


def _study_intensity(hours: float) -> str:
    if hours >= 3: return "충분한 학습량"
    if hours >= 2: return "적정 학습량"
    if hours >= 1: return "학습량 보강 필요"
    return "학습량 매우 부족"


def _math_report(req: dict) -> dict:
    student_id     = req.get("studentId", "")
    percentile     = float(req.get("currentKoreanGrade") or 50)
    study_hours    = float(req.get("studyTime") or 1)
    student_note   = req.get("studentNote") or ""
    recommend_ctx  = req.get("recommendContext") or ""
    feedback       = req.get("instructorFeedback") or ""

    level   = _level_label(percentile)
    stamina = _study_intensity(study_hours)

    # RAG로 수학 취약 개념 관련 학습 자료 검색 (과목 필터 적용)
    rag_docs  = subject_aware_search("수학", recommend_ctx or "방정식 함수 기본 개념", k=3)
    rag_text  = "\n".join(f"  • {d[:120]}" for d in rag_docs if d and "오류" not in d)

    career_analysis = (
        f"[수학 사고력 및 오답 패턴 분석]\n\n"
        f"● 현재 학업 수준: {level} (국어 백분위 {percentile:.0f}%)\n"
        f"● 학습 강도: {stamina} (일 {study_hours:.1f}시간)\n"
    )
    if student_note:
        career_analysis += f"● 학습 특성: {student_note}\n"
    if feedback:
        career_analysis += f"● 최근 강사 피드백: {feedback}\n"
    if recommend_ctx:
        career_analysis += f"\n[취약 개념 집중 분석]\n{recommend_ctx}\n"
    if rag_text:
        career_analysis += f"\n[관련 학습 자료 검색 결과]\n{rag_text}\n"

    learning_guide = (
        f"[수학 맞춤 학습 전략]\n\n"
        f"1단계 — 오답 유형 분류: 계산 실수 / 개념 미이해 / 응용력 부족으로 구분 후 원인별 대응\n"
        f"2단계 — 취약 개념 집중: '{recommend_ctx or '핵심 공식'}' 단원 기출 10문항 반복\n"
        f"3단계 — 오답 노트 작성: 틀린 문제를 다시 풀어 풀이 과정을 직접 설명\n"
        f"4단계 — 실력 점검: 단원 마무리 후 실전 모의고사로 확인\n\n"
        f"현재 {level} 기준 권장 일일 학습: "
        f"{'수학 1.5시간 + 오답 정리 30분' if percentile >= 75 else '개념 1시간 + 문제풀이 1시간 + 복습 30분'}"
    )

    # math 어댑터는 문제 풀이 특화 파인튜닝으로 학습 조언 프롬프트에 부적합 — LLM 미사용

    return {
        "studentId": student_id,
        "title": "수학 메타인지 분석 리포트",
        "careerAnalysis": career_analysis,
        "learningGuide": learning_guide,
    }


def _writing_report(req: dict) -> dict:
    student_id    = req.get("studentId", "")
    percentile    = float(req.get("currentKoreanGrade") or 50)
    study_hours   = float(req.get("studyTime") or 1)
    student_note  = req.get("studentNote") or ""
    recommend_ctx = req.get("recommendContext") or ""
    feedback      = req.get("instructorFeedback") or ""

    level = _level_label(percentile)

    # RAG로 국어 관련 학습 자료 검색 (과목 필터 적용)
    rag_docs  = subject_aware_search("국어", student_note or "문학 독해 언어 능력", k=3)
    rag_text  = "\n".join(f"  • {d[:120]}" for d in rag_docs if d and "오류" not in d)

    career_analysis = (
        f"[언어·작문 역량 및 진로 적합성 분석]\n\n"
        f"● 국어 학업 수준: {level} (백분위 {percentile:.0f}%)\n"
        f"● 일일 학습 시간: {study_hours:.1f}시간\n"
    )
    if student_note:
        career_analysis += f"● 학습 성향 및 특성: {student_note}\n"
    if feedback:
        career_analysis += f"● 강사 관찰 소견: {feedback}\n"
    if recommend_ctx:
        career_analysis += f"\n[진로 추천 근거]\n{recommend_ctx}\n"
    if rag_text:
        career_analysis += f"\n[관련 진로·학습 자료]\n{rag_text}\n"

    if percentile >= 80:
        career_path = "언어·미디어 계열 (기자, 작가, 출판 편집자, 광고 카피라이터)"
        guide_focus = "심화 독해 + 논술 + 다양한 장르 글쓰기 경험"
    elif percentile >= 60:
        career_path = "교육·상담·사회복지 계열"
        guide_focus = "비문학 독해 강화 + 핵심 주장 파악 훈련"
    else:
        career_path = "실용 언어 능력을 바탕으로 한 서비스·유통 계열"
        guide_focus = "어휘력 확장 + 단문 요약 훈련부터 시작"

    learning_guide = (
        f"[진로 맞춤 언어 학습 전략]\n\n"
        f"● 추천 진로 계열: {career_path}\n\n"
        f"● 핵심 역량 강화 방향: {guide_focus}\n\n"
        f"주간 학습 계획:\n"
        f"  - 월·수·금: 비문학 지문 1편 정독 + 핵심어 3개 추출\n"
        f"  - 화·목: 문학 작품 1편 감상 + 서술자 시점 분석\n"
        f"  - 주말: 500자 자유 글쓰기 + 자기 평가\n"
    )

    # writing 어댑터는 글쓰기 평가 특화 파인튜닝으로 진로 조언 프롬프트에 부적합 — LLM 미사용

    return {
        "studentId": student_id,
        "title": "AI 기반 진로 탐색 리포트",
        "careerAnalysis": career_analysis,
        "learningGuide": learning_guide,
    }


def _premium_report(req: dict) -> dict:
    student_id    = req.get("studentId", "")
    percentile    = float(req.get("currentKoreanGrade") or 50)
    study_hours   = float(req.get("studyTime") or 1)
    student_note  = req.get("studentNote") or ""
    recommend_ctx = req.get("recommendContext") or ""
    feedback      = req.get("instructorFeedback") or ""

    level   = _level_label(percentile)
    stamina = _study_intensity(study_hours)

    # 프리미엄 리포트: 주 취약 과목 기준으로 RAG 검색
    _premium_subject = "수학" if "수학" in (recommend_ctx or "") else \
                       "영어" if "영어" in (recommend_ctx or "") else \
                       "국어" if "국어" in (recommend_ctx or "") else "수학"
    rag_docs  = subject_aware_search(_premium_subject, recommend_ctx or student_note or "기본 개념", k=4)
    rag_text  = "\n".join(f"  • {d[:130]}" for d in rag_docs if d and "오류" not in d)

    career_analysis = (
        f"[i-Route 프리미엄 종합 학습 진단]\n\n"
        f"━━ 학업 현황 ━━\n"
        f"● 학업 수준: {level} (국어 백분위 {percentile:.0f}%)\n"
        f"● 학습 강도: {stamina} (일 {study_hours:.1f}시간)\n"
    )
    if student_note:
        career_analysis += f"● 학습 특성: {student_note}\n"
    if feedback:
        career_analysis += f"● 강사 종합 평가: {feedback}\n"
    if recommend_ctx:
        career_analysis += f"\n━━ 취약 영역 집중 진단 ━━\n{recommend_ctx}\n"
    if rag_text:
        career_analysis += f"\n━━ RAG 기반 연관 학습 데이터 ━━\n{rag_text}\n"

    if percentile >= 85:
        strategy = (
            "최상위권 유지 전략: 실수 제로화에 집중하세요.\n"
            "  - 수학: 킬러문항 집중 훈련, 시간 단축 연습\n"
            "  - 영어: 고난도 빈칸·순서 유형 반복\n"
            "  - 국어: 고전 문학 + 현대소설 심층 독해"
        )
    elif percentile >= 60:
        strategy = (
            "상승 도약 전략: 취약 단원을 집중 공략하세요.\n"
            "  - 수학: 오답 유형 분류 후 유형별 집중 풀이\n"
            "  - 영어: 구문 독해 → 유형별 문제풀이 순서로\n"
            "  - 국어: 비문학 지문 구조 파악 훈련"
        )
    else:
        strategy = (
            "기초 강화 전략: 개념 이해를 최우선으로 하세요.\n"
            "  - 수학: 교과서 개념부터 순서대로 재정리\n"
            "  - 영어: 핵심 문법 5개 + 단어 20개/일 암기\n"
            "  - 국어: 짧은 지문 요약 훈련으로 독해력 향상"
        )

    learning_guide = (
        f"[프리미엄 맞춤 전략 리포트]\n\n"
        f"{strategy}\n\n"
        f"━━ 주간 학습 로드맵 ━━\n"
        f"  월: 수학 오답 노트 정리 + 취약 유형 5문항\n"
        f"  화: 영어 구문 독해 1지문 + 어휘 정리\n"
        f"  수: 국어 비문학 1지문 + 선지 분석\n"
        f"  목: 수학 심화 문제 + 강사 피드백 반영\n"
        f"  금: 3과목 약점 점검 미니 테스트\n"
        f"  주말: 전 주 복습 + 다음 주 예습 30분씩\n\n"
        f"━━ 목표 설정 ━━\n"
        f"  현재 {percentile:.0f}% → "
        f"{'99% 목표: 실수 0건' if percentile >= 85 else f'{min(100, percentile+15):.0f}% 목표: 취약 단원 집중 공략'}"
    )

    return {
        "studentId": student_id,
        "title": "i-Route 프리미엄 통합 AI 진단 리포트",
        "careerAnalysis": career_analysis,
        "learningGuide": learning_guide,
    }


@router.post("/report/math")
async def report_math(req: dict):
    result = _math_report(req)
    concept = req.get("recommendContext", "").strip()
    if concept:
        prompt = (
            f"한국 중고등학생이 수학 '{concept}' 개념을 어려워합니다. "
            f"이 개념에서 학생들이 가장 자주 하는 핵심 실수 1가지와 "
            f"그것을 극복하는 구체적인 학습 전략을 2~3문장으로 간결하게 한국어로 답해주세요."
        )
        llm_insight = await _ollama_analyze(prompt)
        if llm_insight:
            result["careerAnalysis"] += f"\n\n[AI 개념 심층 분석]\n{llm_insight}"
    return result


@router.post("/report/writing")
async def report_writing(req: dict):
    result = _writing_report(req)
    note = req.get("studentNote", "").strip()
    percentile = float(req.get("currentKoreanGrade") or 50)
    prompt = (
        f"국어 백분위 {percentile:.0f}%인 학생의 학습 특성: '{note or '특이사항 없음'}'. "
        f"이 학생의 국어(문학·비문학·작문) 약점을 보완하기 위한 "
        f"가장 효과적인 학습 방법 2가지를 2~3문장으로 구체적으로 한국어로 답해주세요."
    )
    llm_insight = await _ollama_analyze(prompt)
    if llm_insight:
        result["careerAnalysis"] += f"\n\n[AI 맞춤 학습 제안]\n{llm_insight}"
    return result


@router.post("/report/premium")
async def report_premium(req: dict):
    result = _premium_report(req)
    concept = req.get("recommendContext", "").strip()
    percentile = float(req.get("currentKoreanGrade") or 50)
    note = req.get("studentNote", "").strip()
    prompt = (
        f"학생 정보 — 백분위: {percentile:.0f}%, 취약 영역: '{concept or '전반적'}', 특성: '{note or '없음'}'. "
        f"이 학생이 다음 단계로 성적을 올리기 위해 가장 먼저 해야 할 핵심 행동 2가지를 "
        f"구체적이고 실행 가능한 형태로 2~3문장 한국어로 답해주세요."
    )
    llm_insight = await _ollama_analyze(prompt)
    if llm_insight:
        result["careerAnalysis"] += f"\n\n[AI 우선순위 행동 제안]\n{llm_insight}"
    return result


def _detect_subject(query: str) -> str | None:
    """쿼리에서 과목을 자동 감지. 매칭 없으면 None."""
    for subject, keywords in _SUBJECT_KEYWORDS.items():
        if any(kw in query for kw in keywords):
            return subject
    return None


@router.post("/search")
async def ai_search(req: dict):
    query = req.get("question", "")
    subject = req.get("subject") or _detect_subject(query)
    if subject:
        contexts = subject_aware_search(subject, query, k=5)
    else:
        contexts = vector_search(query, k=5)
    return {"contexts": contexts}

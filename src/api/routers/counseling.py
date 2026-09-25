import httpx
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from src.api import model_registry
from src.api.concept_map import search_concept_map
from src.api.generation import OLLAMA_MODEL, OLLAMA_OPTIONS, OLLAMA_URL
from src.api.java_client import get_student_weakness_from_java
from src.api.grounding import context_block, is_curated
from src.api.postprocess import strip_markdown
from src.api.script_guard import has_foreign, strip_foreign
from src.api.routers.rag import subject_aware_search, vector_search, _SUBJECT_KEYWORDS

router = APIRouter()

# 과목별 기본 검색어 (recommendContext 없을 때 RAG 검색 폴백용)
_DEFAULT_QUERY = {
    "수학":   "방정식 함수 기본 개념",
    "영어":   "구문 독해 어법 기본",
    "국어":   "문학 독해 언어 능력",
    "과학":   "물리 화학 생물 기본 개념",
    "사회":   "사회 역사 지리 기본 개념",
    "한국사": "한국사 시대별 핵심 사건",
}

# 과목별 오답 유형 (수학 외 과목에서 "계산 실수" 대신 쓸 표현)
_MISTAKE_TYPES = {
    "수학":   "계산 실수 / 개념 미이해 / 응용력 부족",
    "영어":   "어휘·문법 실수 / 구문 해석 오류 / 추론력 부족",
    "국어":   "어휘력 부족 / 지문 구조 파악 실패 / 선지 함정 오독",
    "과학":   "개념 혼동 / 단위·계산 실수 / 실험 해석 오류",
    "사회":   "암기 부족 / 개념 간 혼동 / 자료 해석 오류",
    "한국사": "연대·인물 암기 부족 / 사건 인과관계 혼동 / 사료 해석 오류",
}

# 과목별 백분위 구간(상/중/하)에 따른 학습 전략
_STRATEGY = {
    "수학":   {"high": "킬러문항 집중 훈련, 시간 단축 연습", "mid": "오답 유형 분류 후 유형별 집중 풀이", "low": "교과서 개념부터 순서대로 재정리"},
    "영어":   {"high": "고난도 빈칸·순서 유형 반복", "mid": "구문 독해 → 유형별 문제풀이 순서로", "low": "핵심 문법 5개 + 단어 20개/일 암기"},
    "국어":   {"high": "고전 문학 + 현대소설 심층 독해", "mid": "비문학 지문 구조 파악 훈련", "low": "짧은 지문 요약 훈련으로 독해력 향상"},
    "과학":   {"high": "실험·탐구 서술형 고난도 문항 반복", "mid": "단원별 핵심 개념 정리 후 응용 문제 풀이", "low": "교과서 개념 정의부터 순서대로 재학습"},
    "사회":   {"high": "시사 이슈 연계 자료 해석 심화 훈련", "mid": "개념 간 비교표 작성 후 기출 문제 풀이", "low": "핵심 용어 암기 + 단원 요약 정리"},
    "한국사": {"high": "사료 분석형 고난도 문항 반복", "mid": "시대별 흐름표 정리 후 기출 문제 풀이", "low": "연표 암기 + 시대별 핵심 사건 정리"},
}


async def _ollama_once(client: httpx.AsyncClient, prompt: str, timeout: float) -> str | None:
    import logging
    r = await client.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": OLLAMA_OPTIONS},
        timeout=timeout,
    )
    if r.status_code == 200:
        return r.json().get("response", "").strip()
    logging.warning(f"[Ollama] status {r.status_code}: {r.text[:200]}")
    return None


async def _ollama_analyze(prompt: str, timeout: float = 20.0) -> str | None:
    """Ollama llama3.1으로 짧은 분석 생성. 실패/타임아웃 시 None 반환.

    main.py의 개념 설명 폴백도 이 함수를 쓴다(예전엔 두 파일에 같은 함수가 따로 있었다).
    다른 문자(まず·बनन 등)가 섞이면 한 번 다시 생성하고, 그래도 섞이면 그 부분만 지운다.
    Ollama API로는 토큰을 막을 수 없어서다(Qwen은 script_guard로 생성 단계에서 막는다).
    """
    import logging
    try:
        async with httpx.AsyncClient() as client:
            text = await _ollama_once(client, prompt, timeout)
            if text and has_foreign(text):
                retry = await _ollama_once(client, prompt, timeout)
                text = retry if retry and not has_foreign(retry) else strip_foreign(retry or text).strip()
            # Qwen 개념 설명과 같은 정리 — 리포트에 "**굵게**" 기호가 그대로 보였다.
            return strip_markdown(text) if text else text
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


def _resolve_concept(subject: str, req: dict) -> str:
    """리포트가 다룰 취약 개념. 없으면 빈 문자열.

    Backend의 recommendContext는 개념이 아니라 수준 라벨이다("심화 과정 추천",
    "기초 강화 필요" 등 — GpsDummyDataInitializer). 예전에는 이걸 개념으로 받아
    "수학 '기초 강화 필요' 개념을 어려워합니다"라는 프롬프트를 만들고 그 말로 자료를
    검색했다. ConceptMap에 있는 개념어일 때만 그대로 쓰고, 아니면 subject-recommend처럼
    Backend에서 그 과목의 오답 개념 태그를 가져온다(Java 호출이라 블로킹).
    """
    ctx = (req.get("recommendContext") or "").strip()
    if ctx and search_concept_map(subject, ctx, 1):
        return ctx
    sid = req.get("studentId")
    if sid:
        for w in get_student_weakness_from_java(str(sid), subject) or []:
            tag = (w.get("conceptTag") or "").strip() if isinstance(w, dict) else ""
            if tag:
                return tag
    return ""


def _material_lines(docs: list[str], concept: str, width: int = 120) -> str:
    """리포트의 '관련 학습 자료' 줄. 개념을 특정하지 못해 기본 검색어로 찾은 경우에는
    ConceptMap 자료만 보여 준다 — FAISS 교과서 조각은 "Use the phrases from the
    vocabulary list." 같은 문장이라 학생에게 의미가 없다."""
    keep = [d for d in docs if d and "오류" not in d and (concept or is_curated(d))]
    return "\n".join(f"  • {d[:width]}" for d in keep)


# writing·premium의 학습 조언 프롬프트 공통 지시. 예전 프롬프트는 질문을 되풀이하며 시작해
# 번호 목록에 같은 말을 맴돌았다("개념을 이해하기 위해 노력하는 동안, 개념을 이해하기 위한
# 노력을 통해…"). 학생 4명 프로필로 비교해 이 지시가 짧고 구체적인 조언을 냈다.
_ADVICE_RULES = (" 학생에게 직접 말하듯 3문장 이내로 쓰세요. 각 방법에는 무엇을 얼마나 자주 할지 넣으세요."
                 " 질문을 되풀이하지 말고 목록 기호 없이 문장으로만 쓰세요.")


def _feedback_clause(req: dict) -> str:
    fb = (req.get("instructorFeedback") or "").strip()
    return f" 강사 의견: '{fb}'." if fb else ""


def _direction_line(req: dict, concept: str) -> str:
    """recommendContext가 개념이 아닌 수준 라벨이면 '학습 방향'으로 보여 준다."""
    label = (req.get("recommendContext") or "").strip()
    return f"● 학습 방향: {label}\n" if label and label != concept else ""


def _math_report(req: dict, concept: str) -> tuple[dict, list[str]]:
    """(리포트, 검색한 RAG 자료). 자료는 LLM 문단의 근거로 다시 쓴다.
    concept은 _resolve_concept()로 정한 취약 개념(없으면 빈 문자열)."""
    student_id     = req.get("studentId", "")
    percentile     = float(req.get("currentKoreanGrade") or 50)
    study_hours    = float(req.get("studyTime") or 1)
    student_note   = req.get("studentNote") or ""
    feedback       = req.get("instructorFeedback") or ""

    level   = _level_label(percentile)
    stamina = _study_intensity(study_hours)

    # RAG로 수학 취약 개념 관련 학습 자료 검색 (과목 필터 적용)
    rag_docs  = subject_aware_search("수학", concept or "방정식 함수 기본 개념", k=3)
    rag_text  = _material_lines(rag_docs, concept)

    career_analysis = (
        f"[수학 사고력 및 오답 패턴 분석]\n\n"
        f"● 현재 학업 수준: {level} (국어 백분위 {percentile:.0f}%)\n"
        f"● 학습 강도: {stamina} (일 {study_hours:.1f}시간)\n"
    )
    career_analysis += _direction_line(req, concept)
    if student_note:
        career_analysis += f"● 학습 특성: {student_note}\n"
    if feedback:
        career_analysis += f"● 최근 강사 피드백: {feedback}\n"
    if concept:
        career_analysis += f"\n[취약 개념 집중 분석]\n{concept}\n"
    if rag_text:
        career_analysis += f"\n[관련 학습 자료 검색 결과]\n{rag_text}\n"

    learning_guide = (
        f"[수학 맞춤 학습 전략]\n\n"
        f"1단계 — 오답 유형 분류: 계산 실수 / 개념 미이해 / 응용력 부족으로 구분 후 원인별 대응\n"
        f"2단계 — 취약 개념 집중: '{concept or '핵심 공식'}' 단원 기출 10문항 반복\n"
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
    }, rag_docs


def _subject_report(subject: str, req: dict, concept: str) -> tuple[dict, list[str]]:
    """전 과목 공용 메타인지 분석 리포트 (수학 리포트와 동일한 형식을 과목별로 일반화).
    (리포트, 검색한 RAG 자료)를 돌려준다. 자료는 LLM 문단의 근거로 다시 쓴다.
    concept은 _resolve_concept()로 정한 취약 개념(없으면 빈 문자열)."""
    student_id     = req.get("studentId", "")
    percentile     = float(req.get("currentKoreanGrade") or 50)
    study_hours    = float(req.get("studyTime") or 1)
    student_note   = req.get("studentNote") or ""
    feedback       = req.get("instructorFeedback") or ""

    level   = _level_label(percentile)
    stamina = _study_intensity(study_hours)
    tier    = "high" if percentile >= 75 else "mid" if percentile >= 50 else "low"

    rag_docs = subject_aware_search(subject, concept or _DEFAULT_QUERY.get(subject, subject), k=3)
    rag_text = _material_lines(rag_docs, concept)

    career_analysis = (
        f"[{subject} 사고력 및 오답 패턴 분석]\n\n"
        f"● 현재 학업 수준: {level} (백분위 {percentile:.0f}%)\n"
        f"● 학습 강도: {stamina} (일 {study_hours:.1f}시간)\n"
    )
    career_analysis += _direction_line(req, concept)
    if student_note:
        career_analysis += f"● 학습 특성: {student_note}\n"
    if feedback:
        career_analysis += f"● 최근 강사 피드백: {feedback}\n"
    if concept:
        career_analysis += f"\n[취약 개념 집중 분석]\n{concept}\n"
    if rag_text:
        career_analysis += f"\n[관련 학습 자료 검색 결과]\n{rag_text}\n"

    mistake_types = _MISTAKE_TYPES.get(subject, "개념 미이해 / 적용력 부족 / 실수")
    strategy      = _STRATEGY.get(subject, {}).get(tier, "기본 개념 정리 후 문제풀이 반복")

    learning_guide = (
        f"[{subject} 맞춤 학습 전략]\n\n"
        f"1단계 — 오답 유형 분류: {mistake_types}로 구분 후 원인별 대응\n"
        f"2단계 — 취약 개념 집중: '{concept or '핵심 개념'}' 단원 기출 10문항 반복\n"
        f"3단계 — 오답 노트 작성: 틀린 문제를 다시 풀어 풀이 과정을 직접 설명\n"
        f"4단계 — 실력 점검: {strategy}\n\n"
        f"현재 {level} 기준 권장 일일 학습: "
        f"{f'{subject} 1.5시간 + 오답 정리 30분' if tier == 'high' else '개념 1시간 + 문제풀이 1시간 + 복습 30분'}"
    )

    return {
        "studentId": student_id,
        "title": f"{subject} 메타인지 분석 리포트",
        "careerAnalysis": career_analysis,
        "learningGuide": learning_guide,
    }, rag_docs


def _writing_report(req: dict, concept: str) -> dict:
    student_id    = req.get("studentId", "")
    percentile    = float(req.get("currentKoreanGrade") or 50)
    study_hours   = float(req.get("studyTime") or 1)
    student_note  = req.get("studentNote") or ""
    recommend_ctx = req.get("recommendContext") or ""
    feedback      = req.get("instructorFeedback") or ""

    level = _level_label(percentile)

    # RAG로 국어 관련 학습 자료 검색 (과목 필터 적용). recommendContext는 보통 수준 라벨이라
    # 검색어로 쓰지 않고 _resolve_concept()로 정한 개념을 쓴다.
    rag_docs  = subject_aware_search("국어", concept or _DEFAULT_QUERY["국어"], k=3)
    rag_text  = _material_lines(rag_docs, concept)

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
        career_analysis += f"\n[관련 학습 자료 검색 결과]\n{rag_text}\n"

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


def _premium_subject(req: dict) -> str:
    ctx = req.get("recommendContext") or ""
    return "영어" if "영어" in ctx else "국어" if "국어" in ctx else "수학"


def _premium_report(req: dict, concept: str) -> dict:
    student_id    = req.get("studentId", "")
    percentile    = float(req.get("currentKoreanGrade") or 50)
    study_hours   = float(req.get("studyTime") or 1)
    student_note  = req.get("studentNote") or ""
    recommend_ctx = req.get("recommendContext") or ""
    feedback      = req.get("instructorFeedback") or ""

    level   = _level_label(percentile)
    stamina = _study_intensity(study_hours)

    # 프리미엄 리포트: 주 취약 과목 기준으로 RAG 검색
    rag_docs  = subject_aware_search(_premium_subject(req), concept or "기본 개념", k=4)
    rag_text  = _material_lines(rag_docs, concept, 130)

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
    if recommend_ctx and recommend_ctx != concept:
        career_analysis += f"● 학습 방향: {recommend_ctx}\n"
    if concept:
        career_analysis += f"\n━━ 취약 영역 집중 진단 ━━\n{concept}\n"
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
    concept = await run_in_threadpool(_resolve_concept, "수학", req)
    result, rag_docs = _math_report(req, concept)
    if concept:
        prompt = (
            f"{context_block(rag_docs)}"
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
    concept = await run_in_threadpool(_resolve_concept, "국어", req)
    result = _writing_report(req, concept)
    note = req.get("studentNote", "").strip()
    percentile = float(req.get("currentKoreanGrade") or 50)
    hours = float(req.get("studyTime") or 1)
    prompt = (
        f"국어 백분위 {percentile:.0f}%, 하루 공부 시간 {hours:.0f}시간인 학생입니다. "
        f"학생 특성: '{note or '특이사항 없음'}'.{_feedback_clause(req)} "
        f"이 학생의 국어(문학·비문학·작문) 실력을 올릴 구체적인 공부 방법 2가지를{_ADVICE_RULES}"
    )
    llm_insight = await _ollama_analyze(prompt)
    if llm_insight:
        result["careerAnalysis"] += f"\n\n[AI 맞춤 학습 제안]\n{llm_insight}"
    return result


@router.post("/report/premium")
async def report_premium(req: dict):
    concept = await run_in_threadpool(_resolve_concept, _premium_subject(req), req)
    result = _premium_report(req, concept)
    label = (req.get("recommendContext") or "").strip()
    percentile = float(req.get("currentKoreanGrade") or 50)
    note = req.get("studentNote", "").strip()
    hours = float(req.get("studyTime") or 1)
    direction = f"학습 방향: '{label}', " if label and label != concept else ""
    prompt = (
        f"학생 정보 — 국어 백분위 {percentile:.0f}%, 하루 공부 {hours:.0f}시간, {direction}"
        f"취약 개념: '{concept or '특정되지 않음'}', 특성: '{note or '없음'}'.{_feedback_clause(req)} "
        f"이 학생이 성적을 올리려면 이번 주에 가장 먼저 해야 할 행동 2가지를{_ADVICE_RULES}"
    )
    llm_insight = await _ollama_analyze(prompt)
    if llm_insight:
        result["careerAnalysis"] += f"\n\n[AI 우선순위 행동 제안]\n{llm_insight}"
    return result


@router.post("/report/{subject}")
async def report_subject(subject: str, req: dict):
    """수학/국어/프리미엄 외 과목(영어·과학·사회·한국사)의 메타인지 분석 리포트.
    /report/math, /report/writing, /report/premium은 위에 먼저 등록되어 있어 우선 매칭됨."""
    if subject not in _SUBJECT_KEYWORDS:
        raise HTTPException(status_code=404, detail=f"지원하지 않는 과목입니다: {subject}")

    concept = await run_in_threadpool(_resolve_concept, subject, req)
    result, rag_docs = _subject_report(subject, req, concept)
    if concept:
        # 1순위: main.py의 _concept_explain()(베이스 Qwen). main.py가 registry에 등록한다.
        # TestClient로 라우터만 띄운 경우처럼 미등록이면 곧바로 Ollama로 넘어간다.
        # rag_docs는 concept으로 검색한 것이다.
        llm_insight = None
        concept_explain = model_registry.get("concept_explain")
        if concept_explain:
            llm_insight = await run_in_threadpool(concept_explain, subject, concept, rag_docs)

        # 2순위: _concept_explain()이 None을 준 경우(한국사, 생성 실패) Ollama
        if not llm_insight:
            prompt = (
                f"{context_block(rag_docs)}"
                f"한국 중고등학생이 {subject} '{concept}' 개념을 어려워합니다. "
                f"이 개념에서 학생들이 가장 자주 하는 핵심 실수 1가지와 "
                f"그것을 극복하는 구체적인 학습 전략을 2~3문장으로 간결하게 한국어로 답해주세요."
            )
            llm_insight = await _ollama_analyze(prompt)

        if llm_insight:
            result["careerAnalysis"] += f"\n\n[AI 개념 심층 분석]\n{llm_insight}"
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

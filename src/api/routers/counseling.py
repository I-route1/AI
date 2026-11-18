import re

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from src.api.ask_quality import availability_notice, missing_exercise, ollama_answer, question_prompt, read_answer, tutor_instruction, verify_reported_claim, wants_answer_only
from src.api.linear_equation import solve_linear
from src.api.literature import classify_quoted_figure
from src.api import fallback_stats, model_registry
from src.api.concept_map import search_concept_map
from src.api.generation import OLLAMA_CPU, OLLAMA_URL, ollama_request
from src.api.java_client import get_student_weakness_from_java
from src.api.grounding import context_block, is_curated
from src.api.postprocess import strip_chat_frame, strip_markdown
from src.api.report_text import complete_generation, material_excerpt, positive_measurement
from src.api.script_guard import has_foreign, strip_foreign
from src.api.routers.rag import detect_subject, subject_aware_search, vector_search, _SUBJECT_KEYWORDS

router = APIRouter()

_REVIEW_ACTIVITY = {
    "수학": "예제의 조건 하나를 바꾸어 다시 풀고, 그 조건이 어느 풀이 단계에 영향을 줬는지 표시하세요.",
    "영어": "예문에서 핵심 표현에 밑줄을 긋고, 그 표현을 써서 새 문장을 하나 만드세요.",
    "국어": "지문에서 개념을 보여 주는 구절을 골라, 그 구절이 해석의 근거가 되는 이유를 적으세요.",
    "과학": "교과서에서 이 개념이 나타나는 현상 하나를 찾아, 관찰할 수 있는 변화와 그 원인을 나누어 적으세요.",
    "사회": "사례 하나를 골라 개념의 정의에 해당하는 부분을 표시하고, 둘이 어떻게 연결되는지 적으세요.",
}
_CALENDAR_YEAR = re.compile(r"(?<![\d.])(?:(?:기원전|서기)\s*[1-9]\d{0,3}|[1-9]\d{2,3})년(?!간|째|\s*동안)")


def _review_activity(subject: str, concept: str, materials: str) -> str:
    focus = f"‘{concept}’" if concept else "최근에 배운 단원"
    if subject == "사회" and concept:
        for note in materials.splitlines():
            if concept not in note:
                continue
            years = list(dict.fromkeys(_CALENDAR_YEAR.findall(note)))
            if len(years) >= 2:
                return (f"{focus} 자료에서 {years[0]}과 {years[1]}에 해당하는 내용을 각각 찾아, "
                        "두 기록의 차이를 한 문장으로 설명해 보세요.")
    return _REVIEW_ACTIVITY.get(subject, "교과서 예제를 한 문제 풀고 풀이에 사용한 개념을 표시하세요.")


_SELF_CHECK = {
    "수학": "바꾼 조건과 그대로인 조건을 구분하고, 풀이에서 달라진 줄을 짚을 수 있나요?",
    "영어": "새 문장을 소리 내어 읽고, 밑줄 친 표현이 어떤 뜻을 더하는지 설명할 수 있나요?",
    "국어": "자신의 해석을 뒷받침하는 구절을 지문에서 다시 찾을 수 있나요?",
    "과학": "직접 관찰한 사실과 그것을 설명하는 원리를 구분해서 말할 수 있나요?",
    "사회": "자료에 나온 표현 하나를 짚어 자신의 설명을 뒷받침할 수 있나요?",
}


def _study_guide(subject: str, concept: str, materials: str) -> str:
    focus = f"‘{concept}’" if concept else "최근 배운 단원"
    check = _SELF_CHECK.get(subject, "답을 고른 근거를 한 문장으로 말할 수 있나요?")
    return (f"[학습 전략]\n1단계 — {focus}의 예제 한 문제를 풀고 답을 고른 근거를 한 문장으로 적으세요.\n"
            f"2단계 — {_review_activity(subject, concept, materials)}\n\n[스스로 확인]\n{check}")


def _report_prompt(subject: str, concept: str, docs: list[str]) -> str:
    return (f"{context_block(docs)}"
            f"{subject} '{concept}'을 복습하는 학생에게, 원리 하나를 쉬운 말로 설명하고 "
            "짧은 예에서 그 원리를 어떻게 확인하는지 보여 주세요. "
            "학생의 실제 오답이나 습관을 추정하지 마세요. 같은 조언을 되풀이하지 마세요.")


async def _ollama_once(client: httpx.AsyncClient, prompt: str, timeout: float,
                       max_tokens: int | None) -> str | None:
    import logging
    r = await client.post(
        OLLAMA_URL,
        json=ollama_request(prompt, max_tokens=max_tokens),
        timeout=timeout,
    )
    if r.status_code == 200:
        payload = r.json()
        result = (payload.get("response") or "").strip()
        if re.search(r"</?think\b|<\|im_(?:start|end)\|>", result, re.IGNORECASE):
            return None
        if payload.get("done_reason") == "length":
            return complete_generation(result) or None
        return result
    logging.warning(f"[Ollama] status {r.status_code}: {r.text[:200]}")
    return None


async def _ollama_analyze(prompt: str, timeout: float = 20.0, *,
                          max_tokens: int | None = None, retry_foreign: bool = True) -> str | None:
    """Ollama llama3.1으로 짧은 분석 생성. 실패/타임아웃 시 None 반환.

    main.py의 개념 설명 폴백도 이 함수를 쓴다(예전엔 두 파일에 같은 함수가 따로 있었다).
    다른 문자(まず·बनन 등)가 섞이면 한 번 다시 생성하고, 그래도 섞이면 그 부분만 지운다.
    Ollama API로는 토큰을 막을 수 없어서다(Qwen은 script_guard로 생성 단계에서 막는다).
    """
    import logging
    try:
        async with httpx.AsyncClient() as client:
            text = await _ollama_once(client, prompt, timeout, max_tokens)
            if text and has_foreign(text):
                retry = await _ollama_once(client, prompt, timeout, max_tokens) if retry_foreign else None
                text = retry if retry and not has_foreign(retry) else strip_foreign(retry or text).strip()
            # Qwen 개념 설명과 같은 정리 — 리포트에 "**굵게**" 기호가 그대로 보였다.
            return strip_chat_frame(strip_markdown(text)) if text else text
    except Exception as e:
        logging.warning(f"[Ollama] 연결 실패: {e}")
    return None


async def _report_ollama(prompt: str) -> str | None:
    if not OLLAMA_CPU:
        return await _ollama_analyze(prompt)
    # CPU에서 기존 400토큰 요청은 대개 20초 제한을 넘겼다. Java의 50초 한도 안에
    # 한 번만 요청하고, 교과 근거와 다음 학습 행동 하나에 집중한다.
    return await _ollama_analyze(
        prompt + "\n한국어 두 문장 이내로 핵심을 먼저 쓰고, 이어서 요청한 예나 확인 방법을 보여 주세요.",
        timeout=35.0, max_tokens=106, retry_foreign=False,
    )


async def _advice(prompt: str, subject: str | None = None) -> str | None:
    """수학/writing/premium 리포트의 학습 조언 문단. 베이스 Qwen(main._qwen_advice)을 먼저
    쓰고, 미등록이거나(TestClient로 이 라우터만 띄운 경우) 생성에 실패하면 Ollama로 대체한다.
    report_subject()가 concept_explain을 쓰는 것과 같은 순서다(2026-09-28 비교).

    subject: 이 조언이 다루는 과목. main._qwen_advice가 국어·사회일 때 영문자도 막는 데 쓴다
    (수학 공식 D, x 등은 정당해서 막지 않는다 — script_guard._LATIN 설명)."""
    qwen_advice = model_registry.get("qwen_advice")
    if qwen_advice:
        result = await run_in_threadpool(qwen_advice, prompt, subject)
        if result:
            fallback_stats.record("advice", "qwen")
            return result
    result = await _report_ollama(prompt)
    fallback_stats.record("advice", "ollama" if result else "none")
    return result


def _level_label(percentile: float) -> str:
    if percentile >= 90: return "최상위권"
    if percentile >= 75: return "상위권"
    if percentile >= 50: return "중위권"
    if percentile >= 30: return "중하위권"
    return "기초 학습 필요"


def _level_basis(subject: str, req: dict) -> tuple[float | None, str]:
    """Backend는 미등록 국어 백분위를 0으로 보낸다. 대체 교과는 항상 표시한다."""
    measured = positive_measurement(req.get("subjectPercentile"), 100)
    if measured is not None:
        return measured, f"{subject} 백분위 {measured:g}%"
    korean = positive_measurement(req.get("currentKoreanGrade"), 100)
    if korean is not None:
        if subject == "국어":
            return korean, f"국어 백분위 {korean:g}%"
        return None, f"{subject} 백분위 미등록 (참고용 국어 백분위 {korean:g}%)"
    return None, "성적 정보 없음"


def _study_time(req: dict) -> str:
    hours = positive_measurement(req.get("studyTime"), 24)
    return f"일 {hours:g}시간" if hours is not None else "학습 시간 기록 없음"


def _grade_description(percentile: float | None, basis: str) -> str:
    if percentile is None:
        return basis
    return f"{_level_label(percentile)} ({basis})"


def _resolve_concept(subject: str, req: dict) -> str:
    """리포트가 다룰 취약 개념. 없으면 빈 문자열.

    Backend의 recommendContext는 개념이 아니라 수준 라벨이다("심화 과정 추천",
    "기초 강화 필요" 등 — GpsDummyDataInitializer). 예전에는 이걸 개념으로 받아
    "수학 '기초 강화 필요' 개념을 어려워합니다"라는 프롬프트를 만들고 그 말로 자료를
    검색했다. 그래서 Backend가 그 과목의 최다 오답 개념을 weakConcept로 함께 넘긴다.
    weakConcept가 비어 있으면 recommendContext가 ConceptMap 개념어일 때만 그대로 쓴다.
    weakConcept가 없는(None) 요청 — 프리미엄 리포트, weakConcept 이전 Backend — 은
    Backend 오답 API에서 개념 태그를 직접 가져온다(Java 호출이라 블로킹).
    """
    weak = req.get("weakConcept")
    if weak and weak.strip():
        return weak.strip()
    ctx = (req.get("recommendContext") or "").strip()
    if ctx and search_concept_map(subject, ctx, 1):
        return ctx
    if weak is not None:  # Backend가 이미 찾아 봤고 오답이 없었다
        return ""
    sid = req.get("studentId")
    if sid:
        for w in get_student_weakness_from_java(str(sid), subject) or []:
            tag = (w.get("conceptTag") or "").strip() if isinstance(w, dict) else ""
            if tag:
                return tag
    return ""


def _material_lines(docs: list[str], concept: str, width: int = 145) -> str:
    """Show one readable source passage tied to the concept. Keep the full documents for inference."""
    target = "".join(concept.split()).casefold()
    for doc in docs:
        if not doc or doc.startswith(("검색 오류", "검색 실패")):
            continue
        if not is_curated(doc) and (not target or target not in "".join(doc.split()).casefold()):
            continue
        excerpt = material_excerpt(doc, width, concept)
        if excerpt:
            return f"• {excerpt}"
    return ""


# writing·premium의 학습 조언 프롬프트 공통 지시. 예전 프롬프트는 질문을 되풀이하며 시작해
# 번호 목록에 같은 말을 맴돌았다("개념을 이해하기 위해 노력하는 동안, 개념을 이해하기 위한
# 노력을 통해…"). 학생 4명 프로필로 비교해 이 지시가 짧고 구체적인 조언을 냈다.
_ADVICE_RULES = (" 학생이 바로 할 수 있는 활동 한 가지와 스스로 확인할 기준 한 가지를 두 문장으로 쓰세요."
                 " 학생 기록에 없는 능력·진로·오답을 추정하지 마세요.")


def _feedback_clause(req: dict) -> str:
    fb = (req.get("instructorFeedback") or "").strip()
    return f" 강사 의견: '{fb}'." if fb else ""


def _direction_line(req: dict, concept: str) -> str:
    """recommendContext가 개념이 아닌 수준 라벨이면 '학습 방향'으로 보여 준다."""
    label = (req.get("recommendContext") or "").strip()
    return f"● 학습 방향: {label}\n" if label and label != concept else ""


def _math_report(req: dict, concept: str) -> tuple[dict, list[str]]:
    return _subject_report("수학", req, concept)


def _subject_report(subject: str, req: dict, concept: str) -> tuple[dict, list[str]]:
    percentile, basis = _level_basis(subject, req)
    rag_docs = subject_aware_search(subject, concept, k=3) if concept else []
    rag_text = _material_lines(rag_docs, concept)
    analysis = (
        "[기록으로 본 학습]\n"
        f"● 현재 학업 수준: {_grade_description(percentile, basis)}\n"
        f"● 기록된 학습 시간: {_study_time(req)}\n"
    )
    analysis += _direction_line(req, concept)
    if note := req.get("studentNote"):
        analysis += f"● 학습 특성: {note}\n"
    if feedback := req.get("instructorFeedback"):
        analysis += f"● 최근 강사 피드백: {feedback}\n"
    if concept:
        analysis += f"● 복습할 개념: {concept}\n"
    if rag_text:
        analysis += f"\n[교과 자료]\n{rag_text}"

    hours = positive_measurement(req.get("studyTime"), 24)
    guide = _study_guide(subject, concept, rag_text)
    if percentile is None:
        guide += "\n다음 시험 성적을 등록하면 이번 복습에서 다룬 단원의 변화를 함께 살펴보세요."
    elif percentile >= 75:
        guide += "\n익숙해지면 새 예에서도 같은 원리를 찾아보세요."
    elif percentile < 50:
        guide += "\n막히면 교과서의 첫 예제와 자신의 풀이를 한 줄씩 대조하세요."
    if hours is not None:
        minutes = round(hours * 60)
        if minutes >= 2:
            session = min(20, minutes)
            guide += f"\n\n[복습 시간 예시]\n기록된 하루 {minutes}분 중 {session}분을 이 활동에 사용해 보세요."
    return {
        "studentId": req.get("studentId", ""),
        "title": f"{subject} 메타인지 분석 리포트",
        "careerAnalysis": analysis,
        "learningGuide": guide,
    }, rag_docs


def _writing_report(req: dict, concept: str) -> dict:
    student_id    = req.get("studentId", "")
    percentile, basis = _level_basis("국어", req)
    student_note  = req.get("studentNote") or ""
    feedback      = req.get("instructorFeedback") or ""

    # RAG로 국어 관련 학습 자료 검색 (과목 필터 적용). recommendContext는 보통 수준 라벨이라
    # 검색어로 쓰지 않고 _resolve_concept()로 정한 개념을 쓴다.
    rag_docs  = subject_aware_search("국어", concept, k=3) if concept else []
    rag_text  = _material_lines(rag_docs, concept)

    career_analysis = (
        "[기록으로 본 학습]\n"
        f"● 국어 학업 수준: {_grade_description(percentile, basis)}\n"
        f"● 기록된 학습 시간: {_study_time(req)}\n"
    )
    if student_note:
        career_analysis += f"● 학습 성향 및 특성: {student_note}\n"
    if feedback:
        career_analysis += f"● 강사 관찰 소견: {feedback}\n"
    career_analysis += _direction_line(req, concept)
    if concept:
        career_analysis += f"● 복습할 개념: {concept}\n"
    if rag_text:
        career_analysis += f"\n[교과 자료]\n{rag_text}"

    if concept:
        reading_task = f"교과서에서 ‘{concept}’을 다룬 지문을 골라, 판단 근거가 되는 구절에 밑줄을 그으세요."
    else:
        reading_task = "짧은 글 한 편에서 글쓴이가 전하려는 말을 한 문장으로 적고, 근거가 되는 구절에 밑줄을 그으세요."
    if percentile is not None and percentile >= 80:
        writing_check = "같은 구절을 다르게 읽을 수도 있나요? 두 해석 중 어느 쪽을 지문이 더 잘 뒷받침하는지 설명하세요."
    elif percentile is not None and percentile < 50:
        writing_check = "인용한 구절에서 가장 중요한 낱말을 고르고, 그 뜻을 자신의 말로 풀어 쓸 수 있나요?"
    else:
        writing_check = "인용한 내용과 자신의 해석을 구별해서 읽어 줄 수 있나요?"

    learning_guide = (
        f"[이번 복습]\n1단계 — {reading_task}\n"
        "2단계 — 표시한 구절을 인용해 자신의 해석을 두 문장으로 쓰세요.\n\n"
        f"[스스로 확인]\n{writing_check}\n\n"
        "[진로 탐색]\n관심 있는 주제로 기사를 쓴다면 자료 조사, 인터뷰, 문장 다듬기 중 "
        "어느 일을 먼저 해 보고 싶은지 이유를 적어 보세요."
    )

    return {
        "studentId": student_id,
        "title": "AI 기반 진로 탐색 리포트",
        "careerAnalysis": career_analysis,
        "learningGuide": learning_guide,
    }


def _premium_subject(req: dict) -> str:
    """프리미엄 리포트의 주 취약 과목. Backend가 오답이 가장 많은 과목(weakSubject)을 넘긴다.
    없으면(오답 없음, 이전 Backend) 예전 규칙 — recommendContext는 수준 라벨이라 사실상 수학."""
    weak = (req.get("weakSubject") or "").strip()
    if weak in _SUBJECT_KEYWORDS:
        return weak
    ctx = req.get("recommendContext") or ""
    return "영어" if "영어" in ctx else "국어" if "국어" in ctx else "수학"


def _premium_report(req: dict, concept: str) -> dict:
    subject = _premium_subject(req)
    percentile, basis = _level_basis(subject, req)
    analysis = (
        "[기록으로 본 학습]\n"
        f"● 학업 수준: {_grade_description(percentile, basis)}\n"
        f"● 기록된 학습 시간: {_study_time(req)}\n"
    )
    if note := req.get("studentNote"):
        analysis += f"● 학습 특성: {note}\n"
    if feedback := req.get("instructorFeedback"):
        analysis += f"● 최근 강사 피드백: {feedback}\n"
    analysis += _direction_line(req, concept)
    if concept:
        analysis += f"● 복습할 개념: {subject} · {concept}\n"
        docs = subject_aware_search(subject, concept, k=3)
        if material := _material_lines(docs, concept):
            analysis += f"\n[교과 자료]\n{material}"

    focus = f"{subject} ‘{concept}’" if concept else "복습할 단원 한 가지"
    strategy = (
        f"[이번 주 복습]\n1단계 — {focus}의 교과서 예제 한 문제를 풀고 사용한 개념에 표시하세요.\n"
        "2단계 — 다음 학습일에 답안을 가리고 같은 원리가 쓰이는 다른 문제를 풀어 보세요.\n\n"
        "[스스로 확인]\n두 풀이에서 같은 원리를 적용한 부분을 찾을 수 있나요? "
        "막힌 단계가 있으면 그 줄에 표시해 강사에게 질문하세요."
    )
    if feedback:
        strategy += "\n강사 피드백에서 요청한 내용을 풀이 옆에 표시하고 다음 수업에서 확인하세요."
    return {
        "studentId": req.get("studentId", ""),
        "title": "i-Route 프리미엄 통합 AI 진단 리포트",
        "careerAnalysis": analysis,
        "learningGuide": strategy,
    }


@router.post("/report/math")
async def report_math(req: dict):
    concept = await run_in_threadpool(_resolve_concept, "수학", req)
    result, rag_docs = await run_in_threadpool(_math_report, req, concept)
    if concept:
        prompt = _report_prompt("수학", concept, rag_docs)
        llm_insight = await _advice(prompt, "수학")
        if llm_insight:
            result["careerAnalysis"] += f"\n\n[AI 개념 설명]\n{llm_insight}"
    return result


@router.post("/report/writing")
async def report_writing(req: dict):
    concept = await run_in_threadpool(_resolve_concept, "국어", req)
    result = await run_in_threadpool(_writing_report, req, concept)
    note = (req.get("studentNote") or "").strip()
    _, basis = _level_basis("국어", req)
    prompt = (
        f"성적 정보: {basis}. 학습 시간: {_study_time(req)}. "
        f"학생 특성: '{note or '특이사항 없음'}'.{_feedback_clause(req)} "
        f"이번에 연습할 국어 개념: {concept or '지문 읽기와 글쓰기'}. {_ADVICE_RULES}"
    )
    llm_insight = await _advice(prompt, "국어")
    if llm_insight:
        result["learningGuide"] += f"\n\n[AI 복습 제안]\n{llm_insight}"
    return result


@router.post("/report/premium")
async def report_premium(req: dict):
    concept = await run_in_threadpool(_resolve_concept, _premium_subject(req), req)
    result = await run_in_threadpool(_premium_report, req, concept)
    label = (req.get("recommendContext") or "").strip()
    _, basis = _level_basis(_premium_subject(req), req)
    note = (req.get("studentNote") or "").strip()
    direction = f"학습 방향: '{label}', " if label and label != concept else ""
    prompt = (
        f"학생 정보 — {basis}, 학습 시간: {_study_time(req)}. {direction}"
        f"취약 개념: '{concept or '특정되지 않음'}', 특성: '{note or '없음'}'.{_feedback_clause(req)} "
        f"이번 주 복습 중 먼저 할 일을 제안하세요. {_ADVICE_RULES}"
    )
    llm_insight = await _advice(prompt, _premium_subject(req))
    if llm_insight:
        result["learningGuide"] += f"\n\n[AI 복습 제안]\n{llm_insight}"
    return result


@router.post("/report/{subject}")
async def report_subject(subject: str, req: dict):
    """수학/국어/프리미엄 외 과목(영어·과학·사회)의 메타인지 분석 리포트.
    /report/math, /report/writing, /report/premium은 위에 먼저 등록되어 있어 우선 매칭됨."""
    if subject not in _SUBJECT_KEYWORDS:
        raise HTTPException(status_code=404, detail=f"지원하지 않는 과목입니다: {subject}")

    concept = await run_in_threadpool(_resolve_concept, subject, req)
    result, rag_docs = await run_in_threadpool(_subject_report, subject, req, concept)
    if concept:
        # 1순위: main.py의 _concept_explain()(베이스 Qwen). main.py가 registry에 등록한다.
        # TestClient로 라우터만 띄운 경우처럼 미등록이면 곧바로 Ollama로 넘어간다.
        # rag_docs는 concept으로 검색한 것이다.
        llm_insight = None
        prompt = _report_prompt(subject, concept, rag_docs)
        concept_explain = model_registry.get("concept_explain")
        if concept_explain:
            llm_insight = await run_in_threadpool(concept_explain, subject, concept, rag_docs, prompt)
            if llm_insight:
                fallback_stats.record("concept", "qwen")

        # 2순위: _concept_explain()이 None을 준 경우(생성 실패) Ollama
        if not llm_insight:
            llm_insight = await _report_ollama(prompt)
            fallback_stats.record("concept", "ollama" if llm_insight else "none")

        if llm_insight:
            result["careerAnalysis"] += f"\n\n[AI 개념 설명]\n{llm_insight}"
    return result


@router.post("/search")
async def ai_search(req: dict):
    query = req.get("question", "")
    subject = req.get("subject") or detect_subject(query)
    if subject:
        contexts = subject_aware_search(subject, query, k=5)
    else:
        contexts = vector_search(query, k=5)
    return {"contexts": contexts}


ASK_MAX_CHARS = 300


@router.post("/ask")
async def ai_ask(req: dict):
    """질문 내용과 실제 사용한 참고 자료를 구분해 앱으로 반환한다."""

    question = req.get("question")
    if not isinstance(question, str) or len(question.strip()) < 2:
        raise HTTPException(status_code=400, detail="질문을 입력해 주세요.")
    question = question.strip()
    if len(question) > ASK_MAX_CHARS:
        raise HTTPException(status_code=400, detail=f"질문은 {ASK_MAX_CHARS}자 이내로 입력해 주세요.")
    subject = req.get("subject")
    if subject is not None and not isinstance(subject, str):
        raise HTTPException(status_code=400, detail="지원하지 않는 과목입니다.")
    if subject is not None:
        subject = subject.strip() or None
    if subject is not None and subject not in _SUBJECT_KEYWORDS:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 과목입니다: {subject}")
    if notice := availability_notice(question):
        return {"subject": subject, "question": question, "answer": notice, "grounded": False,
                "needSubject": False, "answerKind": "out_of_scope"}
    subject = subject or detect_subject(question)
    if subject is None:
        return {"subject": None, "question": question, "answer": None, "grounded": False,
                "needSubject": True, "answerKind": "clarification"}
    if missing_exercise(question):
        return {"subject": subject, "question": question,
                "answer": "문제 본문과 주어진 조건을 적어 주세요. 보기가 있다면 함께 보내 주세요.",
                "grounded": False, "needSubject": False, "answerKind": "clarification"}

    answer_only = wants_answer_only(question)
    if subject == "수학":
        solution = solve_linear(question, answer_only=answer_only)
        if solution is not None:
            return {"subject": subject, "question": question, "answer": solution, "grounded": False,
                    "needSubject": False, "answerKind": "answer"}
    docs = await run_in_threadpool(subject_aware_search, subject, question, k=3)
    if subject == "국어":
        figure = classify_quoted_figure(question, docs, answer_only)
        if figure:
            return {"subject": subject, "question": question, "answer": figure, "grounded": True,
                    "needSubject": False, "answerKind": "answer"}
    prompt, has_reference = question_prompt(subject, question, docs)
    grounded = False
    kind = "answer"
    answer = None
    concept_explain = model_registry.get("concept_explain")
    if concept_explain:
        qwen_prompt = (f'{tutor_instruction(subject)}\n{{"근거":"…","답변":"…","유형":"…"}} '
                       f'형식의 JSON으로 반환하세요.\n\n{prompt}')
        generated = await run_in_threadpool(concept_explain, subject, question, docs, qwen_prompt)
        answer, grounded, kind = read_answer(generated, has_reference, answer_only=answer_only, question=question)
        if answer:
            fallback_stats.record("ask", "qwen")
    if not answer:
        answer, grounded, kind = await ollama_answer(subject, prompt, has_reference, answer_only, question=question)
        fallback_stats.record("ask", "ollama" if answer else "none")
    if kind == "answer" and not answer_only:
        answer, cited = verify_reported_claim(subject, question, answer, docs)
        grounded = grounded or cited
    return {"subject": subject, "question": question, "answer": answer, "grounded": grounded,
            "needSubject": False, "answerKind": kind}

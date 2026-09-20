import math
import re
import torch
from typing import List, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()

# ──────────────── Schemas ────────────────

class KeywordItem(BaseModel):
    keyword: str
    weight: int

class EvaluateRequest(BaseModel):
    passage_text: str
    question_text: str
    model_answer: str
    user_answer: str
    keywords: List[KeywordItem] = []

class DeepAnalysis(BaseModel):
    error_types: List[str]
    analysis: str
    improvement: str

class EvaluateResponse(BaseModel):
    keyword_score: Optional[int] = None
    raw_score: int
    normalized_score: int
    final_score: int
    feedback: str
    feedback_type: str
    score_feedback: str
    matched_keywords: List[str]
    missing_keywords: List[str]
    deep_analysis: DeepAnalysis
    # 파인튜닝 채점기의 의견. **final_score에는 영향을 주지 않는다.**
    #
    # final_score는 키워드 일치율 70% + 길이 비율 30%의 규칙 기반이다. 한편
    # writing 어댑터는 사람 채점과 QWK 0.54로 일치하는 1~4점 채점기인데
    # (test_writing_grading_eval.py, 777건) 지금까지 피드백 문장 생성에만
    # 쓰이고 채점에는 호출되지 않았다.
    #
    # 바로 갈아끼우지 않는 이유: 1~4점을 0~100점으로 어떻게 매핑할지, 기존
    # 점수 분포와의 호환을 어떻게 할지가 정해져야 하고 그건 실제 학생 점수가
    # 바뀌는 일이다. 그래서 우선 두 점수를 나란히 실어 보내 실제 트래픽에서
    # 비교할 수 있게 한다. 기존 필드는 그대로라 하위 호환이 깨지지 않는다.
    llm_score: Optional[int] = None        # 1~4 반올림값
    llm_score_raw: Optional[float] = None  # 반올림 전 기댓값 (상관분석용)


class IrtItem(BaseModel):
    difficulty: int
    score: int
    a_param: Optional[float] = None
    b_param: Optional[float] = None
    c_param: Optional[float] = None

class IrtEstimateRequest(BaseModel):
    responses: List[IrtItem]

class IrtEstimateResponse(BaseModel):
    theta: float
    se: float
    ability_level: str
    next_difficulty: int


class CompetencyHistory(BaseModel):
    competency: str
    scores: List[int]

class CurriculumAdjustRequest(BaseModel):
    competency_history: List[CompetencyHistory]

class CurriculumAdjustResponse(BaseModel):
    needs_adjustment: bool
    weak_competencies: List[str]
    adjustment_message: str
    recommended_focus: str


class CompareAnswersRequest(BaseModel):
    question_text: str
    model_answer: str
    previous_answer: str
    previous_score: int
    current_answer: str
    current_score: int
    keywords: List[KeywordItem] = []

class CompareAnswersResponse(BaseModel):
    score_diff: int
    is_improved: bool
    growth_message: str
    newly_included_keywords: List[str]
    still_missing_keywords: List[str]
    analysis: str


class CompetencyScore(BaseModel):
    competency: str
    score: int

class WeaknessReportRequest(BaseModel):
    competency_scores: List[CompetencyScore]

class WeakCompetencyDetail(BaseModel):
    competency: str
    score: int
    level: str

class WeaknessReportResponse(BaseModel):
    weak_competencies: List[WeakCompetencyDetail]
    report: str
    recommendations: List[str]
    priority_competency: str


# ──────────────── Helpers ────────────────

COMPETENCY_KO = {
    "FACTUAL": "사실적 이해",
    "INFERENTIAL": "추론적 이해",
    "CRITICAL": "비판적 사고",
    "VOCABULARY": "어휘력",
    "LOGICAL": "논리적 사고",
}

def _match_keywords(text: str, keywords: List[KeywordItem]):
    text_lower = text.lower()
    matched = [kw.keyword for kw in keywords if kw.keyword.lower() in text_lower]
    missing = [kw.keyword for kw in keywords if kw.keyword.lower() not in text_lower]
    return matched, missing

def _keyword_score(matched: List[str], keywords: List[KeywordItem]) -> int:
    if not keywords:
        return 100
    total_weight = sum(kw.weight for kw in keywords)
    matched_weight = sum(kw.weight for kw in keywords if kw.keyword in matched)
    return int(matched_weight / total_weight * 100) if total_weight > 0 else 0

def _length_score(user_answer: str, model_answer: str) -> int:
    ratio = len(user_answer.strip()) / max(len(model_answer.strip()), 1)
    if ratio >= 0.7: return 90
    if ratio >= 0.5: return 70
    if ratio >= 0.3: return 50
    return 30

def _score_to_level(score: int) -> str:
    if score < 40: return "하"
    if score < 60: return "중하"
    if score < 80: return "중"
    return "상"

def _irt_prob(theta: float, a: float, b: float, c: float) -> float:
    return c + (1 - c) / (1 + math.exp(-a * (theta - b)))

def _estimate_theta(items: List[IrtItem]) -> tuple[float, float]:
    theta = 0.0
    for _ in range(100):
        grad = 0.0
        info = 0.0
        for r in items:
            a = float(r.a_param) if r.a_param is not None else 1.0
            b = float(r.b_param) if r.b_param is not None else (r.difficulty - 3) * 0.5
            c = float(r.c_param) if r.c_param is not None else 0.0
            p = _irt_prob(theta, a, b, c)
            u = 1 if r.score >= 60 else 0
            denom = p * (1 - c) + 1e-9
            grad += a * (u - p) * (p - c) / denom
            info += (a ** 2) * ((p - c) ** 2) * (1 - p) / (denom ** 2 + 1e-9)
        if abs(grad) < 1e-6:
            break
        step = min(max(grad / (info + 1e-6), -0.5), 0.5)
        theta = max(-3.0, min(3.0, theta + step))
    se = 1.0 / math.sqrt(max(sum((r.a_param or 1.0) ** 2 for r in items), 1e-6))
    return round(theta, 3), round(se, 3)

def _theta_to_level(theta: float) -> str:
    if theta < -2.0: return "최하"
    if theta < -1.0: return "하"
    if theta < -0.5: return "하중"
    if theta < 0.0:  return "중하"
    if theta < 0.5:  return "중"
    if theta < 1.0:  return "중상"
    if theta < 1.5:  return "상하"
    if theta < 2.0:  return "상"
    return "최상"


# ──────────────── LLM 피드백 생성 ────────────────

def _llm_feedback(request: Request, question: str, model_answer: str,
                  user_answer: str, missing: List[str], final_score: int) -> tuple[str, str]:
    """글쓰기 전용 모델(Qwen3-8B + writing 어댑터)로 LLM 피드백 생성. 실패 시 규칙 기반 fallback."""
    try:
        model     = request.app.state.writing_model
        tokenizer = request.app.state.writing_tokenizer
        model.set_adapter("writing")

        missing_hint = f"\n누락된 키워드: {', '.join(missing[:3])}" if missing else ""
        messages = [
            {
                "role": "system",
                "content": "다음 서술형 문제에 대한 학생 답안을 평가하고 구체적인 피드백을 작성하세요.",
            },
            {
                "role": "user",
                "content": (
                    f"[문제]: {question}\n"
                    f"[모범 답안]: {model_answer}{missing_hint}\n\n"
                    f"[학생 답안]: {user_answer[:700]}"
                ),
            },
        ]
        # enable_thinking=False: 학습 데이터에 <think> 블록이 전혀 없어서, 켜두면
        # 모델이 습관적으로 빈 사고 블록을 먼저 내보내며 파인튜닝된 동작이 어긋남.
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=896).to("cuda")
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.4,
                do_sample=True,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
            )
        feedback = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        # 혹시 남아있을 수 있는 사고 블록 제거 (enable_thinking=False로 대부분 방지되지만 방어적으로)
        feedback = re.sub(r'<think>.*?</think>', '', feedback, flags=re.DOTALL).strip()
        # 마크다운 헤더 제거
        feedback = re.sub(r'#{1,4}\s*\w*:?\s*', '', feedback)
        feedback = re.sub(r'\s+', ' ', feedback).strip()
        # 중복 문장 제거
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', feedback) if s.strip()]
        seen_s: set[str] = set()
        unique = []
        for s in sentences:
            key = re.sub(r'\s+', '', s)
            if key not in seen_s:
                seen_s.add(key)
                unique.append(s)
        feedback = ' '.join(unique)

        if final_score >= 80:
            score_feedback = "우수한 답변입니다."
        elif final_score >= 60:
            score_feedback = "양호한 답변입니다."
        elif final_score >= 40:
            score_feedback = "미흡한 답변입니다."
        else:
            score_feedback = "부족한 답변입니다."

        return feedback, score_feedback

    except Exception:
        return None, None


# 채점 라벨 공간. 학습 14,223건·평가 777건 모두 5점이 0건이라 실제로는 1~4다.
# system 프롬프트는 "1점부터 5점"이라고 말하지만 모델이 5를 낼 일은 없다.
_GRADE_SCALE = (1, 2, 3, 4)


def _llm_grade(request: Request, question: str, user_answer: str) -> tuple[Optional[int], Optional[float]]:
    """writing 어댑터로 1~4점 채점. (반올림값, 기댓값). 실패 시 (None, None).

    generate()로 한 글자를 뽑지 않고 점수 토큰의 로짓을 직접 읽어 기댓값을 낸다.
    순서형 점수에는 argmax보다 기댓값이 낫다 — 같은 어댑터로 QWK가
    0.543 -> 0.553, 정확도가 56.9% -> 58.7%로 올랐다(777건 측정).
    순전파 한 번이라 생성보다 오히려 싸다.

    프롬프트는 학습 형식(train/preprocess_writing_qwen.py)과 글자 단위로 같아야 한다.
    """
    try:
        model     = request.app.state.writing_model
        tokenizer = request.app.state.writing_tokenizer
        model.set_adapter("writing")

        messages = [
            {"role": "system",
             "content": "제시된 지시문을 바탕으로 학생의 답안을 평가하여 1점부터 5점 사이의 숫자 점수만 출력하시오."},
            {"role": "user",
             "content": f"[지시문]: {question}\n[학생 답안]: {user_answer}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=896).to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :].float()

        ids = [tokenizer.encode(str(c), add_special_tokens=False) for c in _GRADE_SCALE]
        if any(len(i) != 1 for i in ids):
            return None, None
        vec = torch.tensor([logits[i[0]] for i in ids])
        probs = torch.softmax(vec, dim=0)
        expected = float((probs * torch.tensor(_GRADE_SCALE, dtype=probs.dtype)).sum())
        rounded = int(min(_GRADE_SCALE[-1], max(_GRADE_SCALE[0], round(expected))))
        return rounded, round(expected, 3)
    except Exception:
        return None, None


# ──────────────── Endpoints ────────────────

@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: Request, req: EvaluateRequest):
    matched, missing = _match_keywords(req.user_answer, req.keywords)
    kw_score = _keyword_score(matched, req.keywords)
    length_score = _length_score(req.user_answer, req.model_answer)

    raw_score  = int(kw_score * 0.7 + length_score * 0.3) if req.keywords else length_score
    final_score = min(100, max(0, raw_score))

    if final_score >= 80:
        feedback_type = "EXCELLENT"
    elif final_score >= 60:
        feedback_type = "GOOD"
    elif final_score >= 40:
        feedback_type = "NEEDS_IMPROVEMENT"
    else:
        feedback_type = "POOR"

    # LLM 피드백 시도 → 실패 시 규칙 기반 fallback
    llm_fb, score_feedback = _llm_feedback(
        request, req.question_text, req.model_answer, req.user_answer, missing, final_score
    )
    if llm_fb:
        feedback = llm_fb
    else:
        feedback = {
            "EXCELLENT": "핵심 내용을 잘 파악하고 있습니다.",
            "GOOD":      "주요 내용을 포함했으나 보완이 필요합니다.",
            "NEEDS_IMPROVEMENT": "핵심 키워드가 부족합니다. 지문을 다시 읽어보세요.",
            "POOR":      "답변이 너무 짧거나 핵심 내용이 빠져 있습니다.",
        }[feedback_type]
        score_feedback = {
            "EXCELLENT": "우수한 답변입니다.", "GOOD": "양호한 답변입니다.",
            "NEEDS_IMPROVEMENT": "미흡한 답변입니다.", "POOR": "부족한 답변입니다.",
        }[feedback_type]

    improvement = (
        f"누락 키워드 '{', '.join(missing[:3])}' 를 답변에 포함해보세요."
        if missing else "잘 작성했습니다."
    )

    # 파인튜닝 채점기의 의견을 함께 실어 보낸다. final_score는 건드리지 않는다.
    llm_score, llm_score_raw = _llm_grade(request, req.question_text, req.user_answer)

    return EvaluateResponse(
        keyword_score=kw_score if req.keywords else None,
        raw_score=raw_score,
        normalized_score=final_score,
        final_score=final_score,
        feedback=feedback,
        feedback_type=feedback_type,
        score_feedback=score_feedback,
        matched_keywords=matched,
        missing_keywords=missing,
        deep_analysis=DeepAnalysis(
            error_types=["키워드 누락"] if missing else [],
            analysis=f"총 {len(req.keywords)}개 키워드 중 {len(matched)}개 포함",
            improvement=improvement,
        ),
        llm_score=llm_score,
        llm_score_raw=llm_score_raw,
    )


@router.post("/irt/estimate", response_model=IrtEstimateResponse)
async def irt_estimate(req: IrtEstimateRequest):
    if not req.responses:
        return IrtEstimateResponse(theta=0.0, se=1.0, ability_level="중", next_difficulty=3)

    theta, se = _estimate_theta(req.responses)
    ability_level = _theta_to_level(theta)

    avg_score = sum(r.score for r in req.responses) / len(req.responses)
    difficulties = [r.difficulty for r in req.responses]
    if avg_score >= 75:
        next_difficulty = min(5, max(difficulties) + 1)
    elif avg_score < 50:
        next_difficulty = max(1, min(difficulties) - 1)
    else:
        next_difficulty = round(sum(difficulties) / len(difficulties))

    return IrtEstimateResponse(
        theta=theta,
        se=se,
        ability_level=ability_level,
        next_difficulty=next_difficulty,
    )


@router.post("/curriculum/adjust", response_model=CurriculumAdjustResponse)
async def curriculum_adjust(req: CurriculumAdjustRequest):
    weak = [
        item.competency
        for item in req.competency_history
        if len(item.scores) >= 3 and all(s < 50 for s in item.scores[-3:])
    ]

    if weak:
        ko_names = [COMPETENCY_KO.get(w, w) for w in weak]
        return CurriculumAdjustResponse(
            needs_adjustment=True,
            weak_competencies=weak,
            adjustment_message=f"{', '.join(ko_names)} 역량이 지속적으로 낮습니다. 해당 영역 집중 학습을 권장합니다.",
            recommended_focus=weak[0],
        )

    return CurriculumAdjustResponse(
        needs_adjustment=False,
        weak_competencies=[],
        adjustment_message="현재 학습 방향을 유지하세요.",
        recommended_focus="",
    )


@router.post("/compare", response_model=CompareAnswersResponse)
async def compare_answers(req: CompareAnswersRequest):
    score_diff = req.current_score - req.previous_score
    is_improved = score_diff > 0

    prev_matched, prev_missing = _match_keywords(req.previous_answer, req.keywords)
    curr_matched, _ = _match_keywords(req.current_answer, req.keywords)

    newly_included = [k for k in curr_matched if k not in prev_matched]
    still_missing = [k for k in prev_missing if k not in curr_matched]

    if is_improved:
        growth_message = f"이전보다 {abs(score_diff)}점 향상됐습니다! 꾸준히 성장하고 있어요."
    elif score_diff == 0:
        growth_message = "점수가 동일합니다. 핵심 키워드에 더 집중해보세요."
    else:
        growth_message = f"이전보다 {abs(score_diff)}점 낮아졌습니다. 지문을 다시 꼼꼼히 읽어보세요."

    analysis = f"이전 {len(prev_matched)}개 → 현재 {len(curr_matched)}개 키워드 포함."
    if newly_included:
        analysis += f" 새로 포함된 키워드: '{', '.join(newly_included)}'."

    return CompareAnswersResponse(
        score_diff=score_diff,
        is_improved=is_improved,
        growth_message=growth_message,
        newly_included_keywords=newly_included,
        still_missing_keywords=still_missing,
        analysis=analysis,
    )


@router.post("/weakness-report", response_model=WeaknessReportResponse)
async def weakness_report(req: WeaknessReportRequest):
    weak = sorted(
        [
            WeakCompetencyDetail(
                competency=item.competency,
                score=item.score,
                level=_score_to_level(item.score),
            )
            for item in req.competency_scores
            if item.score < 60
        ],
        key=lambda x: x.score,
    )

    priority = weak[0].competency if weak else ""

    if weak:
        ko_names = [COMPETENCY_KO.get(w.competency, w.competency) for w in weak]
        report = (
            f"현재 {', '.join(ko_names)} 역량이 취약한 상태입니다. "
            f"특히 {COMPETENCY_KO.get(priority, priority)} 영역의 집중 학습이 필요합니다. "
            f"관련 지문을 반복적으로 읽고 핵심 키워드 중심으로 답안을 작성하는 연습을 권장합니다."
        )
        recommendations = [
            f"{COMPETENCY_KO.get(w.competency, w.competency)} 관련 지문 반복 학습"
            for w in weak[:3]
        ] + ["핵심 키워드 노트 정리 후 암기", "유사 문제 추가 풀이를 통한 패턴 파악"]
    else:
        report = "전반적으로 양호한 학습 상태입니다. 현재 수준을 유지하며 심화 학습을 진행하세요."
        recommendations = ["현재 학습 페이스 유지", "심화 문제 도전"]

    return WeaknessReportResponse(
        weak_competencies=weak,
        report=report,
        recommendations=recommendations,
        priority_competency=priority,
    )

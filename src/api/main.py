import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import json

class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"

    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False).encode("utf-8")

import torch
import httpx
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.api.routers import counseling, predictor, writing
from src.api.routers.rag import subject_aware_search as _subject_aware_search

app = FastAPI(title="iRoute AI Server", default_response_class=UTF8JSONResponse)

app.include_router(counseling.router, prefix="/api/ai", tags=["counseling"])
app.include_router(predictor.router, prefix="/api/ai", tags=["predictor"])
app.include_router(writing.router, prefix="/api/writing", tags=["writing"])

# 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

BASE_MODEL_ID      = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
WRITING_ADAPTER_ID = "i-route-ai/iroute-writing-ai"

print("📥 base 모델 로드 중 (4bit)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.clean_up_tokenization_spaces = False

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

print("📥 글쓰기 어댑터 로드 중...")
base_model.load_adapter(WRITING_ADAPTER_ID, adapter_name="writing")
print("✅ 글쓰기 어댑터 로드 완료")

# writing.py에서 app.state로 접근
app.state.model     = base_model
app.state.tokenizer = tokenizer


def get_student_weakness_from_java(student_id: str, subject: str):
    try:
        with httpx.Client() as client:
            response = client.get(
                "http://localhost:8080/api/wrong-answer/ai-pipeline",
                params={"studentId": student_id, "subject": subject},
                timeout=5.0
            )
            return response.json() if response.status_code == 200 else []
    except:
        return []


_SUBJECT_DEFAULT_CONCEPT: dict[str, str] = {
    "수학":   "방정식과 함수 기본 개념",
    "영어":   "독해와 어법 기본",
    "국어":   "문학과 비문학 독해",
    "과학":   "기본 과학 개념",
    "사회":   "사회 기본 개념",
    "한국사": "한국사 주요 사건",
}

_SUBJECT_STRATEGY: dict[str, list[str]] = {
    "수학":   ["교과서 개념 정리 및 공식 확인", "관련 기출 예제 5문항 풀이", "오답 원인 분석 후 재풀이", "유사 문제 추가 풀이로 확인"],
    "영어":   ["핵심 어법 규칙 정리", "관련 지문 구문 분석", "어휘 정리 및 암기", "유형별 문제 반복 풀이"],
    "국어":   ["지문 구조 파악 훈련", "핵심 주장 및 근거 추출 연습", "어휘·표현 정리", "유사 지문 독해 연습"],
    "과학":   ["개념 원리 이해 및 정리", "관련 실험·현상 사례 확인", "공식·법칙 적용 연습", "단원 마무리 문제 풀이"],
    "사회":   ["핵심 개념·용어 정리", "관련 사례 및 사료 확인", "개념 간 연관성 파악", "기출 문제 풀이"],
    "한국사": ["시대적 흐름 파악", "주요 사건·인물 정리", "사료 해석 연습", "연표 작성 후 복습"],
}


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


@app.post("/api/ai/report/subject-recommend")
async def generate_subject_recommendation(
        student_id: str = Query(...),
        subject: str = Query(...),
        concept_tag: str = Query(default=None),
):
    # concept 결정: 직접 전달 → Java 백엔드 → 과목 기본값 순
    if concept_tag:
        concept_query = concept_tag
    else:
        weakness_data = get_student_weakness_from_java(student_id, subject)
        concept_query = (
            weakness_data[0].get("conceptTag")
            if weakness_data else
            _SUBJECT_DEFAULT_CONCEPT.get(subject, f"{subject} 기본 개념")
        )

    # RAG 검색
    rag_docs = _subject_aware_search(subject, concept_query, k=3)
    rag_text = "\n".join(f"  • {d}" for d in rag_docs if d and "오류" not in d)

    # Rule-based 리포트
    steps = _SUBJECT_STRATEGY.get(subject, [
        "개념 정리 및 확인", "관련 예제 풀이", "오답 분석", "추가 문제로 실력 확인"
    ])
    strategy = "\n".join(f"{i+1}단계 — {s}" for i, s in enumerate(steps))

    report = f"[{subject} 취약 개념 학습 가이드]\n\n● 집중 학습 개념: {concept_query}\n"
    if rag_text:
        report += f"\n[관련 학습 자료]\n{rag_text}\n"
    report += f"\n[학습 전략]\n{strategy}"

    # LLM 심층 분석 추가
    prompt = (
        f"{subject} 과목에서 '{concept_query}' 개념을 어려워하는 학생에게 "
        f"이 개념의 핵심 포인트와 효과적인 학습 방법을 2~3문장으로 한국어로 답해주세요."
    )
    llm_insight = await _ollama_analyze(prompt)
    if llm_insight:
        report += f"\n\n[AI 개념 분석]\n{llm_insight}"

    return {
        "studentId": student_id,
        "subject": subject,
        "targetConcept": concept_query,
        "aiRecommendationReport": report,
    }

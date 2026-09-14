import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Windows 콘솔이 cp949 등 비-UTF8 인코딩일 때 이모지/한글 print가 UnicodeEncodeError로
# 서버 기동 자체를 죽이는 것을 방지.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json

class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"

    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False).encode("utf-8")

import torch
import httpx
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src.api.routers import counseling, predictor, writing, rag
from src.api.routers.rag import subject_aware_search as _subject_aware_search
from src.api.java_client import get_student_weakness_from_java

app = FastAPI(title="iRoute AI Server", default_response_class=UTF8JSONResponse)

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8080,http://localhost:3000")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(counseling.router, prefix="/api/ai", tags=["counseling"])
app.include_router(predictor.router, prefix="/api/ai", tags=["predictor"])
app.include_router(writing.router, prefix="/api/writing", tags=["writing"])
app.include_router(rag.router, prefix="/api/rag", tags=["rag"])

# 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 베이스 모델 하나(Qwen3-8B)에 수학/글쓰기 어댑터를 둘 다 올려 set_adapter()로 갈아끼우며 서빙한다.
# 예전엔 수학=Gemma4, 글쓰기=원격 Llama3.1 어댑터로 서로 다른 베이스라 8B 모델 두 개를 동시에
# 띄워야 했는데(VRAM 부족으로 OOM), 두 어댑터를 같은 베이스로 재학습해서 하나로 합침.
BASE_MODEL_ID      = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MATH_ADAPTER_PATH  = "train/math_adapter_qwen"
WRITING_ADAPTER_PATH = "train/writing_adapter_qwen_weighted"
MATH_SYSTEM_PROMPT = "당신은 수학 전문 교사입니다. 학생의 수학 문제에 대해 정확한 풀이와 해설을 제공하세요."

print("📥 base 모델 로드 중 (4bit)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
)

print("📥 수학 어댑터 로드 중...")
base_model = PeftModel.from_pretrained(base_model, MATH_ADAPTER_PATH, adapter_name="math")
print("✅ 수학 어댑터 로드 완료")

print("📥 글쓰기 어댑터 로드 중...")
base_model.load_adapter(WRITING_ADAPTER_PATH, adapter_name="writing")
print("✅ 글쓰기 어댑터 로드 완료")

base_model.set_adapter("math")

app.state.model     = base_model
app.state.tokenizer = tokenizer
# math/writing 라우터가 같은 베이스 모델을 공유 - 호출 전 각자 set_adapter()로 전환한다.
app.state.writing_model     = base_model
app.state.writing_tokenizer = tokenizer


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


def _math_concept_explain(concept_query: str) -> str | None:
    """수학 어댑터로 개념 설명 생성. 실패 시 None(호출부에서 Ollama로 fallback)."""
    import logging
    try:
        model     = app.state.model
        tokenizer = app.state.tokenizer
        model.set_adapter("math")

        messages = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": f"'{concept_query}' 개념의 핵심 포인트를 학생에게 설명해주세요."},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    except Exception as e:
        logging.warning(f"[수학 어댑터] 생성 실패: {e}")
        return None


@app.post("/api/ai/report/subject-recommend")
async def generate_subject_recommendation(
        student_id: str = Query(...),
        subject: str = Query(...),
        concept_tag: str = Query(default=None),
):
    if concept_tag:
        concept_query = concept_tag
    else:
        weakness_data = get_student_weakness_from_java(student_id, subject)
        concept_query = (
            weakness_data[0].get("conceptTag")
            if weakness_data else
            _SUBJECT_DEFAULT_CONCEPT.get(subject, f"{subject} 기본 개념")
        )

    rag_docs = _subject_aware_search(subject, concept_query, k=3)
    rag_text = "\n".join(f"  • {d}" for d in rag_docs if d and "오류" not in d)

    steps = _SUBJECT_STRATEGY.get(subject, [
        "개념 정리 및 확인", "관련 예제 풀이", "오답 분석", "추가 문제로 실력 확인"
    ])
    strategy = "\n".join(f"{i+1}단계 — {s}" for i, s in enumerate(steps))

    report = f"[{subject} 취약 개념 학습 가이드]\n\n● 집중 학습 개념: {concept_query}\n"
    if rag_text:
        report += f"\n[관련 학습 자료]\n{rag_text}\n"
    report += f"\n[학습 전략]\n{strategy}"

    llm_insight = None
    if subject == "수학":
        llm_insight = _math_concept_explain(concept_query)

    if not llm_insight:
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

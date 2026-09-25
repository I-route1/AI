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
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json

class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"

    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False, allow_nan=False).encode("utf-8")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src.api import model_registry
from src.api.adapters import (
    BASE_MODEL_ID, MATH_ADAPTER_PATH, WRITING_ADAPTER_PATH,
    MATH_SYSTEM_PROMPT, SUBJECT_ADAPTERS,
)
from src.api.routers import counseling, predictor, writing, rag
from src.api.routers.counseling import _ollama_analyze
from src.api.generation import CONCEPT_GEN
from src.api.postprocess import strip_markdown, trim_cut_tail
from src.api.script_guard import ForeignScriptBlocker, foreign_token_mask, has_foreign, strip_foreign
from transformers import LogitsProcessorList
from src.api.routers.rag import subject_aware_search as _subject_aware_search
from src.api.java_client import get_student_weakness_from_java
from src.api.grounding import concept_user_message, context_block

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

# ⚠️ 라우터 등록은 파일 맨 아래에 있다. counseling 라우터의 /report/{subject}가
# catch-all이라, 먼저 등록하면 이 파일의 @app.post("/api/ai/report/subject-recommend")를
# 가로챈다(FastAPI는 등록 순서대로 매칭). 자세한 설명은 파일 하단 참고.

# 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

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

# 글쓰기 어댑터는 /api/writing/* 전용인데, 그 라우터를 Backend·Front 어디서도
# 호출하지 않는다(2026-09-23 확인). 서술형 답안을 제출받는 기능이 생기면 True로.
# 꺼져 있으면 /api/writing/evaluate는 규칙 기반 피드백만 내고 llm_score는 None이다.
LOAD_WRITING_ADAPTER = False

if LOAD_WRITING_ADAPTER:
    print("📥 글쓰기 어댑터 로드 중...")
    base_model.load_adapter(WRITING_ADAPTER_PATH, adapter_name="writing")
    print("✅ 글쓰기 어댑터 로드 완료")
else:
    print("ℹ️ 글쓰기 어댑터는 로드하지 않습니다 (/api/writing 소비처 없음, 약 167MB 절약)")

# 교과 어댑터(korean/english/science/social)는 **로드하지 않는다.**
#
# 쓰이던 곳이 _concept_explain() 하나뿐이었는데, 거기서 베이스가 더 낫다는 것이
# 측정으로 확인돼 전 과목 베이스로 돌렸다(_CONCEPT_USE_BASE 참고). 그래서 지금은
# 올려두기만 하고 호출되지 않는다. 개당 166MB, 4개면 약 668MB의 VRAM과
# 기동 시간을 그냥 쓴다.
#
# 프롬프트를 학습 형식([학습 지문]/[교육과정 성취기준]/질문:)에 맞추면 회복될까
# 싶어 재봤지만 오히려 격차가 벌어졌다(개념F1 기준 베이스-어댑터 차이가
# +0.070 → +0.105, 4과목 모두 유의). 학습 형식을 주니 학습한 대로 더 짧은
# 교과서 답안을 내놓는다 — 출력이 112자에서 68자로 줄었다. 개념 설명은
# 그 어댑터들이 학습한 과제가 아니다.
#
# 어댑터 파일은 train/에 그대로 있으므로 아래를 True로 바꾸면 되돌릴 수 있다.
LOAD_SUBJECT_ADAPTERS = False

_LOADED_SUBJECT_ADAPTERS: set[str] = set()
if LOAD_SUBJECT_ADAPTERS:
    # 하나라도 실패하면 해당 과목만 Ollama fallback으로 떨어지고 기동은 계속한다.
    for _subject, (_adapter_name, _adapter_path, _) in SUBJECT_ADAPTERS.items():
        try:
            print(f"📥 {_subject} 어댑터 로드 중...")
            base_model.load_adapter(_adapter_path, adapter_name=_adapter_name)
            _LOADED_SUBJECT_ADAPTERS.add(_subject)
            print(f"✅ {_subject} 어댑터 로드 완료")
        except Exception as _e:
            print(f"⚠️ {_subject} 어댑터 로드 실패 — Ollama fallback 사용: {_e}")
else:
    print("ℹ️ 교과 어댑터 4개는 로드하지 않습니다 (개념 설명은 베이스 사용, 약 668MB 절약)")

base_model.set_adapter("math")

app.state.model     = base_model
app.state.tokenizer = tokenizer
# math/writing 라우터가 같은 베이스 모델을 공유 - 호출 전 각자 set_adapter()로 전환한다.
app.state.writing_model     = base_model if LOAD_WRITING_ADAPTER else None
app.state.writing_tokenizer = tokenizer

# 개념 설명에 중국어·일본어 등 다른 문자가 섞이지 않게 해당 토큰을 막는다(script_guard.py).
_FOREIGN_BLOCKER = ForeignScriptBlocker(foreign_token_mask(tokenizer, base_model.config.vocab_size))


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




# 개념 설명에서 어댑터를 끄고 베이스로 생성할 과목.
#
# 근거는 test_concept_explain_eval.py의 개념F1(커버리지와 밀도의 조화평균,
# 길이에 중립적인 지표)이다. 같은 개념으로 두 조건을 쟀으므로 평균이 아니라
# 개념별 차이를 부트스트랩으로 검정했다 — 과목당 표본이 14~22개뿐이라
# 평균만 보면 우연을 실력으로 읽기 쉽다.
#
#   과목  n   베이스-어댑터   95% 신뢰구간
#   영어  20     +0.095    [+0.043, +0.153]
#   수학  20     +0.081    [+0.006, +0.158]
#   사회  14     +0.068    [+0.007, +0.133]
#   과학  20     +0.066    [+0.017, +0.114]
#   국어  16     +0.030    [-0.017, +0.078]   <- 개별로는 0을 포함
#   전체  90     +0.070    [+0.043, +0.097]
#
# 국어도 베이스를 쓴다. 개별 신뢰구간이 0을 포함하지만, n=16에서 +0.030을
# 못 잡는 건 검정력 부족이지 효과가 없다는 증거가 아니다. 국어 구간은 전체
# 구간과 크게 겹쳐 국어가 다르게 움직인다는 근거가 없고, 부트스트랩 재표집의
# 88.7%에서 베이스가 앞섰다. 검정력이 모자란 하위집단만 따로 떼어내지 않는다.
#
# 과목별로 눈에 띄는 점:
# - 수학: math 어댑터는 "정답:/풀이:" 문제 풀이 전용으로 학습돼서 개념 설명을
#   시켜도 그 형식으로 답한다. 측정한 15건 전부(100%)가 그랬다.
#   counseling.py의 _math_report()는 같은 이유로 이미 math 어댑터를 쓰지 않는다.
# - 영어: 학습 답안이 중앙값 20자로 전 과목 최단이라(국어 35 / 사회 39 /
#   과학 44) 한 문장만 내놓는다. '수동태'를 물으면 "be + 과거분사"조차
#   언급하지 않는다. 베이스는 한국어 비율이 0.711로 낮지만, 영어 과목
#   설명에 영어 용어가 섞이는 것은 결함이 아니다.
#
# 어댑터가 나은 축도 있다. 밀도(꺼낸 개념어 중 맞은 비율)는 4개 과목에서
# 어댑터가 앞선다 — 짧지만 정확하다. 다만 출력이 112자 대 359자로 3배
# 짧아서, 그 정확함이 커버리지 격차를 덮지 못한다.
#
# 결과적으로 개념 설명은 전 과목이 베이스를 쓴다. 과목 어댑터(korean/english/
# science/social)는 이 경로에서만 쓰이던 것이라 로드하지 않는다
# (LOAD_SUBJECT_ADAPTERS 참고).
_CONCEPT_USE_BASE: frozenset[str] = frozenset(SUBJECT_ADAPTERS) | {"수학"}


def _concept_explain(subject: str, concept_query: str,
                     context_docs: list[str] | None = None) -> str | None:
    """개념 설명 생성. 생성할 수 없으면 None(호출부에서 Ollama로 fallback).

    context_docs: 호출부가 이미 검색한 RAG 자료. 주면 근거로 붙인다(grounding.py).

    과목마다 어댑터를 쓸지 베이스를 쓸지 다르다 — _CONCEPT_USE_BASE 참고.
    어댑터가 항상 나은 것이 아니라서, 과목별로 측정해 정한 값이다.
    system 프롬프트는 어느 쪽이든 해당 과목 것을 그대로 쓴다(측정도 그 조합으로 했다).

    프롬프트는 학습 포맷(train/preprocess_curriculum.py)과 맞춰 system + user 2턴으로 구성한다.
    enable_thinking=False: 학습 데이터에 <think> 블록이 없어 켜두면 빈 사고 블록이 먼저 나온다.
    """
    import logging

    if subject == "수학":
        adapter_name, system_prompt = None, MATH_SYSTEM_PROMPT
    elif subject in _CONCEPT_USE_BASE and subject in SUBJECT_ADAPTERS:
        _, _, system_prompt = SUBJECT_ADAPTERS[subject]
        adapter_name = None
    elif subject in _LOADED_SUBJECT_ADAPTERS:
        adapter_name, _, system_prompt = SUBJECT_ADAPTERS[subject]
    else:
        return None  # 한국사 등 어댑터 미보유 과목

    try:
        model     = app.state.model
        tokenizer = app.state.tokenizer

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": concept_user_message(concept_query, context_docs)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        # 참고 자료가 붙으면 입력이 1,000토큰을 넘을 수 있다. 오른쪽부터 잘리므로
        # 한도가 모자라면 질문과 생성 프롬프트가 먼저 날아간다.
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        input_len = inputs["input_ids"].shape[-1]

        def _gen():
            with torch.no_grad():
                return model.generate(
                    **inputs,
                    **CONCEPT_GEN,  # 한도·온도는 generation.py (200토큰에서 전부 잘렸다)
                    logits_processor=LogitsProcessorList([_FOREIGN_BLOCKER]),
                    pad_token_id=tokenizer.eos_token_id,
                )

        if adapter_name is None:
            with model.disable_adapter():
                outputs = _gen()
        else:
            model.set_adapter(adapter_name)
            outputs = _gen()

        text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        text = strip_markdown(text)
        if outputs.shape[-1] - input_len >= CONCEPT_GEN["max_new_tokens"]:
            text = trim_cut_tail(text)
        # 바이트 단위로 쪼개진 드문 한자는 토큰 마스크로 못 막는다. 남으면 지운다.
        if has_foreign(text):
            text = strip_foreign(text).strip()
        return text or None
    except Exception as e:
        logging.warning(f"[{subject} 개념 설명] 생성 실패: {e}")
        return None


# counseling 라우터가 main을 import하면 순환 참조가 되므로 registry를 통해 넘긴다.
model_registry.register("concept_explain", _concept_explain)


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

    # 베이스 Qwen으로 생성하고, _concept_explain()이 None을 주는 과목만 Ollama로 넘긴다.
    # GPU 생성은 수 초가 걸리는 블로킹 작업이라 threadpool로 빼서 이벤트 루프를 막지 않는다.
    llm_insight = await run_in_threadpool(_concept_explain, subject, concept_query, rag_docs)

    if not llm_insight:
        prompt = (
            f"{context_block(rag_docs)}"
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


# ── 라우터 등록 (반드시 위의 @app 경로들을 모두 정의한 뒤) ──────────────────────
# FastAPI/Starlette는 등록 순서대로 경로를 매칭한다. counseling 라우터의
# /report/{subject}는 한 세그먼트를 전부 받는 catch-all이라, 이걸 먼저 등록하면
# /api/ai/report/subject-recommend 요청까지 삼켜버린다. 그러면 report_subject()의
# 필수 본문(req: dict) 검증에 걸려 422가 나고, 위에 정의한 전용 핸들러는
# 영원히 호출되지 않는다.
#
# math/writing/premium은 같은 라우터 안에서 {subject}보다 위에 있어 문제가 없지만,
# app 레벨 경로는 여기서 순서를 맞춰야 한다. 새 @app 경로를 추가할 때는 이 줄들보다
# 위에 둘 것.
app.include_router(counseling.router, prefix="/api/ai", tags=["counseling"])
app.include_router(predictor.router, prefix="/api/ai", tags=["predictor"])
app.include_router(writing.router, prefix="/api/writing", tags=["writing"])
app.include_router(rag.router, prefix="/api/rag", tags=["rag"])

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from fastapi import FastAPI, Query  # ✅ Query 추가
import torch
import httpx
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  # ✅ BitsAndBytesConfig 추가
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

from src.api.routers import counseling, predictor, rag

app = FastAPI(title="iRoute AI Server")

app.include_router(counseling.router, prefix="/api/ai", tags=["counseling"])
app.include_router(predictor.router, prefix="/api/ai", tags=["predictor"])
app.include_router(rag.router, prefix="/api/rag", tags=["rag"])

# ✅ 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("📥 범용 모델 로드 중 (4bit 양자화 모드)...")
MODEL_ID = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.clean_up_tokenization_spaces = False
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# ✅ 수학 모델 추가
print("📥 수학 전용 모델 로드 중 (4bit 양자화 모드)...")
MATH_MODEL_ID = "i-route-ai/iroute-math-llm-v2-16bit"
math_tokenizer = AutoTokenizer.from_pretrained(MATH_MODEL_ID)
math_tokenizer.pad_token = math_tokenizer.eos_token
math_tokenizer.clean_up_tokenization_spaces = False
math_model = AutoModelForCausalLM.from_pretrained(
    MATH_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# RAG 로드
DB_DIR = os.path.join(os.path.dirname(__file__), "src", "api", "rag_db")
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
try:
    vector_db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    print("✅ RAG DB 로드 완료!")
except:
    vector_db = None
    print("⚠️ RAG DB 없음, 스킵")


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


def get_rag_context(subject: str, query: str) -> str:
    if not vector_db:
        return ""
    docs = vector_db.similarity_search(f"{subject} {query}", k=2)
    return "\n".join([re.sub(r'<[^>]*>', '', doc.page_content) for doc in docs])


# ✅ Query 파라미터 명시적 선언
@app.post("/api/ai/report/subject-recommend")
def generate_subject_recommendation(
        student_id: str = Query(...),
        subject: str = Query(...)
):
    weakness_data = get_student_weakness_from_java(student_id, subject)
    if not weakness_data:
        return {"status": "error", "message": "데이터 없음"}

    concept_query = weakness_data[0].get("conceptTag")
    retrieved_questions = get_rag_context(subject, concept_query)

    # ✅ 과목별 모델 라우팅
    if subject == "수학":
        active_tokenizer = math_tokenizer
        active_model = math_model
        prompt = f"""### 지시사항:
    당신은 수학 전문가입니다. 아래 [학습 자료]만 사용하여 개념을 설명하세요.

    [학습 자료]:
    {retrieved_questions}

    ### 개념 요약:"""
    else:
        active_tokenizer = tokenizer
        active_model = model
        prompt = f"""### 지시사항:
    당신은 {subject} 전문가입니다. 아래 [학습 자료]만 사용하여 개념을 설명하세요.

    [학습 자료]:
    {retrieved_questions}

    ### 개념 요약:"""

    inputs = active_tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = active_model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=active_tokenizer.eos_token_id
        )

    result_text = active_tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_text = result_text.split("개념 요약:")[-1].strip()

    def refine_text(text: str) -> str:
        text = re.sub(r'<[^>]*>', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.replace("###", "").replace("  ", " ").strip()
        return text

    clean_text = refine_text(raw_text)

    return {
        "studentId": student_id,
        "subject": subject,
        "targetConcept": concept_query,
        "aiRecommendationReport": clean_text
    }

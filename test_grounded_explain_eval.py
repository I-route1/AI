"""개념 설명의 사실 정확도: 지금 방식 대 검색 자료를 근거로 붙인 방식.

개념F1(test_concept_explain_eval.py)은 참조의 용어가 출력에 나오는지만 봐서
사실 오류를 잡지 못한다. 한국사에서 개념F1은 Qwen이 앞섰지만 사실 정확도는
Ollama가 나았다(MODELS.md). 그래서 이 스크립트는 점수를 매기지 않고, 사람이
판정할 수 있게 조건별 출력을 나란히 저장한다.

개념은 Backend 시드(questions_seed.json)의 실제 conceptTag와 한국사 대표 개념이다.

FAISS(69만 벡터)와 8B 모델을 한 프로세스에 올리면 죽으므로 두 단계로 나눈다.
    python test_grounded_explain_eval.py retrieve train/grounded_ctx.json
    python test_grounded_explain_eval.py generate train/grounded_ctx.json train/grounded_out.json
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

CONCEPTS: dict[str, list[str]] = {
    "국어": ["고전소설 인물 심리", "문학 표현기법 반어", "시조 형식 평시조",
             "현대소설 서술시점", "비문학 추론적 이해"],
    "수학": ["이차방정식 판별식", "등차수열 일반항", "삼각함수 덧셈공식",
             "이차함수 꼭짓점", "미분 chain rule 응용"],
    "영어": ["가정법 과거완료", "관계부사 where/when", "분사구문 의미상 주어",
             "현재완료 용법", "복합관계대명사"],
    "과학": ["뉴턴 운동 법칙", "물질 변화", "세포 구조", "유전과 진화"],
    "사회": ["민주주의 원리", "세계 지리", "시장 경제", "한국 현대사"],
    "한국사": ["신간회", "갑오개혁", "병자호란", "대동법", "4·19 혁명",
               "6월 민주항쟁", "서희", "을사늑약"],
}


def retrieve(out_path: str) -> None:
    from src.api.routers.rag import _concept_map_search, subject_aware_search

    rows = []
    for subject, concepts in CONCEPTS.items():
        for c in concepts:
            docs = subject_aware_search(subject, c, k=3)
            source = "conceptmap" if _concept_map_search(subject, c, 3) else "faiss_or_fallback"
            rows.append({"subject": subject, "concept": c, "source": source, "docs": docs})
            print(f"[{subject}] {c}  ({source})  {docs[0][:80] if docs else ''!r}")
    json.dump(rows, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def generate(ctx_path: str, out_path: str) -> None:
    import httpx
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from src.api.adapters import BASE_MODEL_ID, MATH_SYSTEM_PROMPT, SUBJECT_ADAPTERS
    from src.api.grounding import concept_user_message, context_block

    history_prompt = ("당신은 한국사 전문 교사입니다. 학생의 한국사 지문과 질문에 대해 "
                      "정확하고 이해하기 쉬운 답변을 제공하세요.")

    def system_prompt(subject: str) -> str:
        if subject == "수학":
            return MATH_SYSTEM_PROMPT
        if subject == "한국사":
            return history_prompt
        return SUBJECT_ADAPTERS[subject][2]

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb,
                                                 device_map={"": 0}).eval()

    def qwen(subject: str, concept: str, docs: list[str] | None) -> str:
        msgs = [{"role": "system", "content": system_prompt(subject)},
                {"role": "user", "content": concept_user_message(concept, docs)}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        ids = tok(p, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=200, do_sample=False,
                                 repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    def ollama(subject: str, concept: str, docs: list[str] | None) -> str:
        # subject-recommend 폴백 프롬프트 (main.py)
        prompt = (f"{context_block(docs)}{subject} 과목에서 '{concept}' 개념을 어려워하는 학생에게 "
                  f"이 개념의 핵심 포인트와 효과적인 학습 방법을 2~3문장으로 한국어로 답해주세요.")
        r = httpx.post("http://localhost:11434/api/generate",
                       json={"model": "llama3.1:latest", "prompt": prompt, "stream": False,
                             "options": {"num_predict": 150, "temperature": 0.0, "seed": 42,
                                         "num_ctx": 4096}},
                       timeout=60)
        return r.json().get("response", "").strip()

    rows = json.load(open(ctx_path, encoding="utf-8"))
    for row in rows:
        s, c, docs = row["subject"], row["concept"], row["docs"]
        row["qwen_plain"] = qwen(s, c, None)
        row["qwen_grounded"] = qwen(s, c, docs)
        if s == "한국사":
            row["ollama_plain"] = ollama(s, c, None)
            row["ollama_grounded"] = ollama(s, c, docs)
        print(f"[{s}] {c} 완료", flush=True)
        json.dump(rows, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "retrieve":
        retrieve(sys.argv[2])
    else:
        generate(sys.argv[2], sys.argv[3])

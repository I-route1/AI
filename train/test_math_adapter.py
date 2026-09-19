"""
수학 어댑터 성능 테스트
실행: .\venv_blackwell\Scripts\python.exe train/test_math_adapter.py
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID   = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
ADAPTER_PATH    = "train/math_adapter"
SYSTEM_PROMPT   = "당신은 수학 전문 교사입니다. 학생의 수학 문제에 대해 정확한 풀이와 해설을 제공하세요."

TEST_PROBLEMS = [
    {
        "question": "이차방정식 x² - 5x + 6 = 0의 두 근을 구하시오.",
        "expected": "x = 2, x = 3",
    },
    {
        "question": "일차방정식 3x + 7 = 16을 풀어라.",
        "expected": "x = 3",
    },
    {
        "question": "삼각형의 세 내각의 합은 몇 도인가?",
        "expected": "180도",
    },
    {
        "question": "(-3)² 의 값을 구하시오.",
        "expected": "9",
    },
    {
        "question": "직사각형의 가로가 8cm, 세로가 5cm일 때 넓이를 구하시오.",
        "expected": "40cm²",
    },
]


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    print("모델 로드 중...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    tokenizer = processor.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    # transformers 내장 model.load_adapter()는 caching_allocator_warmup에서 모델 전체
    # 크기만큼 재할당을 시도해 OOM 발생. peft 네이티브 PeftModel.from_pretrained로 우회.
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, adapter_name="math")
    model.set_adapter("math")
    model.eval()
    print("✅ 로드 완료\n")

    for i, item in enumerate(TEST_PROBLEMS, 1):
        print(f"{'='*60}")
        print(f"[문제 {i}] {item['question']}")
        print(f"[정답]   {item['expected']}")

        prompt = build_prompt(tokenizer, item["question"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )

        answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        print(f"[모델 출력]\n{answer}\n")


if __name__ == "__main__":
    main()

"""
국어/영어/과학/사회 어댑터 성능 정성 테스트 - Qwen3-8B 버전
각 과목 eval.jsonl에서 무작위 샘플을 뽑아 실제 생성 결과를 정답과 비교 출력한다.
실행: .\\venv_blackwell\\Scripts\\python.exe train/test_curriculum_adapters_qwen.py
"""

import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
SUBJECTS = ["korean", "english", "science", "social"]
N_SAMPLES = 5
SEED = 42


def load_samples(subject: str, n: int):
    path = f"train/{subject}_eval.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    random.Random(SEED).shuffle(lines)
    return lines[:n]


def build_prompt(tokenizer, messages):
    prompt_messages = messages[:2]
    return tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def main():
    print("베이스 모델 로드 중...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )

    first = True
    for subject in SUBJECTS:
        adapter_path = f"train/{subject}_adapter_qwen"
        if first:
            model = PeftModel.from_pretrained(model, adapter_path, adapter_name=subject)
            first = False
        else:
            model.load_adapter(adapter_path, adapter_name=subject)
    model.eval()
    print("로드 완료\n")

    for subject in SUBJECTS:
        model.set_adapter(subject)
        samples = load_samples(subject, N_SAMPLES)
        print(f"\n{'#'*70}")
        print(f"# {subject.upper()} 어댑터 테스트")
        print(f"{'#'*70}")

        for i, item in enumerate(samples, 1):
            messages = item["messages"]
            question = messages[1]["content"]
            expected = messages[2]["content"]

            print(f"\n{'='*60}")
            print(f"[{subject} 문제 {i}]\n{question[:500]}")
            print(f"\n[정답]\n{expected[:500]}")

            prompt = build_prompt(tokenizer, messages)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to("cuda")
            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.3,
                    do_sample=True,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id,
                )

            answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
            print(f"\n[모델 출력]\n{answer}")


if __name__ == "__main__":
    main()

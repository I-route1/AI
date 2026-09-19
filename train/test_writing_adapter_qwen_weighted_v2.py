"""
글쓰기 어댑터 성능 테스트 - Qwen3-8B 버전 (writing_eval_qwen.jsonl 홀드아웃 샘플 기반)
실행: .\\venv_blackwell\\Scripts\\python.exe train/test_writing_adapter_qwen.py
"""

import json
import random
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
ADAPTER_PATH  = "train/writing_adapter_qwen_weighted_v2"
EVAL_FILE     = "train/writing_eval_qwen.jsonl"
MAX_LENGTH    = 896
N_SCORE       = 100
N_FEEDBACK    = 3
RANDOM_SEED   = 7

SCORE_PATTERN = re.compile(r"[1-5]")


def load_eval_samples():
    score_items, feedback_items = [], []
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            expected = obj["messages"][2]["content"].strip()
            if SCORE_PATTERN.fullmatch(expected):
                score_items.append(obj)
            else:
                feedback_items.append(obj)

    random.seed(RANDOM_SEED)
    random.shuffle(score_items)
    random.shuffle(feedback_items)
    return score_items[:N_SCORE], feedback_items[:N_FEEDBACK]


def build_prompt(tokenizer, messages):
    prompt_messages = messages[:2]
    return tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def main():
    print("모델 로드 중...")
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
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, adapter_name="writing")
    model.set_adapter("writing")
    model.eval()
    print("로드 완료\n")

    score_items, feedback_items = load_eval_samples()

    correct = 0
    no_digit = 0
    diffs = []
    confusion = {}  # (expected, predicted) -> count
    print("=" * 60)
    print(f"[채점(숫자) 테스트] {len(score_items)}개")
    for i, item in enumerate(score_items, 1):
        messages = item["messages"]
        expected = messages[2]["content"].strip()

        prompt = build_prompt(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to("cuda")
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.4,
                do_sample=True,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        match = SCORE_PATTERN.search(answer)
        predicted = match.group(0) if match else "?"
        is_correct = predicted == expected
        correct += int(is_correct)

        if predicted == "?":
            no_digit += 1
        else:
            diffs.append(abs(int(predicted) - int(expected)))
        confusion[(expected, predicted)] = confusion.get((expected, predicted), 0) + 1

        marker = "OK" if is_correct else "X"
        print(f"[{i:3d}] 정답 {expected} | 출력 {answer!r:20s} -> 파싱 {predicted} {marker}")

    print(f"\n채점 정확도: {correct}/{len(score_items)} ({correct/len(score_items)*100:.1f}%)")
    print(f"숫자 파싱 실패(형식 무시): {no_digit}개")
    if diffs:
        print(f"평균 오차(|예측-정답|): {sum(diffs)/len(diffs):.2f}")
        exact = sum(1 for d in diffs if d == 0)
        within1 = sum(1 for d in diffs if d <= 1)
        print(f"±0(정확): {exact}/{len(diffs)} | ±1 이내: {within1}/{len(diffs)}")
    print("\n혼동 행렬 (정답 -> 예측: 횟수):")
    for (exp, pred), cnt in sorted(confusion.items()):
        print(f"  {exp} -> {pred}: {cnt}")

    print("\n" + "=" * 60)
    print(f"[피드백(서술형) 테스트] {len(feedback_items)}개")
    for i, item in enumerate(feedback_items, 1):
        messages = item["messages"]
        expected = messages[2]["content"].strip()

        prompt = build_prompt(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to("cuda")
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
        answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        print(f"\n[{i}] 지시문: {messages[0]['content'][:60]}")
        print(f"    [정답 피드백]  {expected}")
        print(f"    [모델 피드백]  {answer}")


if __name__ == "__main__":
    main()

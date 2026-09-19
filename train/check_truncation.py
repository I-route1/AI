"""
writing_train.jsonl / writing_eval.jsonl 중 MAX_LENGTH(640) 초과로
정답(assistant) 텍스트가 잘리는 샘플이 얼마나 되는지 점검.
실행: .\\venv_blackwell\\Scripts\\python.exe train/check_truncation.py
"""

import json
from transformers import AutoProcessor

BASE_MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
MAX_LENGTH = 640
FILES = ["train/writing_train.jsonl", "train/writing_eval.jsonl"]


def main():
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    tokenizer = processor.tokenizer

    for filepath in FILES:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        total = len(lines)
        truncated_examples = []
        lengths = []

        for line in lines:
            obj = json.loads(line)
            messages = obj["messages"]

            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            full_ids = tokenizer(full_text, truncation=False)["input_ids"]
            lengths.append(len(full_ids))

            if len(full_ids) > MAX_LENGTH:
                # assistant 부분만 따로 토크나이즈해서 얼마나 잘리는지 확인
                prompt_only = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
                prompt_len = len(tokenizer(prompt_only, truncation=False)["input_ids"])
                assistant_len = len(full_ids) - prompt_len
                remaining_for_assistant = MAX_LENGTH - prompt_len
                truncated_examples.append({
                    "total_len": len(full_ids),
                    "prompt_len": prompt_len,
                    "assistant_len": assistant_len,
                    "remaining_for_assistant": remaining_for_assistant,
                    "assistant_text": messages[2]["content"][:200],
                })

        n_trunc = len(truncated_examples)
        print(f"\n=== {filepath} ===")
        print(f"전체 {total}개 중 640 토큰 초과: {n_trunc}개 ({n_trunc/total*100:.2f}%)")
        print(f"토큰 길이: min={min(lengths)}, max={max(lengths)}, "
              f"median={sorted(lengths)[len(lengths)//2]}, mean={sum(lengths)/len(lengths):.1f}")

        # 정답(assistant) 텍스트가 아예 잘려서 0토큰 이하로 사라지는 심각한 케이스
        severe = [e for e in truncated_examples if e["remaining_for_assistant"] <= 0]
        print(f"정답이 완전히 잘려서 학습 신호가 아예 없는 샘플: {len(severe)}개")

        if truncated_examples:
            print("\n--- 잘린 샘플 예시 (최대 3개) ---")
            for e in truncated_examples[:3]:
                print(f"  전체 {e['total_len']}토큰 (프롬프트 {e['prompt_len']} + 정답 {e['assistant_len']}) "
                      f"→ 정답에 남는 여유: {e['remaining_for_assistant']}토큰")
                print(f"  정답 앞부분: {e['assistant_text']!r}")


if __name__ == "__main__":
    main()

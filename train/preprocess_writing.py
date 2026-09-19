"""
글쓰기 평가/피드백 데이터 → LoRA 학습용 JSONL 전처리 스크립트 (다운샘플링)

원본 3개 카테고리 파일(논술형/서술형/주제별)에서 카테고리별로 SAMPLES_PER_CATEGORY개씩,
카테고리 내 채점:피드백 1:1 비율을 유지하며 추출.

실행: python train/preprocess_writing.py
출력: train/writing_train.jsonl, train/writing_eval.jsonl
"""

import json
import random
import re
from pathlib import Path

SOURCE_DIR = Path("C:/Users/User/IdeaProjects/untitled")
SOURCE_FILES = [
    SOURCE_DIR / "onjeom_논술형_글쓰기_2stage.jsonl",
    SOURCE_DIR / "onjeom_서술형_글쓰기_2stage.jsonl",
    SOURCE_DIR / "onjeom_주제별_글쓰기_2stage.jsonl",
]

OUTPUT_DIR    = Path("train")
TRAIN_FILE    = OUTPUT_DIR / "writing_train.jsonl"
EVAL_FILE     = OUTPUT_DIR / "writing_eval.jsonl"
SAMPLES_PER_CATEGORY = 10000
EVAL_RATIO    = 0.05
RANDOM_SEED   = 42

SCORE_PATTERN = re.compile(r"[1-5]")


def load_records(filepath: Path) -> tuple[list[dict], list[dict]]:
    score_records, feedback_records = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            instruction = obj.get("instruction", "").strip()
            input_text  = obj.get("input", "").strip()
            output_text = obj.get("output", "").strip()
            if not instruction or not input_text or not output_text:
                continue
            if SCORE_PATTERN.fullmatch(output_text):
                score_records.append(obj)
            else:
                feedback_records.append(obj)
    return score_records, feedback_records


def to_sample(obj: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": obj["instruction"].strip()},
            {"role": "user", "content": obj["input"].strip()},
            {"role": "assistant", "content": obj["output"].strip()},
        ]
    }


def main():
    random.seed(RANDOM_SEED)

    n_categories = len(SOURCE_FILES)
    per_category = [SAMPLES_PER_CATEGORY] * n_categories

    all_samples = []
    for filepath, cat_total in zip(SOURCE_FILES, per_category):
        score_records, feedback_records = load_records(filepath)
        n_score = cat_total // 2
        n_feedback = cat_total - n_score

        random.shuffle(score_records)
        random.shuffle(feedback_records)
        picked = score_records[:n_score] + feedback_records[:n_feedback]

        print(f"{filepath.name}: 원본 채점 {len(score_records)} / 피드백 {len(feedback_records)} "
              f"→ 추출 채점 {n_score} / 피드백 {n_feedback}")

        all_samples.extend(to_sample(obj) for obj in picked)

    random.shuffle(all_samples)

    n_eval = int(len(all_samples) * EVAL_RATIO)
    eval_samples  = all_samples[:n_eval]
    train_samples = all_samples[n_eval:]

    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for sample in eval_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n총 추출: {len(all_samples)}개")
    print(f"학습 데이터: {len(train_samples)}개 → {TRAIN_FILE}")
    print(f"평가 데이터: {len(eval_samples)}개 → {EVAL_FILE}")

    print("\n--- 학습 샘플 미리보기 ---")
    for msg in train_samples[0]["messages"]:
        print(f"[{msg['role']}] {msg['content'][:100]}")


if __name__ == "__main__":
    main()

"""
글쓰기 평가/피드백 데이터 -> Qwen3 LoRA 학습용 JSONL 전처리 (v2)

기존(preprocess_writing.py) 대비 변경점:
- 토크나이저를 Qwen3 기준으로 사용 (한국어 토큰 수가 Gemma 대비 늘어남)
- 카테고리별 단순 길이 필터링 대신, MAX_LENGTH를 넘는 샘플은
  system(지시문)/assistant(정답)는 그대로 두고 user 쪽 입력 텍스트만
  잘라서 항상 정답이 온전히 보존되도록 함 (주제별 카테고리는 전체가
  구조적으로 길어서 단순 필터링으로는 표본을 거의 확보할 수 없었음)

실행: python train/preprocess_writing_qwen.py
출력: train/writing_train_qwen.jsonl, train/writing_eval_qwen.jsonl
"""

import json
import random
import re
from pathlib import Path
from transformers import AutoTokenizer

SOURCE_DIR = Path("C:/Users/User/IdeaProjects/untitled")
SOURCE_FILES = [
    SOURCE_DIR / "onjeom_논술형_글쓰기_2stage.jsonl",
    SOURCE_DIR / "onjeom_서술형_글쓰기_2stage.jsonl",
    SOURCE_DIR / "onjeom_주제별_글쓰기_2stage.jsonl",
]

OUTPUT_DIR = Path("train")
TRAIN_FILE = OUTPUT_DIR / "writing_train_qwen.jsonl"
EVAL_FILE = OUTPUT_DIR / "writing_eval_qwen.jsonl"
SAMPLES_PER_CATEGORY = 10000
EVAL_RATIO = 0.05
RANDOM_SEED = 42

BASE_MODEL_ID = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MAX_LENGTH = 896
SAFETY_MARGIN = 16  # 특수토큰/템플릿 오차 대비 여유

SCORE_PATTERN = re.compile(r"[1-5]")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)


def load_records(filepath: Path) -> tuple[list[dict], list[dict]]:
    score_records, feedback_records = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            instruction = obj.get("instruction", "").strip()
            input_text = obj.get("input", "").strip()
            output_text = obj.get("output", "").strip()
            if not instruction or not input_text or not output_text:
                continue
            if SCORE_PATTERN.fullmatch(output_text):
                score_records.append(obj)
            else:
                feedback_records.append(obj)
    return score_records, feedback_records


def token_len(messages: list[dict]) -> int:
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return len(tokenizer(text, truncation=False)["input_ids"])


def fit_to_budget(instruction: str, input_text: str, output_text: str) -> dict:
    """system+assistant는 고정, user(input_text)만 필요한 만큼 잘라서
    전체 길이가 MAX_LENGTH 이내가 되도록 만든다."""
    budget = MAX_LENGTH - SAFETY_MARGIN

    def build(inp: str):
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": inp},
            {"role": "assistant", "content": output_text},
        ]

    messages = build(input_text)
    length = token_len(messages)
    if length <= budget:
        return {"messages": messages}

    # system+assistant만으로 이미 예산을 넘으면(즉 input_text=="") 이 레코드는 포기
    empty_len = token_len(build(""))
    if empty_len >= budget:
        return None

    # input_text를 이분 탐색으로 잘라서 예산에 맞춤 (문자 단위 절삭)
    lo, hi = 0, len(input_text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if token_len(build(input_text[:mid])) <= budget:
            lo = mid
        else:
            hi = mid - 1
    truncated_input = input_text[:lo].rstrip() + " …(중략)"
    return {"messages": build(truncated_input)}


def main():
    n_categories = len(SOURCE_FILES)
    per_category = [SAMPLES_PER_CATEGORY] * n_categories

    random.seed(RANDOM_SEED)
    all_samples = []
    for filepath, cat_total in zip(SOURCE_FILES, per_category):
        score_records, feedback_records = load_records(filepath)
        n_score = cat_total // 2
        n_feedback = cat_total - n_score

        random.shuffle(score_records)
        random.shuffle(feedback_records)

        picked = []
        truncated_count = 0
        dropped_count = 0

        for pool, n_needed, label in [(score_records, n_score, "score"), (feedback_records, n_feedback, "feedback")]:
            got = 0
            for obj in pool:
                if got >= n_needed:
                    break
                sample = fit_to_budget(obj["instruction"].strip(), obj["input"].strip(), obj["output"].strip())
                if sample is None:
                    dropped_count += 1
                    continue
                original_len = token_len([
                    {"role": "system", "content": obj["instruction"].strip()},
                    {"role": "user", "content": obj["input"].strip()},
                    {"role": "assistant", "content": obj["output"].strip()},
                ])
                if original_len > MAX_LENGTH - SAFETY_MARGIN:
                    truncated_count += 1
                picked.append(sample)
                got += 1
            if got < n_needed:
                print(f"  [경고] {filepath.name} {label}: {n_needed}개 필요했지만 {got}개만 확보 (풀 소진)")

        print(f"{filepath.name}: 확보 {len(picked)}개 (그중 입력 잘림 {truncated_count}개, 포기 {dropped_count}개)")
        all_samples.extend(picked)

    random.shuffle(all_samples)

    n_eval = int(len(all_samples) * EVAL_RATIO)
    eval_samples = all_samples[:n_eval]
    train_samples = all_samples[n_eval:]

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for sample in eval_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n총 추출: {len(all_samples)}개")
    print(f"학습 데이터: {len(train_samples)}개 -> {TRAIN_FILE}")
    print(f"평가 데이터: {len(eval_samples)}개 -> {EVAL_FILE}")


if __name__ == "__main__":
    main()

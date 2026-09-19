"""
수학교과문제풀이데이터 → LoRA 학습용 JSONL 전처리 스크립트

실행: python train/preprocess.py
출력: train/math_train.jsonl, train/math_val.jsonl
"""

import os
import json
import random
from pathlib import Path

DATA_ROOT  = Path("C:/수학교과문제풀이데이터")
OUTPUT_DIR = Path("train")
TRAIN_FILE = OUTPUT_DIR / "math_train.jsonl"
EVAL_FILE  = OUTPUT_DIR / "math_eval.jsonl"   # VL_ 폴더 기반 별도 평가셋
RANDOM_SEED = 42

SYSTEM_PROMPT = "당신은 수학 전문 교사입니다. 학생의 수학 문제에 대해 정확한 풀이와 해설을 제공하세요."


def extract_text_by_class(learning_data: list, class_name: str) -> str:
    texts = []
    for item in learning_data:
        if class_name in item.get("class_name", ""):
            for info in item.get("class_info_list", []):
                text = info.get("text_description", "").strip()
                if text:
                    texts.append(text)
    return " ".join(texts)


def parse_file(filepath: Path) -> dict | None:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    learning_data = data.get("learning_data_info", [])
    source_info = data.get("source_data_info", {})

    question = extract_text_by_class(learning_data, "문항(텍스트)")
    answer = extract_text_by_class(learning_data, "정답(텍스트)")
    explanation = extract_text_by_class(learning_data, "해설(텍스트)")

    # 문제 또는 정답이 없으면 스킵
    if not question or not answer:
        return None

    difficulty = source_info.get("level_of_difficulty", "")
    standard = ""
    for key in ["2022_achievement_standard", "2015_achievement_standard"]:
        standards = source_info.get(key, [])
        if standards and standards[0].strip():
            standard = standards[0].strip()
            break

    input_text = question
    if standard:
        input_text = f"[성취 기준: {standard}]\n{question}"

    output_parts = [f"정답: {answer}"]
    if explanation:
        output_parts.append(f"풀이: {explanation}")

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": "\n".join(output_parts)},
        ]
    }


def main():
    train_samples = []
    eval_samples  = []
    skipped = 0

    folders = sorted(DATA_ROOT.iterdir())
    print(f"폴더 수: {len(folders)}")

    for folder in folders:
        if not folder.is_dir():
            continue
        is_eval = folder.name.startswith("VL_")
        files = list(folder.glob("*.json"))
        for filepath in files:
            sample = parse_file(filepath)
            if sample:
                if is_eval:
                    eval_samples.append(sample)
                else:
                    train_samples.append(sample)
            else:
                skipped += 1

    random.seed(RANDOM_SEED)
    random.shuffle(train_samples)
    random.shuffle(eval_samples)

    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for sample in eval_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"변환 성공: {len(train_samples) + len(eval_samples)}개 / 스킵: {skipped}개")
    print(f"학습 데이터 (TL_): {len(train_samples)}개 → {TRAIN_FILE}")
    print(f"평가 데이터 (VL_): {len(eval_samples)}개 → {EVAL_FILE}")

    print("\n--- 학습 샘플 미리보기 ---")
    for msg in train_samples[0]["messages"]:
        print(f"[{msg['role']}] {msg['content'][:100]}")


if __name__ == "__main__":
    main()

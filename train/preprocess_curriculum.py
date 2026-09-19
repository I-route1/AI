"""
교과 단계별 데이터(D:/교과 단계별 데이터) -> Qwen3 LoRA 학습용 JSONL 전처리

수학은 이미 train_math_lora_qwen.py가 별도 데이터셋(C:/수학교과문제풀이데이터)으로
어댑터를 학습해 두었으므로 여기서는 국어/영어/과학/사회 4과목만 대상으로 한다.

원본 라벨 구조 (예: S1_초등_3_국어_TXT_031974.json):
  raw_data_info.school / grade / semester / subject
  source_data_info.2022_achievement_standard / 2015_achievement_standard
  learning_data_info.text_description (학습 지문)
  learning_data_info.text_qa (질문)
  learning_data_info.text_an (답변)

data_captured 폴더명 패턴: TL_{순번}.{학교} {학년}_{순번}.{과목}_01.텍스트
(그 옆의 _02.이미지 폴더는 이미지 원본 - 텍스트 LLM 학습에는 불필요하므로 무시)

실행: python train/preprocess_curriculum.py
출력: train/{korean,english,science,social}_train.jsonl / _eval.jsonl
"""

import json
import random
from pathlib import Path
from collections import defaultdict

DATA_ROOT = Path("D:/교과 단계별 데이터")
OUTPUT_DIR = Path("train")
RANDOM_SEED = 42
EVAL_RATIO = 0.05
CAP_PER_SUBJECT = 16000          # 수학 어댑터(14,395건/2epoch/~3h) 규모에 맞춤
MIN_ANSWER_LEN = 4               # 이보다 짧은 답변(스텁)은 제외
MIN_QUESTION_LEN = 4

# 과목명(한글, 폴더명 기준) -> (영문 슬러그, 시스템 프롬프트)
SUBJECTS = {
    "국어": ("korean", "당신은 국어 전문 교사입니다. 학생의 국어 지문과 질문에 대해 정확하고 이해하기 쉬운 답변을 제공하세요."),
    "영어": ("english", "당신은 영어 전문 교사입니다. 학생의 영어 지문과 질문에 대해 정확하고 이해하기 쉬운 답변을 제공하세요."),
    "과학": ("science", "당신은 과학 전문 교사입니다. 학생의 과학 지문과 질문에 대해 정확하고 이해하기 쉬운 답변을 제공하세요."),
    "사회": ("social", "당신은 사회 전문 교사입니다. 학생의 사회 지문과 질문에 대해 정확하고 이해하기 쉬운 답변을 제공하세요."),
}


def find_subject_dirs(subject: str) -> list[Path]:
    # 폴더명 예: TL_01.초등학교 3학년_01.국어_01.텍스트 (과목 앞에 과목 순번이 붙음)
    return sorted(DATA_ROOT.glob(f"TL_*.{subject}_01.텍스트"))


def best_standard(source_info: dict) -> str:
    for key in ["2022_achievement_standard", "2015_achievement_standard", "2009_achievement_standard"]:
        vals = source_info.get(key, [])
        if vals and vals[0].strip():
            return vals[0].strip()
    return ""


def parse_file(filepath: Path) -> dict | None:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    raw_info = data.get("raw_data_info", {})
    source_info = data.get("source_data_info", {})
    learning = data.get("learning_data_info", {})
    if isinstance(learning, list):
        learning = learning[0] if learning else {}

    passage = (learning.get("text_description") or "").strip()
    question = (learning.get("text_qa") or "").strip()
    answer = (learning.get("text_an") or "").strip()

    if len(question) < MIN_QUESTION_LEN or len(answer) < MIN_ANSWER_LEN:
        return None
    if answer == question:
        return None

    school = raw_info.get("school", "")
    grade = raw_info.get("grade", "")
    standard = best_standard(source_info)

    user_parts = []
    if passage:
        user_parts.append(f"[학습 지문]\n{passage}")
    if standard:
        user_parts.append(f"[교육과정 성취기준]\n{standard}")
    user_parts.append(f"질문: {question}")

    return {
        "school": school,
        "grade": grade,
        "input": "\n".join(user_parts),
        "output": answer,
    }


def main():
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    for subject_kr, (slug, system_prompt) in SUBJECTS.items():
        dirs = find_subject_dirs(subject_kr)
        if not dirs:
            print(f"[{subject_kr}] 폴더를 찾지 못함 - 건너뜀")
            continue

        parsed = []
        skipped = 0
        grade_counts = defaultdict(int)
        for d in dirs:
            for filepath in d.glob("*.json"):
                rec = parse_file(filepath)
                if rec is None:
                    skipped += 1
                    continue
                parsed.append(rec)
                grade_counts[f"{rec['school']} {rec['grade']}"] += 1

        total_parsed = len(parsed)
        random.shuffle(parsed)
        if len(parsed) > CAP_PER_SUBJECT:
            parsed = parsed[:CAP_PER_SUBJECT]

        samples = [
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": rec["input"]},
                    {"role": "assistant", "content": rec["output"]},
                ]
            }
            for rec in parsed
        ]

        n_eval = max(1, int(len(samples) * EVAL_RATIO))
        eval_samples = samples[:n_eval]
        train_samples = samples[n_eval:]

        train_file = OUTPUT_DIR / f"{slug}_train.jsonl"
        eval_file = OUTPUT_DIR / f"{slug}_eval.jsonl"
        with open(train_file, "w", encoding="utf-8") as f:
            for s in train_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        with open(eval_file, "w", encoding="utf-8") as f:
            for s in eval_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        print(f"\n=== {subject_kr} ({slug}) ===")
        print(f"  원본 폴더: {len(dirs)}개, 파싱 성공 {total_parsed}/{total_parsed + skipped} (스킵 {skipped}), 캡 적용 후 {len(parsed)}개")
        print(f"  학년 분포(캡 적용 전 전체): {dict(sorted(grade_counts.items()))}")
        print(f"  학습 {len(train_samples)}개 -> {train_file}")
        print(f"  평가 {len(eval_samples)}개 -> {eval_file}")


if __name__ == "__main__":
    main()

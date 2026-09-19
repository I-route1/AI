"""
Qwen3 토크나이저 기준으로 원본 글쓰기 데이터의 토큰 길이 분포를 카테고리별로 확인.
MAX_LENGTH를 얼마로 잡을지, 카테고리당 1만개(채점 5천+피드백 5천)를
그 길이 이하에서 확보 가능한지 확인.
"""
import json
import random
from pathlib import Path
from transformers import AutoTokenizer

SOURCE_DIR = Path("C:/Users/User/IdeaProjects/untitled")
SOURCE_FILES = [
    SOURCE_DIR / "onjeom_논술형_글쓰기_2stage.jsonl",
    SOURCE_DIR / "onjeom_서술형_글쓰기_2stage.jsonl",
    SOURCE_DIR / "onjeom_주제별_글쓰기_2stage.jsonl",
]
THRESHOLDS = [640, 768, 896, 1024, 1152, 1280]

tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-8B-unsloth-bnb-4bit")

import re
SCORE_PATTERN = re.compile(r"[1-5]")

for filepath in SOURCE_FILES:
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    score_lens, feedback_lens = [], []
    for line in lines:
        obj = json.loads(line)
        instruction = obj.get("instruction", "").strip()
        input_text = obj.get("input", "").strip()
        output_text = obj.get("output", "").strip()
        if not instruction or not input_text or not output_text:
            continue
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        ids = tokenizer(text, truncation=False)["input_ids"]
        n = len(ids)
        if SCORE_PATTERN.fullmatch(output_text):
            score_lens.append(n)
        else:
            feedback_lens.append(n)

    print(f"\n=== {filepath.name} ===")
    print(f"채점 레코드 {len(score_lens)}개, 피드백 레코드 {len(feedback_lens)}개")
    for th in THRESHOLDS:
        n_score_ok = sum(1 for x in score_lens if x <= th)
        n_fb_ok = sum(1 for x in feedback_lens if x <= th)
        print(f"  <= {th}: 채점 {n_score_ok}개, 피드백 {n_fb_ok}개 (각 5000 필요)")

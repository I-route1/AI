"""국어/영어/과학/사회 교과 어댑터의 정량 평가.

train/{subject}_eval.jsonl(과목당 800건)이 학습 이후 한 번도 정량 평가에
쓰이지 않아서, 어댑터가 실제로 베이스 모델보다 나은지 확인되지 않았다.
이 스크립트는 같은 샘플에 대해 (a) 어댑터 적용 (b) 어댑터 비활성(=베이스)
두 조건으로 생성해 비교한다. 차이가 곧 어댑터의 기여분이다.

지표: 한국어는 교착어라 어절 단위 토큰화가 불안정해 문자 단위로 잰다.
  - char-F1   : 정답/생성의 문자 다중집합 겹침 F1 (SQuAD F1의 문자판)
  - ROUGE-L   : 최장공통부분수열 기반 F1
  - EM        : 공백 제거 후 완전일치

사용법:
    python test_subject_adapters_eval.py                 # 과목당 40건
    python test_subject_adapters_eval.py --n 100         # 과목당 100건
    python test_subject_adapters_eval.py --subjects 국어 과학
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src.api.adapters import BASE_MODEL_ID, MATH_ADAPTER_PATH, SUBJECT_ADAPTERS  # noqa: E402

EVAL_PATH = "train/{slug}_eval.jsonl"
SEED = 42


# ── 지표 ──────────────────────────────────────────────────────────────────────
def _chars(s: str) -> list[str]:
    return [c for c in s if not c.isspace()]


def char_f1(pred: str, ref: str) -> float:
    p, r = Counter(_chars(pred)), Counter(_chars(ref))
    overlap = sum((p & r).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(p.values())
    recall = overlap / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, ref: str) -> float:
    a, b = _chars(pred), _chars(ref)
    if not a or not b:
        return 0.0
    # LCS 길이만 필요하므로 행 두 개로 굴린다 (문자열이 길어도 메모리 일정)
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(a), lcs / len(b)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, ref: str) -> float:
    return float("".join(_chars(pred)) == "".join(_chars(ref)))


# ── 평가 ──────────────────────────────────────────────────────────────────────
def load_samples(slug: str, n: int) -> list[dict]:
    path = Path(EVAL_PATH.format(slug=slug))
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    return rows[:n]


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to("cuda")
    n_in = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # 평가는 greedy — 재현 가능하게
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="과목당 평가 샘플 수")
    ap.add_argument("--subjects", nargs="*", default=None, help="평가할 과목 (기본: 전체)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    targets = args.subjects or list(SUBJECT_ADAPTERS)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)

    print("모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb, device_map={"": 0})
    # PeftModel로 감싸려면 어댑터가 하나는 필요하다 — 수학으로 감싸고 과목별로 추가한다.
    model = PeftModel.from_pretrained(model, MATH_ADAPTER_PATH, adapter_name="math")
    for subject in targets:
        name, path, _ = SUBJECT_ADAPTERS[subject]
        model.load_adapter(path, adapter_name=name)
    model.eval()
    print("로드 완료\n")

    rows = []
    for subject in targets:
        adapter_name, _, system_prompt = SUBJECT_ADAPTERS[subject]
        samples = load_samples(adapter_name, args.n)
        if not samples:
            print(f"[{subject}] eval 파일 없음 — 건너뜀")
            continue

        model.set_adapter(adapter_name)
        agg = {"adapter": [0.0, 0.0, 0.0], "base": [0.0, 0.0, 0.0]}
        t0 = time.time()

        for i, row in enumerate(samples, 1):
            msgs = row["messages"]
            prompt_msgs, ref = msgs[:2], msgs[2]["content"]

            pred_a = generate(model, tokenizer, prompt_msgs, args.max_new_tokens)
            with model.disable_adapter():
                pred_b = generate(model, tokenizer, prompt_msgs, args.max_new_tokens)

            for key, pred in (("adapter", pred_a), ("base", pred_b)):
                agg[key][0] += char_f1(pred, ref)
                agg[key][1] += rouge_l(pred, ref)
                agg[key][2] += exact_match(pred, ref)

            if i % 10 == 0:
                print(f"  [{subject}] {i}/{len(samples)} ({time.time() - t0:.0f}s)")

        n = len(samples)
        rows.append((subject, n,
                     [v / n for v in agg["adapter"]],
                     [v / n for v in agg["base"]]))

    print("\n" + "=" * 86)
    print(f"{'과목':<7}{'N':<5}{'char-F1':>18}{'ROUGE-L':>18}{'EM':>16}")
    print(f"{'':12}{'어댑터  베이스  Δ':>22}{'어댑터  베이스  Δ':>20}{'어댑터  베이스':>18}")
    print("=" * 86)
    for subject, n, a, b in rows:
        print(f"{subject:<7}{n:<5}"
              f"{a[0]:>7.3f}{b[0]:>8.3f}{a[0] - b[0]:>+8.3f}"
              f"{a[1]:>8.3f}{b[1]:>8.3f}{a[1] - b[1]:>+8.3f}"
              f"{a[2]:>8.3f}{b[2]:>8.3f}")
    print("=" * 86)
    if rows:
        gains = [a[0] - b[0] for _, _, a, b in rows]
        better = sum(g > 0 for g in gains)
        print(f"char-F1 기준 어댑터가 베이스보다 나은 과목: {better}/{len(rows)}"
              f"  (평균 Δ {sum(gains) / len(gains):+.3f})")


if __name__ == "__main__":
    main()

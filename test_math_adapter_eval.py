"""수학 어댑터 정량 평가.

train/math_eval.jsonl 1,755건이 학습 이후 정량 평가에 쓰이지 않았다.
기존 확인은 기본 문항 5개 스모크 테스트가 전부였다.

교과 어댑터 평가(test_subject_adapters_eval.py)와 지표를 달리한 이유:
수학 정답은 "정답: ③ 50" 처럼 짧고 형식이 정해져 있어서, 문자 겹침(char-F1)은
풀이 문장이 길수록 점수가 올라가는 엉뚱한 신호를 준다. 여기서는 정답 줄만
뽑아 정규화 후 일치를 본다.

지표
  - 정답 일치   : "정답:" 줄을 정규화(LaTeX·공백 제거)해 완전일치
  - 객관식 일치 : 보기 번호(①②③, (2), 3.)만 뽑아 비교 — 객관식 문항만 대상
  - char-F1     : 정답 줄끼리의 문자 F1 (부분점수 성격의 보조 지표)

베이스 모델(어댑터 비활성)과 나란히 재서 파인튜닝 기여분을 본다.

사용법:
    python test_math_adapter_eval.py --n 100
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src.api.adapters import BASE_MODEL_ID, MATH_ADAPTER_PATH  # noqa: E402

EVAL_FILE = "train/math_eval.jsonl"
SEED = 42

_CIRCLED = {c: str(i + 1) for i, c in enumerate("①②③④⑤⑥⑦⑧⑨")}
# 이 모델은 텍스트 전용이라 그림·표가 있어야 풀리는 문항은 구조적으로 못 맞힌다.
# math_eval.jsonl의 약 28%가 여기 해당한다. --text-only로 걸러낸다.
FIGURE_REF = re.compile(
    r"다음 그림|위 그림|아래 그림|그림과 같|그래프입니다|그래프이다|"
    r"도수분포표|다음 표|표입니다|표이다|보기와 같|나타낸"
)
# LaTeX 잔재. 정답 문자열 비교에서 표기 차이로 오답 처리되는 것을 막는다.
_LATEX_CMD = re.compile(r"\\(mathrm|text|left|right|prime|overline|mathbb|mathcal|displaystyle)\b")
_LATEX_SYM = re.compile(r"[$\\~{}]")


def extract_answer(text: str) -> str:
    """'정답:' 줄만 뽑아낸다. 없으면 첫 줄을 답으로 본다(모델이 형식을 안 지킨 경우)."""
    m = re.search(r"정답\s*[:：]\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()
    first = text.strip().split("\n", 1)[0]
    return first.strip()


def truncated_before_answer(text: str) -> bool:
    """'정답:'에 도달하지 못한 채 생성이 끝났는지.

    CoT 어댑터(math_adapter_qwen_cot)는 풀이를 먼저 쓰고 마지막에 정답을 낸다.
    max_new_tokens가 부족하면 풀이 도중에 잘려 정답이 아예 나오지 않는데,
    이때 extract_answer()는 첫 줄(풀이 앞부분)을 답으로 오인한다. 그러면 모델
    성능이 아니라 생성 길이 때문에 점수가 깎인다. 이 비율을 따로 보고해서
    그런 상황을 눈에 띄게 한다.
    """
    return not re.search(r"정답\s*[:：]", text)


def normalize(ans: str) -> str:
    s = ans
    for c, d in _CIRCLED.items():
        s = s.replace(c, d)
    s = _LATEX_CMD.sub("", s)
    s = _LATEX_SYM.sub("", s)
    s = re.sub(r"\s+", "", s)
    return s.rstrip(".,:;")


def choice_number(ans: str) -> str | None:
    """보기 번호만 추출. ①②③ / (2) / 3. / 3) 형태를 받는다. 없으면 None."""
    s = ans.strip()
    for c, d in _CIRCLED.items():
        if s.startswith(c):
            return d
    m = re.match(r"^\(?([1-9])\)?[.)]?\s", s)
    return m.group(1) if m else None


def char_f1(pred: str, ref: str) -> float:
    p, r = Counter(c for c in pred if not c.isspace()), Counter(c for c in ref if not c.isspace())
    overlap = sum((p & r).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / sum(p.values()), overlap / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def generate(model, tokenizer, messages, max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=640).to("cuda")
    n_in = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=160,
                    help="CoT 어댑터는 풀이를 먼저 쓰고 정답을 마지막에 내므로 "
                         "512 이상을 줘야 한다. 부족하면 정답 이전에 잘린다.")
    ap.add_argument("--adapter", default=MATH_ADAPTER_PATH,
                    help="평가할 수학 어댑터 경로. CoT 재학습본과 비교할 때 바꿔 넣는다.")
    ap.add_argument("--show", type=int, default=5, help="샘플 출력 개수")
    ap.add_argument("--text-only", action="store_true",
                    help="그림·표를 참조하는 문항을 제외. 텍스트 전용 모델이라 "
                         "'다음 그림에서…' 같은 문항은 애초에 풀 수 없어 점수를 깎는다.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(EVAL_FILE, encoding="utf-8") if l.strip()]
    random.Random(SEED).shuffle(rows)
    if args.text_only:
        before = len(rows)
        rows = [r for r in rows if not FIGURE_REF.search(r["messages"][1]["content"])]
        print(f"그림·표 참조 문항 제외: {before} -> {len(rows)}건")
    rows = rows[:args.n]

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    print("모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb, device_map={"": 0})
    model = PeftModel.from_pretrained(model, args.adapter, adapter_name="math")
    model.set_adapter("math")
    model.eval()
    print(f"로드 완료 (어댑터: {args.adapter}, max_new_tokens={args.max_new_tokens})\n", flush=True)

    stats = {k: {"exact": 0, "f1": 0.0, "mcq_hit": 0, "mcq_n": 0, "fmt": 0, "cut": 0}
             for k in ("adapter", "base")}
    shown = 0
    t0 = time.time()

    for i, row in enumerate(rows, 1):
        msgs = row["messages"]
        prompt_msgs, ref_full = msgs[:2], msgs[2]["content"]
        ref = extract_answer(ref_full)
        ref_norm, ref_choice = normalize(ref), choice_number(ref)

        pred_a = generate(model, tokenizer, prompt_msgs, args.max_new_tokens)
        with model.disable_adapter():
            pred_b = generate(model, tokenizer, prompt_msgs, args.max_new_tokens)

        for key, raw in (("adapter", pred_a), ("base", pred_b)):
            ans = extract_answer(raw)
            s = stats[key]
            s["fmt"] += bool(re.search(r"정답\s*[:：]", raw))
            s["cut"] += truncated_before_answer(raw)
            s["exact"] += normalize(ans) == ref_norm
            s["f1"] += char_f1(normalize(ans), ref_norm)
            if ref_choice:
                s["mcq_n"] += 1
                s["mcq_hit"] += choice_number(ans) == ref_choice

        if shown < args.show:
            shown += 1
            print(f"─── 샘플 {shown}")
            print(f"  정답     : {ref[:90]}")
            print(f"  어댑터   : {extract_answer(pred_a)[:90]}")
            print(f"  베이스   : {extract_answer(pred_b)[:90]}", flush=True)

        if i % 10 == 0:
            print(f"  {i}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)

    n = len(rows)
    print("\n" + "=" * 74)
    print(f"{'지표':<22}{'어댑터':>12}{'베이스':>12}{'Δ':>12}")
    print("=" * 74)

    def line(name, a, b, pct=True):
        fmt = (lambda v: f"{v * 100:>10.1f}%") if pct else (lambda v: f"{v:>11.3f}")
        print(f"{name:<22}{fmt(a)}{fmt(b)}{('%+.1f%%' % ((a - b) * 100)) if pct else ('%+.3f' % (a - b)):>12}")

    sa, sb = stats["adapter"], stats["base"]
    line("정답 일치", sa["exact"] / n, sb["exact"] / n)
    if sa["mcq_n"]:
        line(f"객관식 일치(n={sa['mcq_n']})", sa["mcq_hit"] / sa["mcq_n"], sb["mcq_hit"] / sb["mcq_n"])
    line("char-F1(정답줄)", sa["f1"] / n, sb["f1"] / n, pct=False)
    line("'정답:' 형식 준수", sa["fmt"] / n, sb["fmt"] / n)
    line("정답 전 잘림", sa["cut"] / n, sb["cut"] / n)
    print("=" * 74)
    print(f"평가 문항 {n}건 / max_new_tokens {args.max_new_tokens} / 소요 {time.time() - t0:.0f}s")
    if sa["cut"] / n > 0.05:
        print(f"\n!! 어댑터 출력의 {sa['cut'] / n:.1%}가 '정답:'에 도달하지 못하고 잘렸습니다.")
        print("   --max-new-tokens를 늘려 다시 재야 합니다. 지금 수치는 모델 성능이 아니라")
        print("   생성 길이 제한을 반영한 값입니다.")


if __name__ == "__main__":
    main()

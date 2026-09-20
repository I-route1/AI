"""글쓰기 채점 어댑터 정량 평가 (순서형 지표 + 클래스별 + 기준선 비교).

기존 평가(train/test_writing_adapter_qwen_weighted.py)의 문제
-------------------------------------------------------------
1. **표본이 100건뿐**이다. 평가셋에는 채점 샘플이 777건 있는데 100건만 썼다.
   그 100건 중 1점은 4건이다. "가중치도 오버샘플링도 1점을 못 맞힌다"는 결론이
   4건 위에 서 있었다. 1점 클래스에 대해서는 사실상 아무것도 측정되지 않았다.

2. **채점을 샘플링으로 생성**했다(temperature=0.4, do_sample=True).
   1~4 중 하나를 고르는 결정적 분류에 무작위성을 넣으면 같은 입력에 다른 점수가
   나온다. 여기서는 greedy로 고정한다.

3. **정확도를 주지표로 썼다**. 점수는 순서형(ordinal)이라 3점을 2점으로 틀린 것과
   4점으로 틀린 것, 1점으로 틀린 것의 무게가 다르다. 정확도는 이걸 구분하지 못한다.
   게다가 항상 3점만 찍는 모델도 정확도 56%가 나온다(평가셋의 3점 비율).
   즉 정확도 51%라는 수치만으로는 모델이 뭘 배웠는지 알 수 없다.

그래서 이 스크립트는
--------------------
- 주지표를 **QWK(quadratic weighted kappa)** 로 바꾼다. 자동 채점(AES)의 표준
  지표로, 예측이 정답에서 멀수록 더 크게 벌점을 주고, 우연 일치를 보정한다.
  **항상 같은 점수만 찍는 모델의 QWK는 정의상 0**이라 허수 성능이 걸러진다.
- **클래스별 정밀도/재현율/F1**을 낸다. 1점 문제는 전체 정확도가 아니라
  1점 재현율로 봐야 한다.
- **기준선**(항상 다수 클래스, 사전분포 무작위)을 같이 계산해 비교 기준을 준다.
- 어댑터 여러 개를 한 번에 로드해 같은 표본으로 비교한다.

참고: 학습·평가 데이터 모두 **5점이 0건**이다(train 14,223건, eval 777건 기준).
system 프롬프트는 "1점부터 5점"이라고 말하지만 실제 라벨 공간은 1~4다.
QWK 척도는 1~4로 잡고, 5점 예측이 나오면 4로 클램프한 뒤 따로 센다.

사용법:
    python test_writing_grading_eval.py                          # 전체 777건
    python test_writing_grading_eval.py --n 300
    python test_writing_grading_eval.py --adapters train/writing_adapter_qwen_weighted \\
                                                   train/writing_adapter_qwen_weighted_v2
    python test_writing_grading_eval.py --include-base           # 베이스 모델도 함께
"""
import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src.api.adapters import BASE_MODEL_ID  # noqa: E402

EVAL_FILE = "train/writing_eval_qwen.jsonl"
MAX_LENGTH = 896
SEED = 7
SCALE = (1, 2, 3, 4)  # 실제 라벨 공간. 데이터에 5점은 존재하지 않는다.

SCORE_PATTERN = re.compile(r"[1-5]")

DEFAULT_ADAPTERS = [
    "train/writing_adapter_qwen_weighted",
    "train/writing_adapter_qwen_weighted_v2",
]


# ── 지표 ──────────────────────────────────────────────────────────────────────
def confusion(y_true: list[int], y_pred: list[int]) -> list[list[int]]:
    k = len(SCALE)
    idx = {c: i for i, c in enumerate(SCALE)}
    m = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        m[idx[t]][idx[p]] += 1
    return m


def qwk(y_true: list[int], y_pred: list[int]) -> float:
    """Quadratic weighted kappa. 완전일치 1.0, 우연 수준 0.0, 우연보다 나쁘면 음수.

    상수 예측기(항상 3점 등)는 주변분포가 한 열에 몰려 기대행렬과 같아지므로 0이 된다.
    """
    k = len(SCALE)
    n = len(y_true)
    if n == 0:
        return 0.0
    O = confusion(y_true, y_pred)
    row = [sum(O[i]) for i in range(k)]
    col = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / ((k - 1) ** 2)
            num += w * O[i][j]
            den += w * row[i] * col[j] / n
    return 1.0 - num / den if den else 0.0


def per_class(y_true: list[int], y_pred: list[int]) -> dict[int, dict]:
    out = {}
    for c in SCALE:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = {"n": tp + fn, "prec": prec, "rec": rec, "f1": f1}
    return out


def summarize(name: str, y_true: list[int], y_pred: list[int], n_clamped: int = 0) -> dict:
    n = len(y_true)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    within1 = sum(1 for t, p in zip(y_true, y_pred) if abs(t - p) <= 1) / n
    mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
    return {"name": name, "qwk": qwk(y_true, y_pred), "acc": acc, "within1": within1,
            "mae": mae, "per_class": per_class(y_true, y_pred),
            "conf": confusion(y_true, y_pred), "pred_dist": Counter(y_pred),
            "clamped": n_clamped}


# ── 데이터 ────────────────────────────────────────────────────────────────────
def load_scoring_samples(n: int | None) -> list[dict]:
    rows = []
    with open(EVAL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            expected = obj["messages"][2]["content"].strip()
            if SCORE_PATTERN.fullmatch(expected):
                rows.append(obj)
    random.Random(SEED).shuffle(rows)
    return rows[:n] if n else rows


def predict(model, tokenizer, messages) -> int | None:
    prompt = tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to("cuda")
    n_in = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,  # 채점은 결정적이어야 한다. 기존 평가는 샘플링이었다.
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
    m = SCORE_PATTERN.search(text)
    return int(m.group(0)) if m else None


def score_logits(model, tokenizer, messages, digit_ids: dict[int, int]) -> list[float]:
    """점수 토큰 1~4의 로짓을 직접 읽는다.

    generate()로 한 글자를 뽑는 대신 순전파 한 번으로 분포 전체를 가져온다.
    이러면 표본당 한 번만 계산하고도 디코딩 전략(argmax / logit adjustment / 기댓값)을
    여러 개 오프라인으로 비교할 수 있다.
    """
    prompt = tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to("cuda")
    with torch.no_grad():
        out = model(**inputs)
    last = out.logits[0, -1, :].float()
    return [last[digit_ids[c]].item() for c in SCALE]


def decode_adjusted(logit_vec: list[float], log_prior: list[float], tau: float) -> int:
    """logit adjustment: 로짓에서 사전분포를 tau만큼 빼 다수 클래스 편향을 걷어낸다.

    tau=0이면 그냥 argmax(=greedy 생성과 같은 결과), tau=1이면 사전분포를 완전히
    상쇄해 균형 사전분포 하의 예측이 된다. 재학습이 필요 없는 롱테일 보정 기법이다.
    """
    adj = [lg - tau * lp for lg, lp in zip(logit_vec, log_prior)]
    return SCALE[max(range(len(SCALE)), key=lambda i: adj[i])]


def decode_expected(logit_vec: list[float]) -> int:
    """점수를 확률 가중 기댓값으로 보고 반올림. 순서형 과제에 맞는 디코딩."""
    m = max(logit_vec)
    exp = [math.exp(v - m) for v in logit_vec]
    z = sum(exp)
    probs = [e / z for e in exp]
    ev = sum(c * p for c, p in zip(SCALE, probs))
    return min(SCALE[-1], max(SCALE[0], round(ev)))


def train_prior() -> list[float]:
    """학습 데이터의 점수 분포. logit adjustment의 기준이 된다."""
    counts = Counter()
    with open("train/writing_train_qwen.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a = json.loads(line)["messages"][2]["content"].strip()
            if SCORE_PATTERN.fullmatch(a):
                counts[int(a)] += 1
    total = sum(counts.values())
    return [counts[c] / total for c in SCALE]


def baseline_analysis(rows: list[dict], y_true: list[int]) -> None:
    """서빙의 규칙 기반 점수가 얼마나 신호를 담고 있는지 가늠한다 (GPU 불필요).

    `/api/writing/evaluate`의 final_score는 키워드 일치율 70% + 길이 비율 30%다.
    이 평가셋에는 키워드도 모범답안도 없어서 키워드 항목은 잴 수 없지만,
    **길이 항목은 잴 수 있다.** 길이가 답안 품질을 얼마나 설명하는지 보면
    규칙 기반 접근의 상한을 짐작할 수 있다.
    """
    import statistics as stat

    def student_answer(msgs) -> str:
        c = msgs[1]["content"]
        i = c.find("[학생 답안]:")
        return c[i + 8:].strip() if i >= 0 else ""

    lens = [len(student_answer(r["messages"])) for r in rows]
    n = len(lens)

    mx, my = stat.mean(lens), stat.mean(y_true)
    sx, sy = stat.pstdev(lens), stat.pstdev(y_true)
    cor = (sum((a - mx) * (b - my) for a, b in zip(lens, y_true)) / n / (sx * sy)) if sx and sy else 0.0

    print("\n" + "=" * 88)
    print("규칙 기반 점수의 상한 가늠 — 길이 항목이 품질을 설명하는가")
    print("=" * 88)
    print(f"답안 길이 vs 실제 점수 피어슨 상관: {cor:.3f}")
    for s in SCALE:
        sub = [l for l, g in zip(lens, y_true) if g == s]
        if sub:
            print(f"  {s}점(n={len(sub):3}) 답안 길이 중앙값 {int(stat.median(sub)):5}자")

    q = stat.quantiles(lens, n=4)
    pred = [1 if L < q[0] else 2 if L < q[1] else 3 if L < q[2] else 4 for L in lens]
    r = summarize("길이 4분위", y_true, pred)
    maj = Counter(y_true).most_common(1)[0][0]
    b = summarize(f"항상 {maj}점", y_true, [maj] * n)
    print(f"\n{'예측 방법':<28}{'QWK':>8}{'정확도':>10}{'평균오차':>10}")
    print("-" * 60)
    print(f"{'길이 4분위로만 예측':<28}{r['qwk']:>8.3f}{r['acc']:>9.1%}{r['mae']:>10.3f}")
    print(f"{f'항상 {maj}점 (상수)':<28}{b['qwk']:>8.3f}{b['acc']:>9.1%}{b['mae']:>10.3f}")
    print("-" * 60)
    print("비교: 파인튜닝 채점기(weighted + 기댓값) QWK 0.553 / 58.7% / 0.423")
    print("\n키워드 항목(가중치 70%)은 이 평가셋에 키워드가 없어 재지 못했다.")
    print("=" * 88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="평가 건수(기본: 전체 777건)")
    ap.add_argument("--adapters", nargs="*", default=DEFAULT_ADAPTERS)
    ap.add_argument("--include-base", action="store_true", help="베이스 모델(어댑터 비활성)도 평가")
    ap.add_argument("--baseline-analysis", action="store_true",
                    help="서빙의 규칙 기반 점수(키워드 70%% + 길이 30%%)가 얼마나 신호를 "
                         "담고 있는지 길이 항목으로 가늠한다. 모델을 안 띄우므로 GPU가 필요 없다.")
    ap.add_argument("--logit-sweep", action="store_true",
                    help="점수 토큰 로짓을 직접 읽어 디코딩 전략(argmax / logit adjustment / "
                         "기댓값)을 비교한다. 재학습 없이 1점 재현율을 올릴 수 있는지 보는 용도.")
    args = ap.parse_args()

    rows = load_scoring_samples(args.n)
    y_true = [int(r["messages"][2]["content"].strip()) for r in rows]
    dist = Counter(y_true)
    n = len(rows)

    print(f"평가 표본: {n}건")
    print("정답 분포: " + "  ".join(f"{c}점 {dist[c]}건({dist[c]/n:.1%})" for c in SCALE))

    if args.baseline_analysis:
        baseline_analysis(rows, y_true)
        return

    # ── 기준선 (모델 없이 계산 가능) ──────────────────────────────────────────
    majority = dist.most_common(1)[0][0]
    baselines = [
        summarize(f"[기준선] 항상 {majority}점", y_true, [majority] * n),
    ]
    rng = random.Random(SEED)
    prior_pred = rng.choices(SCALE, weights=[dist[c] for c in SCALE], k=n)
    baselines.append(summarize("[기준선] 사전분포 무작위", y_true, prior_pred))

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    print("\n모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb,
                                                 device_map={"": 0})
    names = []
    for i, path in enumerate(args.adapters):
        name = f"a{i}"
        if i == 0:
            model = PeftModel.from_pretrained(model, path, adapter_name=name)
        else:
            model.load_adapter(path, adapter_name=name)
        names.append((name, path))
    model.eval()
    print("로드 완료\n", flush=True)

    results = []
    t0 = time.time()

    # ── 디코딩 전략 비교 (재학습 없이 1점 재현율을 올릴 수 있는지) ──────────────
    if args.logit_sweep:
        digit_ids = {}
        for c in SCALE:
            ids = tokenizer.encode(str(c), add_special_tokens=False)
            if len(ids) != 1:
                raise SystemExit(f"점수 '{c}'가 단일 토큰이 아닙니다: {ids}")
            digit_ids[c] = ids[0]

        prior = train_prior()
        log_prior = [math.log(p) for p in prior]
        print("학습 사전분포: " + "  ".join(f"{c}점 {p:.4f}" for c, p in zip(SCALE, prior)))

        for adapter_name, path in names:
            model.set_adapter(adapter_name)
            vecs = []
            for i, r in enumerate(rows, 1):
                vecs.append(score_logits(model, tokenizer, r["messages"], digit_ids))
                if i % 100 == 0:
                    print(f"  [{path}] 로짓 {i}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)

            print(f"\n{'=' * 88}")
            print(f"디코딩 전략 비교 — {path}")
            print(f"{'=' * 88}")
            print(f"{'전략':<28}{'QWK':>8}{'정확도':>10}{'평균오차':>10}"
                  f"{'1점재현':>10}{'2점재현':>10}{'1점예측수':>11}")
            print("-" * 88)
            for tau in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
                preds = [decode_adjusted(v, log_prior, tau) for v in vecs]
                s = summarize(f"tau={tau}", y_true, preds)
                label = "argmax (=현재 서빙)" if tau == 0.0 else f"logit adj tau={tau}"
                print(f"{label:<28}{s['qwk']:>8.3f}{s['acc']:>9.1%}{s['mae']:>10.3f}"
                      f"{s['per_class'][1]['rec']:>10.3f}{s['per_class'][2]['rec']:>10.3f}"
                      f"{s['pred_dist'][1]:>11}")
            preds = [decode_expected(v) for v in vecs]
            s = summarize("expected", y_true, preds)
            print(f"{'확률 기댓값 반올림':<28}{s['qwk']:>8.3f}{s['acc']:>9.1%}{s['mae']:>10.3f}"
                  f"{s['per_class'][1]['rec']:>10.3f}{s['per_class'][2]['rec']:>10.3f}"
                  f"{s['pred_dist'][1]:>11}")
            print("-" * 88)
            for r in baselines:
                print(f"{r['name']:<28}{r['qwk']:>8.3f}{r['acc']:>9.1%}{r['mae']:>10.3f}"
                      f"{r['per_class'][1]['rec']:>10.3f}{r['per_class'][2]['rec']:>10.3f}"
                      f"{r['pred_dist'][1]:>11}")
            print("=" * 88)
        print(f"\n표본 {n}건 / 소요 {time.time() - t0:.0f}s")
        return

    conditions = [(name, path, False) for name, path in names]
    if args.include_base:
        conditions.append((None, "베이스(어댑터 비활성)", True))

    for adapter_name, label, is_base in conditions:
        preds, n_fail, n_clamp = [], 0, 0
        if not is_base:
            model.set_adapter(adapter_name)
        for i, r in enumerate(rows, 1):
            if is_base:
                with model.disable_adapter():
                    p = predict(model, tokenizer, r["messages"])
            else:
                p = predict(model, tokenizer, r["messages"])
            if p is None:
                n_fail += 1
                p = majority  # 형식 이탈은 다수 클래스로 대체하고 따로 센다
            if p > SCALE[-1]:
                n_clamp += 1
                p = SCALE[-1]
            preds.append(p)
            if i % 50 == 0:
                print(f"  [{label}] {i}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
        res = summarize(label, y_true, preds, n_clamp)
        res["fail"] = n_fail
        results.append(res)

    # ── 출력 ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("종합 (QWK가 주지표 — 순서형 점수에 맞고, 상수 예측기는 정의상 0)")
    print("=" * 88)
    print(f"{'조건':<42}{'QWK':>8}{'정확도':>10}{'±1이내':>10}{'평균오차':>10}")
    print("-" * 88)
    for r in baselines + results:
        print(f"{r['name']:<42}{r['qwk']:>8.3f}{r['acc']:>9.1%}{r['within1']:>10.1%}{r['mae']:>10.3f}")
    print("=" * 88)

    for r in results:
        print(f"\n── {r['name']}")
        if r.get("fail"):
            print(f"   숫자 파싱 실패 {r['fail']}건 (다수 클래스로 대체)")
        if r["clamped"]:
            print(f"   5점 예측 {r['clamped']}건 (4점으로 클램프)")
        print(f"   {'점수':<6}{'정답수':>8}{'정밀도':>10}{'재현율':>10}{'F1':>10}")
        for c in SCALE:
            pc = r["per_class"][c]
            print(f"   {c:<6}{pc['n']:>8}{pc['prec']:>10.3f}{pc['rec']:>10.3f}{pc['f1']:>10.3f}")
        print(f"   예측 분포: " + "  ".join(f"{c}점 {r['pred_dist'][c]}" for c in SCALE))
        print("   혼동 행렬 (행=정답, 열=예측):")
        print("        " + "".join(f"{c:>7}" for c in SCALE))
        for i, c in enumerate(SCALE):
            print(f"     {c:<3}" + "".join(f"{v:>7}" for v in r["conf"][i]))

    print(f"\n표본 {n}건 / greedy 디코딩 / 소요 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

"""개념 설명(_concept_explain) 품질 정량 평가.

왜 필요한가
-----------
서빙에서 과목 어댑터가 실제로 하는 일은 문제 풀이가 아니라 main.py의
_concept_explain() — 개념 설명 생성이다. 그런데 지금까지 정량 지표가 이것 하나뿐
없었다. test_subject_adapters_eval.py는 '지문+성취기준+질문 → 교과서 답안' 과제를
재고, test_math_adapter_eval.py는 문제 풀이를 잰다. 둘 다 개념 설명이 아니다.

평가셋을 어떻게 만드나
----------------------
"정답이 하나로 정해지지 않는다"가 이 과제의 어려운 점이다. 자유 서술이라
정답 문자열과 비교하는 방식이 성립하지 않는다.

그래서 rag.py의 _CONCEPT_MAP을 정답지로 쓴다. 149개 항목은 사람이 직접 쓴
과목별 핵심 개념 노트이고, 서빙에서도 이미 신뢰해 RAG 최상위 레이어로 쓰고
있다. 별도 평가셋을 만들 필요 없이 그대로 참조 답안이 된다.

단, 문자 겹침(char-F1)만 보면 안 된다. 한국어 설명문은 조사·어미가 겹쳐서
아무 말이나 해도 점수가 어느 정도 나온다. 그래서 **변별력**을 같이 잰다:
같은 생성문을 (a) 올바른 개념의 참조 (b) 무작위로 고른 다른 개념의 참조와
각각 비교해, 그 차이를 본다. 차이가 작으면 모델이 '그럴듯한 교육용 문장'을
쓰고 있을 뿐 해당 개념을 설명하지 못하는 것이다. 이 지표가 이 평가의 핵심이다.

지표
  - 커버리지   : 참조의 변별 용어(전체 항목 중 드물게 나오는 용어) 중 생성문에
                 등장한 비율. 조사·상투어는 문서빈도가 높아 자동으로 걸러진다.
  - 변별력 Δ   : 커버리지(정답 참조) - 커버리지(무작위 참조). 이 평가의 주지표.
  - char-F1    : 참조와의 문자 F1 (보조 지표)
  - 한국어 비율: 생성문 중 한글 문자 비율. 낮으면 중국어·영어로 새는 것.
  - 반복률     : 3-gram 중복 비율. 높으면 같은 말을 맴도는 것.
  - 형식 오염  : '정답:' 패턴 출현율. 수학 어댑터는 문제 풀이 형식만 학습해서
                 개념 설명을 시켜도 답안 형식으로 새는지 확인한다.
  - 에코       : 질문 문구를 그대로 되뱉는 비율.

어댑터와 베이스(어댑터 비활성)를 나란히 재서 파인튜닝 기여분을 본다.

사용법:
    python test_concept_explain_eval.py                      # 과목당 15개 개념
    python test_concept_explain_eval.py --n 30
    python test_concept_explain_eval.py --subjects 수학 과학
    python test_concept_explain_eval.py --math-adapter train/math_adapter_qwen_cot
    python test_concept_explain_eval.py --greedy             # 재현 가능한 결정적 디코딩
"""
import argparse
import random
import re
import sys
import time
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src.api.adapters import (  # noqa: E402
    BASE_MODEL_ID, MATH_ADAPTER_PATH, MATH_SYSTEM_PROMPT, SUBJECT_ADAPTERS,
)
# rag가 아니라 concept_map에서 직접 가져온다. rag는 import만 해도 FAISS 인덱스
# (69만 벡터)와 sentence-transformers를 올리는데, 그 상태로 8B 4bit 모델까지
# 로드하면 프로세스가 죽는다(exit 139). 여기서 필요한 건 딕셔너리 하나뿐이다.
from src.api.concept_map import CONCEPT_MAP as _CONCEPT_MAP  # noqa: E402

SEED = 42

# 서빙(_concept_explain)과 동일한 생성 설정. 여기가 어긋나면 서빙 품질을 재는 게 아니다.
SERVING_GEN = dict(max_new_tokens=200, temperature=0.3, do_sample=True, repetition_penalty=1.2)

_TERM = re.compile(r"[가-힣]{2,}|[A-Za-z][A-Za-z0-9]*|[0-9]+")
_HANGUL = re.compile(r"[가-힣]")
_ANSWER_FMT = re.compile(r"정답\s*[:：]")
# 변별 용어로 쓸 최대 문서빈도. 전체 항목의 8%를 넘게 나오는 말(조사·'문제'·'학습' 등)은
# 개념을 특정하지 못하므로 제외한다.
MAX_DF_RATIO = 0.08


def terms(text: str) -> set[str]:
    return set(_TERM.findall(text))


def build_df(entries: list[str]) -> Counter:
    """ConceptMap 전체에서 각 용어의 문서빈도."""
    df = Counter()
    for e in entries:
        df.update(terms(e))
    return df


def distinctive(ref: str, df: Counter, n_docs: int) -> set[str]:
    """참조에서 '이 개념을 특정하는' 용어만 남긴다."""
    cut = max(1, int(n_docs * MAX_DF_RATIO))
    return {t for t in terms(ref) if df[t] <= cut}


def coverage(pred: str, key_terms: set[str]) -> float:
    if not key_terms:
        return 0.0
    return sum(1 for t in key_terms if t in pred) / len(key_terms)


def char_f1(pred: str, ref: str) -> float:
    p = Counter(c for c in pred if not c.isspace())
    r = Counter(c for c in ref if not c.isspace())
    overlap = sum((p & r).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / sum(p.values()), overlap / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def hangul_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return len(_HANGUL.findall(text)) / len(letters)


def repetition(text: str, n: int = 3) -> float:
    """문자 n-gram 중복 비율. 0이면 반복 없음, 1에 가까우면 같은 말 반복."""
    s = "".join(text.split())
    if len(s) <= n:
        return 0.0
    grams = [s[i:i + n] for i in range(len(s) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def build_eval_set(subjects: list[str], n: int) -> dict[str, list[tuple[str, str]]]:
    """과목별 [(개념어, 참조설명)]. 개념어는 ConceptMap 키의 첫 키워드를 쓴다."""
    out: dict[str, list[tuple[str, str]]] = {}
    for sub in subjects:
        items = [(keys[1], text) for keys, text in _CONCEPT_MAP.items()
                 if keys[0] == sub and len(keys) > 1]
        random.Random(SEED).shuffle(items)
        out[sub] = items[:n]
    return out


def generate(model, tokenizer, subject_prompt: str, concept: str, greedy: bool) -> str:
    """main.py의 _concept_explain()과 동일한 프롬프트·생성 설정."""
    messages = [
        {"role": "system", "content": subject_prompt},
        {"role": "user", "content": f"질문: '{concept}' 개념의 핵심 포인트를 학생에게 설명해주세요."},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    n_in = inputs["input_ids"].shape[-1]
    gen = dict(SERVING_GEN)
    if greedy:
        gen = dict(max_new_tokens=SERVING_GEN["max_new_tokens"],
                   do_sample=False, repetition_penalty=SERVING_GEN["repetition_penalty"])
    with torch.no_grad():
        out = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, **gen)
    return tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="과목당 평가할 개념 수")
    ap.add_argument("--subjects", nargs="*", default=["수학", "국어", "영어", "과학", "사회"])
    ap.add_argument("--math-adapter", default=MATH_ADAPTER_PATH,
                    help="수학 어댑터 경로. CoT 재학습본과 비교할 때 바꿔 넣는다.")
    ap.add_argument("--greedy", action="store_true",
                    help="결정적 디코딩. 기본은 서빙과 같은 샘플링(seed 고정).")
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    # 한국사는 어댑터가 없어 _concept_explain이 None을 반환하고 Ollama로 폴백한다.
    # 어댑터 vs 베이스 비교 대상이 아니므로 기본 과목 목록에서 뺀다.
    if "한국사" in args.subjects:
        print("한국사는 어댑터가 없어(Ollama 폴백) 이 평가 대상이 아닙니다. 제외합니다.\n")
        args.subjects = [s for s in args.subjects if s != "한국사"]

    all_entries = [v for v in _CONCEPT_MAP.values()]
    df = build_df(all_entries)
    n_docs = len(all_entries)
    eval_set = build_eval_set(args.subjects, args.n)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    print("모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb,
                                                 device_map={"": 0})
    model = PeftModel.from_pretrained(model, args.math_adapter, adapter_name="math")
    for sub, (name, path, _) in SUBJECT_ADAPTERS.items():
        if sub in args.subjects:
            model.load_adapter(path, adapter_name=name)
    model.eval()
    print(f"로드 완료 (수학 어댑터: {args.math_adapter})\n", flush=True)

    rows = []
    shown = 0
    t0 = time.time()

    for subject, items in eval_set.items():
        if subject == "수학":
            adapter_name, system_prompt = "math", MATH_SYSTEM_PROMPT
        else:
            adapter_name, _, system_prompt = SUBJECT_ADAPTERS[subject]

        # 변별력 대조군: 같은 과목의 다른 개념 참조 (과목이 달라 쉬운 대조가 되지 않도록)
        pool = [t for _, t in items]
        if len(pool) < 2:
            print(f"  [{subject}] 항목이 {len(pool)}개뿐이라 대조군을 만들 수 없습니다. 건너뜁니다.")
            continue

        for idx, (concept, ref) in enumerate(items):
            key_terms = distinctive(ref, df, n_docs)
            wrong_ref = pool[(idx + 1) % len(pool)]
            wrong_terms = distinctive(wrong_ref, df, n_docs)

            torch.manual_seed(SEED + idx)
            model.set_adapter(adapter_name)
            pred_a = generate(model, tokenizer, system_prompt, concept, args.greedy)

            torch.manual_seed(SEED + idx)
            with model.disable_adapter():
                pred_b = generate(model, tokenizer, system_prompt, concept, args.greedy)

            for kind, pred in (("adapter", pred_a), ("base", pred_b)):
                rows.append({
                    "subject": subject, "kind": kind, "concept": concept,
                    "cov": coverage(pred, key_terms),
                    "cov_wrong": coverage(pred, wrong_terms),
                    "f1": char_f1(pred, ref),
                    "ko": hangul_ratio(pred),
                    "rep": repetition(pred),
                    "fmt": float(bool(_ANSWER_FMT.search(pred))),
                    "echo": float("핵심 포인트를 학생에게" in pred),
                    "empty": float(not pred.strip()),
                })

            if shown < args.show:
                shown += 1
                print(f"─── [{subject}] {concept}")
                print(f"  어댑터: {pred_a[:150]!r}")
                print(f"  베이스: {pred_b[:150]!r}", flush=True)

        print(f"  [{subject}] {len(items)}개 완료 ({time.time() - t0:.0f}s)", flush=True)

    # ── 집계 ──────────────────────────────────────────────────────────────────
    def agg(subject: str | None, kind: str, field: str) -> float:
        vals = [r[field] for r in rows
                if r["kind"] == kind and (subject is None or r["subject"] == subject)]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n" + "=" * 86)
    print("개념 설명 품질 — 과목별 (변별력 Δ = 정답참조 커버리지 - 무작위참조 커버리지)")
    print("=" * 86)
    print(f"{'과목':<8}{'커버리지(A/B)':>20}{'무작위(A)':>12}{'변별력Δ(A/B)':>20}{'char-F1(A/B)':>18}")
    print("-" * 86)
    for subject in eval_set:
        ca, cb = agg(subject, "adapter", "cov"), agg(subject, "base", "cov")
        wa, wb = agg(subject, "adapter", "cov_wrong"), agg(subject, "base", "cov_wrong")
        fa, fb = agg(subject, "adapter", "f1"), agg(subject, "base", "f1")
        print(f"{subject:<8}{ca:>9.3f} /{cb:>7.3f}{wa:>12.3f}"
              f"{ca - wa:>10.3f} /{cb - wb:>7.3f}{fa:>10.3f} /{fb:>7.3f}")
    print("-" * 86)
    ca, cb = agg(None, "adapter", "cov"), agg(None, "base", "cov")
    wa, wb = agg(None, "adapter", "cov_wrong"), agg(None, "base", "cov_wrong")
    fa, fb = agg(None, "adapter", "f1"), agg(None, "base", "f1")
    print(f"{'전체':<8}{ca:>9.3f} /{cb:>7.3f}{wa:>12.3f}"
          f"{ca - wa:>10.3f} /{cb - wb:>7.3f}{fa:>10.3f} /{fb:>7.3f}")

    print("\n" + "=" * 86)
    print("생성 건전성 (어댑터 / 베이스)")
    print("=" * 86)
    for label, field, good in (("한국어 비율", "ko", "높을수록 좋음"),
                               ("반복률", "rep", "낮을수록 좋음"),
                               ("'정답:' 형식 오염", "fmt", "낮을수록 좋음"),
                               ("질문 에코", "echo", "낮을수록 좋음"),
                               ("빈 출력", "empty", "낮을수록 좋음")):
        a, b = agg(None, "adapter", field), agg(None, "base", field)
        print(f"{label:<20}{a:>8.3f} /{b:>8.3f}   ({good})")

    n_concepts = sum(len(v) for v in eval_set.values())
    print("=" * 86)
    print(f"개념 {n_concepts}개 × 2조건 / 디코딩 {'greedy' if args.greedy else 'sampling(seed 고정)'} "
          f"/ 소요 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

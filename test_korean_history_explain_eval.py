"""한국사 개념 설명: 지금 서빙 중인 Ollama 대 베이스 Qwen.

한국사는 어댑터가 없어 _concept_explain()이 None을 돌려주고 Ollama(llama3.1)로
넘어갔다. 다른 과목은 전부 베이스 Qwen으로 개념 설명을 만드므로, 한국사도 그렇게
바꿀 수 있는지 잰 스크립트다(결과는 MODELS.md "한국사 개념 설명" 절).

주의: 개념F1은 용어 겹침만 잰다. 이 지표로는 Qwen이 앞섰지만, 대표 개념 8개의
사실 정확도를 직접 확인하니 Ollama가 더 나았다(3.5/8 대 1.5/8). 전환하지 않았다. 채점은 test_concept_explain_eval.py와 같다(ConceptMap 참조,
개념F1, 짝지은 부트스트랩).

조건 (모두 서빙과 같은 프롬프트·설정)
  qwen       : 베이스 Qwen, _concept_explain() 프롬프트, max_new_tokens=200
  ollama_rec : Ollama, /api/ai/report/subject-recommend 폴백 프롬프트 (main.py)
  ollama_rep : Ollama, /api/ai/report/{subject} 폴백 프롬프트 (counseling.py)
Ollama는 num_predict=150, 타임아웃 20초. 타임아웃이면 서빙에서 섹션이 빠지므로
빈 출력으로 세고 따로 보고한다.

사용법:
    python test_korean_history_explain_eval.py            # 42개 전부, greedy
"""
import argparse
import sys
import time

import httpx
import torch

sys.stdout.reconfigure(encoding="utf-8")

from test_concept_explain_eval import (  # noqa: E402
    BASE_MODEL_ID, _CONCEPT_MAP, build_df, build_eval_set, char_f1, coverage, density,
    distinctive, f1, generate, hangul_ratio, paired_bootstrap, repetition,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

SUBJECT = "한국사"
# 다른 과목 system 프롬프트(src/api/adapters.py)와 같은 틀.
QWEN_SYSTEM_PROMPT = ("당신은 한국사 전문 교사입니다. 학생의 한국사 지문과 질문에 대해 "
                      "정확하고 이해하기 쉬운 답변을 제공하세요.")
OLLAMA_TIMEOUT = 20.0


def ollama_rec_prompt(concept: str) -> str:
    return (f"{SUBJECT} 과목에서 '{concept}' 개념을 어려워하는 학생에게 "
            f"이 개념의 핵심 포인트와 효과적인 학습 방법을 2~3문장으로 한국어로 답해주세요.")


def ollama_rep_prompt(concept: str) -> str:
    return (f"한국 중고등학생이 {SUBJECT} '{concept}' 개념을 어려워합니다. "
            f"이 개념에서 학생들이 가장 자주 하는 핵심 실수 1가지와 "
            f"그것을 극복하는 구체적인 학습 전략을 2~3문장으로 간결하게 한국어로 답해주세요.")


def ollama(prompt: str, greedy: bool) -> tuple[str, float]:
    """(응답, 소요초). 서빙처럼 타임아웃·실패면 빈 문자열."""
    opts = {"num_predict": 150, "temperature": 0.0 if greedy else 0.4, "seed": 42}
    t = time.time()
    try:
        r = httpx.post("http://localhost:11434/api/generate",
                       json={"model": "llama3.1:latest", "prompt": prompt,
                             "stream": False, "options": opts},
                       timeout=OLLAMA_TIMEOUT)
        text = r.json().get("response", "").strip() if r.status_code == 200 else ""
    except httpx.TimeoutException:
        text = ""
    return text, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="평가할 개념 수 (한국사는 42개)")
    ap.add_argument("--sampling", action="store_true", help="서빙과 같은 샘플링. 기본은 greedy")
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()
    greedy = not args.sampling

    entries = list(_CONCEPT_MAP.values())
    df, n_docs = build_df(entries), len(entries)
    items = build_eval_set([SUBJECT], args.n)[SUBJECT]
    pool = [t for _, t in items]

    # 첫 호출의 모델 적재 시간이 타임아웃 통계를 오염시키지 않게 미리 올린다.
    ollama("안녕", greedy=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    print("모델 로드 중...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb,
                                                 device_map={"": 0})
    model.eval()

    kinds = ("qwen", "ollama_rec", "ollama_rep")
    rows: list[dict] = []
    t0 = time.time()
    for idx, (concept, ref) in enumerate(items):
        key_terms = distinctive(ref, df, n_docs)
        wrong_terms = distinctive(pool[(idx + 1) % len(pool)], df, n_docs)

        torch.manual_seed(42 + idx)
        t = time.time()
        preds = {"qwen": (generate(model, tokenizer, QWEN_SYSTEM_PROMPT, concept, greedy), time.time() - t)}
        preds["ollama_rec"] = ollama(ollama_rec_prompt(concept), greedy)
        preds["ollama_rep"] = ollama(ollama_rep_prompt(concept), greedy)

        for kind, (pred, sec) in preds.items():
            cov, den = coverage(pred, key_terms), density(pred, key_terms, df, n_docs)
            rows.append({"kind": kind, "concept": concept, "cov": cov, "den": den,
                         "cf1": f1(cov, den), "disc": cov - coverage(pred, wrong_terms),
                         "f1": char_f1(pred, ref), "ko": hangul_ratio(pred),
                         "rep": repetition(pred), "empty": float(not pred),
                         "len": float(len(pred)), "sec": sec})
        if idx < args.show:
            print(f"─── {concept}")
            for kind in kinds:
                print(f"  {kind:<10}: {preds[kind][0][:160]!r}")
    print(f"{len(items)}개 완료 ({time.time() - t0:.0f}s)\n")

    def agg(kind: str, field: str) -> float:
        vals = [r[field] for r in rows if r["kind"] == kind]
        return sum(vals) / len(vals) if vals else 0.0

    print("=" * 78)
    print(f"{'조건':<12}{'커버리지':>9}{'밀도':>8}{'개념F1':>8}{'변별력Δ':>9}{'char-F1':>9}"
          f"{'한국어':>8}{'반복률':>8}{'길이':>7}")
    print("-" * 78)
    for k in kinds:
        print(f"{k:<12}{agg(k, 'cov'):>9.3f}{agg(k, 'den'):>8.3f}{agg(k, 'cf1'):>8.3f}"
              f"{agg(k, 'disc'):>9.3f}{agg(k, 'f1'):>9.3f}{agg(k, 'ko'):>8.3f}"
              f"{agg(k, 'rep'):>8.3f}{agg(k, 'len'):>7.0f}")

    print("\n개념F1 짝지은 비교 (qwen - ollama, 부트스트랩)")
    by = {(r["kind"], r["concept"]): r["cf1"] for r in rows}
    for other in ("ollama_rec", "ollama_rep"):
        diffs = [by[("qwen", c)] - by[(other, c)] for c, _ in items]
        m, lo, hi, frac = paired_bootstrap(diffs)
        verdict = "qwen 우세" if lo > 0 else ("ollama 우세" if hi < 0 else "구분 안 됨")
        print(f"  qwen - {other:<10} n={len(diffs)} {m:+.3f} [{lo:+.3f}, {hi:+.3f}] "
              f"qwen 우세 {frac:.1%}  {verdict}")

    print("\n서빙 영향")
    for k in kinds:
        secs = sorted(r["sec"] for r in rows if r["kind"] == k)
        print(f"  {k:<10} 빈 출력(타임아웃 포함) {agg(k, 'empty'):.1%}  "
              f"소요 중앙값 {secs[len(secs) // 2]:.1f}s / 최대 {secs[-1]:.1f}s")
    print(f"\n디코딩: {'greedy' if greedy else 'sampling'}")


if __name__ == "__main__":
    main()

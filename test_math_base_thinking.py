"""베이스 Qwen3-8B의 수학 객관식 정확도 — 추론 모드(thinking) vs 끔.

왜 필요한가
-----------
수학 어댑터는 짧은 풀이 데이터로 미세조정하고 enable_thinking=False로만 쓴다.
베이스 Qwen3-8B에는 자체 추론 모드(enable_thinking=True)가 있는데, 그 정확도를 한 번도
재본 적이 없다. 이 값이 미세조정 어댑터를 크게 넘으면 새 학습 데이터를 붙이기 전에
접근 자체를 다시 봐야 한다.

공정하게 재기 위해
------------------
- 같은 문항: test_math_adapter_eval.py와 같은 셔플(seed 42)의 첫 N문항 중 객관식.
  덤프의 `i`가 같아서 dump_orig.jsonl / dump_cot.jsonl과 짝지어 McNemar가 된다.
- 채점은 **값 일치**(math_value_match.py). 71859의 문제 본문에는 보기가 없다(보기는
  이미지에만 있다). 객관식 204문항 전부가 그렇다. 정답에는 '③ 40°'처럼 번호가 붙어 있어서
  모델은 풀어도 번호를 맞힐 방법이 없다. 번호 정확도는 실력이 아니라 순서 추측이다.
  어댑터 덤프도 같은 함수로 다시 채점해서 비교한다.
- 형식 분리: 베이스는 '정답:' 형식을 안 지켜서 예전에 0%가 나왔다. 프롬프트로 답 형식을
  지정하고, 추론 모드를 끈 조건도 같이 재서 '추론 모드의 효과'와 '형식 지정의 효과'를 가른다.
- 예산 강제 종료(budget forcing): 추론 모드는 출력이 길다. 한도에 닿아 생각이 안 끝난 문항을
  그냥 오답으로 세면 '추론이 약하다'가 아니라 '예산이 부족하다'를 재게 된다. 그래서 한도에서
  '</think>'를 붙이고 그때까지의 생각으로 답을 내게 한다. 이 경우는 `forced`로 따로 센다.

사용법
------
    python test_math_base_thinking.py --mode think --limit 8                 # 파일럿
    python test_math_base_thinking.py --mode think --dump train/dump_think.jsonl
    python test_math_base_thinking.py --mode nothink --max-new-tokens 1024 --dump train/dump_nothink.jsonl
"""
import argparse
import json
import random
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from math_value_match import value_match  # noqa: E402
from src.api.adapters import BASE_MODEL_ID  # noqa: E402
from test_math_adapter_eval import choice_number, extract_answer, EVAL_FILE, SEED  # noqa: E402

_CIRCLED = "①②③④⑤⑥⑦⑧⑨"

SYSTEM = "당신은 수학 전문 교사입니다. 학생의 수학 문제를 정확하게 풀어 주세요."
FORMAT_HINT = (
    "\n\n[답안 형식] 이 문제의 보기는 주어지지 않습니다. 보기 번호를 추측하지 말고, "
    "풀이를 마친 뒤 마지막 줄에 '정답: ' 다음에 구한 값만 간결하게 적으세요. 예: 정답: 40"
)
# 예산이 바닥났을 때 생각을 끝내게 하는 문구 (s1 논문의 budget forcing과 같은 방식)
FORCE_SUFFIX = "\n\nI have used up my thinking budget, so I will give my best final answer now.\n</think>\n\n정답: "

_THINK = re.compile(r"<think>.*?</think>", re.S)
# 특수 토큰(<|im_end|>, <|endoftext|>, <|vision_pad|> 등)을 이름으로 하나씩 지우면 빠뜨린다.
# Qwen3의 패딩 토큰은 <|vision_pad|>인데, 처음엔 im_end와 endoftext만 지워서 답이
# '1.44<|vision_pad|>...'가 되어 맞는 답이 오답으로 채점됐다(파일럿 1/8 -> 실제 5/8).
# <think>/</think>는 <|...|> 형식이 아니라 이 패턴에 걸리지 않는다.
_SPECIAL = re.compile(r"<\|[^|>]+\|>")


def split_think(raw: str) -> tuple[str, bool]:
    """(사고 블록을 뺀 본문, 사고 블록이 닫혔는가)."""
    if "<think>" not in raw and "</think>" not in raw:
        return raw.strip(), True  # 추론 모드를 끈 출력
    if "</think>" not in raw:
        return "", False
    return raw.split("</think>", 1)[1].strip(), True


def pick_answer(body: str) -> str:
    """본문에서 최종 답을 뽑는다. 마지막 '정답:' 줄. 없으면 빈 문자열."""
    ms = list(re.finditer(r"정답\s*[:：]\s*(.+)", body))
    return ms[-1].group(1).strip().split("\n")[0] if ms else ""


def count_new(seq: torch.Tensor, stop_ids: set[int]) -> int:
    """생성된 토큰 수. 배치 생성은 먼저 끝난 문장을 패딩으로 채우므로, 첫 종료 토큰까지만 센다."""
    for k, t in enumerate(seq.tolist()):
        if t in stop_ids:
            return k + 1
    return len(seq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="셔플 후 앞에서 N문항(그중 객관식만 쓴다)")
    ap.add_argument("--mode", choices=["think", "nothink"], required=True)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--limit", type=int, default=None, help="객관식 중 앞에서 K개만 (파일럿용)")
    ap.add_argument("--no-force", action="store_true", help="예산 강제 종료를 끈다")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(EVAL_FILE, encoding="utf-8") if l.strip()]
    random.Random(SEED).shuffle(rows)
    rows = rows[:args.n]
    items = []
    for i, r in enumerate(rows, 1):  # i는 덤프의 i와 같다
        ref = extract_answer(r["messages"][2]["content"])
        if choice_number(ref):
            items.append((i, r["messages"][1]["content"], ref))
    if args.limit:
        items = items[:args.limit]
    print(f"객관식 {len(items)}문항 / 모드 {args.mode} / max_new_tokens {args.max_new_tokens} / "
          f"batch {args.batch} / 강제종료 {not args.no_force}", flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tok.padding_side = "left"  # 배치 생성은 왼쪽 패딩이어야 한다
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    stop_ids = {tok.eos_token_id, tok.pad_token_id}
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb, device_map={"": 0})
    model.eval()
    print("로드 완료", flush=True)

    think = args.mode == "think"
    # Qwen3 권장 샘플링: 추론 모드는 greedy를 피하라고 한다(반복·무한 생성 위험).
    gen = (dict(temperature=0.6, top_p=0.95, top_k=20) if think
           else dict(temperature=0.7, top_p=0.8, top_k=20))

    out = open(args.dump, "w", encoding="utf-8") if args.dump else None
    n_ok = n_len = n_forced = n_forced_ok = n_noans = 0
    toks: list[int] = []
    t0 = time.time()
    torch.manual_seed(SEED)

    for s in range(0, len(items), args.batch):
        chunk = items[s:s + args.batch]
        prompts = []
        for _, q, _ in chunk:
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q + FORMAT_HINT}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=think))
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=768).to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                               pad_token_id=tok.pad_token_id, **gen)
        raws, n_news, closed_flags = [], [], []
        for seq in o:
            new = seq[enc["input_ids"].shape[1]:]
            n_news.append(count_new(new, stop_ids))
            raw = _SPECIAL.sub("", tok.decode(new, skip_special_tokens=False))
            raws.append(raw)
            closed_flags.append(split_think(raw)[1])

        # 생각이 안 끝난 것만 모아 강제 종료 후 답을 내게 한다.
        forced = [False] * len(chunk)
        need = [j for j, c in enumerate(closed_flags) if think and not c and not args.no_force]
        if need:
            fp = [prompts[j] + raws[j] + FORCE_SUFFIX for j in need]
            fenc = tok(fp, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                fo = model.generate(**fenc, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id)
            for j, seq in zip(need, fo):
                tail = tok.decode(seq[fenc["input_ids"].shape[1]:], skip_special_tokens=True)
                raws[j] = raws[j] + FORCE_SUFFIX + tail
                forced[j] = True
                closed_flags[j] = True

        for j, (i, _, ref) in enumerate(chunk):
            body, closed = split_think(raws[j])
            ans = pick_answer(body) if (closed and body) else ""
            ok = value_match(ans, ref) if ans else False
            n_ok += ok; n_len += 1; n_noans += (not ans); n_forced += forced[j]; n_forced_ok += (forced[j] and ok)
            toks.append(n_news[j])
            if out:
                out.write(json.dumps({"i": i, "ref": ref, "answer": ans, "value_hit": ok,
                                      "tokens": n_news[j], "forced": forced[j], "raw": raws[j][-1500:]},
                                     ensure_ascii=False) + "\n")
                out.flush()
        el = time.time() - t0
        print(f"  {n_len}/{len(items)}  값일치 {n_ok} ({n_ok / n_len:.1%})  강제종료 {n_forced}"
              f"  답없음 {n_noans}  평균 {sum(toks) / len(toks):.0f}토큰  경과 {el:.0f}s  문항당 {el / n_len:.1f}s",
              flush=True)

    if out:
        out.close()
    print("\n" + "=" * 70)
    print(f"모드 {args.mode}: 객관식 {n_len}문항 값 일치 {n_ok}/{n_len} = {n_ok / n_len:.1%}")
    nat = n_len - n_forced
    print(f"  자연 종료 {nat}건 / 예산 강제 종료 {n_forced}건 (강제 종료분 정답 {n_forced_ok}건)")
    print(f"  최종 답 없음 {n_noans}건")
    st = sorted(toks)
    print(f"  생성 토큰: 평균 {sum(toks) / len(toks):.0f} / 중앙값 {st[len(st) // 2]} / 최대 {st[-1]}")
    print(f"  소요 {time.time() - t0:.0f}s")
    print("비교(같은 채점, 같은 204문항): 미세조정 어댑터 기존 25.5% / CoT 28.9%")


if __name__ == "__main__":
    main()

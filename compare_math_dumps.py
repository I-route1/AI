"""수학 어댑터 두 개의 문항별 결과를 짝지어 비교한다 (McNemar 검정).

왜 McNemar인가
--------------
두 어댑터를 **같은 문항**으로 평가했으므로 표본이 독립이 아니다. 둘 다 맞힌
문항과 둘 다 틀린 문항은 어느 쪽이 나은지에 대한 정보를 주지 않는다.
판단에 쓰이는 건 **한쪽만 맞힌 문항**뿐이다(불일치 쌍).

이전 비교에서 "객관식 35.2% vs 26.8%"처럼 비율만 나란히 놓고 봤는데,
그것만으로는 71문항에서 6건 차이가 우연인지 알 수 없었다. McNemar는
정확히 그 질문에 답한다.

불일치 쌍이 적으므로 정규근사 대신 **이항검정 정확값**을 쓴다.

사용법:
    python test_math_adapter_eval.py --n 300 --adapter train/math_adapter_qwen \
        --max-new-tokens 512 --dump train/dump_orig.jsonl
    python test_math_adapter_eval.py --n 300 --adapter train/math_adapter_qwen_cot \
        --max-new-tokens 512 --skip-base --dump train/dump_cot.jsonl
    python compare_math_dumps.py train/dump_orig.jsonl train/dump_cot.jsonl
"""
import argparse
import json
import sys
from math import comb

sys.stdout.reconfigure(encoding="utf-8")


def binom_two_sided(b: int, c: int) -> float:
    """McNemar 정확검정. 불일치 n=b+c 중 한쪽이 b번 나올 확률(양측)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load(path: str) -> dict[int, dict]:
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["i"]] = r
    return rows


def mcnemar(a: dict, b: dict, field: str, label: str, name_a: str, name_b: str) -> None:
    shared = [i for i in a if i in b
              and a[i].get(field) is not None and b[i].get(field) is not None]
    both = sum(1 for i in shared if a[i][field] and b[i][field])
    only_a = sum(1 for i in shared if a[i][field] and not b[i][field])
    only_b = sum(1 for i in shared if not a[i][field] and b[i][field])
    neither = sum(1 for i in shared if not a[i][field] and not b[i][field])
    n = len(shared)
    if n == 0:
        print(f"\n[{label}] 비교 가능한 문항 없음")
        return

    p = binom_two_sided(only_a, only_b)
    print(f"\n[{label}] 공통 {n}문항")
    print(f"  {name_a:<16} {both + only_a:>4}건 ({(both + only_a) / n:.1%})")
    print(f"  {name_b:<16} {both + only_b:>4}건 ({(both + only_b) / n:.1%})")
    print(f"  둘 다 맞음 {both} / 둘 다 틀림 {neither}")
    print(f"  {name_a}만 맞음 {only_a}  vs  {name_b}만 맞음 {only_b}   <- 판단에 쓰이는 불일치 쌍")
    verdict = "유의미한 차이" if p < 0.05 else "우연과 구분되지 않음"
    print(f"  McNemar 정확검정 p = {p:.4f}  →  {verdict}")


def binom_tail(k: int, n: int, p: float) -> float:
    """P(X >= k), X ~ Binomial(n, p). 단측."""
    return sum(comb(n, x) * p ** x * (1 - p) ** (n - x) for x in range(k, n + 1))


def baseline_check(a: dict, b: dict, na: str, nb: str) -> None:
    """객관식 정확도를 '항상 최빈 보기' 기준선과 비교한다.

    5지선다라고 기준선을 20%로 두면 안 된다. 정답 분포가 치우쳐 있으면(이 평가셋은
    1번이 30.4%) 항상 1번만 찍어도 그만큼 나온다. 균등 무작위 기준선은 모델 실력을
    과대평가하게 만든다.
    """
    from collections import Counter

    rows = [r for r in a.values() if r.get("ref_choice")]
    n = len(rows)
    if not n:
        return
    prior = Counter(r["ref_choice"] for r in rows)
    top_choice, top_n = prior.most_common(1)[0]
    base_top = top_n / n
    base_bias = sum((c / n) ** 2 for c in prior.values())

    print("\n" + "=" * 74)
    print(f"기준선 비교 — 객관식 {n}문항, 정답 분포 {dict(sorted(prior.items()))}")
    print(f"  균등 무작위 20.0% / 정답분포 편향 찍기 {base_bias:.1%} / "
          f"항상 {top_choice}번 {base_top:.1%}")
    for name, d in ((na, a), (nb, b)):
        rs = [r for r in d.values() if r.get("lenient") is not None]
        k = sum(1 for r in rs if r["lenient"])
        print(f"  {name:<16} {k}/{len(rs)} = {k / len(rs):.1%}   "
              f"균등 대비 p={binom_tail(k, len(rs), 0.2):.4f} / "
              f"항상 {top_choice}번 대비 p={binom_tail(k, len(rs), base_top):.4f}")
        picks = Counter(r.get("adapter_choice") for r in d.values() if r.get("ref_choice"))
        share = picks.get(top_choice, 0) / len(rs)
        print(f"  {'':<16} 예측 분포 {top_choice}번 비율 {share:.1%} (정답 분포 {base_top:.1%})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_a")
    ap.add_argument("dump_b")
    ap.add_argument("--name-a", default=None)
    ap.add_argument("--name-b", default=None)
    args = ap.parse_args()

    a, b = load(args.dump_a), load(args.dump_b)
    na = args.name_a or args.dump_a.split("/")[-1].replace(".jsonl", "")
    nb = args.name_b or args.dump_b.split("/")[-1].replace(".jsonl", "")

    print(f"A = {na}  ({len(a)}문항)")
    print(f"B = {nb}  ({len(b)}문항)")
    print("=" * 74)

    mcnemar(a, b, "mcq_hit", "객관식 일치 (보기번호)", na, nb)
    mcnemar(a, b, "val_hit", "값만 일치 (번호 무시)", na, nb)
    mcnemar(a, b, "exact", "정답 완전일치", na, nb)

    # 관대한 채점: 모델이 보기 번호 없이 값만 쓴 답(`① 8/45` 대신 `8/45`)을 오답으로
    # 세면, 번호를 덜 쓰는 어댑터가 실력과 무관하게 불리해진다.
    for d in (a, b):
        for r in d.values():
            r["lenient"] = (None if r.get("mcq_hit") is None else
                            bool(r["mcq_hit"]) or
                            (r.get("adapter_choice") is None and bool(r.get("val_hit"))))
    mcnemar(a, b, "lenient", "객관식 관대 채점 (번호 없이 값만 맞아도 정답)", na, nb)

    baseline_check(a, b, na, nb)

    # 값은 맞았는데 번호를 틀린 경우 — 이번 비교의 핵심 가설
    print("\n" + "=" * 74)
    print("값은 맞고 보기 번호만 틀린 문항 수 (값→번호 매핑 실패)")
    for name, d in ((na, a), (nb, b)):
        cnt = sum(1 for r in d.values()
                  if r.get("val_hit") and not r.get("mcq_hit"))
        tot = sum(1 for r in d.values() if r.get("mcq_hit") is not None)
        print(f"  {name:<16} {cnt:>4}건 / 객관식 {tot}건 ({cnt / tot:.1%})" if tot else f"  {name}: 없음")
    print("=" * 74)


if __name__ == "__main__":
    main()

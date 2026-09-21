"""71718(수학 과목 문제생성 데이터)을 학습용 JSONL로 변환한다.

71859와 다른 점 (D 드라이브 실측)
--------------------------------
- 문제와 모범답안이 파일명으로 짝지어진다: `..._35793.json` ↔ `..._35793_A.json`. UTF-8 BOM.
- 문제: OCR_info[0].question_text (LaTeX). **선택형의 보기가 본문에 들어 있다.**
  71859는 본문에 보기가 없어 번호를 맞힐 방법이 없었다. 여기서는 보기를 파싱해서
  번호를 값으로 바꿀 수 있다(중학교 선택형의 84.9%, 그중 97%가 정답 번호와 일치).
- 풀이: answer_info[0].answer_text.
- 최종 정답: answer_bbox 중 type == 'answer' 항목의 text. 선택형은 보기 번호만(`④`) 있는
  경우가 대부분이고, 단답형은 `$324$`처럼 값이다. 소문항이면 `(1) $2300$, (2) $3720$`.

출력 형식 (--answer-format)
---------------------------
  value  : 정답: <값>            보기 번호를 값으로 바꾼다. 값 일치 평가와 맞는다. (기본)
  choice : 정답: ④ <값>         71859 형식과 같다.

정제 (전부 건수를 보고한다 — 조용히 버리지 않는다)
----------------------------------------------
  - 정답이 없거나 풀이가 없는 문항 제외
  - 선택형인데 보기를 파싱하지 못해 값을 못 구한 문항 제외 (--answer-format value일 때)
  - 소문항 정답(`(1)...(2)...`)은 제외: 한 줄 값 비교가 안 된다
  - 토큰 길이 초과는 자르지 않고 제외: 정답이 맨 뒤에 있어 잘리면 라벨이 망가진다
  - 71859 평가 문항과 유사도 >= 0.9인 문항 제외: 평가 오염 방지 (train/overlap_check.py)
  - --min-grade 로 학년 하한 지정 가능(초등 3~4학년은 풀이가 짧은 설명)

사용법
------
    python train/preprocess_math_718.py --dry-run
    python train/preprocess_math_718.py --min-grade 중학교_1학년 --out train/math718_train.jsonl
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
# train/ 안에서 직접 실행하면 프로젝트 루트가 경로에 없어 src를 못 찾는다.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE = "D:/수학 과목 문제생성 데이터"
GRADES = ["초등학교_3학년", "초등학교_4학년", "초등학교_5학년", "초등학교_6학년",
          "중학교_1학년", "중학교_2학년", "중학교_3학년", "고등학교_공통수학"]
SYSTEM_PROMPT = "당신은 수학 전문 교사입니다. 학생의 수학 문제에 대해 정확한 풀이와 해설을 제공하세요."
CIRCLED = "①②③④⑤"
_PAREN_NUM = re.compile(r"\(\s*([1-5])\s*\)")
_SUBQ = re.compile(r"^\s*\(\s*1\s*\)")


def load_json(path: str) -> dict:
    return json.load(open(path, encoding="utf-8-sig"))


def parse_choices(q: str) -> dict[str, str] | None:
    """본문에서 보기를 번호->값으로 쪼갠다. ①~⑤ 표기와 (1)~(5) 표기를 모두 본다."""
    pos = sorted((q.find(c), c) for c in CIRCLED if q.find(c) >= 0)
    if len(pos) >= 3:
        return {c: q[p + 1:(pos[k + 1][0] if k + 1 < len(pos) else len(q))].strip()
                for k, (p, c) in enumerate(pos)}
    # (1) (2) ... 표기는 소문항과 헷갈리므로, 문제 끝부분에서 1~5가 순서대로 이어질 때만 인정한다.
    hits = [(m.start(), int(m.group(1)), m.end()) for m in _PAREN_NUM.finditer(q)]
    for start in range(len(hits)):
        seq = [hits[start]]
        for h in hits[start + 1:]:
            if h[1] == seq[-1][1] + 1:
                seq.append(h)
        if len(seq) >= 4 and seq[0][1] == 1:
            out = {}
            for k, (p, n, e) in enumerate(seq):
                end = seq[k + 1][0] if k + 1 < len(seq) else len(q)
                out[CIRCLED[n - 1]] = q[e:end].strip()
            return out
    return None


def final_answer(ans_info: dict) -> str:
    """answer_bbox의 answer 항목 텍스트를 합쳐 최종 정답 문자열로."""
    parts = [bb.get("text") for bb in ans_info.get("answer_bbox", [])
             if bb.get("type") == "answer" and bb.get("text") not in (None, "None", "")]
    return " ".join(parts).strip()


def build_answer(qtype: str, q_text: str, fa: str, fmt: str) -> tuple[str | None, str]:
    """(정답 줄, 제외 사유). 사유가 ''이면 성공."""
    if not fa:
        return None, "정답 없음"
    if _SUBQ.match(fa):
        return None, "소문항 정답"
    if qtype == "선택형":
        m = re.match(r"^\s*([①②③④⑤])", fa)
        if not m:
            return None, "선택형 정답에 보기 번호 없음"
        # '①, ②, ⑤'처럼 정답이 여럿인 문항('옳은 것을 모두 고르시오')은 값 하나로 표현할 수 없다.
        # 맨 앞 번호만 보고 그 값을 정답으로 만들면 틀린 라벨이 된다.
        if len(set(re.findall(r"[①②③④⑤]", fa[:24]))) > 1 and re.match(r"^\s*[①②③④⑤]\s*[,、·]", fa):
            return None, "복수 정답"
        num = m.group(1)
        choices = parse_choices(q_text)
        val = (choices or {}).get(num)
        if fmt == "choice":
            return (f"정답: {num} {val}" if val else f"정답: {num}"), ""
        if not val:
            return None, "보기 파싱 실패"
        # 원본 본문이 '$'를 닫지 않고 다음 보기로 이어지면 값 끝에 '$'가 홀수 개 남는다.
        # 값을 자르지 않고 그 문항을 제외한다(고치려 들면 다른 오류를 만든다).
        if val.count("$") % 2 == 1:
            return None, "보기 값의 수식 기호가 짝이 안 맞음"
        return f"정답: {val}", ""
    # 단답형인데 정답 자체가 보기 번호인 문항(그림의 경로 번호, 표 안의 번호 등)은 값이 없다.
    # 그림이 있어야 풀리는 문제이기도 해서 제외한다.
    if re.match(r"^\s*[①②③④⑤]\s*$", fa):
        return None, "정답이 번호뿐(그림/표 문항)"
    return f"정답: {fa}", ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer-format", choices=["value", "choice"], default="value")
    ap.add_argument("--min-grade", default=None, help="이 학년 이상만. 예: 중학교_1학년")
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--split", choices=["TL", "VL"], default="TL")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 건수만 본다")
    ap.add_argument("--limit", type=int, default=None, help="학년별 최대 건수(시험용)")
    ap.add_argument("--sample", type=int, default=None,
                    help="정제를 모두 마친 뒤 무작위로 N건만 남긴다. --limit는 정렬된 앞부분만 자르므로 "
                         "학년이 치우치지만 이건 전체에서 고르게 뽑는다.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-similar", default=None,
                    help="유사도 검사로 만든 제외 id 목록(JSON). 없으면 검사를 건너뛴다.")
    args = ap.parse_args()

    grades = GRADES[GRADES.index(args.min_grade):] if args.min_grade else GRADES
    tokenizer = None
    if not args.dry_run:
        from transformers import AutoTokenizer
        from src.api.adapters import BASE_MODEL_ID
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    excl = set(json.load(open(args.exclude_similar, encoding="utf-8"))) if args.exclude_similar else set()
    drop, kept, per_grade = Counter(), [], Counter()
    for g in grades:
        qd, ad = f"{BASE}/{args.split}_1.문제_{g}", f"{BASE}/{args.split}_2.모범답안_{g}"
        names = sorted(os.listdir(qd))
        if args.limit:
            names = names[:args.limit]
        for nm in names:
            try:
                q = load_json(f"{qd}/{nm}")
                a = load_json(f"{ad}/{nm[:-5]}_A.json")
            except (OSError, json.JSONDecodeError):
                drop["파일 읽기 실패"] += 1
                continue
            if q["id"] in excl:
                drop["평가 문항과 유사"] += 1
                continue
            qt = (q["OCR_info"][0].get("question_text") or "").strip()
            ai = a["answer_info"][0]
            sol = (ai.get("answer_text") or "").strip()
            if len(qt) < 8:
                drop["문제 본문 없음"] += 1
                continue
            if not sol:
                drop["풀이 없음"] += 1
                continue
            line, why = build_answer(q["question_info"][0]["question_type1"], qt, final_answer(ai), args.answer_format)
            if line is None:
                drop[why] += 1
                continue
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": qt},
                    {"role": "assistant", "content": f"풀이: {sol}\n{line}"}]
            if tokenizer is not None:
                enc = tokenizer.apply_chat_template(msgs, tokenize=True)
                ids = enc["input_ids"] if not isinstance(enc, (list, tuple)) else enc
                if len(ids) > args.max_length:
                    drop[f"길이 초과(>{args.max_length})"] += 1
                    continue
            kept.append({"id": q["id"], "grade": g, "messages": msgs})
            per_grade[g] += 1

    total = sum(drop.values()) + len(kept)
    print(f"입력 {total}건 -> 정제 후 {len(kept)}건 ({len(kept) / max(total, 1):.1%})  [형식 {args.answer_format}]")
    if args.sample and args.sample < len(kept):
        import random
        random.Random(args.seed).shuffle(kept)
        kept = kept[:args.sample]
        per_grade = Counter(r["grade"] for r in kept)
        print(f"무작위 {args.sample}건 추출 (seed {args.seed})")
    print("제외 사유:")
    for k, v in drop.most_common():
        print(f"  {k:<28}{v:>7}건 ({v / total:.1%})")
    print("학년별 사용:", dict(per_grade))
    if args.out and not args.dry_run:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"저장: {args.out}")


if __name__ == "__main__":
    main()

"""수학 학습 데이터를 CoT(풀이 → 정답) 순서로 재배열한다.

왜 필요한가
-----------
기존 math_train.jsonl의 assistant 출력은 전부 "정답 먼저, 풀이 나중"이다:

    정답: ② $27\\sqrt{3}$
    풀이: AG:GD = 2:1 이므로 ...

12,640건 중 "풀이 → 정답" 순서는 0건이다. 이 순서로 학습하면 모델은 답 토큰을
중간 계산 없이 단 한 번의 forward pass로 내놓도록 배운다. 뒤에 붙는 풀이는 이미
확정된 답을 사후 합리화할 뿐이라 정답률에 기여하지 못한다.

실제로 test_math_adapter_eval.py 결과가 이 진단과 일치한다 —
'정답:' 형식 준수는 100%인데 객관식 정답률은 35.2%(무작위 기대값 20%)였다.
형식만 학습되고 수리 추론은 학습되지 않은 상태다.

그래서 순서를 뒤집는다:

    풀이: AG:GD = 2:1 이므로 ...
    정답: ② $27\\sqrt{3}$

답을 쓰기 전에 풀이 토큰을 먼저 생성하게 되므로, 그 토큰들이 답을 계산하는
실제 연산 과정으로 쓰인다. 데이터를 새로 구할 필요 없이 기존 9,752건이
그대로 CoT 학습 데이터가 된다.

절단(truncation) 주의
---------------------
순서를 뒤집으면 정답이 시퀀스 맨 뒤로 간다. max_length에서 잘리면 기존에는
풀이 꼬리만 날아갔지만 이제는 **정답 자체가 날아가** 라벨이 망가진다.
그래서 이 스크립트는 길이 초과 샘플을 자르지 않고 **제외**한다.

실행:
    python train/preprocess_math_cot.py
    python train/preprocess_math_cot.py --keep-no-solution   # 풀이 없는 건도 포함

출력: train/math_cot_train.jsonl, train/math_cot_eval.jsonl
"""
import argparse
import json
import re
from pathlib import Path

BASE_MODEL_ID = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MAX_LENGTH = 768  # 학습 스크립트의 MAX_LENGTH와 반드시 일치해야 한다

SRC = {
    "train": ("train/math_train.jsonl", "train/math_cot_train.jsonl"),
    "eval": ("train/math_eval.jsonl", "train/math_cot_eval.jsonl"),
}

# "정답: ...\n풀이: ..." 를 두 조각으로 가른다. 풀이는 여러 줄일 수 있다.
_SPLIT = re.compile(r"^\s*정답\s*[:：]\s*(?P<ans>.*?)\s*(?:\n\s*풀이\s*[:：]\s*(?P<sol>.*))?$", re.S)


def split_answer_solution(content: str) -> tuple[str, str] | None:
    """assistant 출력에서 (정답, 풀이)를 뽑는다. 형식이 다르면 None."""
    m = _SPLIT.match(content)
    if not m:
        return None
    ans = (m.group("ans") or "").strip()
    sol = (m.group("sol") or "").strip()
    if not ans:
        return None
    return ans, sol


def to_cot(content: str) -> str | None:
    """정답-먼저 출력을 풀이-먼저(CoT) 출력으로 재배열."""
    parsed = split_answer_solution(content)
    if parsed is None:
        return None
    ans, sol = parsed
    if not sol:
        return None  # 풀이가 없으면 CoT가 성립하지 않는다
    return f"풀이: {sol}\n정답: {ans}"


def count_tokens(tokenizer, messages: list[dict]) -> int:
    """대화 전체의 토큰 수.

    apply_chat_template(tokenize=True)는 transformers 버전에 따라 list[int]가 아니라
    BatchEncoding(키 2개)을 돌려준다. 거기에 len()을 쓰면 토큰 수가 아니라 항상 2가
    나와 길이 필터가 조용히 무력화된다. input_ids를 명시적으로 꺼내 센다.
    """
    enc = tokenizer.apply_chat_template(messages, tokenize=True)
    # BatchEncoding은 UserDict 상속이라 isinstance(enc, dict)가 False다. 그걸로 거르면
    # 분기를 못 타고 len()이 키 개수(2)를 돌려준다. 리스트가 아니면 매핑으로 취급한다.
    if not isinstance(enc, (list, tuple)):
        enc = enc["input_ids"]
    if enc and isinstance(enc[0], (list, tuple)):  # 배치로 감싸 나온 경우
        enc = enc[0]
    return len(enc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-no-solution", action="store_true",
                    help="풀이가 없는 샘플도 '정답: ...' 단독으로 포함한다. 기본은 제외 — "
                         "이런 샘플은 '바로 답부터 쓰기'를 다시 가르쳐 CoT 신호를 흐린다.")
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    for split, (src_path, dst_path) in SRC.items():
        rows = [json.loads(l) for l in open(src_path, encoding="utf-8") if l.strip()]
        out, n_no_sol, n_bad, n_toolong = [], 0, 0, 0
        max_seen = 0

        for row in rows:
            msgs = row["messages"]
            parsed = split_answer_solution(msgs[2]["content"])
            if parsed is None:
                n_bad += 1
                continue
            ans, sol = parsed

            if not sol:
                n_no_sol += 1
                if not args.keep_no_solution:
                    continue
                new_content = f"정답: {ans}"
            else:
                new_content = f"풀이: {sol}\n정답: {ans}"

            new_msgs = [msgs[0], msgs[1], {"role": "assistant", "content": new_content}]

            # 정답이 맨 뒤에 있으므로, 잘리면 라벨이 통째로 망가진다. 자르지 않고 제외한다.
            n_tok = count_tokens(tokenizer, new_msgs)
            max_seen = max(max_seen, n_tok)
            if n_tok > args.max_length:
                n_toolong += 1
                continue

            out.append({"messages": new_msgs})

        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "w", encoding="utf-8") as f:
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")

        print(f"[{split}] {src_path} -> {dst_path}")
        print(f"  원본 {len(rows)}건 -> 출력 {len(out)}건")
        print(f"  풀이 없음 {n_no_sol}건 ({'포함' if args.keep_no_solution else '제외'})"
              f" / 형식 불일치 {n_bad}건 / 길이초과(>{args.max_length}) 제외 {n_toolong}건")
        print(f"  최대 토큰 길이 {max_seen}")
        if out:
            print("  --- 변환 예시 ---")
            print("  " + out[0]["messages"][2]["content"][:220].replace("\n", "\n  "))
        print()


if __name__ == "__main__":
    main()

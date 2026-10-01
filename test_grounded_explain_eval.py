"""개념 설명 사실 정확도 점검 — 서빙과 같은 경로로 생성해 사람이 읽고 판정한다.

개념F1(test_concept_explain_eval.py)은 참조 용어가 출력에 나오는지만 봐서 사실 오류를
잡지 못한다. 그래서 이 스크립트는 점수를 매기지 않는다. 서빙 함수·설정 그대로 생성해
저장하고, 이전 실행과 비교해 **달라진 출력만** 보여 준다. 판정 기록(eval/fact_judgments.json)이
있으면 옆에 띄운다.

ConceptMap을 고친 뒤에는 이걸로 퇴행을 확인한다. 실제로 영어 항목의 자료가 둘에서 하나로
줄자 "what 앞에 선행사가 온다"로 부정이 뒤집힌 적이 있다.

개념 목록
- backend: Backend 시드(questions_seed.json)의 실제 conceptTag 42개
- extra: 보강하며 다룬 영어·국어 개념

생성: 베이스 Qwen(_concept_explain과 같은 프롬프트·후처리·문자 차단·길이 한도). 재현을 위해
greedy로 생성한다.

    python test_grounded_explain_eval.py retrieve train/fact_ctx.json
    python test_grounded_explain_eval.py generate train/fact_ctx.json train/fact_out.json
    # 기준 출력(eval/fact_out_20260929.json, 판정 77.5/79)과 비교 — 달라진 것만 판정하면 된다
    python test_grounded_explain_eval.py compare  eval/fact_out_20260929.json train/fact_out.json
    # 일부 개념만 (쉼표 구분)
    python test_grounded_explain_eval.py generate train/fact_ctx.json train/fact_part.json "사이시옷,균역법"
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

JUDGMENTS = Path("eval/fact_judgments.json")

CONCEPTS: dict[str, list[tuple[str, str]]] = {
    "backend": [(s, c) for s, cs in {
        "국어": ["고전소설 인물 심리", "고전시가 어휘", "문학 갈등 구조", "문학 표현기법 반어",
                 "비문학 추론적 이해", "비문학 핵심어 파악", "시조 형식 평시조", "현대소설 서술시점",
                 "현대시 화자 태도"],
        "수학": ["등차수열 일반항", "미분 chain rule 응용", "방정식의 활용", "분수 나눗셈 역수",
                 "비와 비율 계산", "삼각함수 덧셈공식", "삼각함수 응용 문제", "수열의 귀납적 정의",
                 "수열의 합 공식", "이차방정식 판별식", "이차방정식의 근과 계수", "이차함수 꼭짓점",
                 "지수로그 함수 미분", "함수 합성", "함수의 최댓값 최솟값"],
        "영어": ["가정법 과거완료", "가정법 미래", "관계대명사 that/which", "관계부사 where/when",
                 "도치구문 강조 용법", "복합관계대명사", "부정사 명사적 용법", "분사구문 의미상 주어",
                 "수동태 완료형", "현재완료 용법"],
        "과학": ["뉴턴 운동 법칙", "물질 변화", "세포 구조", "유전과 진화"],
        "사회": ["민주주의 원리", "세계 지리", "시장 경제", "한국 현대사"],
    }.items() for c in cs],
    "extra": [("영어", "관계대명사 what"), ("영어", "to부정사 의미상 주어"),
              ("국어", "음운 변동"), ("국어", "사이시옷"), ("국어", "동사와 형용사 구별")],
}


def retrieve(out_path: str) -> None:
    from src.api.concept_map import search_concept_map
    from src.api.routers.rag import subject_aware_search

    rows = []
    for group, items in CONCEPTS.items():
        for subject, c in items:
            docs = subject_aware_search(subject, c, k=3)
            src = "conceptmap" if search_concept_map(subject, c, 3) else "faiss"
            rows.append({"group": group, "subject": subject, "concept": c, "source": src, "docs": docs})
    json.dump(rows, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(rows)}개 개념 검색 완료 → {out_path}")


def generate(ctx_path: str, out_path: str, only: str | None = None) -> None:
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              LogitsProcessorList)

    from src.api.adapters import BASE_MODEL_ID, MATH_SYSTEM_PROMPT, SUBJECT_ADAPTERS
    from src.api.generation import CONCEPT_GEN_GREEDY
    from src.api.grounding import concept_user_message
    from src.api.postprocess import strip_chat_frame, strip_markdown, trim_cut_tail
    from src.api.script_guard import (ForeignScriptBlocker, foreign_token_mask, has_foreign,
                                      latin_token_mask, strip_foreign,
                                      RareHangulBlocker)

    # 국어·사회는 개념 설명에 영어가 나올 이유가 없어 영문자 토큰도 막는다(main._NO_LATIN_SUBJECTS
    # 와 같은 목록) — 실제로 "갈Conflict", "속belongs합니다" 같은 섞임이 있었다. 이 목록이
    # main.py와 어긋나면 평가가 서빙과 다른 조건을 재는 것이니 같이 고칠 것.
    _NO_LATIN_SUBJECTS = frozenset({"국어", "사회"})

    rows = json.load(open(ctx_path, encoding="utf-8"))
    if only:  # 쉼표로 구분한 개념 이름 — 항목 하나를 고친 뒤 그 개념만 다시 볼 때
        wanted = set(only.split(","))
        rows = [r for r in rows if r["concept"] in wanted]

    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_use_double_quant=True)).eval()
    blocker = ForeignScriptBlocker(foreign_token_mask(tok, model.config.vocab_size))
    latin_blocker = ForeignScriptBlocker(latin_token_mask(tok, model.config.vocab_size))
    rare_blocker = RareHangulBlocker(tok, model.config.vocab_size)

    def qwen(subject: str, concept: str, docs: list[str]) -> tuple[str, int, bool]:
        sysp = MATH_SYSTEM_PROMPT if subject == "수학" else SUBJECT_ADAPTERS[subject][2]
        msgs = [{"role": "system", "content": sysp},
                {"role": "user", "content": concept_user_message(concept, docs)}]
        ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False),
                  return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        processors = [blocker, rare_blocker] + ([latin_blocker] if subject in _NO_LATIN_SUBJECTS else [])
        with torch.no_grad():
            out = model.generate(**ids, **CONCEPT_GEN_GREEDY, pad_token_id=tok.eos_token_id,
                                 logits_processor=LogitsProcessorList(processors))
        n = out.shape[-1] - ids["input_ids"].shape[-1]
        text = strip_markdown(tok.decode(out[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True))
        if n >= CONCEPT_GEN_GREEDY["max_new_tokens"]:
            text = trim_cut_tail(text)
        text = strip_chat_frame(text)
        leaked = has_foreign(text)
        return (strip_foreign(text).strip() if leaked else text.strip()), n, leaked

    for i, row in enumerate(rows):
        s, c = row["subject"], row["concept"]
        row["out"], row["tokens"], row["leaked"] = qwen(s, c, row["docs"])
        row["limit"] = CONCEPT_GEN_GREEDY["max_new_tokens"]
        print(f"[{i + 1}/{len(rows)}] {s} {c} ({row['tokens']}토큰)", flush=True)
        json.dump(rows, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    cut = sum(r["tokens"] >= r["limit"] for r in rows)
    leak = sum(r["leaked"] for r in rows)
    print(f"\n{len(rows)}개 생성 — 길이 한도에서 잘림 {cut}개, 다른 문자 섞임(지운 뒤 저장) {leak}개")


def compare(old_path: str, new_path: str) -> None:
    old = {(r["subject"], r["concept"]): r for r in json.load(open(old_path, encoding="utf-8"))}
    judg = json.load(open(JUDGMENTS, encoding="utf-8")) if JUDGMENTS.exists() else {}
    same = 0
    for r in json.load(open(new_path, encoding="utf-8")):
        key = (r["subject"], r["concept"])
        o = old.get(key)
        if o and o.get("out") == r["out"]:
            same += 1
            continue
        j = judg.get(f"{key[0]}|{key[1]}")
        print(f"===== [{key[0]}] {key[1]}" + (f"  (이전 판정 {j['verdict']}: {j['note']})" if j else ""))
        if o:
            print("  [이전] " + o["out"][:500].replace("\n", " / "))
        print("  [지금] " + r["out"][:500].replace("\n", " / "))
    print(f"\n출력이 같은 개념 {same}개는 생략했습니다.")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "retrieve":
        retrieve(sys.argv[2])
    elif cmd == "generate":
        generate(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    else:
        compare(sys.argv[2], sys.argv[3])

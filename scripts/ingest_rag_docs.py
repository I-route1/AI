"""새 문서를 FAISS RAG DB에 적재한다.

기존 scripts/add_concept_docs.py와 다른 점
------------------------------------------
- 문서를 **소스코드에 하드코딩하지 않는다.** JSONL 파일이나 ConceptMap에서 읽는다.
  기존 스크립트는 문서를 추가할 때마다 파이썬 파일을 고쳐야 했다.
- **중복을 거른다.** 같은 지문을 두 번 넣으면 검색 결과가 같은 문서로 도배된다.
- **langchain_core.documents.Document를 쓴다.** 기존 스크립트는 `_Doc` 클래스를
  스크립트 안에 정의해서 pickle에 `__main__._Doc`으로 저장되는데, rag.py가
  역직렬화할 때 그 클래스를 찾지 못해 DB 로드가 통째로 실패한다. 현재 DB의
  698,347개 문서는 전부 langchain Document라 그 스크립트는 실행된 적이 없는
  것으로 보인다. 같은 클래스를 써서 그 지뢰를 피한다.
- **원본을 덮어쓰지 않는다.** 임시 파일에 쓰고 원본을 .bak으로 옮긴 뒤 바꿔 넣는다.
  2.1GB 인덱스를 직접 덮어쓰다 중단되면 DB가 깨진다.
- `--dry-run`으로 무엇이 들어갈지 먼저 확인할 수 있다.

문서 형식
---------
DB의 기존 문서는 아래 형식이고, rag.py의 과목 필터(_SUBJECT_PATH_PATTERNS)와
본문 추출(_extract_content)이 이 태그에 의존한다. 같은 형식으로 맞춘다.

    [대상]{과목} (폴더: ...)
    [교육과정 성취기준] [코드] ...
    [학습 지문] ...
    [관련 Q&A] 질문: ... / 답변: ...

사용법
------
    # 무엇이 들어갈지 먼저 보기
    python scripts/ingest_rag_docs.py --from-concept-map 한국사 --dry-run

    # ConceptMap의 한국사 항목을 DB에 적재
    python scripts/ingest_rag_docs.py --from-concept-map 한국사

    # 새로 들어온 문제/자료를 JSONL로 적재
    #   {"subject": "한국사", "text": "...", "keywords": [...], "question": "...", "answer": "..."}
    #   (subject와 text만 필수)
    python scripts/ingest_rag_docs.py --from-jsonl data/new_docs.jsonl

    # 되돌리기: src/api/rag_db/ 의 .bak 파일을 원래 이름으로 되돌리면 된다.
"""
import argparse
import json
import os
import pickle
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import faiss  # noqa: E402
import numpy as np  # noqa: E402

DB_DIR = Path("src/api/rag_db")
EMBED_MODEL = "jhgan/ko-sroberta-multitask"


# ── 문서 조립 ────────────────────────────────────────────────────────────────
def build_doc(subject: str, text: str, keywords: list[str] | None = None,
              standard: str | None = None, question: str | None = None,
              answer: str | None = None, source: str = "ingest") -> str:
    """DB의 기존 문서 형식에 맞춰 한 건을 조립한다.

    [대상] 줄에 과목명이 들어가야 rag.py의 과목 필터가 잡는다. 예를 들어 한국사는
    _SUBJECT_PATH_PATTERNS가 "한국사" 문자열을 찾는다.
    """
    lines = [f"[대상]{subject} (폴더: {source}_{subject})"]
    meta = []
    if standard:
        meta.append(standard)
    if keywords:
        meta.append("핵심어: " + ", ".join(keywords))
    if meta:
        lines.append(f"[교육과정 성취기준] [{source}] " + " / ".join(meta))
    lines.append(f"[학습 지문] {text}")
    if question:
        lines.append(f"[관련 Q&A] 질문: {question} / 답변: {answer or text}")
    return "\n".join(lines)


def _norm(s: str) -> str:
    """중복 판정용 정규화 — 공백과 대소문자 차이는 같은 문서로 본다."""
    return re.sub(r"\s+", " ", s).strip().lower()


# 임베딩 모델(ko-sroberta-multitask)의 max_seq_length가 128이다. 기존 코퍼스
# 698,347건도 이 설정으로 임베딩됐으므로 같은 값을 쓴다. 늘리면 새 문서만
# 다른 길이로 임베딩돼 유사도 분포가 어긋난다.
EMBED_MAX_TOKENS = 128
# 문서 텍스트에는 [대상]·핵심어 같은 머리말이 붙는데, 그걸 그대로 임베딩하면
# 머리말이 예산의 45%(중앙값 58토큰)를 먹고 정작 본문이 잘린다. 그래서
# **저장하는 텍스트와 임베딩하는 텍스트를 분리한다** — 저장은 전체 형식대로,
# 임베딩은 "과목 + 본문"만.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_passage(text: str, tokenizer, budget: int) -> list[str]:
    """본문을 임베딩 예산에 맞춰 문장 경계로 자른다.

    한국사 ConceptMap 본문은 중앙값 162토큰, 최대 277토큰이라 42건 중 38건이
    128을 넘는다. 자르지 않으면 뒷부분이 통째로 검색되지 않는다.
    문장 중간에서 끊지 않도록 마침표 뒤에서 나눈다.
    """
    def n_tok(s: str) -> int:
        return len(tokenizer.encode(s, add_special_tokens=True))

    if n_tok(text) <= budget:
        return [text]

    chunks, cur = [], ""
    for sent in _SENT_SPLIT.split(text):
        cand = f"{cur} {sent}".strip()
        if cur and n_tok(cand) > budget:
            chunks.append(cur)
            cur = sent
        else:
            cur = cand
    if cur:
        chunks.append(cur)

    # 한 문장이 예산을 넘으면 문장 경계로는 더 못 줄인다. 그때만 강제로 자른다.
    out = []
    for c in chunks:
        while n_tok(c) > budget:
            lo, hi = 1, len(c)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if n_tok(c[:mid]) <= budget:
                    lo = mid
                else:
                    hi = mid - 1
            out.append(c[:lo])
            c = c[lo:].strip()
        if c:
            out.append(c)
    return out


def _passage(doc_text: str) -> str:
    """[학습 지문] 본문만 뽑는다. 중복 판정은 이 본문으로 한다."""
    m = re.search(r"\[학습 지문\]\s*(.+?)(?=\n\[|$)", doc_text, re.S)
    return _norm(m.group(1)) if m else _norm(doc_text)


# ── 입력 소스 ────────────────────────────────────────────────────────────────
def from_concept_map(subjects: list[str]) -> list[dict]:
    from src.api.concept_map import CONCEPT_MAP

    out = []
    for keys, text in CONCEPT_MAP.items():
        if subjects and keys[0] not in subjects:
            continue
        kws = list(keys[1:])
        out.append({
            "subject": keys[0],
            "text": text,
            "keywords": kws,
            "question": f"{kws[0]}에 대해 설명하시오." if kws else None,
            "answer": text,
            "source": "ConceptMap",
        })
    return out


def from_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln} JSON 파싱 실패: {e}")
            if not r.get("subject") or not r.get("text"):
                raise SystemExit(f"{path}:{ln} subject와 text는 필수입니다: {line[:80]}")
            r.setdefault("source", "ingest")
            out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-concept-map", nargs="*", metavar="과목",
                     help="ConceptMap에서 가져온다. 과목을 적지 않으면 전체.")
    src.add_argument("--from-jsonl", metavar="PATH",
                     help="JSONL에서 가져온다. subject/text 필수, "
                          "keywords/standard/question/answer 선택.")
    ap.add_argument("--dry-run", action="store_true", help="DB를 건드리지 않고 결과만 본다")
    ap.add_argument("--db", default=str(DB_DIR))
    ap.add_argument("--device", default="cpu",
                    help="임베딩 장치. 기본 cpu — 적재는 배치 작업이라 CPU로 충분하고, "
                         "서빙 중인 GPU와 다투지 않는다. venv의 torch는 sm_120(RTX 5070 Ti)을 "
                         "지원하지 않아 cuda로 두면 커널 에러가 난다.")
    args = ap.parse_args()

    db = Path(args.db)
    records = (from_concept_map(args.from_concept_map)
               if args.from_concept_map is not None else from_jsonl(args.from_jsonl))
    if not records:
        raise SystemExit("적재할 문서가 없습니다.")
    print(f"입력 문서 {len(records)}건")

    # 본문이 임베딩 예산을 넘으면 문장 경계로 나눠 각각 한 건으로 넣는다.
    # 그래야 뒷부분도 검색된다.
    print(f"임베딩 모델 로드 중: {EMBED_MODEL} (device={args.device})")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL, device=args.device)
    model.max_seq_length = EMBED_MAX_TOKENS
    tokenizer = model.tokenizer

    pairs: list[tuple[str, str]] = []  # (저장할 전체 문서, 임베딩할 텍스트)
    n_split = 0
    for r in records:
        chunks = chunk_passage(r["text"], tokenizer, EMBED_MAX_TOKENS - 8)
        if len(chunks) > 1:
            n_split += 1
        for ch in chunks:
            rr = dict(r, text=ch)
            # 저장은 전체 형식대로, 임베딩은 과목+본문만 — 머리말이 예산을 먹지 않도록.
            pairs.append((build_doc(**rr), f'{r["subject"]} {ch}'))
    print(f"본문 분할: {n_split}건이 여러 조각으로 나뉘어 총 {len(pairs)}건")

    docs = [p[0] for p in pairs]

    # ── 기존 DB 로드 ─────────────────────────────────────────────────────────
    t0 = time.time()
    print(f"DB 로드 중: {db}")
    index = faiss.read_index(str(db / "index.faiss"))
    with open(db / "index.pkl", "rb") as f:
        raw = pickle.load(f)
    if not (isinstance(raw, tuple) and len(raw) == 2):
        raise SystemExit("지원하지 않는 pkl 형식입니다.")
    a, b = raw
    index_to_id, docstore = (a, b) if isinstance(a, dict) else (b, a)
    store = getattr(docstore, "_dict", None)
    if store is None:
        raise SystemExit("docstore._dict를 찾을 수 없습니다.")
    print(f"  기존 벡터 {index.ntotal}개 / 문서 {len(store)}개 ({time.time() - t0:.0f}s)")

    if index.ntotal != len(index_to_id):
        raise SystemExit(f"인덱스({index.ntotal})와 매핑({len(index_to_id)}) 수가 다릅니다. 중단합니다.")

    # ── 중복 제거 ────────────────────────────────────────────────────────────
    existing = {_passage(d.page_content if hasattr(d, "page_content") else str(d))
                for d in store.values()}
    fresh: list[tuple[str, str]] = []
    dup = 0
    seen_in_batch = set()
    for doc_text, embed_text in pairs:
        key = _passage(doc_text)
        if key in existing or key in seen_in_batch:
            dup += 1
            continue
        seen_in_batch.add(key)
        fresh.append((doc_text, embed_text))
    print(f"중복 제외 {dup}건 → 실제 추가 대상 {len(fresh)}건")

    if not fresh:
        print("추가할 새 문서가 없습니다. DB를 건드리지 않고 종료합니다.")
        return

    if args.dry_run:
        print("\n--- dry-run: 아래 문서가 추가됩니다 (DB 변경 없음) ---")
        for t, e in fresh[:3]:
            print("· 저장:", t.replace("\n", " | ")[:140])
            print("  임베딩:", e[:100], f"({len(tokenizer.encode(e))}토큰)")
        if len(fresh) > 3:
            print(f"... 외 {len(fresh) - 3}건")
        over = sum(1 for _, e in fresh if len(tokenizer.encode(e)) > EMBED_MAX_TOKENS)
        print(f"\n임베딩 예산({EMBED_MAX_TOKENS}토큰) 초과: {over}건")
        return

    # ── 임베딩 ───────────────────────────────────────────────────────────────
    vecs = model.encode([e for _, e in fresh], batch_size=32,
                        show_progress_bar=False).astype(np.float32)
    faiss.normalize_L2(vecs)
    if vecs.shape[1] != index.d:
        raise SystemExit(f"임베딩 차원 불일치: 새 {vecs.shape[1]} vs 인덱스 {index.d}")

    # ── 추가 ─────────────────────────────────────────────────────────────────
    # 기존 DB와 같은 클래스를 써야 rag.py가 역직렬화할 수 있다.
    from langchain_core.documents import Document

    next_idx = index.ntotal
    for doc_text, _ in fresh:
        doc_id = str(uuid.uuid4())
        index_to_id[next_idx] = doc_id
        store[doc_id] = Document(page_content=doc_text)
        next_idx += 1
    index.add(vecs)
    print(f"추가 완료. 총 벡터 {index.ntotal}개 / 문서 {len(store)}개")

    # ── 저장 (임시 파일 → 원본 .bak → 교체) ──────────────────────────────────
    # 2.1GB 인덱스를 직접 덮어쓰다 중단되면 DB가 깨진다. 새 파일을 먼저 완성한 뒤
    # 이름만 바꾼다. 이름 바꾸기는 즉시 끝나므로 깨질 틈이 없다.
    fi, fp = db / "index.faiss", db / "index.pkl"
    ti, tp = db / "index.faiss.tmp", db / "index.pkl.tmp"
    print("저장 중...")
    faiss.write_index(index, str(ti))
    with open(tp, "wb") as f:
        pickle.dump((index_to_id, docstore), f)
    for real, tmp in ((fi, ti), (fp, tp)):
        bak = real.with_suffix(real.suffix + ".bak")
        if bak.exists():
            bak.unlink()
        real.rename(bak)
        tmp.rename(real)
    print(f"저장 완료. 이전 버전은 {fi.name}.bak / {fp.name}.bak 로 남겨뒀습니다.")

    # ── 검증 ─────────────────────────────────────────────────────────────────
    print("\n검증: 새로 넣은 문서가 실제로 검색되는지 확인")
    q = model.encode([fresh[0][1]]).astype(np.float32)
    faiss.normalize_L2(q)
    chk = faiss.read_index(str(fi))
    _, idx = chk.search(q, 1)
    hit = idx[0][0]
    ok = hit >= index.ntotal - len(fresh)
    print(f"  최근접 문서 인덱스 {hit} (새 문서 구간 {index.ntotal - len(fresh)}~{index.ntotal - 1})"
          f" → {'정상' if ok else '확인 필요'}")
    print("\nAI 서버를 재시작해야 반영됩니다.")


if __name__ == "__main__":
    main()

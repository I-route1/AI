import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from fastapi import APIRouter

router = APIRouter()

# 개념 키워드 직접 매핑(FAISS 우선 레이어)은 src/api/concept_map.py로 분리했다.
# rag.py는 import만 해도 FAISS와 임베딩 모델을 올리므로, 이 데이터만 필요한
# 쪽(평가 스크립트 등)이 rag를 import하지 않아도 되게 하기 위해서다.
from src.api.concept_map import CONCEPT_MAP as _CONCEPT_MAP
from src.api.concept_map import search_concept_map as _concept_map_search

# ── FAISS 인덱스 로드 ──────────────────────────────────────────────────────────
_FAISS_DIR = Path("src/api/rag_db")
_index: faiss.Index | None = None
_doc_texts: list[str] = []


def _extract_docs(a, b) -> list[str]:
    if isinstance(a, dict):
        index_to_id, docstore = a, b
    else:
        index_to_id, docstore = b, a

    store = getattr(docstore, "_dict", {})
    texts = []
    for i in sorted(index_to_id.keys()):
        doc = store.get(index_to_id[i])
        if doc is not None:
            texts.append(doc.page_content if hasattr(doc, "page_content") else str(doc))
    return texts


def _load_index():
    global _index, _doc_texts
    try:
        _index = faiss.read_index(str(_FAISS_DIR / "index.faiss"))
        with open(_FAISS_DIR / "index.pkl", "rb") as f:
            raw = pickle.load(f)
        if isinstance(raw, tuple) and len(raw) == 2:
            _doc_texts = _extract_docs(raw[0], raw[1])
        elif isinstance(raw, list):
            _doc_texts = [str(d) for d in raw]
        else:
            _doc_texts = []
        print(f"[FAISS] 로드 완료: {_index.ntotal}개 벡터, {len(_doc_texts)}개 문서")
    except Exception as e:
        print(f"[FAISS] 로드 실패: {e}")


_load_index()

# ── 과목별 문서 인덱스 ─────────────────────────────────────────────────────────
# 과목 한정 검색용. 예전에는 전역 상위 30건을 가져와서 과목 필터를 걸었는데,
# 한국사처럼 문서가 적은 과목(98건 / 698,432건)은 상위 30에 들 일이 없어
# 사실상 검색이 안 됐다. FAISS IDSelector로 해당 과목 문서만 놓고 찾으면
# 그 문제가 없어지고, 대상이 줄어 오히려 빠르다(한국사 2ms vs 전역 74ms).
_subject_ids: dict[str, np.ndarray] = {}


def _build_subject_ids() -> None:
    if not _doc_texts:
        return
    for subject, patterns in _SUBJECT_PATH_PATTERNS.items():
        ids = [i for i, t in enumerate(_doc_texts) if any(p in t for p in patterns)]
        if ids:
            _subject_ids[subject] = np.array(ids, dtype="int64")
    print("[FAISS] 과목별 문서: "
          + ", ".join(f"{s} {len(v)}" for s, v in _subject_ids.items()))


# 실제 호출은 _SUBJECT_PATH_PATTERNS가 정의된 뒤에 한다(파일 아래쪽).

# ── 임베딩 모델 로드 ───────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer as _ST
    _embed_model = _ST("jhgan/ko-sroberta-multitask")
    def _encode(text: str) -> np.ndarray:
        return _embed_model.encode([text])
    print("[Embed] sentence-transformers 로드 완료")
except ImportError:
    from transformers import AutoModel, AutoTokenizer
    import torch
    _MODEL_NAME = "jhgan/ko-sroberta-multitask"
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    _hf_model = AutoModel.from_pretrained(_MODEL_NAME)
    def _encode(text: str) -> np.ndarray:
        inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            out = _hf_model(**inputs)
        return out.last_hidden_state[:, 0, :].numpy()
    print("[Embed] transformers 로드 완료")

# ── 과목 키워드 매핑 ───────────────────────────────────────────────────────────
# 우선순위: 경로 패턴(_0X.과목_) > 과목명 > 개념 키워드
# 경로 패턴은 rag_sample.txt 형식의 폴더명에서 추출됨
_SUBJECT_PATH_PATTERNS: dict[str, list[str]] = {
    "수학":   ["_03.수학_", ".수학_"],
    "영어":   ["_02.영어_", ".영어_"],
    "국어":   ["_01.국어_", ".국어_"],
    # 과학과 사회의 번호가 서로 바뀌어 있었다(과학은 _05, 사회는 _04).
    # 두 번째 패턴이 받아내서 동작은 했지만 첫 패턴은 0건이었다.
    "과학":   ["_05.과학_", ".과학_"],
    "사회":   ["_04.사회_", ".사회_"],
    "한국사": ["한국사"],
}
# FAISS 벡터 검색을 건너뛰고 ConceptMap만 쓰는 과목.
#
# 한국사가 여기 있었다. 코퍼스에 13건뿐이라 검색이 오염됐기 때문인데,
# scripts/ingest_rag_docs.py로 ConceptMap 42개 항목을 청크로 나눠 85건 적재해
# 98건이 됐고, 아래 과목 한정 검색이 들어가면서 상황이 달라졌다.
# 같은 42개 질의로 상위 3건 안에 질의어가 들어오는 비율을 재보면
#   FAISS 과목 한정 + 청크 분할 : 81%
#   _concept_map_semantic      : 33%
# 로 뒤집힌다. _concept_map_semantic이 낮은 이유는 임베딩 모델이 128토큰에서
# 잘라서 긴 항목의 뒷부분이 아예 안 보이기 때문이다. 청크로 나누면 그게 해결된다.
_FAISS_EXCLUDED_SUBJECTS: frozenset[str] = frozenset()

_build_subject_ids()


def _check_concept_map_chunks() -> None:
    """FAISS에 적재한 ConceptMap 조각이 지금의 ConceptMap과 맞는지 기동할 때 확인한다.

    한국사는 ConceptMap을 조각내 FAISS에도 넣었다. 항목을 고치고 다시 적재하지 않으면
    옛 조각은 근거 판별(grounding.is_curated)에서 빠지고, 고친 내용은 FAISS에 없다.
    에러 없이 조용히 품질만 떨어지므로 로그로 알린다.
    """
    from src.api.grounding import is_curated

    def norm(s: str) -> str:
        return re.sub(r"\s+", "", s)

    by_subject: dict[str, list[str]] = {}
    for t in _doc_texts:
        m = re.match(r"\[대상\](\S+) \(폴더: ConceptMap_", t)
        if not m:
            continue
        body = re.search(r"\[학습 지문\]\s*(.+?)(?=\n\[|$)", t, re.S)
        by_subject.setdefault(m.group(1), []).append(norm(body.group(1)) if body else "")

    for subject, chunks in by_subject.items():
        stale = sum(1 for c in chunks if not is_curated(c))
        missing = 0
        for keys, text in _CONCEPT_MAP.items():
            if keys[0] != subject:
                continue
            e = norm(text)
            if sum(len(c) for c in chunks if c and c in e) < len(e):
                missing += 1
        if stale or missing:
            # 이모지를 쓰지 않는다. rag를 직접 import하는 스크립트는 콘솔이 cp949일 수 있다.
            print(f"[FAISS 경고] {subject} ConceptMap 조각이 현재 ConceptMap과 다릅니다 — "
                  f"옛 조각 {stale}개, FAISS에 반영 안 된 항목 {missing}개. 다시 적재하세요: "
                  f"python scripts/ingest_rag_docs.py --from-concept-map {subject} --replace")


_check_concept_map_chunks()

_SUBJECT_KEYWORDS: dict[str, list[str]] = {
    "수학":   ["수학", "방정식", "함수", "수열", "확률", "기하", "미적분", "삼각", "벡터", "행렬", "정수", "집합"],
    "영어":   ["영어", "English", "Grammar", "Reading", "Listening", "어법", "구문", "독해", "listening"],
    "국어":   ["국어", "문학", "비문학", "화자", "서술", "소설", "시", "수필", "논설"],
    "과학":   ["과학", "물리", "화학", "생물", "지구", "역학", "전기", "유전", "세포"],
    "사회":   ["사회", "역사", "지리", "경제", "문화", "정치", "법"],
    "한국사": ["한국사", "시대", "왕조", "조선", "고려", "신라"],
}


def _clean_text(text: str) -> str:
    """LaTeX 수식·HTML·불필요한 특수문자 제거"""
    text = re.sub(r'\$\$[\s\S]*?\$\$', '', text)
    text = re.sub(r'\$[^$\n]{1,200}?\$', '', text)
    text = re.sub(r'\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+(?:\{[^}]*\})*', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[{}\[\]\\^_|]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _extract_content(doc: str) -> str:
    """
    RAG 문서에서 핵심 내용만 추출 — 폴더 경로·과정 코드 제거.
    우선순위: 학습 지문 → Q&A 답변 → 성취기준 텍스트 → 전체 클리닝
    """
    # 학습 지문
    m = re.search(r'\[학습 지문\]\s*(.+?)(?=\[|$)', doc, re.DOTALL)
    if m:
        return _clean_text(m.group(1).strip())

    # Q&A 답변
    m = re.search(r'답변[:\s]+(.+?)(?=\[|질문:|$)', doc, re.DOTALL)
    if m:
        return _clean_text(m.group(1).strip())

    # 성취기준 설명 (코드 번호 제외)
    m = re.search(r'\[교육과정 성취기준\]\s*\[[^\]]*\]\s*(.+?)(?=\[|$)', doc, re.DOTALL)
    if m:
        return _clean_text(m.group(1).strip())

    return _clean_text(doc)


_concept_vec_cache: dict[str, tuple[list[str], np.ndarray]] = {}


def _concept_map_semantic(subject: str, query: str, k: int) -> list[str]:
    """해당 과목 ConceptMap 항목을 쿼리와의 의미 유사도로 정렬해 상위 k개 반환.

    키워드 매칭(_concept_map_search)이 실패한 질의에 쓴다. 예를 들어 '갑오개혁'은
    ConceptMap 키 어디에도 없지만, 의미상 '조선'·'일제강점기' 항목이 가장 가깝다.
    과목당 항목이 10~35개뿐이라 전량 비교해도 비용이 거의 없다. 벡터는 캐시한다.
    """
    entries = [v for keys, v in _CONCEPT_MAP.items() if keys[0] == subject]
    if not entries:
        return []
    try:
        cached = _concept_vec_cache.get(subject)
        if cached is None or cached[0] != entries:
            mat = np.vstack([_encode(e) for e in entries]).astype(np.float32)
            faiss.normalize_L2(mat)
            cached = (entries, mat)
            _concept_vec_cache[subject] = cached

        qv = _encode(query).astype(np.float32)
        faiss.normalize_L2(qv)
        scores = cached[1] @ qv[0]
        order = np.argsort(-scores)[:k]
        return [entries[i] for i in order]
    except Exception:
        return entries[:k]  # 임베딩 실패 시 기존 동작(앞에서 k개)으로 폴백


def _filter_by_subject(docs: list[str], subject: str) -> list[str]:
    """
    경로 패턴 우선 → 과목명/개념 키워드 순으로 필터링.
    크로스 과목 오염 방지를 위해 매칭 없으면 빈 리스트 반환.
    """
    path_patterns = _SUBJECT_PATH_PATTERNS.get(subject, [])
    keywords = _SUBJECT_KEYWORDS.get(subject, [subject])

    # 1단계: 경로 패턴으로 정확 필터 (예: _03.수학_)
    if path_patterns:
        strict = [d for d in docs if any(p in d for p in path_patterns)]
        if strict:
            return strict

    # 2단계: 과목명/개념 키워드 필터
    filtered = [d for d in docs if any(kw in d for kw in keywords)]
    return filtered  # 없으면 [] 반환 — 폴백 없음


def vector_search(query: str, k: int = 3) -> list[str]:
    """기존 검색 (과목 필터 없음 — /api/rag/search 에서 사용)"""
    if _index is None or not _doc_texts:
        return ["검색 가능한 데이터가 없습니다."]
    try:
        vec = _encode(query).astype(np.float32)
        faiss.normalize_L2(vec)
        _, indices = _index.search(vec, k)
        return [_doc_texts[i] for i in indices[0] if 0 <= i < len(_doc_texts)]
    except Exception as e:
        return [f"검색 오류: {str(e)}"]


def subject_aware_search(subject: str, query: str, k: int = 3) -> list[str]:
    """
    1. 개요 쿼리(핵심 개념 및 자주 출제되는 단원) → ConceptMap 과목 샘플 즉시 반환
    2. ConceptMap 키워드 직접 매핑 — 히트가 있으면 즉시 반환 (FAISS 혼합 없음)
    3. ConceptMap 미스 → FAISS 벡터 검색 + 과목 필터
    4. FAISS도 결과 없으면 ConceptMap에서 해당 과목 샘플 반환 (fallback)
    """
    # ── 1. 개요 쿼리 감지: 개념 없이 과목만 선택한 경우 ─────────────────────────
    if "핵심 개념 및 자주 출제되는 단원" in query:
        overview = [v for keys, v in _CONCEPT_MAP.items() if keys[0] == subject]
        return overview[:k]

    # ── 2. ConceptMap 우선 ─────────────────────────────────────────────────────
    concept_hits = _concept_map_search(subject, query, k)
    if concept_hits:
        return concept_hits[:k]

    # ── 2. FAISS fallback ──────────────────────────────────────────────────────
    # 한국사는 코퍼스에 13건뿐(다른 과목은 7.6만~18.7만)이라 경로 필터가 사실상 실패하고,
    # 키워드 필터("조선","고려","신라"…)로 넘어가 무관한 문서가 걸린다.
    # (예: "신간회" 검색 → 발해 역사책 설명) 틀린 자료를 주느니 ConceptMap만 쓴다.
    if subject in _FAISS_EXCLUDED_SUBJECTS:
        return _concept_map_semantic(subject, query, k)

    if _index is not None and _doc_texts:
        try:
            # 질의를 그대로 쓴다. 예전에는 f"{subject} 교육과정 {query} 개념 학습"으로
            # 부풀렸는데, 임베딩 모델이 128토큰이라 짧은 질의에서는 붙인 일반어가
            # 벡터를 지배해 '교육과정'을 논하는 엉뚱한 문서로 끌려갔다.
            # ConceptMap 키워드 149개를 질의로 넣어 상위 3건에 질의어가 들어오는
            # 비율을 재보면 원본 61% / 보강 37%로 원본이 낫다(한국사 81% vs 12%).
            vec = _encode(query).astype(np.float32)
            faiss.normalize_L2(vec)

            ids = _subject_ids.get(subject)
            if ids is not None and len(ids):
                # 해당 과목 문서만 놓고 찾는다. 전역에서 뽑아 거르면 문서 수가 적은
                # 과목은 후보에 아예 못 든다.
                selector = faiss.IDSelectorBatch(ids)
                params = faiss.SearchParameters()
                params.sel = selector
                fetch_k = min(k * 5, len(ids))
                _, indices = _index.search(vec, fetch_k, params=params)
            else:
                fetch_k = min(k * 10, len(_doc_texts))
                _, indices = _index.search(vec, fetch_k)
            candidates = [_doc_texts[i] for i in indices[0] if 0 <= i < len(_doc_texts)]

            # IDSelector가 이미 걸렀으므로 보통 그대로 통과한다. 과목 인덱스가
            # 없는 경우(전역 검색으로 빠진 경우)를 위한 안전망으로 남겨둔다.
            filtered = _filter_by_subject(candidates, subject)

            query_words = set(re.sub(r'\s+', ' ', query).split())
            def _kw_score(doc: str) -> int:
                return sum(1 for w in query_words if len(w) > 1 and w in doc)
            filtered = sorted(filtered, key=_kw_score, reverse=True)

            cleaned = []
            seen = set()
            for doc in filtered:
                c = _extract_content(doc)
                if len(c) > 30 and c not in seen:
                    cleaned.append(c)
                    seen.add(c)
                if len(cleaned) >= k:
                    break

            if cleaned:
                return cleaned
        except Exception as e:
            return [f"검색 오류: {str(e)}"]

    # ── 3. ConceptMap 과목 샘플 (개념 미입력 등 완전 미스 시 fallback) ──────────
    overview = [v for keys, v in _CONCEPT_MAP.items() if keys[0] == subject]
    return overview[:k]


# ── POST /api/rag/search ───────────────────────────────────────────────────────
@router.post("/search")
async def rag_search(req: dict):
    query = req.get("question", "")
    results = vector_search(query, k=3)
    context = "\n\n".join(results)
    return {"context": context}

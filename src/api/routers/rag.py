import pickle
from pathlib import Path

import faiss
import numpy as np
from fastapi import APIRouter

router = APIRouter()

# ── FAISS 인덱스 로드 ──────────────────────────────────────────────────────────
_FAISS_DIR = Path("src/api/rag_db")
_index: faiss.Index | None = None
_doc_texts: list[str] = []


def _extract_docs(a, b) -> list[str]:
    """langchain FAISS pkl 튜플에서 텍스트 추출. 두 원소의 순서를 자동으로 판별한다."""
    # index_to_docstore_id 는 dict, docstore 는 InMemoryDocstore
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

# ── 임베딩 모델 로드 ───────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer as _ST

    _embed_model = _ST("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def _encode(text: str) -> np.ndarray:
        return _embed_model.encode([text])

    print("[Embed] sentence-transformers 임베딩 로드")

except ImportError:
    from transformers import AutoModel, AutoTokenizer
    import torch

    _MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    _hf_model = AutoModel.from_pretrained(_MODEL_NAME)

    def _encode(text: str) -> np.ndarray:
        inputs = _tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        with torch.no_grad():
            out = _hf_model(**inputs)
        return out.last_hidden_state[:, 0, :].numpy()

    print("[Embed] transformers 임베딩 로드 (sentence-transformers 없음)")


# ── 공용 벡터 검색 함수 ─────────────────────────────────────────────────────────
def vector_search(query: str, k: int = 3) -> list[str]:
    if _index is None or not _doc_texts:
        return ["검색 가능한 데이터가 없습니다."]
    try:
        vec = _encode(query).astype(np.float32)
        faiss.normalize_L2(vec)
        _, indices = _index.search(vec, k)
        return [
            _doc_texts[i]
            for i in indices[0]
            if 0 <= i < len(_doc_texts)
        ]
    except Exception as e:
        return [f"검색 오류: {str(e)}"]


# ── POST /api/rag/search ───────────────────────────────────────────────────────
# MathAiService 에서 호출 — 단일 context 문자열 반환
@router.post("/search")
async def rag_search(req: dict):
    query = req.get("question", "")
    results = vector_search(query, k=3)
    context = "\n\n".join(results)
    return {"context": context}

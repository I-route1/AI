"""개념 설명 프롬프트에 검색 자료를 근거로 붙인다.

모델이 제 지식으로만 개념을 설명하면 사실 오류가 잦다. 한국사 8개 개념에서
베이스 Qwen은 약 1.5개, Ollama는 약 3.5개만 맞았다(MODELS.md). 같은 리포트에는
이미 검색한 자료(ConceptMap·교과서 조각)가 들어가는데 모델은 그걸 받지 않았다.

서빙(main.py, counseling.py)과 평가 스크립트가 같은 문자열을 쓰도록 여기 둔다.

**ConceptMap에서 온 자료만 근거로 쓴다.** 31개 개념으로 재보니(test_grounded_explain_eval.py)
사람이 쓴 ConceptMap 노트를 붙이면 맞은 수가 늘었지만(한국사 제외 12→13/15, 한국사
Qwen 1.5→6/8, Ollama 3.5→6.5/8), FAISS 교과서 조각(40~120자 한 문장)을 붙이면
오히려 줄고(6.5→5/8) 중국어가 섞여 나왔다(0→4/8). 조각은 개념을 정의하지 못하는
문장이 대부분이라 모델이 그 문장에 끌려간다. 한국사 FAISS 문서는 ConceptMap을
잘라 적재한 것이라 이 판별에 그대로 걸린다.
"""
import re

from src.api.concept_map import CONCEPT_MAP

MAX_DOCS = 3
# ConceptMap 한국사 항목이 길다(중앙값 약 160토큰). 너무 짧게 자르면 핵심이 잘린다.
MAX_DOC_CHARS = 600

GROUNDING_RULES = (
    "위 참고 자료를 근거로 답하세요. 자료와 어긋나는 내용은 쓰지 마세요. "
    "자료에 없는 연도·인물·수치·정의는 확실하지 않으면 쓰지 마세요. "
    "참고 자료가 이 개념과 관련이 없으면 자료는 무시하세요."
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text)


_CURATED = [_norm(v) for v in CONCEPT_MAP.values()]


def is_curated(doc: str) -> bool:
    """ConceptMap 항목 자체이거나 그 일부(한국사 FAISS 청크)인가."""
    n = _norm(doc)
    return bool(n) and any(n in e for e in _CURATED)


def usable_docs(docs: list[str] | None) -> list[str]:
    """근거로 쓸 자료만 남긴다 — ConceptMap 출처만(위 설명 참고)."""
    out = []
    for d in docs or []:
        d = (d or "").strip()
        if not is_curated(d):
            continue
        out.append(d[:MAX_DOC_CHARS])
        if len(out) >= MAX_DOCS:
            break
    return out


def context_block(docs: list[str] | None) -> str:
    """프롬프트 앞에 붙일 참고 자료 블록. 쓸 자료가 없으면 빈 문자열."""
    docs = usable_docs(docs)
    if not docs:
        return ""
    body = "\n".join(f"- {d}" for d in docs)
    return f"[참고 자료]\n{body}\n\n{GROUNDING_RULES}\n\n"


def concept_user_message(concept: str, docs: list[str] | None) -> str:
    """_concept_explain()의 user 메시지."""
    return f"{context_block(docs)}질문: '{concept}' 개념의 핵심 포인트를 학생에게 설명해주세요."

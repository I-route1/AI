"""FAISS 문서(약 70만 개)에 한 번이라도 나온 한글 음절을 src/api/hangul_seen.txt로 저장한다.

script_guard.RareHangulBlocker가 이 목록 밖의 음절("귥한", "줿게")을 생성하지 못하게 막는다.
교과 문서 70만 개에 한 번도 안 나온 음절이면 학생용 개념 설명에 필요할 일이 없다고 본다.
문서를 크게 다시 적재했을 때만 다시 돌리면 된다.

    python scripts/build_hangul_seen.py
"""
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "src/api/rag_db/index.pkl", "rb") as f:
    a, b = pickle.load(f)
store = getattr(a, "_dict", None) or getattr(b, "_dict", {})

seen: set[str] = set()
for doc in store.values():
    seen.update(ch for ch in doc.page_content if "가" <= ch <= "힣")
# ConceptMap은 FAISS에 들어 있지만, 적재 전에 고친 항목도 빠지지 않게 직접 더한다
seen.update(ch for ch in (ROOT / "src/api/concept_map.py").read_text(encoding="utf-8") if "가" <= ch <= "힣")

# 문서에는 없지만 표준어에서 쓰는 음절. 문서 기준으로만 막으면 막히는 것 중 손으로 골랐다
# (숱하게, 셌다, 뵀다·뵌, 엽전 한 닢, 벋다, 톺아보다, 뙈기, 엊그제, 낢·늚).
seen.update("숱셌뵀뵌닢벋톺뙈엊낢늚")

out = ROOT / "src/api/hangul_seen.txt"
out.write_text("".join(sorted(seen)) + "\n", encoding="utf-8")
print(f"문서 {len(store):,}개, 음절 {len(seen):,}개 → {out}")

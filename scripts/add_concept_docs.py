"""
수학 핵심 개념 문서를 기존 FAISS DB에 추가하는 스크립트.
실행: python scripts/add_concept_docs.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import uuid
import numpy as np
import faiss
from pathlib import Path

DB_DIR = Path("src/api/rag_db")

# ── 추가할 개념 문서 ──────────────────────────────────────────────────────────
NEW_DOCS = [
    # 수열
    """[대상]고등학교 수학 (폴더: TL_11.고등학교_03.수학_01.텍스트)
[교육과정 성취기준] [12수학02-01] 등차수열의 뜻을 알고 일반항과 합을 구할 수 있다.
[학습 지문] 수열이란 일정한 규칙에 따라 나열된 수의 열입니다. 등차수열은 이웃하는 두 항의 차(공차 d)가 일정한 수열입니다. 첫째항 a, 공차 d일 때 일반항 an = a + (n-1)d이고, 합 Sn = n/2 × (2a + (n-1)d)입니다.
[관련 Q&A] 질문: 등차수열의 일반항 공식은? / 답변: 첫째항 a, 공차 d일 때 an = a + (n-1)d입니다. 예를 들어 2, 5, 8, 11은 공차 3인 등차수열로 an = 3n-1입니다.""",

    """[대상]고등학교 수학 (폴더: TL_11.고등학교_03.수학_01.텍스트)
[교육과정 성취기준] [12수학02-02] 등차수열의 합 공식을 이해하고 활용한다.
[학습 지문] 등차수열의 합 Sn = n/2 × (첫째항 + 끝항) = n/2 × (2a + (n-1)d)입니다. 수열의 합 Sn과 일반항 an의 관계: n≥2일 때 an = Sn - S(n-1), n=1일 때 a1 = S1입니다.
[관련 Q&A] 질문: 수열의 합 Sn이 주어질 때 일반항을 구하는 방법은? / 답변: n≥2이면 an = Sn - S(n-1), n=1이면 a1 = S1으로 구합니다.""",

    """[대상]고등학교 수학 (폴더: TL_11.고등학교_03.수학_01.텍스트)
[교육과정 성취기준] [12수학02-03] 등비수열의 뜻을 알고 일반항과 합을 구할 수 있다.
[학습 지문] 등비수열은 이웃하는 두 항의 비(공비 r)가 일정한 수열입니다. 첫째항 a, 공비 r일 때 일반항 an = a × r^(n-1)이고, 합 Sn = a(r^n - 1)/(r-1) (r≠1), r=1이면 Sn = na입니다.
[관련 Q&A] 질문: 등비수열의 합 공식은? / 답변: 첫째항 a, 공비 r일 때 Sn = a(r^n-1)/(r-1)입니다. r=1이면 Sn = na입니다.""",

    # 이차방정식
    """[대상]고등학교 수학 (폴더: TL_11.고등학교_03.수학_01.텍스트)
[교육과정 성취기준] [10수01-03] 이차방정식의 풀이 방법을 이해하고 활용한다.
[학습 지문] 이차방정식 ax²+bx+c=0의 풀이: 인수분해, 완전제곱식, 근의 공식을 사용합니다. 근의 공식 x = (-b ± √(b²-4ac)) / 2a. 판별식 D = b²-4ac로 근의 종류를 판별합니다. D>0: 서로 다른 두 실근, D=0: 중근, D<0: 허근.
[관련 Q&A] 질문: 이차방정식의 근의 공식은? / 답변: x = (-b ± √(b²-4ac)) / 2a입니다. D=b²-4ac로 근의 개수를 판별합니다.""",

    """[대상]고등학교 수학 (폴더: TL_11.고등학교_03.수학_01.텍스트)
[교육과정 성취기준] [10수01-04] 이차방정식의 근과 계수의 관계를 이해한다.
[학습 지문] 이차방정식 ax²+bx+c=0의 두 근을 α, β라 하면 근과 계수의 관계: α+β = -b/a, αβ = c/a. 이를 이용해 두 근의 합과 곱을 쉽게 구할 수 있습니다.
[관련 Q&A] 질문: 이차방정식의 두 근의 합과 곱은? / 답변: ax²+bx+c=0의 두 근 α, β에 대해 α+β = -b/a, αβ = c/a입니다.""",

    # 함수
    """[대상]고등학교 수학 (폴더: TL_11.고등학교_03.수학_01.텍스트)
[교육과정 성취기준] [10수01-05] 이차함수의 그래프와 성질을 이해한다.
[학습 지문] 이차함수 y=ax²+bx+c의 꼭짓점: (-b/2a, c-b²/4a). 표준형 y=a(x-p)²+q에서 꼭짓점은 (p, q)이고 축은 x=p입니다. a>0이면 아래로 볼록, a<0이면 위로 볼록.
[관련 Q&A] 질문: 이차함수의 꼭짓점 좌표는? / 답변: y=ax²+bx+c에서 꼭짓점 x좌표는 -b/(2a)이고 y좌표는 c-b²/(4a)입니다.""",

    # 확률과 통계
    """[대상]고등학교 수학 (폴더: TL_11.고등학교_03.수학_01.텍스트)
[교육과정 성취기준] [12수학03-01] 확률의 뜻과 기본 성질을 이해한다.
[학습 지문] 수학적 확률 P(A) = (사건 A의 경우의 수)/(전체 경우의 수). 덧셈정리: P(A∪B) = P(A)+P(B)-P(A∩B). 여사건 확률: P(A') = 1-P(A). 조건부 확률: P(A|B) = P(A∩B)/P(B).
[관련 Q&A] 질문: 여사건의 확률이란? / 답변: P(A') = 1 - P(A)로, 사건 A가 일어나지 않을 확률입니다.""",

    """[대상]고등학교 수학 (폴더: TL_11.고등학교_03.수학_01.텍스트)
[교육과정 성취기준] [12수학03-02] 순열과 조합의 차이를 이해하고 계산한다.
[학습 지문] 순열 nPr = n!/(n-r)! : 순서가 있는 선택. 조합 nCr = n!/(r!(n-r)!) : 순서 없는 선택. 순열과 조합의 관계: nPr = r! × nCr.
[관련 Q&A] 질문: 순열과 조합의 차이는? / 답변: 순열은 순서가 중요한 선택(nPr), 조합은 순서 무관한 선택(nCr)입니다. 5명 중 2명을 뽑을 때 순열은 20가지, 조합은 10가지입니다.""",

    # 미적분
    """[대상]고등학교 수학 (폴더: TL_11.고등학교_03.수학_01.텍스트)
[교육과정 성취기준] [12수학04-01] 함수의 극한과 연속의 개념을 이해한다.
[학습 지문] 미분의 기본: 함수 f(x)의 도함수 f'(x) = lim(h→0) [f(x+h)-f(x)]/h. 기본 미분 공식: (x^n)' = nx^(n-1), (e^x)' = e^x, (ln x)' = 1/x, (sin x)' = cos x.
[관련 Q&A] 질문: x^n의 미분은? / 답변: (x^n)' = nx^(n-1)입니다. 예를 들어 x³의 미분은 3x²입니다.""",

    # 기하
    """[대상]고등학교 수학 (폴더: TL_11.고등학교_03.수학_01.텍스트)
[교육과정 성취기준] [10수02-01] 삼각함수의 뜻과 기본 성질을 이해한다.
[학습 지문] 삼각함수: sin θ = 대변/빗변, cos θ = 밑변/빗변, tan θ = 대변/밑변. 피타고라스 정리: sin²θ + cos²θ = 1. 삼각함수 값의 범위: -1 ≤ sin θ ≤ 1, -1 ≤ cos θ ≤ 1.
[관련 Q&A] 질문: sin²θ + cos²θ = 1이 성립하는 이유는? / 답변: 단위원에서 점 (cosθ, sinθ)은 원 위의 점이므로 cos²θ + sin²θ = 1²이 항상 성립합니다.""",

    # 연립방정식
    """[대상]중학교 수학 (폴더: TL_08.중학교_03.수학_01.텍스트)
[교육과정 성취기준] [9수02-03] 연립일차방정식의 풀이 방법을 이해한다.
[학습 지문] 연립방정식 풀이 방법: 가감법(두 방정식을 더하거나 빼서 미지수 소거), 대입법(한 방정식을 다른 방정식에 대입). 두 직선의 교점이 연립방정식의 해입니다.
[관련 Q&A] 질문: 연립방정식의 가감법이란? / 답변: 두 방정식을 더하거나 빼서 미지수 하나를 없애는 방법입니다. x+y=5, x-y=1 → 더하면 2x=6, x=3, y=2입니다.""",
]


# ── 임베딩 모델 로드 ───────────────────────────────────────────────────────────
print("임베딩 모델 로드 중...")
try:
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    def encode(text: str) -> np.ndarray:
        return embed_model.encode([text]).astype(np.float32)
except ImportError:
    from transformers import AutoModel, AutoTokenizer
    import torch
    _tok = AutoTokenizer.from_pretrained("jhgan/ko-sroberta-multitask")
    _mdl = AutoModel.from_pretrained("jhgan/ko-sroberta-multitask")
    def encode(text: str) -> np.ndarray:
        inputs = _tok(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            out = _mdl(**inputs)
        return out.last_hidden_state[:, 0, :].numpy().astype(np.float32)
print("임베딩 모델 로드 완료")


# ── FAISS 인덱스 및 pkl 로드 ──────────────────────────────────────────────────
print(f"FAISS DB 로드 중: {DB_DIR}")
index = faiss.read_index(str(DB_DIR / "index.faiss"))
print(f"기존 벡터 수: {index.ntotal}")

with open(DB_DIR / "index.pkl", "rb") as f:
    raw = pickle.load(f)

if not (isinstance(raw, tuple) and len(raw) == 2):
    print("지원하지 않는 pkl 형식입니다.")
    sys.exit(1)

a, b = raw
if isinstance(a, dict):
    index_to_id, docstore = a, b
else:
    index_to_id, docstore = b, a

store = getattr(docstore, "_dict", None)
if store is None:
    print("docstore._dict를 찾을 수 없습니다.")
    sys.exit(1)


# ── 새 문서 추가 ──────────────────────────────────────────────────────────────
class _Doc:
    def __init__(self, content: str):
        self.page_content = content

added = 0
next_idx = index.ntotal

for text in NEW_DOCS:
    doc_id = str(uuid.uuid4())
    index_to_id[next_idx] = doc_id
    store[doc_id] = _Doc(text)

    vec = encode(text)
    faiss.normalize_L2(vec)
    index.add(vec)

    next_idx += 1
    added += 1
    print(f"  추가: {text.split(chr(10))[2][:60]}...")

print(f"\n{added}개 문서 추가 완료. 총 벡터: {index.ntotal}")


# ── 저장 ─────────────────────────────────────────────────────────────────────
faiss.write_index(index, str(DB_DIR / "index.faiss"))
with open(DB_DIR / "index.pkl", "wb") as f:
    pickle.dump((index_to_id, docstore), f)

print("DB 저장 완료. AI 서버를 재시작해주세요.")

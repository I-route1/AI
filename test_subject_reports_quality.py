"""
전 과목 메타인지 분석 리포트(/api/ai/report/*) 품질 테스트.
실제 FAISS 인덱스를 in-process로 로드해 RAG 결과 relevance / cross-subject
contamination / 과목별 전략 텍스트 정합성을 점검한다.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import counseling
from src.api.routers.rag import _SUBJECT_KEYWORDS, _SUBJECT_PATH_PATTERNS

app = FastAPI()
app.include_router(counseling.router, prefix="/api/ai")
client = TestClient(app)

SUBJECTS = ["수학", "영어", "국어", "과학", "사회", "한국사"]
TIERS = [("상위권", 90), ("중위권", 65), ("하위권", 35)]

CONCEPT_BY_SUBJECT = {
    "수학":   "수열 등차수열",
    "영어":   "관계대명사 구문 독해",
    "국어":   "비문학 지문 구조",
    "과학":   "뉴턴 운동 법칙",
    "사회":   "수요와 공급",
    "한국사": "조선 후기 실학",
}

ENDPOINT_BY_SUBJECT = {
    "수학": "/api/ai/report/math",
    "국어": "/api/ai/report/writing",
}

issues = []
sep = lambda t: print(f"\n{'='*70}\n[{t}]\n{'='*70}")


_RAG_HEADERS = ["[관련 학습 자료 검색 결과]"]


def rag_lines(text: str) -> list[str]:
    for header in _RAG_HEADERS:
        if header in text:
            tail = text.split(header, 1)[1]
            return [l.strip() for l in tail.split("\n") if l.strip().startswith("•")]
    return []


# 실제로는 정상 서술에서도 자연스럽게 섞이는 범용어 (오탐 방지용 제외 목록)
_BENIGN_OVERLAP = {"벡터", "과학", "정치", "문화", "역사", "지리", "확률", "독해"}


def other_subject_keywords(subject: str) -> list[str]:
    kws = []
    for s, ks in _SUBJECT_KEYWORDS.items():
        if s != subject:
            kws.extend(ks)
    return [k for k in kws if len(k) >= 2 and k not in _BENIGN_OVERLAP]


sep("전 과목 리포트 생성 + 품질 점검")
for subject in SUBJECTS:
    endpoint = ENDPOINT_BY_SUBJECT.get(subject, f"/api/ai/report/{subject}")
    for tier_label, pct in TIERS:
        payload = {
            "studentId": "Q-TEST",
            "currentKoreanGrade": pct,
            "studyTime": 2,
            "recommendContext": CONCEPT_BY_SUBJECT[subject],
            "studentNote": f"{subject} 학습 특성 테스트",
            "instructorFeedback": "",
        }
        r = client.post(endpoint, json=payload)
        tag = f"{subject}/{tier_label}({pct}%)"

        if r.status_code != 200:
            issues.append(f"[{tag}] HTTP {r.status_code}: {r.text[:200]}")
            print(f"  ❌ {tag}: HTTP {r.status_code}")
            continue

        data = r.json()
        career = data.get("careerAnalysis", "")
        guide  = data.get("learningGuide", "")
        title  = data.get("title", "")

        # 1) 필수 필드 존재
        if not career or not guide or not title:
            issues.append(f"[{tag}] 빈 필드 존재: title={bool(title)} career={bool(career)} guide={bool(guide)}")

        # 2) 과목명이 타이틀/분석에 반영되는지 (writing/premium 제외 — 고유 타이틀 사용)
        if subject not in ENDPOINT_BY_SUBJECT and subject not in title:
            issues.append(f"[{tag}] 타이틀에 과목명 누락: {title}")

        # 3) RAG 결과 확인
        lines = rag_lines(career)
        if not lines:
            print(f"  ⚠️  {tag}: RAG 검색 결과 없음 (개념='{CONCEPT_BY_SUBJECT[subject]}')")
        else:
            print(f"  ✅ {tag}: RAG {len(lines)}건 — 예) {lines[0][:80]}")

        # 4) Cross-subject contamination: 다른 과목 키워드가 RAG 결과에 섞였는지
        other_kws = other_subject_keywords(subject)
        for line in lines:
            hit = [k for k in other_kws if k in line]
            if hit:
                issues.append(f"[{tag}] RAG 결과에 타 과목 키워드 혼입: {hit} in '{line[:100]}'")

        # 5) 성적 구간별 전략 문구가 실제로 달라지는지 (동일 과목 내 상/중/하 텍스트 다양성)
        data.setdefault("_guide_by_tier", {})

sep("성적 구간별 학습전략 다양성 체크 (같은 과목, 다른 백분위 → 다른 문구인지)")
for subject in SUBJECTS:
    endpoint = ENDPOINT_BY_SUBJECT.get(subject, f"/api/ai/report/{subject}")
    guides = {}
    for tier_label, pct in TIERS:
        payload = {
            "studentId": "Q-TEST", "currentKoreanGrade": pct, "studyTime": 2,
            "recommendContext": CONCEPT_BY_SUBJECT[subject],
        }
        r = client.post(endpoint, json=payload)
        guides[tier_label] = r.json().get("learningGuide", "")
    unique = len(set(guides.values()))
    if unique < 2:
        issues.append(f"[{subject}] 상/중/하 학습전략이 사실상 동일함 (unique={unique}/3)")
        print(f"  ⚠️  {subject}: 전략 문구 다양성 부족 (unique={unique}/3)")
    else:
        print(f"  ✅ {subject}: 전략 문구 {unique}/3 종류로 분화됨")

sep("잘못된 과목명 / 엣지 케이스")
r = client.post("/api/ai/report/존재안함", json={"studentId": "X"})
if r.status_code != 404:
    issues.append(f"[잘못된 과목] 404가 아닌 {r.status_code} 반환")
else:
    print("  ✅ 지원하지 않는 과목명 → 404 정상")

r = client.post("/api/ai/report/영어", json={"studentId": "X"})  # 필수값 전부 생략
print(f"  필수값 생략 시: HTTP {r.status_code}")
if r.status_code != 200:
    issues.append(f"[필드 생략] 기본값 폴백 실패: HTTP {r.status_code} {r.text[:150]}")

sep("결과 요약")
if issues:
    print(f"❌ {len(issues)}건의 이슈 발견:\n")
    for i in issues:
        print(f"  - {i}")
    sys.exit(1)
else:
    print("✅ 모든 품질 체크 통과")

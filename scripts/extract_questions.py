"""
RAG DB에서 중학교/고등학교 질문 데이터를 추출해 questions_seed.json 생성
"""
import pickle, sys, re, json
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

SUBJECT_MAP = {
    '국어': 1,
    '수학': 2,
    '영어': 3,
    '과학': 4,
    '사회': 5,
    '사회문화': 5,
    '한국사': 5,
}

def grade_to_difficulty(target: str) -> int:
    if '중학교 1' in target: return 2
    if '중학교 2' in target: return 2
    if '중학교 3' in target: return 3
    if '고등학교' in target: return 4
    return 3

def parse_subject_id(target: str):
    for name, sid in SUBJECT_MAP.items():
        if name in target:
            return sid
    return None

def parse_concept_tag(content: str) -> str:
    m = re.search(r'\[교육과정 성취기준\]\s*\[([^\]]+)\]\s*(.+?)(?:\n|$)', content)
    if m:
        raw = m.group(2).strip()
        # 최대 20자
        return raw[:20]
    m2 = re.search(r'\[학습 지문\]\s*(.+?)(?:\n|$)', content)
    if m2:
        return m2.group(1).strip()[:20]
    return '일반'

def parse_qa(content: str) -> str:
    m = re.search(r'\[관련 Q&A\]\s*질문:\s*(.+?)\s*/\s*답변:\s*(.+?)(?:\n|$)', content)
    if m:
        q = m.group(1).strip()
        a = m.group(2).strip()
        return f"질문: {q}\n답변: {a}"
    return None

print("RAG DB 로딩 중...", flush=True)
db_path = Path(__file__).parent.parent / 'src' / 'api' / 'rag_db' / 'index.pkl'
with open(db_path, 'rb') as f:
    raw = pickle.load(f)

if isinstance(raw[0], dict):
    index_to_id, docstore = raw[0], raw[1]
else:
    index_to_id, docstore = raw[1], raw[0]

store = getattr(docstore, '_dict', {})
print(f"총 {len(store):,}개 문서 처리 시작...", flush=True)

# (subjectId, conceptTag) → 최대 5개
buckets = defaultdict(list)
MAX_PER_BUCKET = 5
skipped = 0

for i, (k, doc) in enumerate(store.items()):
    if i % 100000 == 0:
        print(f"  {i:,} 처리 중...", flush=True)

    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)

    # 중학교/고등학교만
    m = re.search(r'\[대상\]([^\n(]+)', content)
    if not m:
        skipped += 1
        continue
    target = m.group(1).strip()
    if '중학교' not in target and '고등학교' not in target:
        skipped += 1
        continue

    subject_id = parse_subject_id(target)
    if subject_id is None:
        skipped += 1
        continue

    qa = parse_qa(content)
    if not qa:
        skipped += 1
        continue

    concept_tag = parse_concept_tag(content)
    difficulty = grade_to_difficulty(target)
    bucket_key = (subject_id, concept_tag)

    if len(buckets[bucket_key]) < MAX_PER_BUCKET:
        buckets[bucket_key].append({
            'subjectId': subject_id,
            'conceptTag': concept_tag,
            'content': qa,
            'difficulty': difficulty
        })

questions = [q for qs in buckets.values() for q in qs]
print(f"\n추출 완료: {len(questions)}개 질문 ({len(buckets)}개 개념태그)")

out_path = Path(__file__).parent.parent / 'src' / 'api' / 'questions_seed.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

print(f"저장 완료: {out_path}")

# 과목별 통계
from collections import Counter
by_subject = Counter(q['subjectId'] for q in questions)
subject_names = {1:'국어', 2:'수학', 3:'영어', 4:'과학', 5:'사회'}
for sid, cnt in sorted(by_subject.items()):
    print(f"  {subject_names[sid]}: {cnt}개")

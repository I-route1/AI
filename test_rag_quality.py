import urllib.request, json, sys

BASE = "http://localhost:8082"

def post(path, data):
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}", data=body,
        headers={"Content-Type": "application/json; charset=utf-8"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

def sep(title):
    print("\n" + "=" * 60)
    print(f"[{title}]")
    print("=" * 60)

sep("1. RAG /api/rag/search - 기본 검색")
for q in ["suryeol concept", "yeongeo grammar", "won-gi-dung bupi"]:
    pass

for q in ["수열 등차수열 개념", "영어 독해 어법", "원기둥 부피"]:
    r = post("/api/rag/search", {"question": q})
    ctx = r.get("context", "")
    print(f"\n  query: {q}")
    print(f"  result: {ctx[:150].replace(chr(10), ' ')}")

sep("2. /api/ai/report/math - 수학 리포트 RAG")
r = post("/api/ai/report/math", {
    "studentId": "14",
    "currentKoreanGrade": 70,
    "studyTime": 2,
    "recommendContext": "suryeol gupsu",
    "studentNote": "gyesan silsu",
    "instructorFeedback": ""
})
career = r.get("careerAnalysis", "")
print("  careerAnalysis RAG lines:")
for line in career.split("\n"):
    if line.strip().startswith("*") or line.strip().startswith("-") or "rag" in line.lower() or "jaryо" in line.lower():
        print(f"    {line[:120]}")
print(f"  learningGuide sample: {r.get('learningGuide','')[:150].replace(chr(10),' ')}")

sep("3. Cross-subject contamination check")
r = post("/api/ai/report/writing", {
    "studentId": "14",
    "currentKoreanGrade": 60,
    "studyTime": 1,
    "recommendContext": "gukeo munhak",
    "studentNote": "dokhaeryeok bujok",
    "instructorFeedback": ""
})
career = r.get("careerAnalysis", "")

math_words = ["wongi-dung", "suryeol", "sunryeol", "bangjeongsik", "mijeokbun"]
korean_words = ["gukeo", "munhak", "dokhaeryeok", "siseol"]

has_rag = "[" in career and ("RAG" in career or "jaryо" in career or "•" in career)
print(f"  RAG section found: {'yes' if has_rag else 'no (empty or no match)'}")
print(f"  careerAnalysis snippet:")
for line in career.split("\n"):
    if line.strip():
        print(f"    {line[:120]}")
    if line.count("\n") > 15:
        break

print("\nDone.")

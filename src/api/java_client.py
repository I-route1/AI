import os
import httpx
from dotenv import load_dotenv

load_dotenv()

_BASE_URL = os.getenv("JAVA_BACKEND_URL", "http://localhost:8080")
JAVA_BACKEND_URL = f"{_BASE_URL.rstrip('/')}/api/wrong-answer/ai-pipeline"

# ai-pipeline은 이 서버 전용으로 열어 둔 내부 API라 인증 없이 permitAll이었다 — 주소를
# 아는 사람이면 누구나 아무 학생의 취약 개념을 조회할 수 있었다. main.py가 들어오는
# 요청을 막는 데 쓰는 것과 같은 값(AI_SERVER_KEY)을 나가는 요청에도 그대로 쓴다
# (Backend의 AiServerKeyFilter가 같은 환경변수 이름으로 검사한다). 비어 있으면 헤더를
# 안 보내고, Backend도 비어 있으면 예전처럼 검사하지 않는다.
_AI_SERVER_KEY = os.getenv("AI_SERVER_KEY", "").strip()


def get_student_weakness_from_java(student_id: str, subject: str):
    try:
        headers = {"X-AI-Key": _AI_SERVER_KEY} if _AI_SERVER_KEY else {}
        with httpx.Client() as client:
            response = client.get(
                JAVA_BACKEND_URL,
                params={"studentId": student_id, "subject": subject},
                headers=headers,
                timeout=5.0,
            )
            if response.status_code == 200:
                return response.json()
            print(f"❌ 자바 서버 응답 에러: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ 자바 서버 연결 실패: {e}")
        return []

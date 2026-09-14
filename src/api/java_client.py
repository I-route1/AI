import os
import httpx
from dotenv import load_dotenv

load_dotenv()

_BASE_URL = os.getenv("JAVA_BACKEND_URL", "http://localhost:8080")
JAVA_BACKEND_URL = f"{_BASE_URL.rstrip('/')}/api/wrong-answer/ai-pipeline"


def get_student_weakness_from_java(student_id: str, subject: str):
    try:
        with httpx.Client() as client:
            response = client.get(
                JAVA_BACKEND_URL,
                params={"studentId": student_id, "subject": subject},
                timeout=5.0,
            )
            if response.status_code == 200:
                return response.json()
            print(f"❌ 자바 서버 응답 에러: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ 자바 서버 연결 실패: {e}")
        return []

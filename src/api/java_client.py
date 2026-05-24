import httpx

JAVA_BACKEND_URL = "http://localhost:8080/api/wrong-answer/ai-pipeline"

def get_student_weakness_from_java(student_id: str, subject: str):
    """
    자바 스프링 백엔드에서 특정 학생의 취약 개념 리스트를 가져옵니다.
    """
    params = {
        "studentId": student_id,
        "subject": subject
    }
    try:
        # 동기 호출 방식 예시 (FastAPI 엔드포인트 내부에서 사용)
        with httpx.Client() as client:
            response = client.get(JAVA_BACKEND_URL, params=params, timeout=5.0)
            if response.status_code == 200:
                return response.json() # WrongAnswer 리스트 반환
            else:
                print(f"❌ 자바 서버 응답 에러: {response.status_code}")
                return []
    except Exception as e:
        print(f"❌ 자바 서버 연결 실패: {str(e)}")
        return []
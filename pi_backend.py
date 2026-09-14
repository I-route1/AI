"""라즈베리파이 모듈(ocr/, rpi/)이 Java 백엔드로 측정 결과를 올릴 때 쓰는 공용 클라이언트.

의존성을 일부러 stdlib으로만 맞췄다. Pi5에는 torch CPU 빌드와 pillow 정도만
설치하는 것이 전제라, requests/httpx를 추가로 요구하지 않기 위함이다.

설정은 환경변수 또는 같은 디렉터리의 .env에서 읽는다:
    BACKEND_URL        예) https://d22mlgf6je9oud.cloudfront.net
    BACKEND_TOKEN      JWT accessToken을 직접 주는 경우
    BACKEND_USERNAME   토큰 대신 로그인으로 발급받는 경우
    BACKEND_PASSWORD
    STUDENT_ID         업로드 대상 학생 ID (CLI 인자로 덮어쓸 수 있음)

백엔드의 /api/wrong-answer/record 와 /api/activities 는 인증이 걸려 있어
토큰 없이는 401이 난다. SecurityConfig에서 permitAll인 것은
/api/wrong-answer/ai-pipeline(AI 서버 전용) 뿐이다.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_TIMEOUT = 10.0
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # 초, 시도마다 곱해짐


class BackendError(RuntimeError):
    """업로드 실패. 호출부는 측정 루프를 멈추지 말고 로그만 남기는 것을 권장."""


def _load_dotenv(path: Path) -> None:
    """python-dotenv 없이 .env를 환경변수로 읽어들인다. 이미 설정된 값은 덮지 않는다."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(Path(__file__).parent / ".env")


class BackendClient:
    def __init__(self, base_url: str | None = None, token: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT):
        self.base_url = (base_url or os.getenv("BACKEND_URL") or "").rstrip("/")
        if not self.base_url:
            raise BackendError("BACKEND_URL이 설정되지 않았습니다 (.env 또는 환경변수).")
        self.timeout = timeout
        self._token = token or os.getenv("BACKEND_TOKEN") or None

    # ── 인증 ──────────────────────────────────────────────────────────────────
    def _ensure_token(self) -> str:
        if self._token:
            return self._token
        username = os.getenv("BACKEND_USERNAME")
        password = os.getenv("BACKEND_PASSWORD")
        if not (username and password):
            raise BackendError(
                "BACKEND_TOKEN이 없고 BACKEND_USERNAME/BACKEND_PASSWORD도 없습니다. "
                "둘 중 하나를 .env에 설정하세요."
            )
        body = json.dumps({"username": username, "password": password}).encode("utf-8")
        res = self._request("POST", "/api/auth/login", body=body, authed=False)
        token = (res or {}).get("accessToken")
        if not token:
            raise BackendError(f"로그인 응답에 accessToken이 없습니다: {res}")
        self._token = token
        return token

    # ── HTTP ──────────────────────────────────────────────────────────────────
    def _request(self, method: str, path: str, *, params: dict | None = None,
                 body: bytes | None = None, authed: bool = True) -> dict | list | None:
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params, encoding="utf-8")

        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
        if authed:
            headers["Authorization"] = f"Bearer {self._ensure_token()}"

        last_err: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8").strip()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                # 4xx는 재시도해도 같은 결과 — 인증 만료(401)만 토큰을 버리고 한 번 더 시도
                if e.code == 401 and authed and attempt == 1 and not os.getenv("BACKEND_TOKEN"):
                    self._token = None
                    last_err = BackendError(f"401 인증 실패, 토큰 재발급 후 재시도: {detail}")
                    continue
                if 400 <= e.code < 500:
                    raise BackendError(f"{method} {path} -> HTTP {e.code}: {detail}") from e
                last_err = BackendError(f"{method} {path} -> HTTP {e.code}: {detail}")
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = BackendError(f"{method} {path} 연결 실패: {e}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)

        raise BackendError(f"{MAX_RETRIES}회 재시도 후 실패: {last_err}")

    # ── 업로드 API ────────────────────────────────────────────────────────────
    def record_wrong_answer(self, student_id: int, subject: str, question_id: str,
                            concept_tag: str, error_type: str | None = None) -> dict | None:
        """오답 1건 기록. POST /api/wrong-answer/record (쿼리 파라미터 방식).

        error_type은 백엔드 ErrorType enum 값이어야 한다:
        CONCEPT_GAP / CALCULATION_ERROR / CARELESS_MISTAKE / MEMORIZATION_GAP
        """
        params = {
            "studentId": student_id,
            "subject": subject,
            "questionId": question_id,
            "conceptTag": concept_tag,
        }
        if error_type:
            params["errorType"] = error_type
        return self._request("POST", "/api/wrong-answer/record", params=params)

    def save_learning_activity(self, student_id: int, subject: str, study_date: str,
                               study_start_time: str, duration_minutes: int,
                               understanding_score: int, concentration_score: int,
                               instructor_feedback: str | None = None) -> dict | None:
        """학습 활동 1건 기록. POST /api/activities (JSON 바디).

        study_date: "YYYY-MM-DD", study_start_time: "HH:MM:SS"
        understanding_score / concentration_score: 백엔드가 int로 받는다.
        """
        payload = {
            "studentId": student_id,
            "subject": subject,
            "studyDate": study_date,
            "studyStartTime": study_start_time,
            "studyDurationMinutes": duration_minutes,
            "understandingScore": understanding_score,
            "concentrationScore": concentration_score,
            "instructorFeedback": instructor_feedback or "",
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self._request("POST", "/api/activities", body=body)

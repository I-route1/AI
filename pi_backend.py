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
import math
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
    def save_posture_record(self, record: dict) -> dict:
        """A stable submissionId makes a retry safe after an ambiguous timeout."""
        if not isinstance(record, dict) or type(record.get("studentId")) is not int or record["studentId"] <= 0:
            raise BackendError("자세 기록에는 양의 정수 학생 ID가 필요합니다.")
        fields = ("uprightSamples", "slouchedSamples", "headDownSamples", "tiltedSamples",
                  "unknownSamples", "noPersonSamples", "multiplePeopleSamples")
        counts = [record.get(field) for field in fields]
        total = record.get("totalSamples")
        if any(type(value) is not int or not 0 <= value <= 86400 for value in counts):
            raise BackendError("자세 표본 수는 0~86400의 정수여야 합니다.")
        if type(total) is not int or not 1 <= total <= 86400 or sum(counts) != total:
            raise BackendError("자세 표본 수의 합이 전체 표본 수와 일치해야 합니다.")
        if record.get("inputSource") not in ("CAMERA", "IMAGE", "VIDEO"):
            raise BackendError("자세 입력 출처가 올바르지 않습니다.")
        duration = record.get("durationSeconds")
        if type(duration) is not int or not 1 <= duration <= 86400:
            raise BackendError("자세 측정 시간은 1~86400초의 정수여야 합니다.")
        body = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")
        try:
            saved = self._request("POST", "/api/posture-records", body=body)
        except (ValueError, TypeError) as error:
            raise BackendError("자세 저장 API 응답 형식을 확인할 수 없습니다.") from error
        if not isinstance(saved, dict) or type(saved.get("id")) is not int or saved["id"] <= 0:
            raise BackendError("자세 저장 API 응답에 유효한 기록 ID가 없습니다.")
        return saved

    def get_posture_records(self, student_id: int, subject: str | None = None) -> list:
        params = {"studentId": student_id}
        if subject:
            params["subject"] = subject
        rows = self._request("GET", "/api/posture-records", params=params)
        if not isinstance(rows, list) or any(not isinstance(row, dict) or row.get("studentId") != student_id for row in rows):
            raise BackendError("학생의 자세 기록 응답을 확인할 수 없습니다.")
        return rows

    def get_wrong_answers(self, student_id: int, subject: str | None = None) -> list:
        """학생의 전체 오답 조회. 과목을 생략하면 전 과목을 반환한다."""
        return self._request("GET", f"/api/wrong-answer/{student_id}",
                             params={"subject": subject} if subject else None)

    def save_ocr_result(self, student_id: int, subject: str, submission_id: str,
                        total_questions: int, correct_count: int, questions: list[dict]) -> dict:
        """문항별 인식 문자열과 정오답을 함께 저장. 같은 submission_id 재전송은 중복 집계하지 않는다.

        questions: questionId, expectedAnswer, recognizedText, similarity, correct,
                   conceptTag, errorType(선택). 오답 기록은 백엔드가 같은 트랜잭션에서 처리한다.
        """
        payload = {"studentId": student_id, "subject": subject, "submissionId": submission_id,
                   "totalQuestions": total_questions, "correctCount": correct_count, "questions": questions}
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return self._request("POST", "/api/ocr-results", body=body)

    def get_ocr_results(self, student_id: int, subject: str | None = None) -> list:
        params = {"studentId": student_id}
        if subject:
            params["subject"] = subject
        return self._request("GET", "/api/ocr-results", params=params)

    def get_ocr_result(self, result_id: int, student_id: int) -> dict:
        return self._request("GET", f"/api/ocr-results/{result_id}", params={"studentId": student_id})

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
                               instructor_feedback: str | None = None, *,
                               focused_ratio: float | None = None, drowsy_ratio: float | None = None,
                               measurement_source: str = "MANUAL") -> dict | None:
        """학습 활동 1건 기록. POST /api/activities (JSON 바디).

        study_date: "YYYY-MM-DD", study_start_time: "HH:MM:SS"
        understanding_score / concentration_score: 1~5점 별점(LearningActivity와 앱이 같은
        척도를 쓴다). 범위를 벗어나면 보내지 않는다 — 0~100을 보내면 Backend 메타인지
        분석(이해도 × 20)이 수백 점이 된다.
        """
        for name, score in (("understanding_score", understanding_score),
                            ("concentration_score", concentration_score)):
            if not 1 <= score <= 5:
                raise BackendError(f"{name}는 1~5여야 합니다: {score}")
        if measurement_source not in ("MANUAL", "CAMERA"):
            raise BackendError("measurement_source는 MANUAL 또는 CAMERA여야 합니다.")
        for name, ratio in (("focused_ratio", focused_ratio), ("drowsy_ratio", drowsy_ratio)):
            if ratio is not None and (not math.isfinite(ratio) or not 0 <= ratio <= 1):
                raise BackendError(f"{name}는 0~1의 유한한 비율이어야 합니다: {ratio}")
        if measurement_source == "CAMERA" and (focused_ratio is None or drowsy_ratio is None):
            raise BackendError("CAMERA 측정에는 focused_ratio와 drowsy_ratio가 모두 필요합니다.")
        if focused_ratio is not None and drowsy_ratio is not None and focused_ratio + drowsy_ratio > 1 + 1e-9:
            raise BackendError("집중·졸음 비율의 합은 1 이하여야 합니다.")
        payload = {
            "studentId": student_id,
            "subject": subject,
            "studyDate": study_date,
            "studyStartTime": study_start_time,
            "studyDurationMinutes": duration_minutes,
            "understandingScore": understanding_score,
            "concentrationScore": concentration_score,
            "focusedRatio": focused_ratio,
            "drowsyRatio": drowsy_ratio,
            "measurementSource": measurement_source,
            "instructorFeedback": instructor_feedback or "",
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self._request("POST", "/api/activities", body=body)

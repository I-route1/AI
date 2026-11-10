# AI
아이루트 맞춤형 학습에 대한 ai 모델

## OCR 결과와 집중도 업로드

`pi_backend.py`는 JWT 인증을 사용해 백엔드에 결과를 저장·조회한다.
`.env.example`의 `BACKEND_URL`, `BACKEND_TOKEN` 또는 로그인 계정, `STUDENT_ID`를 설정한다.
AI 서버 내부 통신의 `X-AI-Key`와 달리 이 경로는 사용자 JWT를 사용한다.

```bash
# 전체 문항의 채점 결과(정답 포함)와 오답을 함께 저장
python ocr/pi_ocr_grade.py --image worksheet.jpg --questions questions.json --upload --student-id 14 --subject 국어

# 같은 채점 결과를 다시 전송할 때는 출력된 제출 ID를 재사용
python ocr/pi_ocr_grade.py --image worksheet.jpg --questions questions.json --upload --student-id 14 --subject 국어 --submission-id <출력된-ID>

# 카메라 측정: 집중·졸음 비율과 CAMERA 출처를 별점과 함께 저장
python rpi/pi_main.py --upload --student-id 14 --subject 수학 --understanding 4
```

OCR은 `POST /api/ocr-results`에 `totalQuestions`, `correctCount`, 문항별 `recognizedText`,
정답·유사도·정오답·개념 태그를 보낸다. 백엔드가 오답도 같은 트랜잭션에서 기록하므로
별도로 오답 API를 호출하지 않는다. 전부 정답인 경우도 저장한다. 동일 제출 ID와 동일
결과 재전송은 중복 저장/집계하지 않으며, 같은 제출 ID로 다른 결과를 보내면 409다.
`questions.json`의 `question_id`는 워크시트 간에도 안정적인 고유 문항 ID 사용을 권장한다.
기존 로컬 출력 형식(`score`, `total`, `questions`)은 유지한다.

```python
from pi_backend import BackendClient

client = BackendClient()
wrong_answers = client.get_wrong_answers(14)          # 전체 과목
korean_wrong = client.get_wrong_answers(14, "국어")
results = client.get_ocr_results(14)                  # 문항별 인식 문자열도 포함
detail = client.get_ocr_result(results[0]["id"], 14) if results else None
```

카메라의 `focusedRatio`, `drowsyRatio`는 얼굴이 검출된 표본 중 각 상태의 비율(0~1)이며
얼굴 미검출은 제외한다. 분류된 표본이 없으면 저장하지 않는다. `measurementSource`는
`CAMERA`로 보내며 `concentrationScore`는 기존 1~5점 환산을 유지한다. 직접 입력하는
클라이언트는 `save_learning_activity()`의 기본 출처 `MANUAL`을 사용할 수 있다.

모델·카메라 없이 업로드 계약과 집계 검증:

```bash
python -m unittest discover -s tests -v
```

백엔드의 상세 계약: `Backend/docs/learning-records-api.md`.

## AI HAT 자세 분석

HailoRT 4.23와 Hailo-8 AI HAT의 YOLOv8 pose HEF를 사용한다. 기본 모델은
`/usr/share/hailo-models/yolov8s_pose_h8.hef`이며 `--hef` 또는 `POSE_HEF_PATH`로 바꿀 수 있다.
Hailo-8L은 해당 장치용 HEF를 지정해야 한다. 얼굴 모델 없이도 독립 실행할 수 있다.

```bash
# 카메라: 학생 자리만 포함하는 정규화 ROI(x,y,width,height), 60초 측정 후 저장
python rpi/pose_main.py --seconds 60 --roi 0.1,0.1,0.8,0.8 --student-id 2 --subject 수학 --upload

# 녹화 영상으로 NPU 추론 확인 (카메라를 열지 않음)
python rpi/pi_main.py --pose-only --input sample.mp4 --max-samples 20 --student-id 2 --subject 수학 --upload

# 전송 실패 시 출력된 JSON을 그대로 재전송: 같은 제출 ID는 중복 저장되지 않음
python rpi/pose_main.py --retry /home/pi/.local/state/i-route/posture/<제출-ID>.json
```

`--input`을 생략하면 카메라를 사용한다. 영상 파일·사진 파일은 각각 `VIDEO`·`IMAGE`,
카메라는 `CAMERA` 출처로 저장한다. `durationSeconds`는 파일의 재생 시간이 아닌 분석 실행 시간이다.
기본 카메라 표본 간격은 1초이며 `--interval`로 조절한다. 세션은 최대 24시간·86,400표본이다.
JSON 결과는 소유자만 읽을 수 있도록 새 파일로 저장하고 기존 파일을 덮어쓰지 않는다.

코코(COCO) 17개 관절에서 코·양 어깨를 사용해 바른 자세, 구부정함, 고개 숙임, 기울어짐을
기하 규칙으로 분류한다. 관절 신뢰도 부족·화면 밖 좌표·퇴화된 어깨 간격은 판단불가로 분리한다.
사람 미검출과 여러 명 검출도 별도 집계하며 학생에게 임의로 배정하지 않는다.
실제 학생 측정 전 카메라 위치와 각도에 맞춘 기준 확인이 필요하다.

`POST /api/posture-records`에 집계만 전송하며 사진·영상은 업로드하지 않는다.
집중도 별점이나 직접 입력 자기평가와는 별도 기록이다. 네 자세 비율의 분모는 판정 가능한
표본 수이고, 판정 표본이 0개이면 비율은 `null`이다. `GET /api/posture-records?studentId=2`로
최신순 조회하며 `subject`, `limit`(기본 50, 최대 100)을 지정할 수 있다.
본인 자녀·소속 학원 학생에 대한 JWT 권한 검사를 거치며, 학부모 집중도 탭과 학원 학생 상세의
패턴 탭에서 과목별로 조회한다.

재전송은 원본 학생·과목·측정 시각·집계·제출 ID를 유지해야 한다. 같은 ID의 다른 내용은
409로 거부한다. 파일로 테스트한 결과에도 원본 출처가 표시되므로 실제 카메라 기록과 구분된다.

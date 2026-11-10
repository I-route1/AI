"""
라즈베리파이5 실행 진입점: 워크시트 이미지 + 문항 정의(JSON) → 채점 결과.

사전 준비 (Pi5에서):
    pip install torch --index-url https://download.pytorch.org/whl/cpu  # ARM CPU 빌드
    pip install pillow
    # ocr/ocr_model_aug.pt(없으면 ocr/ocr_model.pt), ocr/model.py, ocr/decode.py, ocr/inference.py, ocr/grading.py 를 Pi5로 복사

문항 정의 JSON 형식 (questions.json):
[
  {"question_id": "1", "bbox": [120, 340, 480, 420], "expected": "돼지",
   "concept_tag": "받침 있는 명사", "error_type": "MEMORIZATION_GAP"},
  {"question_id": "2", "bbox": [520, 340, 900, 420], "expected": "수박"}
]
bbox는 카메라/스캐너로 찍은 워크시트 이미지 안에서 답안이 적힐 위치를
미리 알고 있어야 한다(고정 템플릿). 자동 검출은 지원하지 않는다.

concept_tag / error_type은 --upload 로 백엔드에 오답을 올릴 때만 쓰인다.
concept_tag가 없으면 --subject 값으로 대체하고, error_type은 생략 가능하다
(백엔드 ErrorType: CONCEPT_GAP / CALCULATION_ERROR / CARELESS_MISTAKE / MEMORIZATION_GAP).

사용법:
    # 채점만
    python ocr/pi_ocr_grade.py --image worksheet.jpg --questions questions.json

    # 채점 전체 결과와 오답을 백엔드에 기록 (.env에 BACKEND_URL과 토큰/계정 필요)
    python ocr/pi_ocr_grade.py --image worksheet.jpg --questions questions.json \
        --upload --student-id 14 --subject 국어
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    from ocr.grading import grade_worksheet
    from ocr.inference import OCRRecognizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="채점할 워크시트 이미지 경로")
    ap.add_argument("--questions", required=True, help="문항 정의 JSON 경로")
    ap.add_argument("--threshold", type=float, default=0.8, help="정답 판정 유사도 임계값")
    ap.add_argument("--ckpt", default=None, help="OCR 모델 체크포인트 경로 (기본: ocr/ocr_model_aug.pt, 없으면 ocr/ocr_model.pt)")
    # 답 칸(bbox)은 보통 글씨보다 넓다. 여백을 남기면 여백 30%에서 정답 인정률이 0.87 → 0.47로
    # 떨어지고, 잘라 내면 0.78로 회복된다(ocr/eval_ocr.py, eval/ocr_eval_trim.json).
    ap.add_argument("--no-trim", action="store_true",
                    help="답 칸의 빈 여백을 잘라 내지 않음 (bbox를 글씨에 딱 맞게 잡은 경우)")
    ap.add_argument("--upload", action="store_true", help="채점 전체 결과와 오답을 백엔드에 기록")
    ap.add_argument("--submission-id", default=None,
                    help="채점 제출 ID (기본: UUID). 같은 결과를 재업로드할 때 같은 ID를 사용")
    ap.add_argument("--student-id", type=int, default=None,
                    help="업로드 대상 학생 ID (기본: .env의 STUDENT_ID)")
    ap.add_argument("--subject", default=None, help="업로드 시 과목명 (예: 국어)")
    args = ap.parse_args()

    if args.upload and not args.subject:
        ap.error("--upload 를 쓰려면 --subject 도 지정해야 합니다.")

    with open(args.questions, encoding="utf-8") as f:
        regions = json.load(f)

    kwargs = {"ckpt_path": args.ckpt} if args.ckpt else {}
    recognizer = OCRRecognizer(trim=not args.no_trim, **kwargs)
    report = grade_worksheet(args.image, regions, recognizer, threshold=args.threshold)

    print(f"채점 결과: {report.score}/{report.total}")
    for r in report.results:
        mark = "O" if r.is_correct else "X"
        print(f"  [{mark}] Q{r.question_id}: 정답='{r.expected}' 인식='{r.recognized}' "
              f"유사도={r.similarity}")

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))

    if args.upload:
        _upload_report(report, regions, args)


def _upload_report(report, regions: list[dict], args) -> None:
    """정답을 포함한 전체 채점 결과 저장. 오답 누적은 백엔드에서 원자적으로 처리한다."""
    import os
    import uuid
    from pi_backend import BackendClient, BackendError

    student_id = args.student_id or (int(os.getenv("STUDENT_ID")) if os.getenv("STUDENT_ID") else None)
    if not student_id:
        print("[업로드] 학생 ID가 없습니다 — --student-id 또는 .env의 STUDENT_ID를 설정하세요.")
        return

    if not report.results:
        print("[업로드] 채점한 문항이 없습니다.")
        return
    meta = {str(r.get("question_id")): r for r in regions}
    questions = []
    for result in report.results:
        question_id = str(result.question_id)
        m = meta.get(question_id, {})
        questions.append({
            "questionId": question_id, "expectedAnswer": result.expected,
            "recognizedText": result.recognized, "similarity": result.similarity,
            "correct": result.is_correct, "conceptTag": m.get("concept_tag") or args.subject,
            "errorType": m.get("error_type"),
        })
    submission_id = args.submission_id or str(uuid.uuid4())
    print(f"[업로드] 제출 ID: {submission_id} (재전송 시 --submission-id에 사용)")
    try:
        saved = BackendClient().save_ocr_result(student_id, args.subject, submission_id,
                                               report.total, report.score, questions)
        print(f"[업로드] 채점 결과 #{saved['id']} 저장 — {report.score}/{report.total}, 오답 함께 기록")
    except BackendError as e:
        print(f"[업로드] 실패: {e}")


if __name__ == "__main__":
    main()

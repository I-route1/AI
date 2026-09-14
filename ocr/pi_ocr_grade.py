"""
라즈베리파이5 실행 진입점: 워크시트 이미지 + 문항 정의(JSON) → 채점 결과.

사전 준비 (Pi5에서):
    pip install torch --index-url https://download.pytorch.org/whl/cpu  # ARM CPU 빌드
    pip install pillow
    # ocr/ocr_model.pt, ocr/model.py, ocr/decode.py, ocr/inference.py, ocr/grading.py 를 Pi5로 복사

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

    # 채점 후 오답을 백엔드에 기록 (.env에 BACKEND_URL과 토큰/계정 필요)
    python ocr/pi_ocr_grade.py --image worksheet.jpg --questions questions.json \
        --upload --student-id 14 --subject 국어
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ocr.grading import grade_worksheet
from ocr.inference import OCRRecognizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="채점할 워크시트 이미지 경로")
    ap.add_argument("--questions", required=True, help="문항 정의 JSON 경로")
    ap.add_argument("--threshold", type=float, default=0.8, help="정답 판정 유사도 임계값")
    ap.add_argument("--ckpt", default=None, help="OCR 모델 체크포인트 경로 (기본: ocr/ocr_model.pt)")
    ap.add_argument("--upload", action="store_true", help="채점 후 오답을 백엔드에 기록")
    ap.add_argument("--student-id", type=int, default=None,
                    help="업로드 대상 학생 ID (기본: .env의 STUDENT_ID)")
    ap.add_argument("--subject", default=None, help="업로드 시 과목명 (예: 국어)")
    args = ap.parse_args()

    if args.upload and not args.subject:
        ap.error("--upload 를 쓰려면 --subject 도 지정해야 합니다.")

    with open(args.questions, encoding="utf-8") as f:
        regions = json.load(f)

    kwargs = {"ckpt_path": args.ckpt} if args.ckpt else {}
    recognizer = OCRRecognizer(**kwargs)
    report = grade_worksheet(args.image, regions, recognizer, threshold=args.threshold)

    print(f"채점 결과: {report.score}/{report.total}")
    for r in report.results:
        mark = "O" if r.is_correct else "X"
        print(f"  [{mark}] Q{r.question_id}: 정답='{r.expected}' 인식='{r.recognized}' "
              f"유사도={r.similarity}")

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))

    if args.upload:
        _upload_wrong_answers(report, regions, args)


def _upload_wrong_answers(report, regions: list[dict], args) -> None:
    """오답만 골라 백엔드에 기록. 네트워크 실패로 채점 결과 출력까지 날리지 않도록
    예외를 삼키고 요약만 남긴다 (채점 결과는 이미 stdout에 찍혀 있다)."""
    import os
    from pi_backend import BackendClient, BackendError

    student_id = args.student_id or (int(os.getenv("STUDENT_ID")) if os.getenv("STUDENT_ID") else None)
    if not student_id:
        print("[업로드] 학생 ID가 없습니다 — --student-id 또는 .env의 STUDENT_ID를 설정하세요.")
        return

    meta = {r.get("question_id"): r for r in regions}
    wrong = [r for r in report.results if not r.is_correct]
    if not wrong:
        print("[업로드] 오답이 없어 전송할 내용이 없습니다.")
        return

    try:
        client = BackendClient()
    except BackendError as e:
        print(f"[업로드] 설정 오류로 건너뜁니다: {e}")
        return

    sent = failed = 0
    for r in wrong:
        m = meta.get(r.question_id, {})
        try:
            client.record_wrong_answer(
                student_id=student_id,
                subject=args.subject,
                question_id=r.question_id,
                concept_tag=m.get("concept_tag") or args.subject,
                error_type=m.get("error_type"),
            )
            sent += 1
        except BackendError as e:
            failed += 1
            print(f"[업로드] Q{r.question_id} 실패: {e}")

    print(f"[업로드] 오답 {len(wrong)}건 중 {sent}건 기록, {failed}건 실패")


if __name__ == "__main__":
    main()

"""
라즈베리파이5 실행 진입점: 워크시트 이미지 + 문항 정의(JSON) → 채점 결과.

사전 준비 (Pi5에서):
    pip install torch --index-url https://download.pytorch.org/whl/cpu  # ARM CPU 빌드
    pip install pillow
    # ocr/ocr_model.pt, ocr/model.py, ocr/decode.py, ocr/inference.py, ocr/grading.py 를 Pi5로 복사

문항 정의 JSON 형식 (questions.json):
[
  {"question_id": "1", "bbox": [120, 340, 480, 420], "expected": "돼지"},
  {"question_id": "2", "bbox": [520, 340, 900, 420], "expected": "수박"}
]
bbox는 카메라/스캐너로 찍은 워크시트 이미지 안에서 답안이 적힐 위치를
미리 알고 있어야 한다(고정 템플릿). 자동 검출은 지원하지 않는다.

사용법:
    python ocr/pi_ocr_grade.py --image worksheet.jpg --questions questions.json
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
    args = ap.parse_args()

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


if __name__ == "__main__":
    main()

"""OCR 인식 결과와 정답을 비교해 채점. OCR 자체에 약 12% CER이 있으므로
완전일치 대신 편집거리 기반 유사도 임계값으로 판정한다.
"""
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ocr.decode import similarity
from ocr.inference import OCRRecognizer


@dataclass
class QuestionResult:
    question_id: str
    expected: str
    recognized: str
    similarity: float
    is_correct: bool


@dataclass
class GradingReport:
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def score(self) -> int:
        return sum(r.is_correct for r in self.results)

    @property
    def total(self) -> int:
        return len(self.results)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "total": self.total,
            "questions": [r.__dict__ for r in self.results],
        }


def grade_answer(recognized: str, expected: str, threshold: float = 0.8) -> tuple[bool, float]:
    """공백 제거 후 유사도 비교. threshold 이상이면 정답으로 판정."""
    r = recognized.replace(" ", "")
    e = expected.replace(" ", "")
    sim = similarity(r, e)
    return sim >= threshold, sim


def grade_worksheet(
    image: Image.Image | Path | str,
    regions: list[dict],
    recognizer: OCRRecognizer,
    threshold: float = 0.8,
) -> GradingReport:
    """
    image: 워크시트 전체 이미지 (이미 스캔/촬영된 것)
    regions: [{"question_id": str, "bbox": [x0,y0,x1,y1], "expected": str}, ...]
        bbox는 호출자가 미리 정의(고정 템플릿 좌표 등)해서 넘겨줘야 함 —
        이 모듈은 "어디에 답이 있는지 찾는" 검출은 하지 않는다.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    crops = [image.crop(tuple(r["bbox"])) for r in regions]
    recognized_texts = recognizer.recognize_batch(crops) if crops else []

    report = GradingReport()
    for region, recognized in zip(regions, recognized_texts):
        is_correct, sim = grade_answer(recognized, region["expected"], threshold)
        report.results.append(QuestionResult(
            question_id=region.get("question_id", ""),
            expected=region["expected"],
            recognized=recognized,
            similarity=round(sim, 3),
            is_correct=is_correct,
        ))
    return report

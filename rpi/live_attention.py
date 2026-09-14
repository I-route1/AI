"""
라즈베리파이5용: dlib 68점 랜드마크 검출기로 실시간 프레임에서
학습 때(rpi/dataset.py)와 동일한 형식의 특징 벡터를 만들어 집중도 분류.

사전 준비 (Pi5에서):
    sudo apt install -y build-essential cmake libopenblas-dev liblapack-dev
    pip install dlib opencv-python joblib
    # dlib 68점 랜드마크 모델 다운로드 (dlib 공식 배포 파일)
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2
    # 위 .dat 파일을 이 스크립트와 같은 폴더에 둘 것
"""
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from rpi.dataset import CATEGORY_NAMES, extract_features

MODEL_PATH = Path(__file__).parent / "attention_model.joblib"
LANDMARK_MODEL_PATH = Path(__file__).parent / "shape_predictor_68_face_landmarks.dat"


class AttentionEstimator:
    def __init__(self, model_path: Path = MODEL_PATH, landmark_path: Path = LANDMARK_MODEL_PATH):
        import dlib  # Pi5 전용 의존성 — 개발 PC엔 미설치

        if not landmark_path.exists():
            raise FileNotFoundError(
                f"랜드마크 모델이 없습니다: {landmark_path}\n"
                "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 에서 받아 압축 해제하세요."
            )
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.model_name = bundle["model_name"]

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(landmark_path))

    def estimate(self, gray_frame: np.ndarray) -> tuple[str, dict] | None:
        """gray_frame: OpenCV로 읽은 그레이스케일 프레임. 얼굴 미검출 시 None."""
        faces = self.detector(gray_frame, 0)
        if not faces:
            return None

        # 가장 큰 얼굴 하나만 사용 (여러 명 동시 지원은 추후 확장)
        face = max(faces, key=lambda f: f.width() * f.height())
        shape = self.predictor(gray_frame, face)
        pts = [[shape.part(i).x, shape.part(i).y] for i in range(68)]
        box = [[face.left(), face.top()], [face.right(), face.bottom()]]

        img_like = {
            "face_points": pts,
            "face_box": box,
            "skeleton": None,
            "height": gray_frame.shape[0],
        }
        feat = extract_features(img_like)
        if feat is None:
            return None

        pred = self.model.predict(feat.reshape(1, -1))[0]
        proba = self.model.predict_proba(feat.reshape(1, -1))[0]
        label = CATEGORY_NAMES[pred]
        return label, {name: float(p) for name, p in zip(CATEGORY_NAMES, proba)}

"""
AI-Hub "학습태도 및 성향 관찰 데이터" 라벨(JSON, zip 압축)에서
얼굴 랜드마크 기반 특징 벡터 + 집중도 라벨을 추출한다.

원본 영상(01.원천데이터)은 없고 라벨(02.라벨링데이터)만 존재하므로,
라벨 JSON에 이미 포함된 face_points(68점)를 그대로 특징으로 쓴다.
"""
import json
import random
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np

# AI-Hub category id → 0-indexed 클래스
CATEGORY_MAP = {1: 0, 2: 1, 3: 2}
CATEGORY_NAMES = ["집중", "집중하지않음", "졸음"]

FEATURE_DIM = 68 * 2 + 2  # 정규화된 68점 좌표 + face_box 종횡비 + skeleton box 종횡비


def extract_features(img: dict) -> np.ndarray | None:
    """이미지 라벨 dict에서 정규화된 특징 벡터 추출. 데이터 결측 시 None."""
    pts = img.get("face_points")
    box = img.get("face_box")
    if not pts or not box or len(pts) != 68 or len(box) != 2:
        return None

    pts = np.asarray(pts, dtype=np.float32)
    box = np.asarray(box, dtype=np.float32)
    box_min, box_max = box[0], box[1]
    box_wh = box_max - box_min
    if np.any(box_wh <= 1):
        return None

    # face_box 기준 [0,1] 정규화 → 위치/스케일 불변
    norm_pts = (pts - box_min) / box_wh
    face_aspect = box_wh[0] / box_wh[1]

    skeleton = img.get("skeleton")
    if skeleton and len(skeleton) == 2:
        sk = np.asarray(skeleton, dtype=np.float32)
        sk_w = abs(sk[0][0] - sk[1][0])
        sk_h = img.get("height") or 1
        skeleton_aspect = sk_w / max(sk_h, 1)
    else:
        skeleton_aspect = 0.0

    return np.concatenate([norm_pts.flatten(), [face_aspect, skeleton_aspect]]).astype(np.float32)


def _iter_zip_labels(zip_path: Path, sample: int | None, rng: random.Random) -> Iterator[dict]:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if sample is not None and sample < len(names):
            names = rng.sample(names, sample)
        for name in names:
            try:
                yield json.loads(zf.read(name).decode("utf-8"))
            except Exception:
                continue


def load_dataset(zip_dir: Path, sample_per_zip: int | None = 8000, seed: int = 42):
    """zip_dir 안의 모든 *.zip에서 특징/라벨 배열을 만든다."""
    rng = random.Random(seed)
    zip_paths = sorted(Path(zip_dir).glob("*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"zip 파일을 찾을 수 없습니다: {zip_dir}")

    X, y = [], []
    for zp in zip_paths:
        n_before = len(X)
        for data in _iter_zip_labels(zp, sample_per_zip, rng):
            img = data.get("이미지", {})
            cat = img.get("category", {}).get("id")
            if cat not in CATEGORY_MAP:
                continue
            feat = extract_features(img)
            if feat is None:
                continue
            X.append(feat)
            y.append(CATEGORY_MAP[cat])
        print(f"  [{zp.name}] +{len(X) - n_before}건 (누적 {len(X)})")

    return np.stack(X), np.asarray(y, dtype=np.int64)

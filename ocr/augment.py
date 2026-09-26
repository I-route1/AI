"""OCR 이미지 변형 — 평가(ocr/eval_ocr.py)와 추가 학습(ocr/train_ocr.py --augment)이 함께 쓴다.

학습 데이터는 300dpi 스캔을 글씨에 딱 맞게 자른 crop이고, 실제 입력은 Pi 카메라 사진에서 답 칸을
잘라 낸 것이다. 흐림·저해상도·기울기·원근·조명·노이즈·JPEG·답 칸 여백/어긋남으로 그 차이를 흉내 낸다.
변형은 crop을 SCALE배로 키운 뒤 적용하고, 호출부가 높이 32로 줄인다(원본 사진에서 잘라 줄이는
실제 경로에 가깝게). 변형 함수는 모두 (im, rng, **설정) 모양의 모듈 최상위 함수다 — 윈도우의
DataLoader 워커는 spawn이라 lambda·클로저를 넘기지 못한다(pickle 불가).
"""
import io
import random
from functools import partial

import numpy as np
from PIL import Image, ImageFilter

SCALE = 3        # 변형을 적용할 배율


# 변형은 (im, rng, **설정) 모양의 모듈 최상위 함수 + functools.partial로 만든다. 윈도우의
# DataLoader 워커는 spawn이라 lambda·클로저를 넘기지 못한다(pickle 불가).

def _rotate(im, rng, deg):
    return im.rotate(rng.uniform(-deg, deg), resample=Image.BILINEAR, expand=True, fillcolor=255)


def _blur(im, rng, sigma):
    return im.filter(ImageFilter.GaussianBlur(sigma * SCALE))


def _lowres(im, rng, height):
    # 카메라가 멀어 글자 높이가 height px밖에 안 되는 경우
    w = max(1, round(im.width * height / im.height))
    return im.resize((w, height), Image.BILINEAR).resize(im.size, Image.BILINEAR)


def _perspective(im, rng, strength):
    # 위아래 폭이 다른 사다리꼴 — 카메라가 비스듬히 내려다볼 때
    w, h = im.size
    d = strength * w * rng.uniform(0.5, 1.0) * rng.choice([-1, 1])
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(max(0, d), 0), (w - max(0, d), 0), (w - max(0, -d), h), (max(0, -d), h)]
    return im.transform((w, h), Image.PERSPECTIVE, _perspective_coeffs(dst, src),
                        Image.BILINEAR, fillcolor=255)


def _perspective_coeffs(pa, pb):
    m = []
    for (x, y), (u, v) in zip(pa, pb):
        m.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        m.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.array(m, dtype=np.float64)
    b = np.array(pb, dtype=np.float64).reshape(8)
    return np.linalg.solve(a, b).tolist()


def _dim(im, rng, contrast, brightness):
    # 어두운 조명: 대비를 줄이고 전체를 어둡게
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = ((a - 0.5) * contrast + 0.5) * brightness
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def _shadow(im, rng, depth):
    # 한쪽이 어두운 그림자(손·스탠드) — 가로 방향 선형 그라데이션
    a = np.asarray(im, dtype=np.float32) / 255.0
    g = np.linspace(1.0, 1.0 - depth, a.shape[1], dtype=np.float32)
    if rng.random() < 0.5:
        g = g[::-1]
    return Image.fromarray((np.clip(a * g[None, :], 0, 1) * 255).astype(np.uint8))


def _noise(im, rng, std):
    a = np.asarray(im, dtype=np.float32) / 255.0
    n = np.random.default_rng(rng.randrange(1 << 30)).normal(0, std, a.shape).astype(np.float32)
    return Image.fromarray((np.clip(a + n, 0, 1) * 255).astype(np.uint8))


def _jpeg(im, rng, q):
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    return Image.open(io.BytesIO(buf.getvalue())).convert("L")


def _pad(im, rng, ratio):
    # 템플릿 답 칸이 글씨보다 넓은 경우 — 좌우·위아래 흰 여백
    w, h = im.size
    px, py = round(w * ratio), round(h * ratio)
    out = Image.new("L", (w + 2 * px, h + 2 * py), 255)
    out.paste(im, (px, py))
    return out


def _clip(im, rng, ratio):
    # 답 칸이 어긋나 글씨 양 끝이 잘린 경우
    w, h = im.size
    cut = round(w * ratio)
    return im.crop((cut, 0, max(cut + 1, w - cut), h))


def _chain(im, rng, fs):
    for g in fs:
        im = g(im, rng)
    return im


P = partial


def random_camera(im, rng: random.Random):
    """추가 학습용 무작위 변형 하나. 흐림·저해상도를 중심으로(요청 사항이자 평가에서 가장 크게
    떨어진 두 조건) 다른 변형을 가볍게 섞는다. 강도는 평가 조건(흐림 σ1·2, 글자 높이 10·16px)을
    덮도록 범위로 뽑는다. im은 SCALE배로 키운 흑백 이미지."""
    fs = []
    r = rng.random()
    if r < 0.4:
        fs.append(partial(_blur, sigma=rng.uniform(0.5, 2.2)))
    elif r < 0.8:
        fs.append(partial(_lowres, height=rng.randint(8, 20)))
    else:
        fs += [partial(_blur, sigma=rng.uniform(0.5, 1.5)), partial(_lowres, height=rng.randint(10, 20))]
    if rng.random() < 0.3:
        fs.append(partial(_rotate, deg=6))
    if rng.random() < 0.15:
        fs.append(partial(_perspective, strength=0.12))
    if rng.random() < 0.2:
        fs.append(partial(_pad, ratio=rng.uniform(0.05, 0.3)))
    if rng.random() < 0.2:
        fs.append(partial(_dim, contrast=rng.uniform(0.5, 0.9), brightness=rng.uniform(0.7, 1.0)))
    if rng.random() < 0.2:
        fs.append(partial(_noise, std=rng.uniform(0.02, 0.08)))
    if rng.random() < 0.2:
        fs.append(partial(_jpeg, q=rng.randint(30, 80)))
    rng.shuffle(fs)
    return _chain(im, rng, fs)

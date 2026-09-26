"""OCR 체크포인트 평가: 검증 세트 전체 + 카메라 촬영을 흉내 낸 변형별 강건성.

train_ocr.py의 evaluate()는 학습 중 속도를 위해 검증 50배치만 본다. 여기서는
1) 검증 세트 전체(약 36.5만 crop)의 CER·어절 완전일치율과, 채점 기준(grading.py의
   유사도 0.8)에서 맞게 쓴 답을 정답으로 판정하는 비율을 답 길이별로 잰다.
2) 무작위 표본에 흐림·저해상도·기울기·원근·조명·그림자·노이즈·JPEG·답 칸 어긋남을 넣어
   같은 지표를 잰다. 학습 데이터는 300dpi 스캔인데 실제 입력은 Pi 카메라 사진이라서다.

변형은 crop을 3배(높이 96px)로 키운 뒤 적용하고, inference.py와 같이 높이 32로 줄여
모델에 넣는다 — 원본 사진에서 잘라 줄이는 실제 경로에 가깝게 하려는 것이다.
crop 바깥 문맥이 없어서 기울기·원근·여백은 흰 배경으로 채운다(실제보다 약간 쉬운 쪽).

    python -m ocr.eval_ocr                      # 전체 + 변형 2만 개
    python -m ocr.eval_ocr --robust-n 5000      # 변형 표본 줄이기
    python -m ocr.eval_ocr --skip-full          # 변형만
"""
import argparse
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ocr.augment import (SCALE, P, _blur, _chain, _clip, _dim, _jpeg, _lowres, _noise, _pad,
                         _perspective, _rotate, _shadow)
from ocr.dataset import OCRCropDataset, load_charset
from ocr.decode import greedy_decode, levenshtein, similarity
from ocr.inference import DEFAULT_CKPT, TARGET_HEIGHT, trim_to_ink
from ocr.model import CRNN

CROPS_DIR = Path(__file__).parent / "crops"
THRESHOLD = 0.8  # grading.py / pi_ocr_grade.py 기본값
CONDITIONS = {
    "원본": None,
    "흐림 약(σ1)": P(_blur, sigma=1.0),
    "흐림 강(σ2)": P(_blur, sigma=2.0),
    "저해상도(글자 높이 16px)": P(_lowres, height=16),
    "저해상도(글자 높이 10px)": P(_lowres, height=10),
    "기울기 ±3°": P(_rotate, deg=3),
    "기울기 ±8°": P(_rotate, deg=8),
    "원근(비스듬히 촬영)": P(_perspective, strength=0.15),
    "어두운 조명": P(_dim, contrast=0.5, brightness=0.7),
    "그림자": P(_shadow, depth=0.6),
    "노이즈": P(_noise, std=0.08),
    "JPEG 품질 30": P(_jpeg, q=30),
    "답 칸 넓음(여백 30%)": P(_pad, ratio=0.3),
    "답 칸 어긋남(양끝 8% 잘림)": P(_clip, ratio=0.08),
    "카메라 종합(흐림·기울기·조명·노이즈·JPEG)": P(_chain, fs=(
        P(_blur, sigma=1.0), P(_rotate, deg=3), P(_dim, contrast=0.7, brightness=0.85),
        P(_noise, std=0.04), P(_jpeg, q=50))),
}


class DegradedDataset(Dataset):
    """OCRCropDataset 위에 변형 하나를 씌운다. 표본마다 seed를 고정해 재현 가능."""

    def __init__(self, base: OCRCropDataset, indices: list[int], fn, seed: int, trim: bool = False):
        self.base, self.indices, self.fn, self.seed, self.trim = base, indices, fn, seed, trim

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        e = self.base.index[self.indices[i]]
        f = self.base._file()
        f.seek(e["offset"])
        img = Image.open(io.BytesIO(f.read(e["length"]))).convert("L")
        if self.fn is not None:
            rng = random.Random(self.seed * 1_000_003 + i)
            big = img.resize((img.width * SCALE, img.height * SCALE), Image.BICUBIC)
            img = self.fn(big, rng)
        # inference.OCRRecognizer._preprocess와 같은 전처리(trim=True면 여백 제거 후 크기 조정)
        if self.trim:
            img = trim_to_ink(img)
        new_w = max(1, round(img.width * TARGET_HEIGHT / img.height))
        img = img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0), e["text"]


def _collate(batch):
    imgs, texts = zip(*batch)
    max_w = max(im.shape[2] for im in imgs)
    out = torch.zeros(len(imgs), 1, TARGET_HEIGHT, max_w)
    widths = []
    for i, im in enumerate(imgs):
        out[i, :, :, :im.shape[2]] = im
        widths.append(im.shape[2])
    return out, widths, texts


def _bucket(gt: str) -> str:
    """답 길이 구간. 손으로 쓴 화살표·분수 등은 라벨이 LaTeX 명령(rightarrow, frac{2}{5} 등)이라
    글자 수로 세면 '9자 이상'에 몰린다. 긴 한국어 답과 섞이지 않게 따로 센다."""
    if "\\" in gt:
        return "수식 기호(LaTeX 라벨)"
    n = len(gt)
    return "1~2자" if n <= 2 else "3~4자" if n <= 4 else "5~8자" if n <= 8 else "9자 이상"


def run(model, loader, idx2char, device) -> dict:
    stats = {"n": 0, "cer_num": 0, "cer_den": 0, "exact": 0, "accept": 0, "by_len": {}}
    with torch.no_grad():
        for imgs, widths, texts in loader:
            out = model(imgs.to(device, non_blocking=True))
            lens = torch.tensor([CRNN.output_length(w) for w in widths])
            preds = greedy_decode(out.log_softmax(2).cpu(), lens, idx2char)
            for pred, gt in zip(preds, texts):
                p, g = pred.replace(" ", ""), gt.replace(" ", "")
                acc = similarity(p, g) >= THRESHOLD  # grading.grade_answer와 같은 판정
                stats["n"] += 1
                stats["cer_num"] += levenshtein(pred, gt)
                stats["cer_den"] += max(len(gt), 1)
                stats["exact"] += int(pred == gt)
                stats["accept"] += int(acc)
                b = stats["by_len"].setdefault(_bucket(g), [0, 0])
                b[0] += 1
                b[1] += int(acc)
    n = max(stats["n"], 1)
    return {"n": stats["n"], "cer": stats["cer_num"] / max(stats["cer_den"], 1),
            "exact": stats["exact"] / n, "accept": stats["accept"] / n,
            "accept_by_len": {k: (v[1] / v[0], v[0]) for k, v in sorted(stats["by_len"].items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--robust-n", type=int, default=20000, help="변형 평가 표본 수")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--trim", action="store_true", help="추론 전 여백 제거(inference.trim_to_ink)")
    ap.add_argument("--conditions", default=None, help="쉼표로 구분한 변형 이름 일부만")
    ap.add_argument("--out", default="eval/ocr_eval.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    chars = ckpt["chars"]
    idx2char = {i + 1: c for i, c in enumerate(chars)}
    model = CRNN(num_classes=len(chars) + 1).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    _, char2idx = load_charset(CROPS_DIR)
    base = OCRCropDataset(CROPS_DIR, "validation", char2idx)
    print(f"체크포인트 epoch {ckpt.get('epoch')}, 학습 시 기록 CER {ckpt.get('val_cer'):.4f} "
          f"(검증 50배치 기준), 검증 세트 {len(base):,}개", flush=True)

    def loader(ds):
        return DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                          collate_fn=_collate, pin_memory=True, persistent_workers=False)

    results = {}
    if not args.skip_full:
        t = time.time()
        r = run(model, loader(DegradedDataset(base, list(range(len(base))), None, 0, args.trim)),
                idx2char, device)
        results["검증 전체"] = r
        print(f"\n[검증 전체 {r['n']:,}개, {time.time() - t:.0f}초] CER {r['cer']:.4f}, "
              f"어절 완전일치 {r['exact']:.4f}, 채점 정답 인정({THRESHOLD}) {r['accept']:.4f}", flush=True)
        for k, (v, n) in r["accept_by_len"].items():
            print(f"    {k}: 정답 인정 {v:.4f} ({n:,}개)", flush=True)

    idx = random.Random(42).sample(range(len(base)), min(args.robust_n, len(base)))
    print(f"\n[변형별, 무작위 {len(idx):,}개]", flush=True)
    wanted = set(args.conditions.split(",")) if args.conditions else None
    for i, (name, fn) in enumerate(CONDITIONS.items()):
        if wanted and name not in wanted:
            continue
        t = time.time()
        r = run(model, loader(DegradedDataset(base, idx, fn, seed=i, trim=args.trim)), idx2char, device)
        results[name] = r
        by = ", ".join(f"{k} {v:.2f}" for k, (v, _) in r["accept_by_len"].items())
        print(f"  {name:<28} CER {r['cer']:.4f}  완전일치 {r['exact']:.4f}  정답 인정 {r['accept']:.4f}"
              f"  [{by}]  ({time.time() - t:.0f}초)", flush=True)

    Path(args.out).parent.mkdir(exist_ok=True)
    json.dump({"ckpt_epoch": ckpt.get("epoch"), "threshold": THRESHOLD, "robust_n": len(idx), "trim": args.trim,
               "results": results}, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()

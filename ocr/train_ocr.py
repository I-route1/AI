"""CRNN+CTC로 AI-Hub OCR(edu) crop 데이터 학습.

--augment P: 학습 crop의 P 비율에 카메라 변형(ocr/augment.random_camera — 흐림·저해상도 중심)을
넣는다. 평가에서 흐림 σ2·글자 높이 10px 조건의 채점 정답 인정률이 0.26·0.33으로 떨어져
추가 학습용으로 넣었다(ocr/eval_ocr.py). 이때는 원본 검증 CER과 변형 검증 CER의 평균으로
체크포인트를 고른다 — 원본 성능을 크게 잃고 변형만 좋아지는 것을 막으려는 것이다.

    # 현재 모델에서 이어서, 새 파일로 저장 (기존 ocr_model.pt는 그대로)
    python -m ocr.train_ocr --init ocr/ocr_model.pt --save ocr/ocr_model_aug.pt --augment 0.5 --lr 3e-4 --epochs 3
"""
import argparse
import io
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from ocr.augment import SCALE, random_camera
from ocr.dataset import BLANK_IDX, OCRCropDataset, collate_batch, load_charset
from ocr.decode import greedy_decode, levenshtein
from ocr.model import CRNN

CROPS_DIR = Path(__file__).parent / "crops"
CKPT_PATH = Path(__file__).parent / "ocr_model.pt"


MAX_WIDTH = 320  # prepare_crops.py와 같은 폭 상한


class AugmentedCropDataset(Dataset):
    """OCRCropDataset과 같은 (tensor, label, text)를 주되, prob 비율로 카메라 변형을 넣는다.
    변형은 SCALE배로 키운 뒤 적용하고 높이 32로 줄인다(eval_ocr.DegradedDataset과 같은 경로)."""

    def __init__(self, base: OCRCropDataset, prob: float):
        self.base, self.prob = base, prob

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        if random.random() >= self.prob:
            return self.base[i]
        e = self.base.index[i]
        f = self.base._file()
        f.seek(e["offset"])
        img = Image.open(io.BytesIO(f.read(e["length"]))).convert("L")
        big = img.resize((img.width * SCALE, img.height * SCALE), Image.BICUBIC)
        img = random_camera(big, random)
        new_w = max(1, min(MAX_WIDTH, round(img.width * 32 / img.height)))
        img = img.resize((new_w, 32), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        label = torch.tensor([self.base.char2idx[c] for c in e["text"] if c in self.base.char2idx],
                             dtype=torch.long)
        return torch.from_numpy(arr).unsqueeze(0), label, e["text"]


def evaluate(model, loader, idx2char, device, max_batches: int = 50):
    model.eval()
    total_cer_num, total_cer_den = 0, 0
    exact_correct, total = 0, 0
    with torch.no_grad():
        for bi, (imgs, widths, targets, target_lengths, texts) in enumerate(loader):
            if bi >= max_batches:
                break
            imgs = imgs.to(device)
            out = model(imgs)  # (T, B, C)
            log_probs = out.log_softmax(2)
            input_lengths = torch.tensor([CRNN.output_length(w.item()) for w in widths])
            preds = greedy_decode(log_probs.cpu(), input_lengths, idx2char)
            for pred, gt in zip(preds, texts):
                total_cer_num += levenshtein(pred, gt)
                total_cer_den += max(len(gt), 1)
                exact_correct += int(pred == gt)
                total += 1
    model.train()
    cer = total_cer_num / max(total_cer_den, 1)
    acc = exact_correct / max(total, 1)
    return cer, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true",
                     help="기존 ocr_model.pt 가중치를 불러와 이어서 학습")
    ap.add_argument("--epoch-offset", type=int, default=0,
                     help="로그에 표시할 epoch 번호 오프셋 (이어서 학습 시 이전 epoch 수)")
    ap.add_argument("--init", default=None, help="이 체크포인트 가중치에서 시작 (--resume과 달리 저장 위치와 무관)")
    ap.add_argument("--save", default=str(CKPT_PATH), help="체크포인트 저장 경로")
    ap.add_argument("--augment", type=float, default=0.0, help="카메라 변형을 넣을 학습 crop 비율 (0이면 끔)")
    ap.add_argument("--max-batches", type=int, default=None, help="epoch당 최대 배치 (속도 측정용)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    chars, char2idx = load_charset(CROPS_DIR)
    idx2char = {i: c for c, i in char2idx.items()}
    num_classes = len(chars) + 1  # +blank
    print(f"charset 크기: {len(chars)} (+blank) = {num_classes}")

    train_ds = OCRCropDataset(CROPS_DIR, "training", char2idx)
    val_ds = OCRCropDataset(CROPS_DIR, "validation", char2idx)
    print(f"train {len(train_ds)}건, val {len(val_ds)}건")

    if args.augment > 0:
        train_ds = AugmentedCropDataset(train_ds, args.augment)
        print(f"카메라 변형 비율: {args.augment}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate_batch, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, collate_fn=collate_batch)

    model = CRNN(num_classes=num_classes).to(device)
    best_cer = float("inf")
    save_path = Path(args.save)

    # 변형 학습이면 고정 표본(원본·변형 각 6,400개)으로 고른다. 변형은 표본마다 seed 고정.
    robust_eval = None
    if args.augment > 0:
        from ocr.eval_ocr import DegradedDataset, _collate, run
        idx = random.Random(123).sample(range(len(val_ds)), 6400)
        mk = lambda fn: DataLoader(DegradedDataset(val_ds, idx, fn, seed=7), batch_size=256,
                                   num_workers=args.num_workers, collate_fn=_collate)
        clean_loader, aug_loader = mk(None), mk(random_camera)

        def robust_eval():
            model.eval()
            c, a = run(model, clean_loader, idx2char, device), run(model, aug_loader, idx2char, device)
            model.train()
            return c, a

    if args.init:
        ckpt = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"{args.init}에서 시작 (epoch {ckpt['epoch']}, 기록 val_CER {ckpt['val_cer']:.4f})")
        if robust_eval:
            c, a = robust_eval()
            best_cer = (c["cer"] + a["cer"]) / 2
            print(f"  시작점: 원본 CER {c['cer']:.4f} / 변형 CER {a['cer']:.4f} → 기준 {best_cer:.4f}")
    elif args.resume:
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        best_cer = ckpt["val_cer"]
        print(f"체크포인트에서 이어서 학습: epoch={ckpt['epoch']} val_CER={best_cer:.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=0)
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    for epoch in range(1, args.epochs + 1):
        display_epoch = epoch + args.epoch_offset
        t0 = time.time()
        running_loss, n_batches = 0.0, 0
        for imgs, widths, targets, target_lengths, texts in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            input_lengths = torch.tensor([CRNN.output_length(w.item()) for w in widths])

            optimizer.zero_grad()
            out = model(imgs)  # (T, B, C)
            log_probs = out.log_softmax(2)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            if args.max_batches and n_batches >= args.max_batches:
                break
            if n_batches % 200 == 0:
                print(f"  epoch {display_epoch} batch {n_batches}/{len(train_loader)} "
                      f"loss={running_loss/n_batches:.4f} ({time.time()-t0:.1f}s)")

        lr_now = optimizer.param_groups[0]["lr"]
        if robust_eval:
            c, a = robust_eval()
            cer, acc = (c["cer"] + a["cer"]) / 2, c["exact"]
            print(f"[epoch {display_epoch}] train_loss={running_loss/max(n_batches,1):.4f} "
                  f"원본 CER {c['cer']:.4f}(정답 인정 {c['accept']:.4f}) / 변형 CER {a['cer']:.4f}"
                  f"(정답 인정 {a['accept']:.4f}) → 기준 {cer:.4f} lr={lr_now:.2e} ({time.time()-t0:.1f}s)")
        else:
            cer, acc = evaluate(model, val_loader, idx2char, device)
            print(f"[epoch {display_epoch}] train_loss={running_loss/max(n_batches,1):.4f} "
                  f"val_CER={cer:.4f} val_exact_acc={acc:.4f} lr={lr_now:.2e} ({time.time()-t0:.1f}s)")
        scheduler.step(cer)

        if cer < best_cer:
            best_cer = cer
            torch.save({"model": model.state_dict(), "chars": chars,
                        "val_cer": cer, "val_acc": acc, "epoch": display_epoch,
                        "augment": args.augment}, save_path)
            print(f"  → 체크포인트 저장 (val_CER={cer:.4f})")

    print(f"학습 완료. 최고 기준 CER={best_cer:.4f} → {save_path}")


if __name__ == "__main__":
    main()

"""CRNN+CTC로 AI-Hub OCR(edu) crop 데이터 학습."""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ocr.dataset import BLANK_IDX, OCRCropDataset, collate_batch, load_charset
from ocr.decode import greedy_decode, levenshtein
from ocr.model import CRNN

CROPS_DIR = Path(__file__).parent / "crops"
CKPT_PATH = Path(__file__).parent / "ocr_model.pt"


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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate_batch, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, collate_fn=collate_batch)

    model = CRNN(num_classes=num_classes).to(device)
    best_cer = float("inf")
    if args.resume:
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
            if n_batches % 200 == 0:
                print(f"  epoch {display_epoch} batch {n_batches}/{len(train_loader)} "
                      f"loss={running_loss/n_batches:.4f} ({time.time()-t0:.1f}s)")

        cer, acc = evaluate(model, val_loader, idx2char, device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[epoch {display_epoch}] train_loss={running_loss/max(n_batches,1):.4f} "
              f"val_CER={cer:.4f} val_exact_acc={acc:.4f} lr={lr_now:.2e} ({time.time()-t0:.1f}s)")
        scheduler.step(cer)

        if cer < best_cer:
            best_cer = cer
            torch.save({"model": model.state_dict(), "chars": chars,
                        "val_cer": cer, "val_acc": acc, "epoch": display_epoch}, CKPT_PATH)
            print(f"  → 체크포인트 저장 (val_CER={cer:.4f})")

    print(f"학습 완료. 최고 val_CER={best_cer:.4f} → {CKPT_PATH}")


if __name__ == "__main__":
    main()

"""패킹된 crop 바이너리(ocr/prepare_crops.py 산출물)를 읽는 PyTorch Dataset."""
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

BLANK_IDX = 0


def load_charset(crops_dir: Path) -> tuple[list[str], dict[str, int]]:
    with open(crops_dir / "charset.json", encoding="utf-8") as f:
        chars = json.load(f)
    char2idx = {c: i + 1 for i, c in enumerate(chars)}  # 0은 CTC blank
    return chars, char2idx


class OCRCropDataset(Dataset):
    def __init__(self, crops_dir: Path, split: str, char2idx: dict[str, int]):
        self.bin_path = crops_dir / f"{split}.bin"
        with open(crops_dir / f"{split}_index.json", encoding="utf-8") as f:
            self.index = json.load(f)
        self.char2idx = char2idx
        self._fh = None  # worker별로 지연 오픈 (DataLoader num_workers>0 대응)

    def _file(self):
        if self._fh is None:
            self._fh = open(self.bin_path, "rb")
        return self._fh

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e = self.index[i]
        f = self._file()
        f.seek(e["offset"])
        jpg = f.read(e["length"])
        img = Image.open(io.BytesIO(jpg)).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        label = torch.tensor([self.char2idx[c] for c in e["text"] if c in self.char2idx], dtype=torch.long)
        return tensor, label, e["text"]


def collate_batch(batch):
    imgs, labels, texts = zip(*batch)
    heights = {im.shape[1] for im in imgs}
    assert len(heights) == 1, "모든 crop은 동일한 높이(32)여야 함"
    max_w = max(im.shape[2] for im in imgs)

    padded = torch.zeros(len(imgs), 1, imgs[0].shape[1], max_w)
    input_lengths = torch.zeros(len(imgs), dtype=torch.long)
    for i, im in enumerate(imgs):
        w = im.shape[2]
        padded[i, :, :, :w] = im
        input_lengths[i] = w

    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets = torch.cat(labels) if labels else torch.tensor([], dtype=torch.long)

    return padded, input_lengths, targets, target_lengths, texts

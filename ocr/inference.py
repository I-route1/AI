"""학습된 CRNN 체크포인트로 crop 이미지 → 텍스트 인식. Pi5/개발PC 공용."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ocr.decode import greedy_decode
from ocr.model import CRNN

TARGET_HEIGHT = 32
DEFAULT_CKPT = Path(__file__).parent / "ocr_model.pt"


class OCRRecognizer:
    def __init__(self, ckpt_path: Path = DEFAULT_CKPT, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.chars = ckpt["chars"]
        self.idx2char = {i + 1: c for i, c in enumerate(self.chars)}  # 0은 blank

        self.model = CRNN(num_classes=len(self.chars) + 1).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("L")
        new_w = max(1, round(img.width * TARGET_HEIGHT / img.height))
        img = img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)

    @torch.no_grad()
    def recognize(self, img: Image.Image | Path | str) -> str:
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        tensor = self._preprocess(img).unsqueeze(0).to(self.device)  # (1,1,H,W)
        out = self.model(tensor)  # (T,1,C)
        log_probs = out.log_softmax(2)
        input_length = torch.tensor([out.shape[0]])
        return greedy_decode(log_probs.cpu(), input_length, self.idx2char)[0]

    @torch.no_grad()
    def recognize_batch(self, imgs: list[Image.Image]) -> list[str]:
        """가변폭 이미지들을 배치 패딩해서 한 번에 추론."""
        tensors = [self._preprocess(im) for im in imgs]
        max_w = max(t.shape[2] for t in tensors)
        batch = torch.zeros(len(tensors), 1, TARGET_HEIGHT, max_w)
        widths = []
        for i, t in enumerate(tensors):
            batch[i, :, :, :t.shape[2]] = t
            widths.append(t.shape[2])
        batch = batch.to(self.device)

        out = self.model(batch)  # (T,B,C)
        log_probs = out.log_softmax(2)
        input_lengths = torch.tensor([CRNN.output_length(w) for w in widths])
        return greedy_decode(log_probs.cpu(), input_lengths, self.idx2char)

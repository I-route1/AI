"""CTC 그리디 디코딩 + 문자 오류율(CER) 계산. train_ocr.py와 inference.py가 공유."""
import torch

BLANK_IDX = 0


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """1 - 정규화 편집거리. 완전일치=1.0, 완전불일치일수록 0에 가까움."""
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b), 1)


def greedy_decode(log_probs: torch.Tensor, input_lengths: torch.Tensor, idx2char: dict[int, str]) -> list[str]:
    """log_probs: (T, B, C). 배치별 argmax 후 CTC collapse(연속중복+blank 제거)."""
    preds = log_probs.argmax(dim=2).transpose(0, 1)  # (B, T)
    out = []
    for b in range(preds.shape[0]):
        seq = preds[b, :input_lengths[b]].tolist()
        chars = []
        prev = None
        for idx in seq:
            if idx != prev and idx != BLANK_IDX:
                chars.append(idx2char.get(idx, ""))
            prev = idx
        out.append("".join(chars))
    return out

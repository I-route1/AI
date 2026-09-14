"""CRNN (CNN + BiLSTM + CTC) — 단어/어절 단위 한글 OCR 인식 모델."""
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256):
        super().__init__()
        # 입력: (B, 1, 32, W). 높이를 1로 줄이면서 폭은 보존(초반 2번의 2x2 풀링으로만 폭이 1/4)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),        # H:32->16, W/2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),      # H:16->8,  W/2
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),                                                # H:8->4
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),                                                # H:4->2
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(inplace=True),    # H:2->1
        )
        self.rnn = nn.LSTM(512, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        feat = self.cnn(x)                # (B, 512, 1, W')
        feat = feat.squeeze(2)             # (B, 512, W')
        feat = feat.permute(0, 2, 1)       # (B, W', 512)
        out, _ = self.rnn(feat)            # (B, W', 2*hidden)
        out = self.fc(out)                 # (B, W', num_classes)
        return out.permute(1, 0, 2)        # (W', B, num_classes) — CTC가 요구하는 (T, N, C)

    @staticmethod
    def output_length(input_width: int) -> int:
        """CNN 통과 후 시퀀스 길이(=CTC input_length) 계산.
        폭은 2x2 풀링 2번(/4)과 마지막 2x2 conv(padding=0, -1)에서 줄어듦."""
        w = input_width // 2
        w = w // 2
        w = w - 1
        return max(w, 1)

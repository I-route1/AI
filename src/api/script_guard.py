"""생성문에 다른 문자(중국어·일본어·키릴·힌디 등)가 섞이는 것을 막는다.

실제로 나온 예: "삼각함数", "만드又要", "的一种", "まず", "강대국 बनन", "첫째 лиц".
한국어 교육용 설명이라 학생에게 그대로 보인다.

- Qwen: 생성할 때 그런 문자가 든 토큰을 아예 고르지 못하게 한다(ForeignScriptBlocker).
  사후에 지우면 "삼각함数" → "삼각함"처럼 단어가 부서지므로 생성 단계에서 막는다.
  단, 바이트 단위로 쪼개진 드문 한자는 토큰 하나로는 판별되지 않아 막지 못한다.
- Ollama: API로 토큰을 막을 수 없어서, 섞였으면 한 번 다시 생성하고 그래도 섞이면
  해당 부분만 지운다(counseling._ollama_analyze).

괄호 안의 한자 병기("반어(反語)", "훈민정음(訓民正音)")는 정상 표기라 섞임으로 보지 않는다.
"""
import re

import torch
from transformers import LogitsProcessor

# 한자(CJK 통합·확장 A·호환), 가나, 키릴, 힌디(데바나가리), 태국, 아랍, 히브리 문자.
_FOREIGN_CHARS = ("぀-ヿ㐀-䶿一-鿿豈-﫿"
                  "Ѐ-ӿऀ-ॿ฀-๿؀-ۿ֐-׿")
_FOREIGN = re.compile(f"[{_FOREIGN_CHARS}]")
_FOREIGN_RUN = re.compile(f"[{_FOREIGN_CHARS}]+")
_PAREN_HANJA = re.compile(r"\([㐀-䶿一-鿿豈-﫿·,\s]+\)")


def has_foreign(text: str) -> bool:
    """괄호 안 한자 병기를 뺀 나머지에 다른 문자가 있는가."""
    return bool(_FOREIGN.search(_PAREN_HANJA.sub("", text)))


def strip_foreign(text: str) -> str:
    """괄호 안 한자 병기는 두고, 나머지 다른 문자 덩어리를 지운다. 최후의 수단."""
    kept: list[str] = []

    def _keep(m: "re.Match[str]") -> str:
        kept.append(m.group(0))
        return f"\x00{len(kept) - 1}\x00"

    text = _PAREN_HANJA.sub(_keep, text)
    text = _FOREIGN_RUN.sub("", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: kept[int(m.group(1))], text)
    return re.sub(r"[ \t]{2,}", " ", text)


def foreign_token_mask(tokenizer, vocab_size: int) -> torch.Tensor:
    """다른 문자가 든 토큰 위치가 True인 마스크. 어휘 15만 개를 한 번 디코드한다(수 초)."""
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for i in range(min(len(tokenizer), vocab_size)):
        if _FOREIGN.search(tokenizer.decode([i])):
            mask[i] = True
    return mask


class ForeignScriptBlocker(LogitsProcessor):
    """마스크된 토큰의 점수를 -inf로 만들어 고르지 못하게 한다."""

    def __init__(self, mask: torch.Tensor):
        self.mask = mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.mask.device != scores.device:
            self.mask = self.mask.to(scores.device)
        return scores.masked_fill(self.mask[: scores.shape[-1]], float("-inf"))

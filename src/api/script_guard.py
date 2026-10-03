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
from pathlib import Path

import torch
from transformers import LogitsProcessor

# 한자(CJK 통합·확장 A·호환), 가나, 키릴, 힌디(데바나가리), 태국, 아랍, 히브리 문자.
# 베트남어 글자(라틴 확장 추가 U+1E00~1EFF, ơ·ư)도 넣는다 — 미적분 설명에 "응 dụng 사례"가 섞였다
# (2026-10-03). 수학·과학·영어에서 쓰는 영문자(A-Z)와 그리스 문자는 이 범위 밖이다.
_FOREIGN_CHARS = ("぀-ヿ㐀-䶿一-鿿豈-﫿"
                  "Ѐ-ӿऀ-ॿ฀-๿؀-ۿ֐-׿"
                  "Ḁ-ỿƠơƯư")
_FOREIGN = re.compile(f"[{_FOREIGN_CHARS}]")
_FOREIGN_RUN = re.compile(f"[{_FOREIGN_CHARS}]+")
_PAREN_HANJA = re.compile(r"\([㐀-䶿一-鿿豈-﫿·,\s]+\)")

# 영문자. 위 _FOREIGN에는 안 든다 — 수학(D, x 같은 변수)·과학(DNA, pH, mRNA)·영어 과목은
# 정상적으로 영문자가 필요해서 전 과목에 일괄 적용할 수 없다. 국어·사회처럼 영문자가 나올
# 이유가 없는 과목에서만 latin_token_mask()로 따로 막는다(main._NO_LATIN_SUBJECTS).
# 실제로 섞인 예: "갈Conflict"(갈등), "속belongs합니다"(속한다), 사이시옷 예시의 "ㅆ"이 "sst"로.
_LATIN = re.compile("[A-Za-z]")


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
    """다른 문자가 든 토큰 위치가 True인 마스크. 어휘 15만 개를 한 번 디코드한다(수 초).

    skip_special_tokens=True로 디코드한다 — False(기본값)로 하면 EOS가 "<|im_end|>"로
    나와 "im"/"end"에 영문자가 있다. latin_token_mask가 그걸 몰라서 막았더니 모델이 끝을
    선택 못 해 매번 800토큰(한도)까지 채운 적이 있다. foreign_token_mask는 지금 이 어휘의
    특수 토큰에 한자·가나 등이 없어 우연히 문제가 없었지만, 같은 함정이라 여기도 맞춘다.
    """
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for i in range(min(len(tokenizer), vocab_size)):
        if _FOREIGN.search(tokenizer.decode([i], skip_special_tokens=True)):
            mask[i] = True
    return mask


def latin_token_mask(tokenizer, vocab_size: int) -> torch.Tensor:
    """영문자가 든 토큰 위치가 True인 마스크. foreign_token_mask와 같은 방식이지만
    국어·사회처럼 영문자가 나올 이유가 없는 과목에서만 추가로 쓴다(위 _LATIN 설명).
    skip_special_tokens=True인 이유는 foreign_token_mask 설명 참고 — EOS를 막지 않기 위함."""
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for i in range(min(len(tokenizer), vocab_size)):
        if _LATIN.search(tokenizer.decode([i], skip_special_tokens=True)):
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


# ── 드문 한글 음절 ──────────────────────────────────────────────────────────
# 샘플링 생성에서 "귥한"(극한), "줿게" 같은 깨진 음절이 나왔다(재생성 88개 중 2개, greedy 79개 중 0개).
# 이 음절들은 단일 토큰이 아니라 바이트 조각 두 개로 만들어진다 — "귥" = [89061, 98]. 89061은
# U+ADC0~ADFF 앞 두 바이트라 "극"도 같은 조각에서 시작한다. 반복 벌점(1.2)이 이미 많이 쓴 "극"
# 토큰을 깎으면 모델이 이 바이트 경로로 돌아가고, 마지막 바이트에서 엉뚱한 음절이 완성된다.
#
# 그래서 토큰 하나를 보는 마스크(위 foreign_token_mask)로는 못 막고, 앞에서 만들다 만 바이트가
# 있으면 그걸 완성하는 다음 토큰 중 "교과 문서 70만 개에 한 번도 안 나온 음절"이 되는 것만 막는다.
# 같은 조각에서 "극"을 완성하는 길은 열려 있다. 목록은 scripts/build_hangul_seen.py가 만든다.
# 기준을 "한 번도 안 나옴"으로 잡아 숱·삯·짊 같은 드물지만 실제로 쓰는 음절은 막지 않는다.
_SEEN_HANGUL_PATH = Path(__file__).with_name("hangul_seen.txt")


def _is_rare_hangul(ch: str, seen: frozenset[str]) -> bool:
    return "가" <= ch <= "힣" and ch not in seen


def _token_bytes(tokenizer, vocab_size: int) -> list[bytes | None]:
    """토큰 id → 바이트열. Qwen은 GPT-2식 바이트 수준 BPE라 토큰 문자열을 바이트로 되돌릴 수 있다.
    특수·추가 토큰처럼 되돌릴 수 없는 것은 None."""
    # GPT-2 bytes_to_unicode()와 같은 표. 지금 쓰는 transformers 버전에는 그 함수가 없다.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    dec = {chr(c): b for b, c in zip(bs, cs)}
    # len(tokenizer)는 fast 토크나이저에서 부를 때마다 어휘를 새로 세므로 한 번만 부른다
    strs = tokenizer.convert_ids_to_tokens(list(range(min(len(tokenizer), vocab_size))))
    out: list[bytes | None] = [
        bytes(dec[c] for c in s) if s and all(c in dec for c in s) else None for s in strs
    ]
    return out + [None] * (vocab_size - len(out))


def _pending_prefix(data: bytes) -> bytes:
    """끝에 남은, 아직 완성되지 않은 한글 음절(3바이트, 첫 바이트 EA~ED)의 앞부분."""
    for k in (2, 1):
        tail = data[-k:]
        if len(tail) == k and 0xEA <= tail[0] <= 0xED and all(0x80 <= b <= 0xBF for b in tail[1:]):
            return tail
    return b""


class RareHangulBlocker(LogitsProcessor):
    """드문 한글 음절이 완성되지 못하게 한다. 단일 토큰으로 들어 있는 것은 늘 막고, 바이트 조각으로
    만들다 만 음절은 그걸 완성하는 다음 토큰만 막는다(위 설명)."""

    def __init__(self, tokenizer, vocab_size: int, seen_path: Path = _SEEN_HANGUL_PATH):
        self.seen = frozenset(seen_path.read_text(encoding="utf-8").strip())
        self.tok_bytes = _token_bytes(tokenizer, vocab_size)
        self.always = torch.zeros(vocab_size, dtype=torch.bool)
        self.cont: list[tuple[int, bytes]] = []  # 연속 바이트(0x80~0xBF)로 시작하는 토큰
        for i, b in enumerate(self.tok_bytes):
            if not b:
                continue
            if any(_is_rare_hangul(ch, self.seen) for ch in b.decode("utf-8", errors="ignore")):
                self.always[i] = True
            if 0x80 <= b[0] <= 0xBF:
                self.cont.append((i, b))
        self._by_prefix: dict[bytes, torch.Tensor] = {}

    def _prefix_mask(self, prefix: bytes) -> torch.Tensor:
        if prefix not in self._by_prefix:
            need = 3 - len(prefix)
            mask = self.always.clone()
            for i, b in self.cont:
                if len(b) >= need:
                    ch = (prefix + b[:need]).decode("utf-8", errors="ignore")
                    if len(ch) == 1 and _is_rare_hangul(ch, self.seen):
                        mask[i] = True
            self._by_prefix[prefix] = mask
        return self._by_prefix[prefix]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.always.device != scores.device:
            self.always = self.always.to(scores.device)
            self._by_prefix = {}
        for row in range(scores.shape[0]):
            tail = b"".join(self.tok_bytes[t] or b"" for t in input_ids[row, -3:].tolist())
            prefix = _pending_prefix(tail)
            mask = self._prefix_mask(prefix).to(scores.device) if prefix else self.always
            scores[row] = scores[row].masked_fill(mask[: scores.shape[-1]], float("-inf"))
        return scores

"""LLM 출력 후처리 — 개념 설명의 마크다운·이모지 정리.

main.py에 있던 것을 옮겼다. 평가 스크립트가 서빙과 똑같이 후처리하려면 main.py를
import해야 했는데, 그러면 8B 모델이 통째로 로드된다.
"""
import re


def trim_cut_tail(text: str) -> str:
    """생성 길이 한도에 걸려 문장 중간에서 끝난 경우, 마지막 완성된 줄까지만 남긴다.

    한도(generation.py)는 대부분의 설명이 자연스럽게 끝나도록 잡았지만, 공식을 전부 나열하는
    식으로 1,400자 넘게 길어지는 출력이 드물게 있다("삼각함수 응용 문제"). 그대로 두면
    "5"처럼 번호만 남은 채 끝난다. 남는 부분이 너무 짧아지면 자르지 않는다.
    """
    cut = text.rfind("\n")
    if cut >= len(text) * 0.6:
        return text[:cut].rstrip()
    return text


# 베이스 모델은 마크다운 헤더·이모지·수평선을 섞어 쓴다("## 📌 1. **정의**").
# 어댑터는 그런 걸 쓰지 않아서 지금까지 문제가 없었는데, 수학을 베이스로 돌리면서
# 학생에게 보이는 리포트에 그대로 흘러 들어가게 됐다. writing.py의 _llm_feedback()도
# 같은 이유로 LLM 출력에서 마크다운을 걷어낸다.
_MD_HEADER = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)   # 헤더 기호만 제거, 제목 텍스트는 남긴다
_MD_RULE   = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE)
# 별표는 겹별표(**굵게**, 짝이 안 맞고 남은 ** 포함)만 지운다. 겹별표가 곱셈을
# 뜻하는 경우는 없어서 안전하다.
#
# 홑별표 이탤릭(*was read*)은 일부러 그냥 둔다. 영어 예문에 제법 나오지만,
# 지우려 들면 곱셈 기호를 먹는다:
#     "넓이는 a*b 이고 둘레는 2*(a+b)"  ->  "넓이는 ab 이고 둘레는 2(a+b)"
# 한 줄에 별표가 둘이면 무엇을 해도 이탤릭 쌍과 구분되지 않는다. 남은 별표는
# 보기 사나운 정도지만 수식이 틀리는 것은 내용 오류다. 덜 나쁜 쪽을 택한다.
_MD_BOLD_PAIR     = re.compile(r"\*{2,3}(.+?)\*{2,3}", re.DOTALL)
_MD_BOLD_LEFTOVER = re.compile(r"\*{2,}")
# $...$ 안의 수식은 손대지 않는다. 치환 전에 빼뒀다가 마지막에 되돌린다.
_LATEX_SPAN = re.compile(r"\$[^$\n]*\$")
_EMOJI     = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF️←-⇿⬀-⯿]"
)


# 표: 프론트는 리포트를 평문으로 보여 줘서 "| 질문지법 | 장점 |"과 "|---|---|"가 그대로
# 찍혔다(배포 QA, 33개 개념 중 7개). 행마다 "- 첫 칸: 나머지" 한 줄로 바꾼다.
_MD_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _tables_to_lines(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            out.append(lines[i])
            i += 1
            continue
        block = []
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            block.append(lines[i])
            i += 1
        headers: list[str] = []
        if len(block) >= 2 and _MD_TABLE_SEP.match(block[1]):
            headers = _cells(block[0])
            block = block[2:]
        for row in block:
            if _MD_TABLE_SEP.match(row):
                continue
            cells = [c for c in _cells(row) if c]
            if not cells:
                continue
            if len(cells) == 1:
                out.append(f"- {cells[0]}")
            elif len(cells) == 2 or len(headers) != len(cells):
                out.append(f"- {cells[0]}: " + " / ".join(cells[1:]))
            else:  # 머리글이 있는 3열 이상: "- 질문지법 — 장점: …, 단점: …"
                out.append(f"- {cells[0]} — " + ", ".join(f"{h}: {c}" for h, c in zip(headers[1:], cells[1:])))
    return "\n".join(out)


# 생성이 고장 나 같은 기호를 끝없이 반복한 경우("…?<><><><>…" 수백 자, 배포 QA의
# "역사-인과 관계 분석"). 한글·숫자가 없는 1~4자 단위가 8번 이상 이어지면 통째로 지운다.
# 숫자는 빼야 10000000 같은 수가 안 지워진다.
# 한도에 걸려 단위 중간에서 끊긴 꼬리("…<><")도 단위의 앞부분이면 같이 지운다.
_SYMBOL_RUN = re.compile(r"([^\s가-힣0-9\x00]{1,4}?)\1{7,}([^\s가-힣0-9\x00]{0,3})")


def _drop_run(m: "re.Match[str]") -> str:
    tail = m.group(2)
    return "" if m.group(1).startswith(tail) else tail
# 같은 줄이 연달아 세 번 이상 나오면 한 번만 남긴다(떨어져 반복되는 줄은 정상일 수 있다:
# 지수함수·로그함수 각각의 "a > 1이면 증가함수").
_REPEATED_LINE = re.compile(r"^(.+)(?:\n\1){2,}$", re.MULTILINE)


def strip_markdown(text: str) -> str:
    """개념 설명 출력에서 마크다운·이모지를 걷어내고 평문 문단으로 정리한다."""
    latex: list[str] = []

    def _hide(m: "re.Match[str]") -> str:
        latex.append(m.group(0))
        return f"\x00{len(latex) - 1}\x00"

    text = _LATEX_SPAN.sub(_hide, text)
    text = _tables_to_lines(text)
    text = _SYMBOL_RUN.sub(_drop_run, text)
    text = _REPEATED_LINE.sub(r"\1", text)
    text = _MD_RULE.sub("", text)
    text = _MD_HEADER.sub("", text)
    text = _MD_BOLD_PAIR.sub(r"\1", text)
    text = _MD_BOLD_LEFTOVER.sub("", text)
    text = _EMOJI.sub("", text)
    # 빈 줄이 여러 개 생기므로 문단 구분은 한 줄로 통일하고 줄 끝 공백을 없앤다
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: latex[int(m.group(1))], text)
    return text.strip()


# 베이스 모델은 개념 설명을 챗봇 답장처럼 감싼다. 리포트 한 칸에 들어가는 글이라
# 앞뒤 인사말은 학생에게 엉뚱하게 읽힌다(배포 QA 33개 중 "물론입니다!" 10개,
# "필요하다면 … 알려드릴 수 있어요" 8개).
_OPENER = re.compile(r"^(?:물론입니다|물론이죠|물론이에요|좋습니다|좋아요|네)\s*[!.,]\s*")
# 첫 줄 끝의 안내 문장: "…핵심 포인트는 다음과 같습니다:", "아래는 …을 설명한 내용입니다."
# 내용 문장("이 구조는 다음과 같은 형식을 따릅니다:")을 지우지 않도록 '핵심·설명·정리'
# 같은 말이 같이 있어야 안내 문장으로 본다.
_META_LEAD = re.compile(r"다음과 같|다음은|아래(?:는|에|와)")
_META_WORD = re.compile(r"핵심|포인트|설명|정리|요약")
_CLOSER = re.compile(
    r"^(?:필요하(?:다면|시다면|면)|더 궁금|궁금한 (?:점|것)|추가로 궁금|도움이 되)"
    r".*(?:수 있어요|수 있습니다|드릴게요|드리겠습니다|주세요|바랍니다|좋겠습니다)[.!~]?$"
)


def strip_chat_frame(text: str) -> str:
    """개념 설명의 챗봇식 머리말("물론입니다! 아래는 …")과 맺음말("필요하다면 …")을 걷어낸다.

    strip_markdown() 뒤에 부른다(줄 단위로 정리된 텍스트를 가정). 남는 게 없으면 원문을 돌려준다.
    """
    original = text
    text = _OPENER.sub("", text.lstrip())
    lines = text.split("\n")
    first = lines[0].rstrip()
    if first.endswith((":", "：", ".")):
        # 첫 줄의 마지막 문장만 본다 — "…으로 이루어집니다. 아래에서 이를 설명드리겠습니다:"
        cut = max(first.rfind(". ", 0, len(first) - 1), first.rfind("! ", 0, len(first) - 1))
        head, last = (first[:cut + 1], first[cut + 2:]) if cut >= 0 else ("", first)
        if _META_LEAD.search(last) and _META_WORD.search(last):
            lines[0] = head.rstrip()
            if not lines[0]:
                lines = lines[1:]
    while lines and _CLOSER.match(lines[-1].strip()):
        lines = lines[:-1]
    text = "\n".join(lines).strip()
    return text or original

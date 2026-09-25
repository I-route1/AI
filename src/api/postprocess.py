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


def strip_markdown(text: str) -> str:
    """개념 설명 출력에서 마크다운·이모지를 걷어내고 평문 문단으로 정리한다."""
    latex: list[str] = []

    def _hide(m: "re.Match[str]") -> str:
        latex.append(m.group(0))
        return f"\x00{len(latex) - 1}\x00"

    text = _LATEX_SPAN.sub(_hide, text)
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

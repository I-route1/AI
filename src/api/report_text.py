"""학생에게 노출되는 수치와 문장의 경계를 보존한다."""
import math
import re


def positive_measurement(value, maximum):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) and 0 < number <= maximum else None


# 소수점과 영어 약어(e.g., p.p.) 안에서 끊지 않는다. 숫자로 끝나는 식과
# 문항 번호("1.")를 독립된 문장으로 놓치지 않도록 숫자 뒤 점은 경계에서 뺀다.
_SENTENCE_END = re.compile(r"""(?:[가-힣][.?!。！？]|[A-Za-z]{3,}[.!?]|[!?。！？])["”’')\]]*(?=\s|$)""")


def complete_generation(text: str) -> str:
    """토큰 한도에 도달했을 때만 호출한다. 완결된 앞부분만 학생에게 표시한다."""
    text = text.strip()
    formula = re.fullmatch(r"(?:정답\s*[:：]\s*)?(?P<delimiter>\${1,2})(?P<body>[^$]+)(?P=delimiter)", text)
    if formula:
        body = formula["body"].strip()
        if (body and body.count("{") == body.count("}") and body.count("(") == body.count(")")
                and body[-1] not in "=+-*/^_\\"):
            return text
    ends = list(_SENTENCE_END.finditer(text))
    if ends:
        return text[:ends[-1].end()].strip()
    if re.fullmatch(r"(?:정답\s*[:：]\s*)?(?:[a-zA-Z]\s*=\s*)?[+-]?\d+(?:\.\d+)?(?:\s*/\s*[+-]?\d+)?(?:\s*(?:개|점|명|도|%|cm|kg))?", text):
        return text
    return ""


_MATERIAL_END = re.compile(r"""(?:[가-힣)\]»”’][.!?。！？]|[A-Za-z]{3,}[.!?]|[!?。！？])["”’')\]]*(?=\s|$)""")


def material_excerpt(text: str, limit: int, concept: str = "") -> str:
    """Pick up to two adjacent, complete source sentences around the selected concept."""
    text = text.strip()
    boundaries = [match.end() for match in _MATERIAL_END.finditer(text)]
    if not boundaries:
        return text if len(text) <= limit else ""
    sentences = [text[start:end].strip() for start, end in zip([0] + boundaries, boundaries)]
    target = "".join(concept.split()).casefold()
    index = next((i for i, sentence in enumerate(sentences)
                  if target and target in "".join(sentence.split()).casefold()), 0)
    excerpt = sentences[index]
    if len(excerpt) > limit:
        return ""
    if index + 1 < len(sentences):
        pair = excerpt + " " + sentences[index + 1]
        if len(pair) <= limit:
            return pair
    return excerpt

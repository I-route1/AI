"""Apply the curriculum's comparison criteria to a single, quoted example."""
import re


_QUOTATION = re.compile(r"""“([^”]+)”|‘([^’]+)’|"([^"]+)"|'([^']+)'""")
_NAME_REQUEST = re.compile(r"(?:어떤|무슨|어느)\s*비유법|비유법(?:의)?\s*(?:이름|종류|명칭)|직유(?:법)?(?:인지|와|인가).*은유")
_OTHER_TASK = re.compile(r"효과|정서|심리|분위기|창작|바꿔|바꾸|고쳐|새로|예시|예문|각각|여러|의인|반어|반복|상징|문자\s*그대로|비유.{0,4}아닌|사실.{0,3}비교")
_TOPIC = r"(?P<topic>[가-힣]{1,12}(?:\s+[가-힣]{1,12}){0,2}?)(?:은|는)"
_COMPARISON = re.compile(_TOPIC + r"\s+(?P<image>[가-힣]{1,12})처럼\s+[가-힣 ]{1,35}")
_IDENTITY = re.compile(_TOPIC + r"\s+(?P<image>[가-힣]{1,12}?)(?:이었다|였다|이다)")


def classify_quoted_figure(question: str, references: list[str], answer_only: bool) -> str | None:
    """Handle a comparison requested by name. Leave interpretation and ambiguous identity statements to the tutor."""
    quotes = list(_QUOTATION.finditer(question))
    if len(quotes) != 1:
        return None
    sample = next(part for part in quotes[0].groups() if part).strip().rstrip(".!?。")
    task = _QUOTATION.sub("", question)
    if not _NAME_REQUEST.search(task) or _OTHER_TASK.search(task):
        return None
    if not any("은유법은" in note and "직유법은" in note for note in references):
        return None
    comparison = _COMPARISON.fullmatch(sample)
    if comparison:
        topic, image = comparison["topic"], comparison["image"]
        # '평소처럼' indicates a familiar circumstance. Kinship/role comparisons
        # can also be literal. The marker alone cannot establish a literary device.
        if re.search(r"평소|어제|예전|전날|오늘|어릴때|작년|저번|그때|옛날|엄마|아빠|아버지|어머니|형|누나|언니|선생님|친구|동생", image):
            return None
        reason = f"‘{image}처럼’에서 ‘처럼’이 두 대상을 연결합니다. ‘{topic}’의 모습을 ‘{image}’에 빗댔습니다."
        return "직유법입니다." if answer_only else "직유법입니다. " + reason
    # A literal identity can use exactly the same grammar. Only apply the rule when
    # the exercise itself says a figurative meaning was expressed in this line.
    if not re.search(r"(?:을|를)\s*(?:표현한|나타낸|빗댄|비유한)\s*비유법", task):
        return None
    identity = _IDENTITY.fullmatch(sample)
    if identity:
        reason = f"‘{sample}’에서는 ‘처럼’ 같은 비교 표현을 쓰지 않고 두 대상(‘{identity['topic']}’, ‘{identity['image']}’)을 같은 것으로 나타냈습니다."
        return "은유법입니다." if answer_only else "은유법입니다. " + reason
    return None

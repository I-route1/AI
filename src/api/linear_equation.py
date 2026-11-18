"""한 개의 일차방정식 풀이를 유리수로 검산한다. 다른 의도나 수식은 기존 튜터가 처리한다."""
from fractions import Fraction
import re


_NUMBER = r"(?:\d{1,6}(?:\.\d{1,4})?|\.\d{1,4})"
_TERM = re.compile(rf"(?P<sign>[+-]?)(?:(?:(?P<coefficient>{_NUMBER})\*?)?(?P<var>[A-Za-z])"
                   rf"|(?P<constant>{_NUMBER}))(?:/(?P<denominator>{_NUMBER}))?")
_MATH = r"A-Za-z0-9\s.+\-*/^()\[\]_²³"
_BEFORE_EQUALS = re.compile(rf"[{_MATH}]+$")
_AFTER_EQUALS = re.compile(rf"^[{_MATH}]+")
_REQUEST_WORDS = (
    r"\d+번(?:째)?|문제|문항|다음|아래|주어진|수학|일차방정식|방정식|미지수|식|"
    r"일때|에서|경우|의|과|와|를|을|는|은|가|이|도|만|값|얼마|몇|해|정답|답|"
    r"구하는|구하시오|구하세요|구해|구할|계산|과정|풀이|풀어|풀면|알려|보여|써|적어|설명|"
    r"함께|같이|자세히|주세요|줘요|줘|해줘|해주세요|인가요|인가|예요|야|나요|입니까|"
    r"[:.,?!·~]"
)


def _side(expression: str, variable: str) -> tuple[Fraction, Fraction] | None:
    if re.search(r"(?<=\d)\s+(?=[\d.])", expression):
        return None
    expression = "".join(expression.split())
    coefficient = Fraction()
    constant = Fraction()
    end = 0
    for term in _TERM.finditer(expression):
        if end != term.start() or (end > 0 and not term["sign"]):
            return None
        end = term.end()
        denominator = Fraction(term["denominator"] or 1)
        if not denominator or (term["var"] and term["var"] != variable):
            return None
        value = Fraction(term["constant"] or term["coefficient"] or 1) / denominator
        if term["sign"] == "-":
            value = -value
        if term["var"]:
            coefficient += value
        else:
            constant += value
    return (coefficient, constant) if end and end == len(expression) else None


def _coefficient(value: Fraction, variable: str) -> str:
    if not value:
        return "0"
    sign = {Fraction(1): "", Fraction(-1): "-"}.get(value, str(value))
    return sign + variable if value.denominator == 1 else f"({value}){variable}"


def solve_linear(question: str, *, answer_only: bool = False) -> str | None:
    """해나 풀이를 요구하는 완전한 한 식만 받는다. 문자열을 코드로 실행하지 않는다."""
    text = question.translate(str.maketrans({"−": "-", "×": "*", "·": "*", "÷": "/", "＝": "="}))
    if text.count("=") != 1:
        return None
    before, after = text.split("=")
    left, right = _BEFORE_EQUALS.search(before), _AFTER_EQUALS.match(after)
    if not left or not right:
        return None
    lhs = re.sub(r"^\s*\.\s+", "", left[0])
    rhs = right[0].rstrip().removesuffix(".")
    letters = set(re.findall(r"[A-Za-z]", lhs + rhs))
    if len(letters) != 1:
        return None
    variable = letters.pop()
    request = re.sub(r"\s+", "", before[:left.start()] + after[right.end():])
    if re.sub(rf"{_REQUEST_WORDS}|{variable}", "", request):
        return None
    if not re.search(r"해|풀|값|얼마|몇|답|구하|구해|구할", request):
        return None
    a_side, b_side = _side(lhs, variable), _side(rhs, variable)
    if a_side is None or b_side is None:
        return None
    a, b = a_side
    c, d = b_side
    factor, difference = a - c, d - b
    reduced = f"{_coefficient(factor, variable)} = {difference}"
    if factor == 0:
        result = "모든 실수가 해입니다." if difference == 0 else "해가 없습니다."
        reason = f"양변에서 같은 항을 소거하면 {reduced}이 됩니다."
    else:
        result = f"{variable} = {difference / factor}"
        reason = f"미지수항을 왼쪽에, 상수항을 오른쪽에 모으면 {reduced}입니다."
        if factor != 1:
            reason += f" 양변을 같은 계수로 나누면 {variable} = {difference} ÷ ({factor}) = {difference / factor}입니다."
    return result if answer_only else f"{result}\n{reason}"

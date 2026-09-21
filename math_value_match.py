"""수학 답의 '값' 일치 판정.

왜 필요한가
-----------
71859의 객관식은 문제 본문에 보기가 없어서(보기는 이미지에만 있다) 보기 번호를 맞힐
방법이 없다. 그래서 번호가 아니라 값으로 채점한다. 그런데 단순 문자열 비교는 같은 값을
다른 표기로 쓴 답을 오답 처리한다. 베이스 추론 모드 파일럿에서 실제로 나온 사례:

    정답 '② 12명'        답 '12'        단위만 다름
    정답 '② 1. 44'       답 '1.44'      공백만 다름
    정답 '① x^{2}-8x+25=0'  답 'x² -8x +25'   표기와 '=0'만 다름
    정답 '① \\frac{8}{45}'  답 '8/45'      분수 표기

여기서는 이런 표기 차이를 흡수하되, **과하게 후하지 않도록** 한다. 서로 다른 문항의
답과 정답을 짝지었을 때 일치하는 비율(우연 일치율)을 `chance_match_rate`로 잴 수 있다.

기존 test_math_adapter_eval.py의 `strip_choice`는 모델 답에 쓰면 안 된다. 값이 `1.44`처럼
숫자+점으로 시작하면 앞의 `1.`을 보기 번호로 보고 지워 `44`로 만든다. 여기서는 모델 답에서는
맨 앞의 동그라미 숫자만 보기 번호로 취급한다.
"""
import re
from fractions import Fraction

_CIRCLED = "①②③④⑤⑥⑦⑧⑨"
_SUP = str.maketrans({"²": "^2", "³": "^3", "⁴": "^4", "₀": "0", "₁": "1", "₂": "2", "₃": "3"})

_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_SQRT = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_LATEX_WORD = re.compile(r"\\(?:mathrm|mathbf|text|left|right|displaystyle|quad|qquad|,|;|!)")
_DEG = re.compile(r"\^\s*\{?\s*\\circ\s*\}?|\\circ|°|도$")
_TRAIL_UNIT = re.compile(
    r"(?<=[0-9a-z)])(?:[가-힣]+|(?:mm|cm|km|kg|ml|m|g|l|s|h|min)(?:\^?[23])?|%)$", re.I)
_LEAD_VAR = re.compile(r"^[a-zA-Z]=(?=.)")
_TRAIL_ZERO = re.compile(r"=0$")


def strip_leading_choice(s: str, *, lenient_forms: bool) -> str:
    """맨 앞의 보기 번호를 뗀다.

    정답(ref)은 `③ 40`, `(2) x=5`, `3. 12` 같은 여러 표기가 있어 lenient_forms=True.
    모델 답은 동그라미 숫자만 뗀다. `1.44` 같은 값의 앞부분을 번호로 오인하지 않도록.
    """
    s = s.strip()
    if s and s[0] in _CIRCLED:
        return s[1:].strip()
    if lenient_forms:
        return re.sub(r"^\(\s*[1-5]\s*\)\s*|^[1-5]\s*[.)]\s+", "", s).strip()
    return s


def canon(s: str) -> str:
    """표기 차이를 흡수한 비교용 문자열."""
    s = (s or "").translate(_SUP)
    s = s.replace("√", "\\sqrt").replace("π", "\\pi").replace("×", "*").replace("·", "*")
    s = s.replace("÷", "/").replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    s = s.replace("\\pi", "pi").replace("\\%", "%")
    s = _FRAC.sub(lambda m: f"{m.group(1)}/{m.group(2)}", s)
    s = _SQRT.sub(lambda m: f"sqrt{m.group(1)}", s)
    s = s.replace("\\sqrt", "sqrt")
    s = _DEG.sub("", s)
    s = _LATEX_WORD.sub("", s)
    s = re.sub(r"[$\\{}()\[\]~\s,]", "", s).lower()
    s = s.rstrip(".;:")
    s = _TRAIL_ZERO.sub("", s)
    s = _LEAD_VAR.sub("", s)
    s = _TRAIL_UNIT.sub("", s)
    return s


def _as_number(c: str):
    """canon 결과가 정수·소수·분수면 값을 돌려준다. 아니면 None."""
    try:
        if re.fullmatch(r"-?\d+/\d+", c):
            a, b = c.split("/")
            return Fraction(int(a), int(b))
        if re.fullmatch(r"-?\d+(\.\d+)?", c):
            return Fraction(c)
    except (ValueError, ZeroDivisionError):
        return None
    return None


def value_match(ans: str, ref: str) -> bool:
    """모델 답(ans)이 정답(ref)과 같은 값인가."""
    a = canon(strip_leading_choice(ans, lenient_forms=False))
    r = canon(strip_leading_choice(ref, lenient_forms=True))
    if not a or not r:
        return False
    if a == r:
        return True
    na, nr = _as_number(a), _as_number(r)
    return na is not None and nr is not None and na == nr


def chance_match_rate(pairs: list[tuple[str, str]], seed: int = 0) -> float:
    """서로 다른 문항의 (답, 정답)을 섞어 짝지었을 때 일치하는 비율.

    이 값이 높으면 value_match가 과하게 후하다는 뜻이다. 0에 가까워야 한다.
    """
    import random
    rng = random.Random(seed)
    refs = [r for _, r in pairs]
    rng.shuffle(refs)
    hit = sum(1 for (a, _), r in zip(pairs, refs) if value_match(a, r))
    return hit / len(pairs) if pairs else 0.0


if __name__ == "__main__":
    # 파일럿에서 실제로 나온 사례와, 일치하면 안 되는 사례.
    should_match = [
        ("12", "② 12명"), ("1.44", "② 1. 44"), ("x² -8x +25", "① $x^{2}-8x+25=0$"),
        ("8/45", "① $\\frac{8}{45}$"), ("$\\frac{8}{45}$", "① $\\frac{8}{45}$"),
        ("40", "③ $40^{\\circ}$"), ("40°", "③ $40^{\\circ}$"), ("③ 40", "③ $40^{\\circ}$"),
        ("0.5", "② $\\frac{1}{2}$"), ("√5", "① $\\sqrt{5}$"), ("x=5", "② 5"),
        ("12 cm", "② $12 \\mathrm{~cm}$"), ("30%", "③ $30 \\%$"),
    ]
    should_not = [
        ("44", "② 1. 44"), ("-a", "① $-4 a$"), ("25", "① $x^{2}-8x+25=0$"),
        ("12", "② 120"), ("1", "② 12명"), ("40", "③ 45"), ("", "② 5"),
        ("1/2", "② $\\frac{1}{3}$"), ("3", "③ 30"),
    ]
    bad = [(a, r) for a, r in should_match if not value_match(a, r)]
    bad2 = [(a, r) for a, r in should_not if value_match(a, r)]
    print(f"일치해야 하는 {len(should_match)}건 중 실패: {bad or '없음'}")
    print(f"일치하면 안 되는 {len(should_not)}건 중 잘못 일치: {bad2 or '없음'}")

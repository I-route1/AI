"""학생에게 표시할 답변과 자료 사용 여부를 서로 다른 필드로 생성한다."""
import asyncio
from difflib import SequenceMatcher
import json
import logging
import re

from src.api.generation import OLLAMA_CPU, OLLAMA_URL, ollama_request
from src.api.grounding import usable_docs
from src.api.concept_map import search_concept_map
from src.api.postprocess import strip_chat_frame, strip_markdown


# 번호나 지시어와 풀이 요청만 있는 경우. 본문·식·인용문이 하나라도 덧붙으면
# fullmatch가 실패하므로 일반 질문 경로에서 그 내용을 사용한다.
_EXERCISE_REFERENCE = re.compile(
    r"""(?:(?:교과서|문제집|학습지|시험지)(?:의|에서)?\s*)?
        (?:(?:선생님|쌤)(?:이|이요|께서)?\s*(?:주신|내신|내주신)\s*)?
        (?:\d+\s*(?:쪽|페이지)\s*)?
        (?: (?:이|그|저|위|아래|다음)\s*(?:문제|문항|지문|사진|그림|보기)
          | (?:문제\s*)?\d+\s*번(?:째)?\s*(?:문제|문항|보기|선지)? )
        (?:의|에서|에는|은|는|을|를|에)?\s*
        (?:(?:정답|답|풀이|해설|힌트)(?:이|가|을|를|은|는)?\s*)?
        (?:(?:알려|설명해|보여|풀어)\s*(?:줘|줘요|줄래요?|주세요|주실래요)
          | 어떻게\s*(?:풀어(?:요)?|풀죠|푸나요|푸는\s*거예요)
          | (?:뭐|무엇)(?:야|예요|에요|죠|인가요|일까요)?)?
        [.!?\s]*""", re.VERBOSE
)


def missing_exercise(question: str) -> bool:
    return _EXERCISE_REFERENCE.fullmatch(question) is not None


def availability_notice(question: str) -> str | None:
    """Inventory/seat availability is external live data. Given quantities still go to the tutor."""
    if not re.search(r"지금|현재|실시간|방금|우리\s*동네|근처", question):
        return None
    if not re.search(r"매장|점포|편의점|가게|마트|영화관|열차|공연", question):
        return None
    if (re.search(r"\d|만약|가정|비율|확률|계산|왜|원리|방법|어떻게|일\s*때|다면|라면", question)
            or not re.search(r"몇|얼마|있나요|있을까|남았|알려", question)):
        return None
    if "재고" in question:
        return "현재 재고는 해당 매장에 문의하거나 매장 앱에서 확인해 주세요."
    if re.search(r"잔여\s*좌석|남은\s*(?:좌석|자리)", question):
        return "현재 남은 좌석은 해당 예매처의 좌석 현황에서 확인해 주세요."
    return None


def question_prompt(subject: str, question: str, docs: list[str]) -> tuple[str, bool]:
    references = usable_docs(docs)
    notes = "\n".join(references)
    prompt = f"과목: {subject}\n참고 자료: {notes or '없음'}\n\n학생 질문: {question}"
    return prompt, bool(references)


def verify_reported_claim(subject: str, question: str, answer: str) -> tuple[str, bool]:
    """Include the curriculum definition when a student asks whether a claim they heard is true."""
    if not re.search(r"라고\s*들었|다고\s*들었|라던데|다던데|라는\s*말이\s*(?:맞|사실)", question):
        return answer, False
    notes = search_concept_map(subject, question, 1)
    if not notes:
        return answer, False
    # Use a complete short definition from a direct keyword match. Long lists,
    # fragments, and loosely related semantic search hits are poor citations.
    first = re.match(r"^([^\n]{4,140}?[가-힣]\.(?=\s|$))", notes[0])
    if not first:
        return answer, False
    definition = first.group(1)
    lines = answer.splitlines()
    if len(lines) == 2:
        compact = [re.sub(r"[\s.,'’‘!?]+", "", line).casefold() for line in lines]
        symbols = r"[\dA-Za-z+=<>−/-]+|아니|않|없|못"
        if (re.findall(symbols, compact[0]) == re.findall(symbols, compact[1])
                and SequenceMatcher(None, *compact).ratio() >= .88):
            answer = lines[0]
    if definition not in answer:
        answer += "\n교과 자료 · " + definition
    return answer, True


_SYSTEM = (
    "중고등학생의 교과 질문에 한국어로 답하는 교사입니다. "
    "답변에는 결론을 짧게 쓰고, 근거에는 핵심 표현이나 중간 계산식 하나를 씁니다. "
    "두 필드의 내용을 반복하지 않습니다. 사실과 다른 전제는 바로잡습니다. "
    "자료에 없는 수치나 출처를 지어내지 않습니다. "
    "유형은 참고 자료의 원리를 실제 사용하면 '자료', 일반 지식이나 계산으로 풀면 '풀이', "
    "문제 본문이나 조건을 더 받아야 하면 '확인', 학습 외 요청이나 확인할 수 없는 실시간 정보면 '범위밖'입니다."
)


def tutor_instruction(subject: str) -> str:
    if subject == "국어":
        return (_SYSTEM + " 예문의 표현을 자료에 나온 각 표현법의 구분 기준과 대조해 정확한 이름을 씁니다. "
                "한글 음절 블록 하나는 한 글자이며 자모 수와 구분합니다.")
    if subject == "과학":
        return _SYSTEM + " 전제가 틀렸으면 올바른 명칭과 관계를 먼저 밝히고, 그 사실로 관찰 결과를 설명합니다."
    return _SYSTEM


_KINDS = {"자료": "answer", "풀이": "answer", "확인": "clarification", "범위밖": "out_of_scope"}
_SCHEMA = {
    "type": "object",
    "properties": {"근거": {"type": "string", "minLength": 1},
                   "답변": {"type": "string", "minLength": 1},
                   "유형": {"type": "string", "enum": list(_KINDS)}},
    "required": ["근거", "답변", "유형"],
    "additionalProperties": False,
}
_NO_ANSWER = (None, False, "answer")
_THINK_TAG = re.compile(r"</?think\b|<\|(?:im_|think)", re.IGNORECASE)


def _completed_fields(text: str) -> dict[str, str]:
    """잘린 JSON 객체에서 닫는 따옴표까지 파싱된 필드만 보존한다."""
    text = text.lstrip()
    decoder = json.JSONDecoder()
    fields = {}
    prefix = r"\{\s*"
    index = 0
    while match := re.match(prefix + r'"(근거|답변|유형)"\s*:\s*', text[index:]):
        key = match.group(1)
        if key in fields:
            return {}
        try:
            value, index = decoder.raw_decode(text, index + match.end())
        except ValueError:
            break
        if not isinstance(value, str):
            return {}
        fields[key] = value
        prefix = r"\s*,\s*"
    return fields


def read_answer(
    text: str | None, has_reference: bool, *, length_limited: bool = False, answer_only: bool = False
) -> tuple[str | None, bool, str]:
    """학생에게 필요한 결론과 근거를 합친다. 미완성 JSON 본문은 반환하지 않는다."""
    if not isinstance(text, str) or not text.strip():
        return _NO_ANSWER
    try:
        data = json.loads(text)
    except ValueError:
        # 온전한 내용 필드 뒤 마지막 분류 필드/닫는 괄호에서 잘릴 수 있다.
        # JSON 파서가 여전히 거절하는 답변·근거 문자열은 절대 반환하지 않는다.
        if not length_limited and not _THINK_TAG.search(text):
            return _NO_ANSWER
        data = _completed_fields(text)
        data.setdefault("유형", "풀이")
    if not isinstance(data, dict):
        return _NO_ANSWER
    answer, reason, mode = data.get("답변"), data.get("근거"), data.get("유형")
    if not (isinstance(answer, str) and isinstance(reason, str) and isinstance(mode, str) and mode in _KINDS):
        return _NO_ANSWER
    if _THINK_TAG.search(answer) or _THINK_TAG.search(reason):
        return _NO_ANSWER
    answer = strip_chat_frame(strip_markdown(answer))
    reason = strip_chat_frame(strip_markdown(reason))
    if not answer or not reason:
        return _NO_ANSWER
    if not answer_only and mode in ("자료", "풀이") and reason not in answer:
        answer = f"{answer}\n{reason}"
    return answer, has_reference and mode == "자료", _KINDS[mode]


def wants_answer_only(question: str) -> bool:
    if re.search(r"(?:근거|이유|풀이|과정|설명)(?:도|까지)\b", question):
        return False
    return bool(re.search(r"(?:정답만|답만|한\s*단어로만)(?=\s*(?:알려|말해|적어|써|부탁|[.!?]|$))"
                          r"|(?:설명|해설|풀이)\s*없이", question))


async def ollama_answer(subject: str, prompt: str, has_reference: bool, answer_only: bool) -> tuple[str | None, bool, str]:
    import httpx
    from src.api.script_guard import has_foreign, strip_foreign

    payload = ollama_request(prompt, max_tokens=165 if OLLAMA_CPU else 300)
    payload.update(system=tutor_instruction(subject), format=_SCHEMA)
    payload["options"]["temperature"] = 0.1
    try:
        # 두 번의 생성이 시작되더라도 Spring의 50초 제한 안에서 함께 끝낸다.
        async with asyncio.timeout(40), httpx.AsyncClient(timeout=40) as client:
            async def generate(body):
                response = await client.post(OLLAMA_URL, json=body)
                response.raise_for_status()
                data = response.json()
                return data if isinstance(data, dict) else {}

            result = await generate(payload)
            answer, grounded, kind = read_answer(
                result.get("response"), has_reference,
                length_limited=result.get("done_reason") == "length", answer_only=answer_only)
            if not answer:
                logging.info("[AI 질문] JSON 응답 재생성: 종료=%s", result.get("done_reason"))
                answer = await _plain_retry(generate, prompt, answer_only)
        if answer and has_foreign(answer):
            answer = strip_foreign(answer).strip()
        return (answer, grounded, kind) if answer else _NO_ANSWER
    except (httpx.HTTPError, ValueError, TimeoutError) as exc:
        logging.warning("[AI 질문] Ollama 응답 실패: %s", type(exc).__name__)
        return _NO_ANSWER


async def _plain_retry(generate, prompt: str, answer_only: bool) -> str | None:
    instruction = ("학생의 교과 질문에 바로 답하세요. 한국어를 사용하고 영어 예문과 수식은 그대로 씁니다. "
                   "질문의 조건이 부족하면 필요한 조건을 묻고, 확인할 수 없는 사실을 지어내지 않습니다. ")
    instruction += "정답만 짧게 쓰세요." if answer_only else "결론과 그 이유를 한두 문장으로 쓰세요."
    retry = ollama_request(prompt, max_tokens=95 if OLLAMA_CPU else 170)
    retry.update(system=instruction)
    retry["options"]["temperature"] = 0
    result = await generate(retry)
    text = result.get("response")
    if not isinstance(text, str) or not text.strip() or _THINK_TAG.search(text):
        return None
    if text.lstrip().startswith(("{", "```")):
        return None
    if result.get("done_reason") == "length":
        # 한국어 문장이나 영문 마침표 뒤에 남은 미완성 꼬리를 버린다.
        endings = list(re.finditer(r"(?:[가-힣][.!?]|[A-Za-z][!?]|[A-Za-z]{2,}\.)(?=\s|$)", text))
        if not endings:
            return None
        text = text[:endings[-1].end()]
    return strip_chat_frame(strip_markdown(text))

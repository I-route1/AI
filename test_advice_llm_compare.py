"""수학/writing/premium 리포트의 '학습 조언' 문단 — 베이스 Qwen vs Ollama(llama3.1) 비교.

이 세 리포트는 개념을 설명하는 게 아니라 학생 프로필(백분위·학습시간·강사 피드백)을 보고
조언 문장을 만든다. 한국사 개념 설명에서 Ollama가 이겼던 것(MODELS.md)은 사실 정확도
문제였고, 여기는 그런 사실 검증이 필요한 과제가 아니라서 별도로 재본다.

Backend 더미 데이터(GpsDummyDataInitializer) 모양의 학생 4명 프로필로 실제 핸들러가
만드는 것과 같은 프롬프트를 만들어, 같은 프롬프트를 Qwen(main._qwen_advice)과
Ollama(_ollama_analyze) 양쪽에 넣고 소요 시간과 결과를 나란히 보여준다.

main.py를 통째로 import하므로 베이스 모델(약 6GB VRAM)이 실제로 올라간다. Ollama가
로컬에 떠 있어야 한다.

    python test_advice_llm_compare.py
"""
import asyncio
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import src.api.main as main  # noqa: E402  (임포트 시 모델 로드 + registry 등록)
from src.api import model_registry  # noqa: E402
from src.api.routers.counseling import (  # noqa: E402
    _ADVICE_RULES, _feedback_clause, _level_basis, _ollama_analyze,
    _premium_subject, _resolve_concept,
)

PROFILES = [
    {
        "label": "상위권·성실",
        "studentId": 1, "currentKoreanGrade": 92.0, "studyTime": 3.5,
        "studentNote": "복습을 꼼꼼히 하는 편", "instructorFeedback": "서술형 답안의 근거 제시가 우수함",
        "recommendContext": "심화 과정 추천", "weakConcept": "고전시가 어휘", "weakSubject": "국어",
        "subjectPercentile": 92.0,
    },
    {
        "label": "중위권·학습량 부족",
        "studentId": 2, "currentKoreanGrade": 58.0, "studyTime": 0.5,
        "studentNote": "집중 시간이 짧음", "instructorFeedback": "",
        "recommendContext": "개념 이해 집중", "weakConcept": "이차방정식 판별식", "weakSubject": "수학",
        "subjectPercentile": 55.0,
    },
    {
        "label": "하위권·강사 피드백 있음",
        "studentId": 3, "currentKoreanGrade": 32.0, "studyTime": 1.0,
        "studentNote": "기초 어휘가 부족함", "instructorFeedback": "서술형 답안이 짧고 근거가 부족함",
        "recommendContext": "전면 재학습 권장", "weakConcept": "비문학 핵심어 파악", "weakSubject": "국어",
        "subjectPercentile": 30.0,
    },
    {
        "label": "데이터 없음(신규 학생)",
        "studentId": 4, "currentKoreanGrade": 0.0, "studyTime": 0.0,
        "studentNote": "", "instructorFeedback": "",
        "recommendContext": "", "weakConcept": "", "weakSubject": None,
        "subjectPercentile": None,
    },
]


def math_prompt(req: dict) -> str:
    concept = _resolve_concept("수학", req)
    return (
        f"한국 중고등학생이 수학 '{concept}' 개념을 어려워합니다. "
        f"이 개념에서 학생들이 가장 자주 하는 핵심 실수 1가지와 "
        f"그것을 극복하는 구체적인 학습 전략을 2~3문장으로 간결하게 한국어로 답해주세요."
    )


def writing_prompt(req: dict) -> str:
    note = req.get("studentNote", "").strip()
    _, basis = _level_basis("국어", req)
    hours = float(req.get("studyTime") or 1)
    return (
        f"{basis}, 하루 공부 시간 {hours:.0f}시간인 학생입니다. "
        f"학생 특성: '{note or '특이사항 없음'}'.{_feedback_clause(req)} "
        f"이 학생의 국어(문학·비문학·작문) 실력을 올릴 구체적인 공부 방법 2가지를{_ADVICE_RULES}"
    )


def premium_prompt(req: dict) -> str:
    concept = _resolve_concept(_premium_subject(req), req)
    label = (req.get("recommendContext") or "").strip()
    _, basis = _level_basis(_premium_subject(req), req)
    note = req.get("studentNote", "").strip()
    hours = float(req.get("studyTime") or 1)
    direction = f"학습 방향: '{label}', " if label and label != concept else ""
    return (
        f"학생 정보 — {basis}, 하루 공부 {hours:.0f}시간, {direction}"
        f"취약 개념: '{concept or '특정되지 않음'}', 특성: '{note or '없음'}'.{_feedback_clause(req)} "
        f"이 학생이 성적을 올리려면 이번 주에 가장 먼저 해야 할 행동 2가지를{_ADVICE_RULES}"
    )


REPORTS = [("math", math_prompt), ("writing", writing_prompt), ("premium", premium_prompt)]


def main_():
    qwen_advice = model_registry.get("qwen_advice")
    assert qwen_advice, "qwen_advice가 registry에 없습니다 — main.py import를 확인하세요."

    rows = []
    for profile in PROFILES:
        for name, build in REPORTS:
            prompt = build(profile)

            t0 = time.time()
            q_out = qwen_advice(prompt)
            q_time = time.time() - t0

            t0 = time.time()
            o_out = asyncio.run(_ollama_analyze(prompt))
            o_time = time.time() - t0

            rows.append((profile["label"], name, prompt, q_out, q_time, o_out, o_time))
            print(f"\n===== [{profile['label']}] /report/{name}")
            print(f"프롬프트: {prompt}")
            print(f"  Qwen   ({q_time:4.1f}초): {q_out}")
            print(f"  Ollama ({o_time:4.1f}초): {o_out}")

    print("\n===== 소요 시간 요약 =====")
    for label, name, _, _, qt, _, ot in rows:
        print(f"  [{label}] {name}: Qwen {qt:.1f}초 / Ollama {ot:.1f}초")
    q_avg = sum(r[4] for r in rows) / len(rows)
    o_avg = sum(r[6] for r in rows) / len(rows)
    print(f"\n평균: Qwen {q_avg:.1f}초 / Ollama {o_avg:.1f}초")


if __name__ == "__main__":
    main_()

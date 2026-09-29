"""Qwen 생성 실패 → Ollama/규칙 기반 폴백 횟수 카운터(프로세스 메모리, 재기동하면 0).

kind: "advice"(수학·국어·프리미엄 조언 문단) / "concept"(개념 설명)
outcome: 최종 문장을 만든 쪽 — "qwen"(폴백 없음) / "ollama"(Qwen 실패 후 대체) / "none"(둘 다 실패, 규칙 기반 문구만)
"""
import logging
import threading
from collections import Counter

_OUTCOMES = ("qwen", "ollama", "none")
_lock = threading.Lock()
_counts: dict[str, Counter] = {}


def record(kind: str, outcome: str) -> None:
    with _lock:
        c = _counts.setdefault(kind, Counter())
        c[outcome] += 1
        total = sum(c.values())
        fallback = c["ollama"] + c["none"]
    if outcome != "qwen":
        logging.warning(f"[폴백] {kind}: Qwen 실패 → {outcome} (누적 폴백 {fallback}/{total}건)")


def snapshot() -> dict:
    with _lock:
        out = {}
        for kind, c in _counts.items():
            total = sum(c.values())
            fallback = c["ollama"] + c["none"]
            out[kind] = {
                **{o: c[o] for o in _OUTCOMES},
                "total": total,
                "fallback_rate": round(fallback / total, 3) if total else 0.0,
            }
        return out
